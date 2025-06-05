# =========================================================
# 0. 라이브러리
# =========================================================
import os, copy, json, random, math
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =========================================================
# 1. 데이터 로드
# =========================================================
df_zero_filled = pd.read_csv('../data/merged_df_cwj_tozero.csv')
trainset = pd.read_csv('../data/ch2025_metrics_train.csv')
testset  = pd.read_csv('../data/ch2025_submission_sample.csv')   # id/date 컬럼만 사용

# ---------------------------------------------------------
# 1-1. 시계열 전처리
# ---------------------------------------------------------
df_zero_filled['timestamp']   = pd.to_datetime(df_zero_filled['timestamp'])
df_zero_filled['lifelog_date'] = df_zero_filled['timestamp'].dt.date.astype(str)

DROP_COLS = ['timestamp', 'subject_id', 'lifelog_date']
SENSOR_COLS = [c for c in df_zero_filled.columns if c not in DROP_COLS]

MAX_SEQ = 144    # 10-min × 24 h 기준

def build_sequences(df):
    """Return dict key->np.ndarray(seq_len, n_feat)"""
    seqs = {}
    for (sid, day), g in df.groupby(['subject_id', 'lifelog_date']):
        g = g.sort_values('timestamp')
        x = g[SENSOR_COLS].to_numpy(np.float32)
        if len(x) > MAX_SEQ: x = x[:MAX_SEQ]
        if len(x) < MAX_SEQ:
            x = np.concatenate([x, np.zeros((MAX_SEQ-len(x), x.shape[1]), np.float32)])
        seqs[(sid, day)] = x
    return seqs

SEQ_DICT = build_sequences(df_zero_filled)

# ---------------------------------------------------------
# 1-2. 학습/추론용 텐서 변환
# ---------------------------------------------------------
TARGETS = ['Q1','Q2','Q3','S1','S2','S3']   # S1 = 3-class, others binary

def rows_to_xy(df):
    xs, ys, groups = [], [], []
    for _, r in df.iterrows():
        key = (r.subject_id, r.lifelog_date)
        if key not in SEQ_DICT:        # missing sequence
            continue
        xs.append( SEQ_DICT[key] )
        ys.append( r[TARGETS].to_list() )
        groups.append( r.subject_id )  # fold split anchor
    return np.stack(xs), np.array(ys, np.int64), np.array(groups)

X_all, y_all, group_all = rows_to_xy(trainset)
X_test, _, _ = rows_to_xy(testset)     # y_dummy ignored

# ---------------------------------------------------------
# 1-3. 스케일링 (training set 기준)
# ---------------------------------------------------------
scaler = StandardScaler().fit(X_all.reshape(-1, X_all.shape[-1]))
def scale(x):
    shp = x.shape
    return scaler.transform(x.reshape(-1, shp[-1])).reshape(shp)

X_all  = scale(X_all)
X_test = scale(X_test)

# =========================================================
# 2. Dataset / DataLoader
# =========================================================
class SleepDS(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        if self.y is None: return self.X[i]
        return self.X[i], self.y[i]

# =========================================================
# 3. GRU 모델 (멀티-헤드)
# =========================================================
class GRUNet(nn.Module):
    def __init__(self, n_feat, hid=128, n_layers=2, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(n_feat, hid, n_layers, batch_first=True, dropout=drop)
        self.heads = nn.ModuleList([
            nn.Linear(hid, 2),   # Q1
            nn.Linear(hid, 2),   # Q2
            nn.Linear(hid, 2),   # Q3
            nn.Linear(hid, 3),   # S1 (3-class)
            nn.Linear(hid, 2),   # S2
            nn.Linear(hid, 2)    # S3
        ])
    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]                 # last layer output (B,H)
        return [head(h) for head in self.heads]   # list of logits

# loss per task
LOSS_FNS = [
    nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
]

# =========================================================
# 4. k-fold validation (subject 그룹 유지)
# =========================================================
N_FOLD     = 5
EPOCH_MAX  = 50
BATCH_SIZE = 64
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

best_epochs = []          # fold별 best epoch 저장

gkf = GroupKFold(n_splits=N_FOLD)
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, group_all), 1):
    print(f'\n=== Fold {fold}/{N_FOLD} ===')
    tr_loader = DataLoader(SleepDS(X_all[tr_idx], y_all[tr_idx]),
                           batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SleepDS(X_all[val_idx], y_all[val_idx]),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = GRUNet(n_feat=X_all.shape[-1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    best_f1, best_ep, best_state = -1, 0, None
    for epoch in range(1, EPOCH_MAX+1):
        # --- train ---
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = sum( LOSS_FNS[k](logits[k], yb[:,k]) for k in range(6) )
            opt.zero_grad(); loss.backward(); opt.step()

        # --- validation ---
        model.eval()
        y_true, y_pred = [[] for _ in range(6)], [[] for _ in range(6)]
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(DEVICE))
                for k in range(6):
                    preds = logits[k].argmax(1).cpu().numpy()
                    y_pred[k].extend(preds)
                    y_true[k].extend(yb[:,k].numpy())
        f1s = [f1_score(y_true[k], y_pred[k], average='macro') for k in range(6)]
        mac_f1 = np.mean(f1s)
        print(f'E{epoch:02d}  F1={mac_f1:.4f}', end='\r')

        if mac_f1 > best_f1:
            best_f1, best_ep = mac_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
    print(f' -> best F1={best_f1:.4f} @ epoch {best_ep}')
    best_epochs.append(best_ep)

# ---------------------------------------------------------
# 4-1. 최적 epoch 결정 (평균 반올림)
# ---------------------------------------------------------
BEST_EPOCH = int(round(np.mean(best_epochs)))
print('\nSelected BEST_EPOCH =', BEST_EPOCH)

# =========================================================
# 5. 전체 trainset으로 재학습 (BEST_EPOCH), test 예측
# =========================================================
full_loader = DataLoader(SleepDS(X_all, y_all),
                         batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SleepDS(X_test),
                         batch_size=BATCH_SIZE, shuffle=False)

model_final = GRUNet(n_feat=X_all.shape[-1]).to(DEVICE)
opt = torch.optim.AdamW(model_final.parameters(), lr=3e-4, weight_decay=1e-2)

for epoch in trange(1, BEST_EPOCH+1, desc='Final-train'):
    model_final.train()
    for xb, yb in full_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model_final(xb)
        loss = sum( LOSS_FNS[k](logits[k], yb[:,k]) for k in range(6) )
        opt.zero_grad(); loss.backward(); opt.step()

# --- inference ---
model_final.eval()
preds_all = [[] for _ in range(6)]
with torch.no_grad():
    for xb in test_loader:
        logits = model_final(xb.to(DEVICE))
        for k in range(6):
            preds_all[k].extend( logits[k].argmax(1).cpu().numpy() )

# =========================================================
# 6. submission 생성
# =========================================================
sub = testset[['subject_id','sleep_date','lifelog_date']].copy()
for k,t in enumerate(TARGETS):
    sub[t] = preds_all[k]

SAVE_PATH = '../data/submission.csv'
sub.to_csv(SAVE_PATH, index=False)
print('✅ submission saved ->', SAVE_PATH)
