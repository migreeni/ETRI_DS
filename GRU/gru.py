# =============================================================
# 0. 공통 import  (gru.py와 동일 + 파일저장용 패키지)
# =============================================================
import pandas as pd, numpy as np, os, json, random, math, gc, warnings, pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)

# =============================================================
# 1. 시계열 → (144, n_feat) 시퀀스 딕셔너리 (gru.py 그대로)
#    └ df_zero_filled : 10분 granular 센서 row DataFrame
# =============================================================
df_zero_filled['timestamp'] = pd.to_datetime(df_zero_filled['timestamp'])
df_zero_filled['lifelog_date'] = df_zero_filled['timestamp'].dt.date.astype(str)

MAX_SEQ_LEN = 144
drop_cols = ["timestamp","subject_id","lifelog_date"] + \
            [c for c in df_zero_filled.columns if c.startswith("id")]
sensor_cols = [c for c in df_zero_filled.columns if c not in drop_cols]

def build_sequences(df):
    seq_dict = {}
    for (sid, day), g in df.groupby(['subject_id','lifelog_date']):
        g = g.sort_values('timestamp')
        x = g[sensor_cols].to_numpy(dtype=np.float32)
        if len(x) > MAX_SEQ_LEN:         x = x[:MAX_SEQ_LEN]
        if len(x) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN-len(x), x.shape[1]), np.float32)
            x   = np.vstack([x, pad])
        seq_dict[(sid, day)] = x
    return seq_dict

sequence_dict = build_sequences(df_zero_filled)
print(f'# sequences  : {len(sequence_dict):,}')

# -------------------------------------------------------------
#  Dataset / Model 정의 (gru.py 그대로 재사용)
# -------------------------------------------------------------
class SleepDS(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        if self.y is None:  return self.X[idx]
        return self.X[idx], self.y[idx]

class GRUModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h = self.gru(x)          # (1,B,H)
        return torch.sigmoid(self.fc(h.squeeze(0)))  # (B,1)

# =============================================================
# 2. metric 별 학습 & 예측
# =============================================================
metrics = ['Q1','Q2','Q3','S1','S2','S3']
submission = testset.copy()          # 템플릿 복사
submission[metrics] = 0              # 값 초기화

for m in metrics:
    print(f'\n=== [{m}] training ===')

    # ------------ (1) train/val set 준비 -------------
    df_tr = globals()[f'trainset_{m.lower()}']      # subject_id, sleep_date, lifelog_date, m
    df_vl = globals()[f'df_val_{m.lower()}']        # validation split (같은 스키마)

    def rows_to_tensors(df):
        X, y = [], []
        for _, r in df.iterrows():
            key = (r['subject_id'], r['lifelog_date'])
            if key not in sequence_dict:  continue
            X.append(sequence_dict[key])
            y.append([r[m]])
        return np.stack(X), np.array(y, np.float32) if y else None

    X_tr, y_tr = rows_to_tensors(df_tr)
    X_vl, y_vl = rows_to_tensors(df_vl)

    scaler = StandardScaler().fit(X_tr.reshape(-1, len(sensor_cols)))
    X_tr = scaler.transform(X_tr.reshape(-1, len(sensor_cols))).reshape(X_tr.shape)
    X_vl = scaler.transform(X_vl.reshape(-1, len(sensor_cols))).reshape(X_vl.shape)

    # ------------ (2) dataloader -------------
    BATCH = 64
    tr_loader = DataLoader(SleepDS(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    vl_loader = DataLoader(SleepDS(X_vl, y_vl), batch_size=BATCH)

    # ------------ (3) model / opt -------------
    model = GRUModel(len(sensor_cols)).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit   = nn.BCELoss()

    # ------------ (4) train loop -------------
    best_f1, BEST = 0, None
    EPOCHS=13
    for ep in range(1,EPOCHS+1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = crit(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()

        model.eval(); preds=[]; trues=[]
        with torch.no_grad():
            for xb, yb in vl_loader:
                out=model(xb.to(device)).cpu().numpy()
                preds.append(out); trues.append(yb.numpy())
        f1 = f1_score(np.vstack(trues), (np.vstack(preds)>=0.5).astype(int))
        if f1>best_f1: best_f1, BEST = f1, model.state_dict()
        print(f'Epoch {ep:02d} | val F1={f1:.4f}', end='\r')
    print(f'  best F1={best_f1:.4f}')

    model.load_state_dict(BEST)      # best 모델 로드

    # =========================================================
    #  (5) testset 예측  -> submission[m] 갱신
    # =========================================================
    X_test, keys = [], []            # keys = (idx_row_in_submission, (sid, lifelog_date))
    for idx, row in submission.iterrows():
        key = (row['subject_id'], row['lifelog_date'])
        if key not in sequence_dict:
            X_test.append(np.zeros((MAX_SEQ_LEN,len(sensor_cols)), np.float32))  # 없는 경우 zero-pad
        else:
            X_test.append(sequence_dict[key])
        keys.append(idx)
    X_test = np.stack(X_test)
    X_test = scaler.transform(X_test.reshape(-1,len(sensor_cols))).reshape(X_test.shape)

    test_loader = DataLoader(SleepDS(X_test), batch_size=BATCH)
    model.eval(); all_pred=[]
    with torch.no_grad():
        for xb in test_loader:
            p = model(xb.to(device)).cpu().numpy()
            all_pred.append(p)
    all_pred = (np.vstack(all_pred)>=0.5).astype(int).flatten()

    # 결과 삽입
    submission.loc[keys, m] = all_pred

# =============================================================
# 3. csv export
# =============================================================
out_path = 'submission.csv'
submission.to_csv(out_path, index=False)
print(f'\n✅ submission saved → {out_path}  (shape={submission.shape})')
