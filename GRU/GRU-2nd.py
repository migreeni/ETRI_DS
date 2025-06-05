# 🔁 GRU 모델을 지표별로 따로 학습/추론하는 구조로 수정한 코드
# 각 지표별로 best epoch 및 best model 추적 → 개별 모델로 testset 예측

import os, copy, random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------
# Load Data
# ------------------------
df_zero_filled = pd.read_csv('../data/merged_df_cwj_tozero.csv')
trainset = pd.read_csv('../data/ch2025_metrics_train.csv')
testset  = pd.read_csv('../data/ch2025_submission_sample.csv')

# Preprocess
TARGETS = ['Q1','Q2','Q3','S1','S2','S3']
df_zero_filled['timestamp'] = pd.to_datetime(df_zero_filled['timestamp'])
df_zero_filled['lifelog_date'] = df_zero_filled['timestamp'].dt.date.astype(str)

DROP_COLS = ['timestamp', 'subject_id', 'lifelog_date']
SENSOR_COLS = [c for c in df_zero_filled.columns if c not in DROP_COLS]
MAX_SEQ = 144

def build_sequences(df):
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

def rows_to_xy(df):
    xs, ys, groups = [], [], []
    for _, r in df.iterrows():
        key = (r.subject_id, r.lifelog_date)
        if key not in SEQ_DICT:
            continue
        xs.append(SEQ_DICT[key])
        ys.append(r[TARGETS].to_list())
        groups.append(r.subject_id)
    return np.stack(xs), np.array(ys, np.int64), np.array(groups)

X_all, y_all, group_all = rows_to_xy(trainset)
X_test, _, _ = rows_to_xy(testset)

# Scale
scaler = StandardScaler().fit(X_all.reshape(-1, X_all.shape[-1]))
def scale(x):
    shp = x.shape
    return scaler.transform(x.reshape(-1, shp[-1])).reshape(shp)

X_all = scale(X_all)
X_test = scale(X_test)

# ------------------------
# Dataset / Model
# ------------------------
class SleepDS(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i] if self.y is None else (self.X[i], self.y[i])

class SingleHeadGRU(nn.Module):
    def __init__(self, input_dim, out_dim, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=drop)
        self.fc = nn.Linear(hidden, out_dim)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# ------------------------
# Train per target
# ------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLD = 5
EPOCH_MAX = 50
BATCH_SIZE = 64

preds_dict = {}

for k, target in enumerate(TARGETS):
    print(f"\n=== {target} training ===")
    y_target = y_all[:, k]
    out_dim = 3 if target == 'S1' else 2
    criterion = nn.CrossEntropyLoss()
    best_epochs = []

    gkf = GroupKFold(n_splits=N_FOLD)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_target, group_all)):
        model = SingleHeadGRU(X_all.shape[-1], out_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

        best_f1, best_ep, best_state = -1, 0, None
        for epoch in range(1, EPOCH_MAX+1):
            model.train()
            for xb, yb in DataLoader(SleepDS(X_all[tr_idx], y_target[tr_idx]), batch_size=BATCH_SIZE, shuffle=True):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            # validation
            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for xb, yb in DataLoader(SleepDS(X_all[val_idx], y_target[val_idx]), batch_size=BATCH_SIZE):
                    pred = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                    all_pred.extend(pred)
                    all_true.extend(yb.numpy())
            f1 = f1_score(all_true, all_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_ep = epoch
                best_state = copy.deepcopy(model.state_dict())

        best_epochs.append(best_ep)
        print(f"Fold {fold+1} - Best F1: {best_f1:.4f} @ Epoch {best_ep}")

    # Final train on all data
    final_model = SingleHeadGRU(X_all.shape[-1], out_dim).to(DEVICE)
    opt = torch.optim.AdamW(final_model.parameters(), lr=3e-4, weight_decay=1e-2)
    loader = DataLoader(SleepDS(X_all, y_target), batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(1, int(round(np.mean(best_epochs)))+1):
        final_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = final_model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    # Predict
    final_model.eval()
    preds = []
    with torch.no_grad():
        for xb in DataLoader(SleepDS(X_test), batch_size=BATCH_SIZE):
            out = final_model(xb.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
    preds_dict[target] = preds

# ------------------------
# Submission
# ------------------------
sub = testset[['subject_id','sleep_date','lifelog_date']].copy()
for t in TARGETS:
    sub[t] = preds_dict[t]

SAVE_PATH = '../data/submission2.csv'
sub.to_csv(SAVE_PATH, index=False)
print('✅ submission saved ->', SAVE_PATH)