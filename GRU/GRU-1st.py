# 🚀 GRU with Optuna Hyperparameter Tuning (per‑target)
# - Tune hidden size, layers, dropout, learning rate, weight decay, epochs
# - Use GroupKFold inside Optuna objective (validation F1 ↑)
# - Train final model with best params, predict test, save submission.csv
# Requirements: pip install optuna

import os, copy, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna                          # ✨ NEW

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

df_zero_filled = pd.read_csv('../data/merged_df_cwj_tozero.csv')
trainset = pd.read_csv('../data/ch2025_metrics_train.csv')
testset  = pd.read_csv('../data/ch2025_submission_sample.csv')

TARGETS = ['Q1','Q2','Q3','S1','S2','S3']

df_zero_filled['timestamp']   = pd.to_datetime(df_zero_filled['timestamp'])
df_zero_filled['lifelog_date'] = df_zero_filled['timestamp'].dt.date.astype(str)

DROP_COLS   = ['timestamp','subject_id','lifelog_date']
SENSOR_COLS = [c for c in df_zero_filled.columns if c not in DROP_COLS]
MAX_SEQ     = 144   # 10‑min resolution

# ---------- utils ----------

def build_sequences(df):
    seqs = {}
    for (sid, day), g in df.groupby(['subject_id','lifelog_date']):
        g = g.sort_values('timestamp')
        x = g[SENSOR_COLS].astype('float32').to_numpy()
        if len(x) > MAX_SEQ: x = x[:MAX_SEQ]
        if len(x) < MAX_SEQ:
            x = np.concatenate([x, np.zeros((MAX_SEQ-len(x), x.shape[1]), np.float32)])
        seqs[(sid, day)] = x
    return seqs

SEQ_DICT = build_sequences(df_zero_filled)


def rows_to_xy(df):
    xs, ys, groups = [], [], []
    for _, r in df.iterrows():
        k = (r.subject_id, r.lifelog_date)
        if k not in SEQ_DICT:
            continue
        xs.append(SEQ_DICT[k])
        ys.append(r[TARGETS].to_list())
        groups.append(r.subject_id)
    return np.stack(xs), np.array(ys, np.int64), np.array(groups)

X_all, y_all, group_all = rows_to_xy(trainset)
X_test, _, _            = rows_to_xy(testset)

# scale
scaler = StandardScaler().fit(X_all.reshape(-1, X_all.shape[-1]))
X_all  = scaler.transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# ---------- dataset ----------
class SleepDS(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx] if self.y is None else (self.X[idx], self.y[idx])

# ---------- model ----------
class SingleHeadGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden, layers, drop):
        super().__init__()
        self.gru = nn.GRU(inp_dim, hidden, layers, batch_first=True, dropout=drop)
        self.fc  = nn.Linear(hidden, out_dim)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE_DEFAULT = 64
N_FOLD = 5

# --------------------------------------------------
# Optuna tuning per target
# --------------------------------------------------
preds_dict = {}

for idx_target, target in enumerate(TARGETS):
    print(f"\n🎯 Optimizing {target}")
    y_target = y_all[:, idx_target]
    out_dim  = 3 if target == 'S1' else 2
    criterion = nn.CrossEntropyLoss()

    def objective(trial):
        # hyperparameters to tune
        hidden = trial.suggest_int('hidden', 64, 256, step=64)
        layers = trial.suggest_int('layers', 1, 3)
        drop   = trial.suggest_float('drop', 0.1, 0.5, step=0.1)
        lr     = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        wd     = trial.suggest_loguniform('wd', 1e-5, 1e-2)
        epochs = trial.suggest_int('epochs', 10, 40, step=10)

        gkf = GroupKFold(n_splits=N_FOLD)
        f1_scores = []

        for tr_idx, val_idx in gkf.split(X_all, y_target, group_all):
            model = SingleHeadGRU(X_all.shape[-1], out_dim, hidden, layers, drop).to(DEVICE)
            opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            tr_loader  = DataLoader(SleepDS(X_all[tr_idx], y_target[tr_idx]), batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
            val_loader = DataLoader(SleepDS(X_all[val_idx], y_target[val_idx]), batch_size=BATCH_SIZE_DEFAULT)

            # training loop
            for epoch in range(epochs):
                model.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = criterion(model(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()

            # validation
            model.eval(); y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                    y_pred.extend(preds); y_true.extend(yb.numpy())
            f1_scores.append(f1_score(y_true, y_pred, average='macro'))

        return float(np.mean(f1_scores))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=None)  # 🕑 adjust n_trials
    best_params = study.best_params
    print('Best params ->', best_params, 'Best F1 ->', study.best_value)

    # ---------- train final model with best params ----------
    hidden = best_params['hidden']
    layers = best_params['layers']
    drop   = best_params['drop']
    lr     = best_params['lr']
    wd     = best_params['wd']
    epochs = best_params['epochs']

    model_final = SingleHeadGRU(X_all.shape[-1], out_dim, hidden, layers, drop).to(DEVICE)
    opt_final   = torch.optim.AdamW(model_final.parameters(), lr=lr, weight_decay=wd)
    full_loader = DataLoader(SleepDS(X_all, y_target), batch_size=BATCH_SIZE_DEFAULT, shuffle=True)

    for _ in range(epochs):
        model_final.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model_final(xb), yb)
            opt_final.zero_grad(); loss.backward(); opt_final.step()

    # ---------- predict test ----------
    model_final.eval(); preds = []
    with torch.no_grad():
        for xb in DataLoader(SleepDS(X_test), batch_size=BATCH_SIZE_DEFAULT):
            preds.extend(model_final(xb.to(DEVICE)).argmax(1).cpu().numpy())
    preds_dict[target] = preds

# --------------------------------------------------
# Build submission
# --------------------------------------------------
sub = testset[['subject_id','sleep_date','lifelog_date']].copy()
for t in TARGETS:
    sub[t] = preds_dict[t]

SAVE_PATH = '../data/submission_optuna.csv'
sub.to_csv(SAVE_PATH, index=False)
print('✅ submission saved ->', SAVE_PATH)
