import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

df_merged = df_zero_filled.copy()
# timestamp → datetime 변환
df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"])

# lifelog_date = 날짜(문자열) ; 2024-06-26 같은 형식
df_merged["lifelog_date"] = df_merged["timestamp"].dt.date.astype(str)

# -------- 2. feature / sensor column 정의 -----------
drop_cols = ["timestamp",                # 시간은 예측에 불필요
             "subject_id",               # 매치용
             "lifelog_date"] + [c for c in df_merged.columns if c.startswith("id")]  # one-hot id

sensor_cols = [c for c in df_merged.columns if c not in drop_cols]
# 하루 당 최대 144 타임스텝으로 패딩
MAX_SEQ_LEN = 144


# -------- 3. 시퀀스 묶는 함수 -----------
def build_sequences(df):
    """(subject_id, lifelog_date) → ndarray(seq_len, n_feat)"""
    seq_dict = {}
    for (sid, day), g in df.groupby(['subject_id', 'lifelog_date']):
        # 10-분 간격 보장 안될 수도 있으니 timestamp 기준 정렬
        g = g.sort_values('timestamp')
        x = g[sensor_cols].to_numpy(dtype=np.float32)

        # 길이 조정
        if len(x) > MAX_SEQ_LEN:          # 잘라내기
            x = x[:MAX_SEQ_LEN]
        if len(x) < MAX_SEQ_LEN:          # 0-패딩
            pad = np.zeros((MAX_SEQ_LEN - len(x), x.shape[1]), np.float32)
            x = np.vstack([x, pad])

        seq_dict[(sid, day)] = x          # shape = (144, n_feat)
    return seq_dict

sequence_dict = build_sequences(df_merged)
print("# total sequences built :", len(sequence_dict))

# ---------- 3. Train / Val Tensor 준비 ----------
def rows_to_tensors(df_label):
    X, y = [], []
    for _, row in df_label.iterrows():
        key = (row['subject_id'], row['lifelog_date'])
        if key not in sequence_dict:        # 누락된 날짜 skip
            continue
        X.append(sequence_dict[key])
        y.append([row['Q1']])               # binary -> shape (1,)
    return np.stack(X), np.array(y, dtype=np.float32)

X_train, y_train = rows_to_tensors(df_train_q1)
X_val,   y_val   = rows_to_tensors(df_val_q1)

print("Train tensor shape :", X_train.shape, y_train.shape)
print("Val   tensor shape :", X_val.shape,   y_val.shape)

# ---------- 4. Feature 정규화 ----------
scaler = StandardScaler().fit(X_train.reshape(-1, len(sensor_cols)))
def scale(x):
    orig_shape = x.shape
    x = scaler.transform(x.reshape(-1, len(sensor_cols)))
    return x.reshape(orig_shape)

X_train = scale(X_train)
X_val   = scale(X_val)

# ---------- 5. PyTorch Dataset ----------
class SleepDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SleepDS(X_train, y_train)
val_ds   = SleepDS(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

# ---------- 6. GRU 모델 ----------
class GRUModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h = self.gru(x)          # h: (1,B,hidden)
        h = h.squeeze(0)            # (B,hidden)
        return torch.sigmoid(self.fc(h))    # (B,1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = GRUModel(len(sensor_cols)).to(device)
criterion = nn.BCELoss()
optim     = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- 7. 학습 루프 ----------
EPOCHS = 13
for epoch in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # --- val ---
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            prob = model(xb).cpu().numpy()
            preds.append(prob)
            trues.append(yb.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    y_hat = (preds >= 0.5).astype(int)
    f1 = f1_score(trues, y_hat, average='macro')
    print(f"Epoch {epoch:2d} | Val macro‑F1: {f1:.4f}")

