import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb
from statistics import mean

################################################################## 
# load data - select 
train_df = pd.read_csv("ch2025_metrics_train.csv")
submission_df = pd.read_csv("ch2025_submission_sample.csv")

# print('data : original')
# merge_df = pd.read_csv("merged_df_original.csv")

print('data : dwt')
merge_df = pd.read_csv("merged_dwt.csv")
################################################################## 

# usage, amb ignore 
# print('ignore : usage, amb')
# merge_df = merge_df.iloc[:,:89]

##################################################################   

merge_df.fillna(-1, inplace=True)

# train, submission과 병합
merge_df['lifelog_date'] = pd.to_datetime(merge_df['lifelog_date'])
train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
submission_df['lifelog_date'] = pd.to_datetime(submission_df['lifelog_date'])

train_df = pd.merge(train_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
submission_df = pd.merge(submission_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])

# subject_id One-Hot 인코딩
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# train_encoded = encoder.fit_transform(train_merged[['subject_id']])
# train_id_ohe = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['subject_id']), index=train_merged.index)
# submission_encoded = encoder.transform(submission_merged[['subject_id']])
# submission_id_ohe = pd.DataFrame(submission_encoded, columns=encoder.get_feature_names_out(['subject_id']), index=submission_merged.index)

# train_df = pd.concat([train_merged, train_id_ohe], axis=1)
# submission_df = pd.concat([submission_merged, submission_id_ohe], axis=1)

train_df.columns = train_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
submission_df.columns = submission_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# 10. 타깃 및 입력 정의
targets = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']
multi_class_targets = ['S1']
binary_targets = [t for t in targets if t not in multi_class_targets]

X = train_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])
y = train_df[targets]
submission_X = submission_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])


# 12. KFold 설정
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF/Submission 예측 결과 저장
oof_preds_xgb = {}
test_preds_xgb = {}

for t in targets:
    print(f"\nTraining target: {t}")

    # 데이터 준비
    if t == 'S1':
        oof_preds_xgb[t] = np.zeros((len(train_df), 3))
        test_preds_xgb[t] = np.zeros((len(submission_df), 3))
        xgb_objective = 'multi:softprob'
        xgb_metric = 'mlogloss'
        num_class = 3
    else:
        oof_preds_xgb[t] = np.zeros(len(train_df))
        test_preds_xgb[t] = np.zeros(len(submission_df))
        xgb_objective = 'binary:logistic'
        xgb_metric = 'logloss'
        num_class = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, train_df[t])):
        print(f"  Fold {fold+1}/{n_splits}")
        X_train, y_train = X.iloc[train_idx], train_df[t].iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], train_df[t].iloc[val_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(submission_X)

        params = {
            'objective': xgb_objective,
            'eval_metric': xgb_metric,
            'seed': 42 + fold,
            'verbosity': 0,
        }
        if num_class:
            params['num_class'] = num_class

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # fold validation 예측 → OOF 저장
        pred_val = model.predict(dval)
        if t == 'S1':
            oof_preds_xgb[t][val_idx, :] = pred_val
        else:
            oof_preds_xgb[t][val_idx] = pred_val

        # submission 예측(평균)
        pred_test = model.predict(dtest)
        if t == 'S1':
            test_preds_xgb[t] += pred_test / n_splits
        else:
            test_preds_xgb[t] += pred_test / n_splits

# --- OOF F1 score 출력 ---
print("\nFinal Validation OOF F1 Scores (XGBoost):")
f1_score_list = []
for t in targets:
    if t == 'S1':
        oof_pred_labels = np.argmax(oof_preds_xgb[t], axis=1)
        f1 = f1_score(train_df[t], oof_pred_labels, average='macro')
        print(f"{t} Macro F1: {f1:.4f}")
    else:
        oof_pred_labels = (oof_preds_xgb[t] > 0.5).astype(int)
        f1 = f1_score(train_df[t], oof_pred_labels)
        print(f"{t} F1: {f1:.4f}")
    f1_score_list.append(f1)
print(f"Total F1 (Mean of 6 targets): {mean(f1_score_list):.4f}")

# --- 제출 데이터 예측 ---
print("\nMaking predictions for submission data (XGBoost):")
for t in targets:
    print(f"Predicting target: {t}")
    if t == 'S1':
        test_pred_labels = np.argmax(test_preds_xgb[t], axis=1)
    else:
        test_pred_labels = (test_preds_xgb[t] > 0.5).astype(int).flatten()
    submission_df[t] = test_pred_labels

submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]
submission_df.to_csv('submission_pred.csv', index=False)
print("Submission saved as dwt_xgb.csv")