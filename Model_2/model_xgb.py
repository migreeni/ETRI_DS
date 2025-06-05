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

# XGBoost
oof_preds_xgb = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_xgb = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}

for target in targets:
    print(f"Training target: {target}")
    if target in multi_class_targets:
        num_class = 3
    else:
        num_class = 1

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, train_df[target])):
        print(f"Fold {fold + 1}/{n_splits} for target {target}")
        X_train, y_train = X.iloc[train_idx], train_df[target].iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], train_df[target].iloc[val_idx]


        # XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(submission_X)
        params_xgb = {
            'objective': 'multi:softprob' if target == 'S1' else 'binary:logistic',
            'eval_metric': 'mlogloss' if target == 'S1' else 'logloss',
            'seed': 42 + fold,
            'verbosity': 0,
        }
        if target == 'S1':
            params_xgb['num_class'] = num_class

        model_xgb = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=100,
        )
        pred_val_xgb = model_xgb.predict(dval)
        pred_test_xgb = model_xgb.predict(dtest)

        # Fold별 평균 F1 score 출력 (모든 모델)
        f1_scores_lgb = []
        f1_scores_xgb = []
        f1_scores_lr = []

        for t in targets:
             # XGBoost
            if t == 'S1':
                preds = oof_preds_xgb[t][val_idx]
                pred_labels = np.argmax(preds, axis=1)
            else:
                preds = oof_preds_xgb[t][val_idx]
                pred_labels = (preds > 0.5).astype(int)
            f1_scores_xgb.append(f1_score(train_df[t].iloc[val_idx], pred_labels, average='macro'))

        print(f"Fold {fold + 1} Mean F1 Score XGBoost: {np.mean(f1_scores_xgb):.4f}")

# 14. 최종 OOF 평가 및 제출 데이터 예측
print("\nFinal Evaluation on OOF predictions (XGBoost):")
f1_score_list = []
for t in targets:
    print(f"=== Target: {t} ===")

    # 다중 클래스(S1)는 확률 평균 후 argmax
    if t == 'S1':
        oof_pred_labels = np.argmax(oof_preds_xgb[t][val_idx], axis=1)
        test_pred_labels = np.argmax(test_preds_xgb[t][val_idx], axis=1)

        acc = (train_df[t] == oof_pred_labels).mean()
        f1 = f1_score(train_df[t].iloc[val_idx], oof_pred_labels, average='macro')
        f1_score_list.append(f1)
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # 이진 분류는 확률 평균 후 0.5 기준 이진화
    else:
        oof_pred_labels = (oof_preds_xgb[t][val_idx] > 0.5).astype(int)
        test_pred_labels = (test_preds_xgb[t][val_idx] > 0.5).astype(int).flatten()

        acc = (train_df[t] == oof_pred_labels).mean()
        f1 = f1_score(train_df[t].iloc[val_idx], oof_pred_labels)
        f1_score_list.append(f1)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # 앙상블 예측값을 원래 컬럼명으로 저장
    submission_df[t] = test_pred_labels
    # 앙상블 예측값을 원래 컬럼명으로 저장
    submission_df[t] = test_pred_labels

# 제출 파일 저장
print(f"Total F1: {mean(f1_score_list):.4f}")
submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]
submission_df.to_csv('dwt_xgb.csv', index=False)