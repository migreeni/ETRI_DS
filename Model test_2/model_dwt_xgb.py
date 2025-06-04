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

# 데이터 로드
train_df = pd.read_csv("ch2025_metrics_train.csv")
submission_df = pd.read_csv("ch2025_submission_sample.csv")
merge_df = pd.read_csv("Model test_2/merged_dwt6.csv")
# train, submission과 병합
merge_df['lifelog_date'] = pd.to_datetime(merge_df['lifelog_date'])
train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
submission_df['lifelog_date'] = pd.to_datetime(submission_df['lifelog_date'])

train_merged = pd.merge(train_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
submission_merged = pd.merge(submission_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
# train_merged = train_merged.drop(columns='date')
# submission_merged = submission_merged.drop(columns='date')

# subject_id One-Hot 인코딩
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_encoded = encoder.fit_transform(train_merged[['subject_id']])
train_id_ohe = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['subject_id']), index=train_merged.index)
submission_encoded = encoder.transform(submission_merged[['subject_id']])
submission_id_ohe = pd.DataFrame(submission_encoded, columns=encoder.get_feature_names_out(['subject_id']), index=submission_merged.index)

train_df = pd.concat([train_merged, train_id_ohe], axis=1)
submission_df = pd.concat([submission_merged, submission_id_ohe], axis=1)

train_df.columns = train_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
submission_df.columns = submission_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# 10. 타깃 및 입력 정의
targets = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']
multi_class_targets = ['S1']
binary_targets = [t for t in targets if t not in multi_class_targets]

X = train_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])
y = train_df[targets]
submission_X = submission_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])

# 11. 스케일러 준비 (Logistic Regression용)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
submission_X_scaled = pd.DataFrame(scaler.transform(submission_X), columns=submission_X.columns, index=submission_X.index)

# 12. KFold 설정
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 13. 예측 결과 저장 공간
# LightGBM
oof_preds_lgb = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_lgb = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}
# XGBoost
oof_preds_xgb = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_xgb = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}
# Logistic Regression
oof_preds_lr = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_lr = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}

for target in targets:
    print(f"Training target: {target}")
    if target in multi_class_targets:
        num_class = 3
        objective_lgb = 'multiclass'
        metric_lgb = 'multi_logloss'
    else:
        num_class = 1
        objective_lgb = 'binary'
        metric_lgb = 'binary_logloss'

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, train_df[target])):
        print(f"Fold {fold + 1}/{n_splits} for target {target}")
        X_train, y_train = X.iloc[train_idx], train_df[target].iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], train_df[target].iloc[val_idx]

        X_train_scaled = X_scaled.iloc[train_idx]
        X_val_scaled = X_scaled.iloc[val_idx]

        # LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        params_lgb = {
            'objective': objective_lgb,
            'metric': metric_lgb,
            'verbosity': -1,
            'seed': 42 + fold,
        }
        if objective_lgb == 'multiclass':
            params_lgb['num_class'] = num_class

        model_lgb = lgb.train(
            params_lgb,
            train_data,
            valid_sets=[val_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        pred_val_lgb = model_lgb.predict(X_val)
        pred_test_lgb = model_lgb.predict(submission_X)

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

        # Logistic Regression (scaled data, OneVsRestClassifier)
        if target == 'S1':
            lr = OneVsRestClassifier(LogisticRegression(max_iter=2000, solver='lbfgs'))
        else:
            lr = LogisticRegression(max_iter=2000, solver='lbfgs')

        lr.fit(X_train_scaled, y_train)
        pred_val_lr = lr.predict_proba(X_val_scaled)
        pred_test_lr = lr.predict_proba(submission_X_scaled)

        # 저장 (S1: 다중클래스, 나머지: 이진분류)
        if target == 'S1':
            oof_preds_lgb[target][val_idx, :] = pred_val_lgb
            oof_preds_xgb[target][val_idx, :] = pred_val_xgb
            oof_preds_lr[target][val_idx, :] = pred_val_lr

            test_preds_lgb[target] += pred_test_lgb / n_splits
            test_preds_xgb[target] += pred_test_xgb / n_splits
            test_preds_lr[target] += pred_test_lr / n_splits
        else:
            oof_preds_lgb[target][val_idx] = pred_val_lgb
            oof_preds_xgb[target][val_idx] = pred_val_xgb
            oof_preds_lr[target][val_idx] = pred_val_lr[:, 1]

            test_preds_lgb[target][:, 0] += pred_test_lgb / n_splits
            test_preds_xgb[target][:, 0] += pred_test_xgb / n_splits
            test_preds_lr[target][:, 0] += pred_test_lr[:, 1] / n_splits

        # Fold별 평균 F1 score 출력 (모든 모델)
        f1_scores_lgb = []
        f1_scores_xgb = []
        f1_scores_lr = []

        for t in targets:
            # LightGBM
            if t == 'S1':
                preds = oof_preds_lgb[t][val_idx]
                pred_labels = np.argmax(preds, axis=1)
            else:
                preds = oof_preds_lgb[t][val_idx]
                pred_labels = (preds > 0.5).astype(int)
            f1_scores_lgb.append(f1_score(train_df[t].iloc[val_idx], pred_labels, average='macro'))

            # XGBoost
            if t == 'S1':
                preds = oof_preds_xgb[t][val_idx]
                pred_labels = np.argmax(preds, axis=1)
            else:
                preds = oof_preds_xgb[t][val_idx]
                pred_labels = (preds > 0.5).astype(int)
            f1_scores_xgb.append(f1_score(train_df[t].iloc[val_idx], pred_labels, average='macro'))

            # Logistic Regression
            if t == 'S1':
                preds = oof_preds_lr[t][val_idx]
                pred_labels = np.argmax(preds, axis=1)
            else:
                preds = oof_preds_lr[t][val_idx]
                pred_labels = (preds > 0.5).astype(int)
            f1_scores_lr.append(f1_score(train_df[t].iloc[val_idx], pred_labels, average='macro'))

        print(f"Fold {fold + 1} Mean F1 Score LightGBM: {np.mean(f1_scores_lgb):.4f}")
        print(f"Fold {fold + 1} Mean F1 Score XGBoost: {np.mean(f1_scores_xgb):.4f}")
        print(f"Fold {fold + 1} Mean F1 Score LogisticRegression: {np.mean(f1_scores_lr):.4f}")

# 14. 최종 OOF 평가 및 제출 데이터 예측 (LightGBM 기준)
print("\nFinal Evaluation on OOF predictions (Ensemble of LightGBM, XGBoost, LogisticRegression):")

for t in targets:
    print(f"=== Target: {t} ===")

    # 다중 클래스(S1)는 확률 평균 후 argmax
    if t == 'S1':
        oof_ensemble = (oof_preds_lgb[t] + oof_preds_xgb[t] + oof_preds_lr[t]) / 3
        oof_pred_labels = np.argmax(oof_ensemble, axis=1)

        test_ensemble = (test_preds_lgb[t] + test_preds_xgb[t] + test_preds_lr[t]) / 3
        test_pred_labels = np.argmax(test_ensemble, axis=1)

        acc = (train_df[t] == oof_pred_labels).mean()
        f1 = f1_score(train_df[t], oof_pred_labels, average='macro')
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # 이진 분류는 확률 평균 후 0.5 기준 이진화
    else:
        oof_ensemble = (oof_preds_lgb[t] + oof_preds_xgb[t] + oof_preds_lr[t]) / 3
        oof_pred_labels = (oof_ensemble > 0.5).astype(int)

        test_ensemble = (test_preds_lgb[t] + test_preds_xgb[t] + test_preds_lr[t]) / 3
        test_pred_labels = (test_ensemble > 0.5).astype(int).flatten()

        acc = (train_df[t] == oof_pred_labels).mean()
        f1 = f1_score(train_df[t], oof_pred_labels)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # 앙상블 예측값을 원래 컬럼명으로 저장
    submission_df[t] = test_pred_labels

# 제출 파일 저장
submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]