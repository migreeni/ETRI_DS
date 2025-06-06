
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from statistics import mean

################################################################## 
# load data - select 
train_df = pd.read_csv("ch2025_metrics_train.csv")
submission_df = pd.read_csv("ch2025_submission_sample.csv")

print('data : original')
merge_df = pd.read_csv("merged_original.csv")

# print('data : dwt')
# merge_df = pd.read_csv("merged_dwt.csv")
################################################################## 

# usage, amb ignore 
# print('ignore : usage, amb')
# merge_df = merge_df.iloc[:,:89]

##################################################################  
 
merge_df.fillna(-1, inplace=True)   

# train, submission
merge_df['lifelog_date'] = pd.to_datetime(merge_df['lifelog_date'])
train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
submission_df['lifelog_date'] = pd.to_datetime(submission_df['lifelog_date'])

train_df = pd.merge(train_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
submission_df = pd.merge(submission_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
# train_merged = train_merged.drop(columns='date')
# submission_merged = submission_merged.drop(columns='date')

# subject_id One-Hot �씤肄붾뵫
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# train_encoded = encoder.fit_transform(train_merged[['subject_id']])
# train_id_ohe = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['subject_id']), index=train_merged.index)
# submission_encoded = encoder.transform(submission_merged[['subject_id']])
# submission_id_ohe = pd.DataFrame(submission_encoded, columns=encoder.get_feature_names_out(['subject_id']), index=submission_merged.index)

# train_df = pd.concat([train_merged, train_id_ohe], axis=1)
# submission_df = pd.concat([submission_merged, submission_id_ohe], axis=1)

train_df.columns = train_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
submission_df.columns = submission_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

targets = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']
multi_class_targets = ['S1']
binary_targets = [t for t in targets if t not in multi_class_targets]

X = train_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])
y = train_df[targets]
submission_X = submission_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])

def deduplicate_columns(df):
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols
    return df

# 예시: 병합/정제 끝난 후 한 번만
X = deduplicate_columns(X)
submission_X = deduplicate_columns(submission_X)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
submission_X_scaled = pd.DataFrame(scaler.transform(submission_X), columns=submission_X.columns, index=submission_X.index)

# 12. KFold 
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# LightGBM
oof_preds_lgb = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_lgb = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}
# XGBoost
oof_preds_xgb = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_xgb = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}
# Random Forest
oof_preds_rf = {t: np.zeros(len(train_df)) if t != 'S1' else np.zeros((len(train_df), 3)) for t in targets}
test_preds_rf = {t: np.zeros((len(submission_df), 3 if t == 'S1' else 1)) for t in targets}

for target in targets:
    print(f"Training target: {target}")
    if target in multi_class_targets:
        num_class = 3
        objective_lgb = 'multiclass'
        objective_rf = True
        metric_lgb = 'multi_logloss'
    else:
        num_class = 1
        objective_lgb = 'binary'
        objective_rf = False
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

        # Random Forest
        if objective_rf:
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42+fold, 
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            pred_val_rf = model.predict_proba(X_val)
            pred_test_rf = model.predict_proba(submission_X)
            oof_preds_rf[target][val_idx, :] = pred_val_rf
            test_preds_rf[target] += pred_test_rf / n_splits
        else:
            X_train, y_train = X.iloc[train_idx], train_df[target].iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], train_df[target].iloc[val_idx]
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42+fold, 
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            pred_val_rf = model.predict_proba(X_val)[:, 1]
            pred_test_rf = model.predict_proba(submission_X)[:, 1]
            oof_preds_rf[target][val_idx] = pred_val_rf
            test_preds_rf[target][:, 0] += pred_test_rf / n_splits


        if target == 'S1':
            oof_preds_lgb[target][val_idx, :] = pred_val_lgb
            oof_preds_xgb[target][val_idx, :] = pred_val_xgb
            oof_preds_rf[target][val_idx, :] = pred_val_rf

            test_preds_lgb[target] += pred_test_lgb / n_splits
            test_preds_xgb[target] += pred_test_xgb / n_splits
            test_preds_rf[target] += pred_test_rf / n_splits
        else:
            oof_preds_lgb[target][val_idx] = pred_val_lgb
            oof_preds_xgb[target][val_idx] = pred_val_xgb
            oof_preds_rf[target][val_idx] = pred_val_rf

            test_preds_lgb[target][:, 0] += pred_test_lgb / n_splits
            test_preds_xgb[target][:, 0] += pred_test_xgb / n_splits
            test_preds_rf[target][:, 0] += pred_test_rf / n_splits

        # Fold
        f1_scores_lgb = []
        f1_scores_xgb = []
        f1_scores_rf = []

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

            # Random Forest
            if t == 'S1':
                preds = oof_preds_rf[t][val_idx]
                pred_labels = np.argmax(preds, axis=1)
            else:
                preds = oof_preds_rf[t][val_idx]
                pred_labels = (preds > 0.5).astype(int)
            f1_scores_rf.append(f1_score(train_df[t].iloc[val_idx], pred_labels, average='macro'))

        print(f"Fold {fold + 1} Mean F1 Score LightGBM: {np.mean(f1_scores_lgb):.4f}")
        print(f"Fold {fold + 1} Mean F1 Score XGBoost: {np.mean(f1_scores_xgb):.4f}")
        print(f"Fold {fold + 1} Mean F1 Score Random Forest: {np.mean(f1_scores_rf):.4f}")

print("\nFinal Evaluation on OOF predictions (Ensemble of LightGBM, XGBoost, Random Forest):")

f1_score_list = []
for t in targets:
    print(f"=== Target: {t} ===")

    if t == 'S1':
        oof_ensemble = (oof_preds_lgb[t] + oof_preds_xgb[t] + oof_preds_rf[t]) / 3
        oof_pred_labels = np.argmax(oof_ensemble, axis=1)

        test_ensemble = (test_preds_lgb[t] + test_preds_xgb[t] + test_preds_rf[t]) / 3
        test_pred_labels = np.argmax(test_ensemble, axis=1)

        acc = (train_df[t] == oof_pred_labels).mean()
        f1 = f1_score(train_df[t], oof_pred_labels, average='macro')
        f1_score_list.append(f1)
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    else:
        oof_ensemble = (oof_preds_lgb[t] + oof_preds_xgb[t] + oof_preds_rf[t]) / 3
        oof_pred_labels = (oof_ensemble > 0.5).astype(int)

        test_ensemble = (test_preds_lgb[t] + test_preds_xgb[t] + test_preds_rf[t]) / 3
        test_pred_labels = (test_ensemble > 0.5).astype(int).flatten()

        acc = (train_df[t] == oof_pred_labels).mean()
        f1 = f1_score(train_df[t], oof_pred_labels)
        f1_score_list.append(f1)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    submission_df[t] = test_pred_labels

print(f"Total F1: {mean(f1_score_list):.4f}")
submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]
submission_df.to_csv('submission_pred.csv', index=False)