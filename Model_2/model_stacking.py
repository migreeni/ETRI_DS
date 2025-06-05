import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from statistics import mean

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

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

merge_df['lifelog_date'] = pd.to_datetime(merge_df['lifelog_date'])
train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
submission_df['lifelog_date'] = pd.to_datetime(submission_df['lifelog_date'])

train_df = pd.merge(train_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
submission_df = pd.merge(submission_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])

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

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

meta_model = LogisticRegression(max_iter=1000)

submission_preds = {}
f1_score_list = []

for target in targets:
    print(f"=== STACKING for Target: {target} ===")
    if target in multi_class_targets:
        num_class = 3
        oof_base = np.zeros((len(train_df), num_class * 3)) 
        test_base = np.zeros((len(submission_df), num_class * 3))
    else:
        oof_base = np.zeros((len(train_df), 3))
        test_base = np.zeros((len(submission_df), 3))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, train_df[target])):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, y_train = X.iloc[train_idx], train_df[target].iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], train_df[target].iloc[val_idx]
        
        # LightGBM
        params_lgb = {
            'objective': 'multiclass' if target == 'S1' else 'binary',
            'metric': 'multi_logloss' if target == 'S1' else 'binary_logloss',
            'verbosity': -1,
            'seed': 42+fold,
        }
        if target == 'S1':
            params_lgb['num_class'] = num_class
        model_lgb = lgb.LGBMClassifier(**params_lgb) if target == 'S1' else lgb.LGBMClassifier(**params_lgb)
        model_lgb.fit(X_train, y_train)
        pred_val_lgb = model_lgb.predict_proba(X_val)
        pred_test_lgb = model_lgb.predict_proba(submission_X)

        # XGBoost
        params_xgb = {
            'objective': 'multi:softprob' if target == 'S1' else 'binary:logistic',
            'eval_metric': 'mlogloss' if target == 'S1' else 'logloss',
            'seed': 42+fold,
            'verbosity': 0,
        }
        model_xgb = xgb.XGBClassifier(**params_xgb, n_estimators=100, use_label_encoder=False)
        model_xgb.fit(X_train, y_train)
        pred_val_xgb = model_xgb.predict_proba(X_val)
        pred_test_xgb = model_xgb.predict_proba(submission_X)

        # RandomForest
        model_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42+fold,
            n_jobs=-1,
            class_weight='balanced'
        )
        model_rf.fit(X_train, y_train)
        pred_val_rf = model_rf.predict_proba(X_val)
        pred_test_rf = model_rf.predict_proba(submission_X)

        # stacking feature
        if target == 'S1':
            oof_base[val_idx, :3] = pred_val_lgb
            oof_base[val_idx, 3:6] = pred_val_xgb
            oof_base[val_idx, 6:9] = pred_val_rf

            test_base[:, :3] += pred_test_lgb / n_splits
            test_base[:, 3:6] += pred_test_xgb / n_splits
            test_base[:, 6:9] += pred_test_rf / n_splits
        else:
            oof_base[val_idx, 0] = pred_val_lgb[:, 1] if pred_val_lgb.shape[1] == 2 else pred_val_lgb[:, 0]
            oof_base[val_idx, 1] = pred_val_xgb[:, 1] if pred_val_xgb.shape[1] == 2 else pred_val_xgb[:, 0]
            oof_base[val_idx, 2] = pred_val_rf[:, 1] if pred_val_rf.shape[1] == 2 else pred_val_rf[:, 0]

            test_base[:, 0] += pred_test_lgb[:, 1] / n_splits if pred_test_lgb.shape[1] == 2 else pred_test_lgb[:, 0] / n_splits
            test_base[:, 1] += pred_test_xgb[:, 1] / n_splits if pred_test_xgb.shape[1] == 2 else pred_test_xgb[:, 0] / n_splits
            test_base[:, 2] += pred_test_rf[:, 1] / n_splits if pred_test_rf.shape[1] == 2 else pred_test_rf[:, 0] / n_splits

    # STACKING
    if target == 'S1':
        meta_model_multi = LogisticRegression(max_iter=1000, multi_class='multinomial')
        meta_model_multi.fit(oof_base, train_df[target])
        oof_pred = meta_model_multi.predict(oof_base)
        test_pred = meta_model_multi.predict(test_base)
        print(f"S1 macro F1:", f1_score(train_df[target], oof_pred, average='macro'))
        f1_score_list.append(f1_score(train_df[target], oof_pred, average='macro'))
    else:
        meta_model_bin = LogisticRegression(max_iter=1000)
        meta_model_bin.fit(oof_base, train_df[target])
        oof_pred = meta_model_bin.predict(oof_base)
        test_pred = meta_model_bin.predict(test_base)
        print(f"{target} F1:", f1_score(train_df[target], oof_pred))
        f1_score_list.append(f1_score(train_df[target], oof_pred))
    submission_preds[target] = test_pred

# result
print('result f1 score =========================')
print(f'Q1: {f1_score_list[0]:.4f}')
print(f'Q2: {f1_score_list[1]:.4f}')
print(f'Q3: {f1_score_list[2]:.4f}')
print(f'S1: {f1_score_list[3]:.4f}')
print(f'S2: {f1_score_list[4]:.4f}')
print(f'S3: {f1_score_list[5]:.4f}')
print(f'Total: {mean(f1_score_list):.4f}')

for t in targets:
    submission_df[t] = submission_preds[t]
submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]
submission_df.to_csv('submission_pred.csv', index=False)
