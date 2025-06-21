import os
import pandas as pd
import numpy as np
import argparse
from statistics import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

'''
Parser parameter setting

data :  original, dwt
model : rf, lgbm, xgb, cat, lr, ensemble, stacking
'''

parser = argparse.ArgumentParser(description='Sleep Quality Prediction given Lifelog Data')
parser.add_argument('-m', '--model', type=str, default='lgbm')
parser.add_argument('-d', '--data', type=str, default='original')
parser.add_argument('-t', '--target', type=str, default=None)

args = parser.parse_args()


def main(args):
    # Selection
    
    select_model = args.model 
    select_data = args.data
    print(f'model : {select_model}')
    print(f'data : {select_data}')

    # Load data
    train_df = pd.read_csv("ch2025_metrics_train.csv")
    submission_df = pd.read_csv("ch2025_submission_sample.csv")
    if select_data == 'original' :
        merge_df = pd.read_csv("merged_original_final.csv")
    elif select_data == 'dwt' : 
        merge_df = pd.read_csv("merged_dwt_final.csv")
    else :
        raise Exception("Data Error")
        


    # Preprocess
    merge_df.fillna(-1, inplace=True)
    merge_df['lifelog_date'] = pd.to_datetime(merge_df['lifelog_date'])
    train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
    submission_df['lifelog_date'] = pd.to_datetime(submission_df['lifelog_date'])

    train_df = pd.merge(train_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])
    submission_df = pd.merge(submission_df, merge_df, how='left', on=['subject_id', 'lifelog_date'])

    train_df.columns = train_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
    submission_df.columns = submission_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)


    # Train, Test split
    targets = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']
    multi_class_targets = ['S1']
    binary_targets = [t for t in targets if t not in multi_class_targets]

    X = train_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])
    y = train_df[targets]
    submission_X = submission_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])


    # 특수문자 처리 이후 중복되는 컬럼명 처리 (임시방편, 중복 없으면 제거)
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

    X = deduplicate_columns(X)
    submission_X = deduplicate_columns(submission_X)

    # K-Fold : stratified로 수정하여 label unbalanced 고려
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    ## Training 
    oof_preds = {}
    test_preds = {}

    select_target = args.target
    target_list = targets if select_target is None else [select_target]

    for t in target_list:
        print(f"\nTraining target: {t}")

        if t == 'S1':
            oof_preds[t] = np.zeros((len(train_df), 3))
            test_preds[t] = np.zeros((len(submission_df), 3))
            objective = 'multi:softprob'
            metric = 'mlogloss'
            num_class = 3
        else:
            oof_preds[t] = np.zeros(len(train_df))
            test_preds[t] = np.zeros(len(submission_df))
            objective = 'binary:logistic'
            metric = 'logloss'
            num_class = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, train_df[t])):
            print(f"  Fold {fold+1}/{n_splits}")
            X_train, y_train = X.iloc[train_idx], train_df[t].iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], train_df[t].iloc[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            submission_X_scaled = scaler.transform(submission_X)

            # unbalanced label -> weighted fitting
            if num_class is None:
                class_counts = y_train.value_counts()
                neg, pos = class_counts[0], class_counts[1]
                scale_pos_weight = neg / pos
                sample_weight = None 
            else:
                scale_pos_weight = None
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

            # 1. random forest
            if select_model == 'rf':
                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42 + fold,
                    class_weight='balanced' if num_class is None else None
                )
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

            # 2. lightgbm
            elif select_model == 'lgbm':
                model = LGBMClassifier(
                    objective='multiclass' if num_class else 'binary',
                    n_estimators=1000,
                    class_weight='balanced' if num_class is None else None,
                    min_data_in_leaf=5,
                    min_gain_to_split=0.0,
                    max_depth=6,
                    random_state=42 + fold
                )
                callbacks = [
                    early_stopping(stopping_rounds=50, verbose=True),
                    log_evaluation(period=10)
                ]
                
                model.fit(
                    X_train_scaled, y_train,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks
                )

            # 3. xgboost
            elif select_model == 'xgb':
                model = XGBClassifier(
                    objective=objective,
                    eval_metric=metric,
                    seed=42 + fold,
                    verbosity=0,
                    num_class=num_class if num_class else None,
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=1000,
                    early_stopping_rounds=50,
                    use_label_encoder=False
                    )

                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    sample_weight=sample_weight,
                    verbose=100
                )

            # 4. catboost
            elif select_model == 'cat':
                model = CatBoostClassifier(
                    loss_function='MultiClass' if num_class else 'Logloss',
                    verbose=0,
                    iterations=1000,
                    early_stopping_rounds=50,
                    random_seed=42 + fold
                )
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=(X_val_scaled, y_val),
                    sample_weight=sample_weight
                )

            # 5. logistic regression
            elif select_model == 'lr':
                model = LogisticRegression(
                    class_weight='balanced' if num_class is None else None,
                    max_iter=2000,
                    solver='lbfgs',
                    random_state=42 + fold
                )
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

            # 6. ensemble : xgb + lgbm + rf, voting : soft
            elif select_model == 'ensemble':
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                lgbm = LGBMClassifier(n_estimators=200, random_state=42)
                xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric=metric)
                
                model = VotingClassifier(
                    estimators=[('rf', rf), ('lgbm', lgbm), ('xgb', xgb)],
                    voting='soft'
                )
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

            # 7. stacking : xgb + lgbm + rf, meta : lr
            elif select_model == 'stacking':

                base_models = [
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                    ('lgbm', LGBMClassifier(n_estimators=200, random_state=42)),
                    ('xgb', XGBClassifier(n_estimators=200, random_state=42, eval_metric=metric))
                ]
                meta_model = LogisticRegression(max_iter=1000)

                model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

            else : 
                raise Exception("Model Error")

            # fold validation 
            if t == 'S1':
                pred_val = model.predict_proba(X_val_scaled)
                oof_preds[t][val_idx, :] = pred_val
            else:
                pred_val = model.predict(X_val_scaled)
                oof_preds[t][val_idx] = pred_val

            # submission
            if t == 'S1':
                pred_test = model.predict_proba(submission_X_scaled)
                test_preds[t] += pred_test / n_splits
            else:
                pred_test = model.predict(submission_X_scaled)
                test_preds[t] += pred_test / n_splits

    # --- OOF F1 score ---
    print(f"\nFinal Validation OOF F1 Scores {select_model}:")
    f1_score_list = []
    for t in targets:
        if t == 'S1':
            oof_pred_labels = np.argmax(oof_preds[t], axis=1)
            f1 = f1_score(train_df[t], oof_pred_labels, average='macro')
            print(f"{t} Macro F1: {f1:.4f}")
        else:
            oof_pred_labels = (oof_preds[t] > 0.5).astype(int)
            f1 = f1_score(train_df[t], oof_pred_labels)
            print(f"{t} F1: {f1:.4f}")
        f1_score_list.append(f1)
    print(f"Total F1 (Mean of 6 targets): {mean(f1_score_list):.4f}")

    # Save F1 scores to CSV
    f1_dict = {'target': targets, 'f1_score': f1_score_list}
    f1_df = pd.DataFrame(f1_dict)
    f1_df.loc[len(f1_df)] = ['Total', mean(f1_score_list)]
    
    f1_save_path = f'results/f1score_{select_data}_{select_model}.csv'
    f1_df.to_csv(f1_save_path, index=False)
    print(f"F1 scores saved to {f1_save_path}")


    print(f"\nMaking predictions for submission data {select_model}:")
    for t in targets:
        if t == 'S1':
            test_pred_labels = np.argmax(test_preds[t], axis=1)
        else:
            test_pred_labels = (test_preds[t] > 0.5).astype(int).flatten()
        submission_df[t] = test_pred_labels

    submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]

    os.makedirs("results", exist_ok=True)
    submission_path = f'results/sub_{select_data}_{select_model}.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved as {submission_path}")


if __name__ =='__main__':
    main(args)