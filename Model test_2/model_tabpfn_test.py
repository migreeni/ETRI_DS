import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from statistics import mean

from tabpfn import TabPFNClassifier

# 데이터 로드
train_df = pd.read_csv("ch2025_metrics_train.csv")
submission_df = pd.read_csv("ch2025_submission_sample.csv")
merge_df = pd.read_csv("merged_dwt.csv")
# usage, amb 사용 안하므로 컬럼을 89까지 슬라이싱
merge_df = merge_df.iloc[:,:89]
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
f1_score_list = []

X = train_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])
submission_X = submission_df.drop(columns=targets + ['subject_id', 'sleep_date', 'lifelog_date'])

for target in targets:
    print(f"Training target: {target}")
    if target in multi_class_targets:
        num_class = 3
    else:
        num_class = 1
    y = train_df[target]
    data_br = {}
    data_br['X_train'], data_br['X_val'], data_br['y_train'], data_br['y_val'] = train_test_split(X, y, test_size=0.7, random_state=42)
    
    classifier = TabPFNClassifier()
    classifier.fit(data_br['X_train'], data_br['y_train'])

    data_br['y_val_pred'] = classifier.predict(data_br['X_val'])
    pred = classifier.predict(submission_X)
    if num_class == 3:
        f1_score_list.append(f1_score(data_br['y_val'], data_br['y_val_pred'], average='macro'))
        print(f'target : {target} - f1 score : {f1_score(data_br['y_val'], data_br['y_val_pred'], average='macro')}')

    else:
        f1_score_list.append(f1_score(data_br['y_val'], data_br['y_val_pred']))
        print(f'target : {target} - f1 score : {f1_score(data_br['y_val'], data_br['y_val_pred'])}')

    submission_df[target] = pred


print(f'TabPFN total f1 score - {mean(f1_score_list)}')
# 제출 파일 저장
submission_df = submission_df[['subject_id', 'sleep_date', 'lifelog_date'] + targets]

submission_df.to_csv('dwt_pfn.csv', index=False)



