import pandas as pd
import numpy as np
import pywt
from scipy.stats import entropy
import ast
import re
from functools import reduce

## dataset loading ------------------------------------------------
path = 'data/'  # 경로 설정

# DataFrame 불러오기
mACStatus = pd.read_parquet(path + 'ch2025_mACStatus.parquet')
mACStatus['timestamp'] = pd.to_datetime(mACStatus['timestamp'])
mActivity = pd.read_parquet(path + 'ch2025_mActivity.parquet')
mActivity['timestamp'] = pd.to_datetime(mActivity['timestamp'])
mAmbience = pd.read_parquet(path + 'ch2025_mAmbience.parquet')
mAmbience['timestamp'] = pd.to_datetime(mAmbience['timestamp'])
mBle = pd.read_parquet(path + 'ch2025_mBle.parquet')
mBle['timestamp'] = pd.to_datetime(mBle['timestamp'])
mGps = pd.read_parquet(path + 'ch2025_mGps.parquet')
mGps['timestamp'] = pd.to_datetime(mGps['timestamp'])
mLight = pd.read_parquet(path + 'ch2025_mLight.parquet')
mLight['timestamp'] = pd.to_datetime(mLight['timestamp'])
mScreenStatus = pd.read_parquet(path + 'ch2025_mScreenStatus.parquet')
mScreenStatus['timestamp'] = pd.to_datetime(mScreenStatus['timestamp'])
mUsageStats = pd.read_parquet(path + 'ch2025_mUsageStats.parquet')
mUsageStats['timestamp'] = pd.to_datetime(mUsageStats['timestamp'])
mWifi = pd.read_parquet(path + 'ch2025_mWifi.parquet')
mWifi['timestamp'] = pd.to_datetime(mWifi['timestamp'])
wHr = pd.read_parquet(path + 'ch2025_wHr.parquet')
wHr['timestamp'] = pd.to_datetime(wHr['timestamp'])
wLight = pd.read_parquet(path + 'ch2025_wLight.parquet')
wLight['timestamp'] = pd.to_datetime(wLight['timestamp'])
wPedo = pd.read_parquet(path + 'ch2025_wPedo.parquet')
wPedo['timestamp'] = pd.to_datetime(wPedo['timestamp'])

## preprocessing ----------------------------------------------------

# 1. mUsageStats
print("Preprocessing mUsageStats...")

def parse_app_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except:
        return []

# mUsageStats 데이터에서 총 사용 시간만 합산
expanded_rows = []
for _, row in mUsageStats.iterrows():
    app_list = parse_app_list(row['m_usage_stats'])
    row_data = {'subject_id': row['subject_id'], 'timestamp': row['timestamp']}
    total_time = 0  # 총 사용 시간
    
    # 각 앱의 사용 시간을 모두 합산
    for app in app_list:
        if 'total_time' in app:
            try:
                total_time += float(app['total_time'])  # 사용 시간을 합산
            except:
                continue
    
    row_data['total_usage_time'] = total_time  # 합산된 총 사용 시간을 저장
    expanded_rows.append(row_data)

prep_mUsageStats = pd.DataFrame(expanded_rows)


# 2. mActivity : 시간에 따른 MET 값 계산 (시간 가중치 없이, MET 값만 그대로 사용)
print("Preprocessing mActivity...")

prep_mActivity = mActivity[['subject_id', 'timestamp', 'm_activity']].copy()

# MET 값 매핑 (시간에 따른 가중치 없이, m_activity 값에 해당하는 MET 값 그대로 사용)
activity_to_met = {0: 1.3, 1: 7.2, 2: 2.3, 3: 1.1, 4: 1.0, 5: 1.3, 7: 3.4, 8: 8.0}

# MET 값을 그대로 적용
prep_mActivity['met_activity'] = prep_mActivity['m_activity'].map(activity_to_met).fillna(1.0)

# 필요한 컬럼만 추출
prep_mActivity = prep_mActivity[['subject_id', 'timestamp', 'met_activity']]


# 3. mBle : Bluetooth RSSI 가중치 계산
print("Preprocessing mBle...")

prep_mBle = mBle[['subject_id', 'timestamp']].copy()

def weighted_ble_rssi(ble_stats):
    return sum(np.exp(ble.get('rssi', 0) / 10) for ble in ble_stats)

prep_mBle['wb_rssi'] = mBle['m_ble'].apply(weighted_ble_rssi)

# 4. mWifi : Wifi RSSI 가중치 계산
print("Preprocessing mWifi...")

prep_mWifi = mWifi[['subject_id', 'timestamp']].copy()

def weighted_wifi_rssi(wifi_stats):
    return sum(np.exp(wifi.get('rssi', 0) / 10) for wifi in wifi_stats)

prep_mWifi['ww_rssi'] = mWifi['m_wifi'].apply(weighted_wifi_rssi)

# 5. wHr : 평균 심박수 계산
print("Preprocessing wHr...")

prep_wHr = wHr[['subject_id', 'timestamp']].copy()
prep_wHr['avg_heart_rate'] = wHr['heart_rate'].apply(lambda x: np.mean(x))

# 6. wPedo : 보행 거리 및 소모 칼로리
print("Preprocessing wPedo...")

prep_wPedo = wPedo[['subject_id', 'timestamp', 'distance', 'burned_calories']].copy()

# 7. mGps : GPS 데이터 평균
print("Preprocessing mGps...")

prep_mGps = mGps[['subject_id', 'timestamp']].copy()

def calc_gps_avgs(gps_list, key):
    vals = [item.get(key, np.nan) for item in gps_list if item.get(key) is not None]
    return np.mean(vals) if vals else np.nan

prep_mGps['avg_latitude'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'latitude'))
prep_mGps['avg_longitude'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'longitude'))
prep_mGps['avg_altitude'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'altitude'))
prep_mGps['avg_speed'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'speed'))

# 8. wLight : 라이트
print("Preprocessing wLight...")
prep_wLight = wLight

# 9. mAmbience : 가중합 방식으로 전처리
print("Preprocessing mAmbience...")

# 가중치 파일 불러오기
weight_df = pd.read_csv(path + 'weight_final.csv')
weight_dict = dict(zip(weight_df['category'], weight_df['weight']))  # Category를 키로, Weight를 값으로 매핑

def calculate_weighted_sum(row):
    weighted_sum = 0
    total_weight = 0  # 총 가중치
    
    # m_ambience에 있는 각 카테고리에 대해
    sound_array = row['m_ambience']
    if isinstance(sound_array, (list, np.ndarray)):
        for item in sound_array:
            if isinstance(item, (list, np.ndarray)) and len(item) == 2:
                category = item[0].strip()
                try:
                    prob = float(item[1])
                    if category in weight_dict:  # weight_final.csv에서 가중치를 불러왔으면 그에 맞게 계산
                        weight = weight_dict[category]
                        weighted_sum += prob * weight
                        total_weight += weight
                except:
                    continue
    
    # 가중합이 잘 계산되었다면 그 값 반환
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0

# 각 row에 대해 가중합 계산 (timestamp를 그대로 사용)
prep_mAmbience = pd.DataFrame([{
    'subject_id': row['subject_id'],
    'timestamp': row['timestamp'],  # timestamp를 그대로 사용
    'weighted_sum': calculate_weighted_sum(row)
} for _, row in mAmbience.iterrows()])

# dwt ---------------------------------------------------------
# DWT 계산 함수
def dwt_summary(series, wavelet='db1'):
    series = series.dropna()
    if len(series) < 2:
        return {
            'cA_mean': np.nan, 'cA_std': np.nan, 'cA_energy': np.nan, 'cA_entropy': np.nan,
            'cD_mean': np.nan, 'cD_std': np.nan, 'cD_energy': np.nan, 'cD_entropy': np.nan
        }
    cA, cD = pywt.dwt(series, wavelet)
    def safe_entropy(arr):
        p = np.abs(arr) / np.sum(np.abs(arr)) if np.sum(np.abs(arr)) > 0 else np.ones_like(arr)/len(arr)
        return entropy(p)
    return {
        'cA_mean': np.mean(cA), 'cA_std': np.std(cA), 'cA_energy': np.sum(cA**2), 'cA_entropy': safe_entropy(cA),
        'cD_mean': np.mean(cD), 'cD_std': np.std(cD), 'cD_energy': np.sum(cD**2), 'cD_entropy': safe_entropy(cD)
    }

# DWT 특성 생성 함수
def create_dwt_features(df, name):
    df = df.copy()
    df['lifelog_date'] = pd.to_datetime(df['timestamp']).dt.date
    value_cols = [col for col in df.columns if col not in ['subject_id', 'timestamp', 'lifelog_date']]
    feature_rows = []
    for (sid, lifelog_date), group in df.groupby(['subject_id', 'lifelog_date']):
        row = {'subject_id': sid, 'lifelog_date': lifelog_date}
        for col in value_cols:
            features = dwt_summary(group[col])
            row.update({f"{name}_{col}_{k}": v for k, v in features.items()})
        feature_rows.append(row)
    return pd.DataFrame(feature_rows)

# DWT 특성 계산
dwt_dfs = []
datasets = {
    'mActivity': prep_mActivity,
    'mBle': prep_mBle,
    'mWifi': prep_mWifi,
    'wHr': prep_wHr,
    'wPedo': prep_wPedo,
    'mGps': prep_mGps,
    'wLight': prep_wLight,
    'mUsageStats': prep_mUsageStats,
    'mAmbience': prep_mAmbience
}

for name, df in datasets.items():
    dwt_df = create_dwt_features(df, name)
    dwt_dfs.append(dwt_df)

# subject_id, lifelog_date 기준으로 병합
merged_dwt_df = reduce(lambda left, right: pd.merge(left, right, on=['subject_id', 'lifelog_date'], how='outer'), dwt_dfs)

# 결측값 처리
merged_dwt_df = merged_dwt_df.fillna(-1)

## id one-hot encoding
for i in range(1, 11):  
    column_name = f'id{i:02d}'  
    merged_dwt_df[column_name] = (merged_dwt_df['subject_id'] == column_name).astype(int)

# 특성 이름 정리
merged_dwt_df.columns = merged_dwt_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# 최종 결과 저장
merged_dwt_df.to_csv("merged_dwt_final.csv", index=False)

