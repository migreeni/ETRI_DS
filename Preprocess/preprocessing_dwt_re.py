import pandas as pd
import numpy as np
import pywt
from scipy.stats import entropy
import ast
import re

## dataset loading ------------------------------------------------

# DataFrame 불러오기
mACStatus = pd.read_parquet('ch2025_mACStatus.parquet')
mACStatus['timestamp'] = pd.to_datetime(mACStatus['timestamp'])
mActivity = pd.read_parquet('ch2025_mActivity.parquet')
mActivity['timestamp'] = pd.to_datetime(mActivity['timestamp'])
mAmbience = pd.read_parquet('ch2025_mAmbience.parquet')
mAmbience['timestamp'] = pd.to_datetime(mAmbience['timestamp'])
mBle = pd.read_parquet('ch2025_mBle.parquet')
mBle['timestamp'] = pd.to_datetime(mBle['timestamp'])
mGps = pd.read_parquet('ch2025_mGps.parquet')
mGps['timestamp'] = pd.to_datetime(mGps['timestamp'])
mLight = pd.read_parquet('ch2025_mLight.parquet')
mLight['timestamp'] = pd.to_datetime(mLight['timestamp'])
mScreenStatus = pd.read_parquet('ch2025_mScreenStatus.parquet')
mScreenStatus['timestamp'] = pd.to_datetime(mScreenStatus['timestamp'])
mUsageStats = pd.read_parquet('ch2025_mUsageStats.parquet')
mUsageStats['timestamp'] = pd.to_datetime(mUsageStats['timestamp'])
mWifi = pd.read_parquet('ch2025_mWifi.parquet')
mWifi['timestamp'] = pd.to_datetime(mWifi['timestamp'])
wHr = pd.read_parquet('ch2025_wHr.parquet')
wHr['timestamp'] = pd.to_datetime(wHr['timestamp'])
wLight = pd.read_parquet('ch2025_wLight.parquet')
wLight['timestamp'] = pd.to_datetime(wLight['timestamp'])
wPedo = pd.read_parquet('ch2025_wPedo.parquet')
wPedo['timestamp'] = pd.to_datetime(wPedo['timestamp'])

## preprocessing ----------------------------------------------------

# 1. mUsageStats
print("Preprocessing mUsageStats...")

def parse_app_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except:
        return []

# 특수문자 및 기호를 제거하는 함수
def clean_app_name(app_name):
    # 모든 특수문자 및 기호를 제거하고 알파벳, 숫자, 공백, _, - 만 허용
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s_-가-힣]', '', app_name.strip())  # 알파벳, 숫자, 공백, _, - 만 허용
    return cleaned_name

# 각 row에서 app_name을 컬럼으로 total_time을 값으로 분해
expanded_rows = []
for _, row in mUsageStats.iterrows():
    app_list = parse_app_list(row['m_usage_stats'])
    row_data = {'subject_id': row['subject_id'], 'timestamp': row['timestamp']}
    for app in app_list:
        if 'app_name' in app and 'total_time' in app:
            app_name = clean_app_name(app['app_name'])  # 한글 및 특수문자 처리
            row_data[app_name] = app['total_time']
    expanded_rows.append(row_data)

mUsageStats_expanded = pd.DataFrame(expanded_rows)

# 2. mActivity : 시간대별 가중합
print("Preprocessing mActivity...")
prep_mActivity = mActivity[['subject_id', 'timestamp', 'm_activity']].copy()

# MET values for each activity type
activity_to_met = {0: 1.3, 1: 7.2, 2: 2.3, 3: 1.1, 4: 1.0, 5: 1.3, 7: 3.4, 8: 8.0}

# Function for time-based weight
def get_time_weight(ts):
    hour = ts.hour
    if 0 <= hour < 8:
        return 0.3
    elif 8 <= hour < 18:
        return 0.7
    else:
        return 1.0

# Convert 'timestamp' to datetime
prep_mActivity['timestamp'] = pd.to_datetime(prep_mActivity['timestamp'])

# Calculate weighted MET value for each row
prep_mActivity['met_activity'] = prep_mActivity.apply(
    lambda row: activity_to_met.get(row['m_activity'], 1.0) * get_time_weight(row['timestamp']), axis=1
)

# Keep only final columns
prep_mActivity = prep_mActivity[['subject_id', 'timestamp', 'met_activity']]

# 3. mBle : 가까운 기기가 더 영향을 많이 주도록 가중합
print("Preprocessing mBle...")
# Extract 'subject_id' and 'timestamp' columns
prep_mBle = mBle[['subject_id', 'timestamp']].copy()

# Calculate weighted bluetooth RSSI for each row
def weighted_ble_rssi(ble_stats):
    return sum(np.exp(ble.get('rssi', 0) / 10) for ble in ble_stats)

prep_mBle['wb_rssi'] = mBle['m_ble'].apply(weighted_ble_rssi)

# 4. mWifi : 가까운 기기가 더 영향을 많이 주도록 가중합
print("Preprocessing mWifi...")
# Extract 'subject_id' and 'timestamp' columns
prep_mWifi = mWifi[['subject_id', 'timestamp']].copy()

# Calculate weighted wifi RSSI for each row
def weighted_wifi_rssi(wifi_stats):
    return sum(np.exp(wifi.get('rssi', 0) / 10) for wifi in wifi_stats)

prep_mWifi['ww_rssi'] = mWifi['m_wifi'].apply(weighted_wifi_rssi)

# 5. wHr : 평균값
print("Preprocessing wHr...")
prep_wHr = wHr[['subject_id', 'timestamp']].copy()

# Calculate average heart rate for each row and store in 'avg_heart_rate'
prep_wHr['avg_heart_rate'] = wHr['heart_rate'].apply(lambda x: np.mean(x))

# 6. wPedo
print("Preprocessing wPedo...")
prep_wPedo = wPedo[['subject_id', 'timestamp']].copy()

# Keep only subject_id, timestamp, distance, and burned_calories
prep_wPedo = wPedo[['subject_id', 'timestamp', 'distance', 'burned_calories']].copy()

# 7. mGps
print("Preprocessing mGps...")
# Extract needed columns
prep_mGps = mGps[['subject_id', 'timestamp']].copy()

# Function to calculate averages for each list of gps dicts
def calc_gps_avgs(gps_list, key):
    vals = [item.get(key, np.nan) for item in gps_list if item.get(key) is not None]
    return np.mean(vals) if vals else np.nan

prep_mGps['avg_latitude'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'latitude'))
prep_mGps['avg_longitude'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'longitude'))
prep_mGps['avg_altitude'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'altitude'))
prep_mGps['avg_speed'] = mGps['m_gps'].apply(lambda gps: calc_gps_avgs(gps, 'speed'))

# 8. wLight : 워치만 사용하기로 함
print("Preprocessing wLight...")
prep_wLight = wLight

# 9. mAmbience
print("Preprocessing mAmbience...")
# mAmbience['timestamp']를 datetime으로 변환
mAmbience['timestamp'] = pd.to_datetime(mAmbience['timestamp'])

# 확률값을 float으로 변환하고, category를 컬럼으로
def expand_m_ambience(row):
    row_data = {
        'subject_id': row['subject_id'],
        'timestamp': row['timestamp']
    }
    sound_array = row['m_ambience']
    if isinstance(sound_array, (list, np.ndarray)):
        for item in sound_array:
            if isinstance(item, (list, np.ndarray)) and len(item) == 2:
                category = item[0].strip()
                try:
                    prob = float(item[1])
                    row_data[category] = prob
                except:
                    continue
    return row_data

# 변환 실행
mAmbience_expanded = pd.DataFrame([expand_m_ambience(row) for _, row in mAmbience.iterrows()])

# dwt ---------------------------------------------------------
# DWT 요약 함수 정의
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

# DWT 적용 함수 정의
def create_dwt_features(df, name):
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    value_cols = [col for col in df.columns if col not in ['subject_id', 'timestamp', 'date']]
    feature_rows = []
    for (sid, date), group in df.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'date': date}
        for col in value_cols:
            features = dwt_summary(group[col])
            row.update({f"{name}_{col}_{k}": v for k, v in features.items()})
        feature_rows.append(row)
    return pd.DataFrame(feature_rows)

# 각 데이터셋별 DWT 피처 생성
dwt_dfs = []
datasets = {
    'mUsageStats': mUsageStats_expanded,
    'mActivity': prep_mActivity,
    'mBle': prep_mBle,
    'mWifi': prep_mWifi,
    'wHr': prep_wHr,
    'wPedo': prep_wPedo,
    'mGps': prep_mGps,
    'wLight': prep_wLight,
    'mAmbience': mAmbience_expanded
}

for name, df in datasets.items():
    dwt_df = create_dwt_features(df, name)
    dwt_dfs.append(dwt_df)

# subject_id, date 기준으로 병합
from functools import reduce
merged_dwt_df = reduce(lambda left, right: pd.merge(left, right, on=['subject_id', 'date'], how='outer'), dwt_dfs)


# 결측치 처리
merged_dwt_df = merged_dwt_df.fillna(0)

## id one-hot encoding
for i in range(1, 11):  
    column_name = f'id{i:02d}'  
    df[column_name] = (df['subject_id'] == column_name).astype(int)

# 특수문자 및 공백을 `_`로 변환하는 처리 추가
merged_dwt_df.columns = merged_dwt_df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# 파일로 저장
merged_dwt_df.to_csv("merged_dwt.csv", index=False)

