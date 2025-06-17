import pandas as pd
import numpy as np
import os
from functools import reduce
import ast

print("Loading data...")
path = 'data/'

mUsageStats = pd.read_parquet(path + 'ch2025_mUsageStats.parquet')
mActivity = pd.read_parquet(path + 'ch2025_mActivity.parquet')
mBle = pd.read_parquet(path + 'ch2025_mBle.parquet')
mWifi = pd.read_parquet(path + 'ch2025_mWifi.parquet')
wHr = pd.read_parquet(path + 'ch2025_wHr.parquet')
wPedo = pd.read_parquet(path + 'ch2025_wPedo.parquet')
mGps = pd.read_parquet(path + 'ch2025_mGps.parquet')
wLight = pd.read_parquet(path + 'ch2025_wLight.parquet')
mAmbience = pd.read_parquet(path + 'ch2025_mAmbience.parquet')
print("Data loaded successfully.")

for df in [mUsageStats, mActivity, mBle, mWifi, wHr, wPedo, mGps, wLight, mAmbience]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------- Preprocessing----------

# 1. mUsageStats
print("Preprocessing mUsageStats...")

def parse_app_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except:
        return []

expanded_rows = []
for _, row in mUsageStats.iterrows():
    app_list = parse_app_list(row['m_usage_stats'])
    row_data = {'subject_id': row['subject_id'], 'timestamp': row['timestamp'].date()}
    for app in app_list:
        if 'app_name' in app and 'total_time' in app:
            app_name = app['app_name'].strip()
            row_data[app_name] = app['total_time']
    expanded_rows.append(row_data)
prep_mUsageStats = pd.DataFrame(expanded_rows)
prep_mUsageStats = prep_mUsageStats.groupby(['subject_id', 'timestamp']).sum().reset_index()

# 2. mActivity
print("Preprocessing mActivity...")
activity_to_met = {0: 1.3, 1: 7.2, 2: 2.3, 3: 1.1, 4: 1.0, 5: 1.3, 7: 3.4, 8: 8.0}

def get_time_weight(ts):
    hour = ts.hour
    if 0 <= hour < 8: return 0.3
    elif 8 <= hour < 18: return 0.7
    else: return 1.0

def calc_weighted_met(row):
    met = activity_to_met.get(mActivity.loc[row.name, 'm_activity'], 1.0)
    weight = get_time_weight(row['timestamp'])
    return met * weight

prep_mActivity = mActivity[['subject_id', 'timestamp']].copy()
prep_mActivity['met_activity'] = prep_mActivity.apply(calc_weighted_met, axis=1)
prep_mActivity['timestamp'] = prep_mActivity['timestamp'].dt.date
prep_mActivity = prep_mActivity.groupby(['subject_id', 'timestamp'])['met_activity'].sum().reset_index()

# 3. mBle
print("Preprocessing mBle...")
def sum_ble_rssi(ble_stats):
    return sum(np.exp(ble.get('rssi', 0) / 10) for ble in ble_stats)

prep_mBle = mBle[['subject_id', 'timestamp']].copy()
prep_mBle['m_wtb_rssi'] = mBle['m_ble'].apply(sum_ble_rssi)
prep_mBle['timestamp'] = prep_mBle['timestamp'].dt.date
prep_mBle = prep_mBle.groupby(['subject_id', 'timestamp'])['m_wtb_rssi'].sum().reset_index()

# 4. mWifi
print("Preprocessing mWifi...")
def sum_wifi_rssi(wifi_stats):
    return sum(np.exp(wifi.get('rssi', 0) / 10) for wifi in wifi_stats)

prep_mWifi = mWifi[['subject_id', 'timestamp']].copy()
prep_mWifi['m_wtb_rssi'] = mWifi['m_wifi'].apply(sum_wifi_rssi)
prep_mWifi['timestamp'] = prep_mWifi['timestamp'].dt.date
prep_mWifi = prep_mWifi.groupby(['subject_id', 'timestamp'])['m_wtb_rssi'].sum().reset_index()

# 5. wHr
print("Preprocessing wHr...")
wHr['heart_rate'] = wHr['heart_rate'].apply(np.mean)
wHr['timestamp'] = wHr['timestamp'].dt.date
prep_wHr = wHr.groupby(['subject_id', 'timestamp'])['heart_rate'].mean().reset_index()

# 6. wPedo
print("Preprocessing wPedo...")
wPedo['distance'] = wPedo['distance'].astype(float)
wPedo['burned_calories'] = wPedo['burned_calories'].astype(float)
wPedo['timestamp'] = wPedo['timestamp'].dt.date
prep_wPedo = wPedo.groupby(['subject_id', 'timestamp'])[['distance', 'burned_calories']].sum().reset_index()

# 7. mGps
print("Preprocessing mGps...")
def mean_gps(row):
    df = pd.DataFrame(list(row))
    return df[['latitude', 'longitude', 'altitude', 'speed']].mean()

mGps_extract = mGps['m_gps'].apply(mean_gps)
mGps_daily = pd.concat([mGps[['subject_id', 'timestamp']], mGps_extract], axis=1)
mGps_daily['timestamp'] = mGps_daily['timestamp'].dt.date
prep_mGps = mGps_daily.groupby(['subject_id', 'timestamp'])[['latitude', 'longitude', 'altitude', 'speed']].mean().reset_index()

# 8. wLight
print("Preprocessing wLight...")
wLight['timestamp'] = wLight['timestamp'].dt.date
prep_wLight = wLight.groupby(['subject_id', 'timestamp'])[['w_light']].mean().reset_index()

# 9. mAmbience
print("Preprocessing mAmbience...")
def expand_m_ambience(row):
    row_data = {
        'subject_id': row['subject_id'],
        'timestamp': row['timestamp'].date()
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

prep_mAmbience = pd.DataFrame([expand_m_ambience(row) for _, row in mAmbience.iterrows()])
prep_mAmbience = prep_mAmbience.groupby(['subject_id', 'timestamp']).mean().reset_index().fillna(0)

# ----------- Merge -----------
print("Merging daily preprocessed dataframes...")

df_list = [
    prep_mActivity,
    prep_mBle,
    prep_mWifi,
    prep_wHr,
    prep_wPedo,
    prep_mGps,
    prep_wLight,
    prep_mUsageStats,
    prep_mAmbience
]

df = reduce(lambda left, right: pd.merge(left, right, on=['subject_id', 'timestamp'], how='outer'), df_list)

# id one-hot encoding
for i in range(1, 11):
    column_name = f'id{i:02d}'
    df[column_name] = (df['subject_id'] == column_name).astype(int)

df = df.rename(columns={'timestamp': 'lifelog_date'})
df.to_csv('merged_original.csv', index=False)