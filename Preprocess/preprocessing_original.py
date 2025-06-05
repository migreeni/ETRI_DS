import pandas as pd
import numpy as np
import os
from functools import reduce

## dataset loading ------------------------------------------------

# <?‚¬?š©?•ˆ?•¨> 
# mACStatus = pd.read_parquet(path + 'ch2025_data_items/ch2025_mACStatus.parquet')
# mScreenStatus = pd.read_parquet(path + 'ch2025_data_items/ch2025_mScreenStatus.parquet')
# mLight = pd.read_parquet(path + 'ch2025_data_items/ch2025_mLight.parquet')
 
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

mUsageStats['timestamp'] = pd.to_datetime(mUsageStats['timestamp'])
mActivity['timestamp'] = pd.to_datetime(mActivity['timestamp'])
mBle['timestamp'] = pd.to_datetime(mBle['timestamp'])
mWifi['timestamp'] = pd.to_datetime(mWifi['timestamp'])
wHr['timestamp'] = pd.to_datetime(wHr['timestamp'])
wPedo['timestamp'] = pd.to_datetime(wPedo['timestamp'])
mGps['timestamp'] = pd.to_datetime(mGps['timestamp'])
wLight['timestamp'] = pd.to_datetime(wLight['timestamp'])
mAmbience['timestamp'] = pd.to_datetime(mAmbience['timestamp'])



## preprocessing ----------------------------------------------------


# 1. mUsageStats : ?•±ë³? ?‹œê°„í•© -> ì´? ?‚¬?š©?‹œê°?
print("Preprocessing mUsageStats...")

# mUsageStats ë¶ˆëŸ¬?˜¤ê¸?
mUsageStats = pd.read_parquet(path + 'ch2025_mUsageStats.parquet')
mUsageStats['timestamp'] = pd.to_datetime(mUsageStats['timestamp'])

# ?”•?…”?„ˆë¦? ë¬¸ì?—´?„ ë¦¬ìŠ¤?Š¸ë¡? ë³??™˜
def parse_app_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except:
        return []

# ê°? row?—?„œ app_name?„ ì»¬ëŸ¼?œ¼ë¡? total_time?„ ê°’ìœ¼ë¡? ë¶„í•´
expanded_rows = []
for _, row in mUsageStats.iterrows():
    app_list = parse_app_list(row['m_usage_stats'])
    row_data = {'subject_id': row['subject_id'], 'timestamp': row['timestamp']}
    for app in app_list:
        if 'app_name' in app and 'total_time' in app:
            app_name = app['app_name'].strip()
            row_data[app_name] = app['total_time']
    expanded_rows.append(row_data)

# ?°?´?„°?”„? ˆ?„?œ¼ë¡? ë³??™˜
prep_mUsageStats = pd.DataFrame(expanded_rows)


# 2. mActivity : ?‹œê°„ë??ë³? ê°?ì¤‘í•©
print("Preprocessing mActivity...")
prep_mActivity = mActivity[['subject_id', 'timestamp']].copy()
prep_mActivity['met_activity'] = 0.0
activity_to_met = {0: 1.3, 1: 7.2, 2: 2.3, 3: 1.1, 4: 1.0, 5: 1.3, 7: 3.4, 8: 8.0} # ê°?ì¤‘ì¹˜

def get_time_weight(ts):
    hour = ts.hour
    if 0 <= hour < 8: return 0.3
    elif 8 <= hour < 18: return 0.7
    else: return 1.0

def calc_weighted_met(row):
    met = activity_to_met.get(mActivity.loc[row.name, 'm_activity'], 1.0)  # ë§¤í•‘ ?—†?œ¼ë©? 1.0
    weight = get_time_weight(row['timestamp'])
    return met * weight

prep_mActivity['met_activity'] = prep_mActivity.apply(calc_weighted_met, axis=1)
prep_mActivity = (prep_mActivity
    .groupby(['subject_id', pd.Grouper(key='timestamp', freq='10min')])
    .agg({'met_activity': 'sum'})
    .reset_index()
)



# 3. mBle : ê°?ê¹Œìš´ ê¸°ê¸°ê°? ?” ?˜?–¥?„ ë§ì´ ì£¼ë„ë¡? ê°?ì¤‘í•©
print("Preprocessing mBle...")
mBle['timestamp'] = mBle['timestamp'].dt.floor('10min')
prep_mBle = mBle[['subject_id', 'timestamp']].copy()
def sum_ble_rssi(ble_stats):
    sum_rssi = 0
    for ble in ble_stats:
        sum_rssi += np.exp(ble.get('rssi', 0) / 10)
    return sum_rssi

prep_mBle['m_wtb_rssi'] = mBle['m_ble'].apply(sum_ble_rssi)


# 4. mWifi : ê°?ê¹Œìš´ ê¸°ê¸°ê°? ?” ?˜?–¥?„ ë§ì´ ì£¼ë„ë¡? ê°?ì¤‘í•©
print("Preprocessing mWifi...")
mWifi['timestamp'] = mWifi['timestamp'].dt.floor('10min')
prep_mWifi = mWifi[['subject_id', 'timestamp']].copy()
def sum_wifi_rssi(wifi_stats):
    sum_rssi = 0
    for wifi in wifi_stats:
        sum_rssi += np.exp(wifi.get('rssi', 0) / 10)
    return sum_rssi

prep_mWifi['m_wtb_rssi'] = mWifi['m_wifi'].apply(sum_wifi_rssi)


# 5. wHr : ?‰ê· ê°’
print("Preprocessing wHr...")
wHr['heart_rate'] = wHr['heart_rate'].apply(lambda x: np.mean(x))
prep_wHr = (wHr.groupby(['subject_id', 
                         pd.Grouper(key='timestamp', freq='10min')])
            ['heart_rate']
            # .apply(list) -> ?‰ê· ê°’?œ¼ë¡? ?ˆ˜? •
            .mean()
            .reset_index()
        )


# 6. wPedo : distance, burned_caloriesë§? ?‰ê· ê°’
print("Preprocessing wPedo...")
wPedo['distance'] = wPedo['distance'].astype(float)
wPedo['burned_calories'] = wPedo['burned_calories'].astype(float)
prep_wPedo = (wPedo.groupby(['subject_id',
                         pd.Grouper(key='timestamp', freq='10min')])
            [['distance', 'burned_calories']]
            .sum()
            .reset_index()
        )

# 7. mGps : ?‰ê· ê°’
print("Preprocessing mGps...")
def mean_gps(row):
    df = pd.DataFrame(list(row))          
    return df[['latitude', 'longitude', 'altitude', 'speed']].mean()
3
mGps_extract = mGps['m_gps'].apply(mean_gps)   
mGps_1min = pd.concat([mGps, mGps_extract], axis=1)
prep_mGps = (mGps_1min.groupby(['subject_id',
                            pd.Grouper(key='timestamp', freq='10min')])
                [['latitude', 'longitude', 'altitude', 'speed']]
                .mean()
                .reset_index()
            )

# 8. wLight : ?›Œì¹˜ë§Œ ?‚¬?š©?•˜ê¸°ë¡œ ?•¨, ?‰ê· ê°’
print("Preprocessing wLight...")
prep_wLight = (wLight
            .groupby(['subject_id', 
                     pd.Grouper(key='timestamp', freq='10min')])
            [['w_light']]
            .mean()
            .reset_index()
        )

# 9. mAmbience : ëª¨ë“  ?¼ë²¨ì„ ì¹¼ëŸ¼?œ¼ë¡? ?ƒ?„±?•˜?—¬ ?‚¬?š© -> 10ë¶„ë‹¨?œ„ ?‰ê·? ë°? NaN?„ 0?œ¼ë¡? ì±„ì??(ë¯¼ì„ ?ˆ˜? •)
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

prep_mAmbience = pd.DataFrame([expand_m_ambience(row) for _, row in mAmbience.iterrows()])
prep_mAmbience = (prep_mAmbience
                  .groupby(['subject_id', 
                     pd.Grouper(key='timestamp', freq='10min')])
                  .mean()
                  .reset_index()
                  .fillna(0)  
                  )



## Merge all preprocessed dataframes --------------------------------
print("Merging preprocessed dataframes...")
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

## id one-hot encoding
for i in range(1, 11):  
    column_name = f'id{i:02d}'  
    df[column_name] = (df['subject_id'] == column_name).astype(int)


# ????¥?‹œ ì£¼ì„ ???ê¸?
df = df.rename(columns={'timestamp': 'lifelog_date'})
df.to_csv('merged_df_original.csv', index=False)

