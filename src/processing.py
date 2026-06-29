import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
file_path = BASE_DIR / "data" / "Afficionado_Coffee_Roasters.parquet"

def load_data(path):
    return pd.read_parquet(path)
df = load_data(file_path)

# 1. Convert to datetime and extract hour
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')

# 2. Ensure sorted
df = df.sort_values(by=['transaction_id'])
df = df.drop_duplicates(subset=['transaction_id'])

# 3. Convert to datetime if you haven't (just in case)
base_date = pd.Timestamp("2025-01-01")

# 4. Calculate the time difference between rows
time_diff = df['transaction_time'].diff()

# 5. Define the 6-hour gap (in seconds: 6 * 3600 = 21600)
new_day_flag = (time_diff < pd.Timedelta(0)) | (time_diff > pd.Timedelta(hours=6))

# 6. Create 'day_id' by cumulative sum of the flag
df['day_id'] = new_day_flag.cumsum()

# 7. Create the continuous 'full_timestamp'
df['full_timestamp'] = (base_date + pd.to_timedelta(df['day_id'], unit='D') +pd.to_timedelta(df['transaction_time'].dt.strftime('%H:%M:%S')))
df = df.sort_values('full_timestamp')

# 8. Indicator Features
df['hour'] = df['full_timestamp'].dt.hour
df['day'] = df['full_timestamp'].dt.day
df['date'] = df['full_timestamp'].dt.date
df['day_of_week'] = df['full_timestamp'].dt.dayofweek
day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['day_name'] = df['day_of_week'].map(day_map)

# 9. Store-specific Dummy Variables
df['store_id_1'] = df['store_id']
df = pd.get_dummies(df, columns=['store_id_1'], prefix='store')

df = df.drop(columns=['transaction_time'])

print("Timeline reconstructed successfully!")
print(df.info())
print(df.head())

csv_path = BASE_DIR / "data" / "Coffee_Clean.csv"
parquet_path = BASE_DIR / "data" / "Coffee_Clean.parquet"
df.to_csv(csv_path, index=False)
df.to_parquet(parquet_path, index=False)