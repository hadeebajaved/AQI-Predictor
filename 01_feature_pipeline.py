import pandas as pd
import os
import requests
from pymongo import MongoClient

# --- 1. Fetching Raw Data (Lahore) ---
LAT = 31.5497
LON = 74.3436

print("Fetching last 90 days of data from Open-Meteo API...")
# Sirf Air Quality variables fetch kar rahe hain taake API error na de
url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&past_days=90&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide&timezone=auto"
response = requests.get(url)
data = response.json()

# --- 2. Feature Generation ---
print("Generating features...")
df = pd.DataFrame(data['hourly'])

# Rename columns
df = df.rename(columns={
    "time": "datetime",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2"
})
df['datetime'] = pd.to_datetime(df['datetime'])

# Time-based features
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month

# Safely handle missing values (khaali spaces ko pichlay ghante ki value se bhar dein)
df = df.ffill().bfill()

# Derived Feature: AQI Change Rate
df['PM2.5_Change'] = df['PM2.5'].diff().fillna(0)

# Target: Predicting next 24 hours PM2.5
df['Target_PM2.5_Next_Day'] = df['PM2.5'].shift(-24)

# HUM NE YAHAN SE DROPNA HATA DIYA HAI TAQAY LATEST DATA DELETE NA HO

print(f"Data ready! Total Rows: {df.shape[0]}")

if not df.empty:
    # # --- 3. Push to Feature Store ---
    MONGO_URI = os.getenv("MONGO_URI")

    
    print("Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_URI)
    db = client['AQI_Project']
    collection = db['Historical_Features']

    print("Uploading data to Feature Store...")
    collection.delete_many({})
    records = df.to_dict('records')
    collection.insert_many(records)

    print(f"🎉 Success! {len(records)} records successfully pushed to MongoDB.")
else:
    print("Error: DataFrame is empty. Check API response.")
