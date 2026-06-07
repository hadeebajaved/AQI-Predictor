import pandas as pd
import os
import pickle
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

# --- 1. Fetch from Feature Store (MongoDB) ---
MONGO_URI = os.getenv("MONGO_URI")


print("Connecting to MongoDB to fetch features...")
client = MongoClient(MONGO_URI)
db = client['AQI_Project']
collection = db['Historical_Features']

# Convert MongoDB data back to Pandas DataFrame
data = list(collection.find({}, {'_id': 0}))
df = pd.DataFrame(data)

# YAHAN HUM TARGET WALI KHALI ROWS DROP KAR RAHE HAIN SIRF TRAINING KE LIYE
df = df.dropna(subset=['Target_PM2.5_Next_Day'])

print(f"Data loaded successfully! Total Rows for Training: {df.shape[0]}")

# --- 2. Train-Test Split ---
print("Preparing data for training...")
# Input Features (X) aur Target Output (y)
features = ['PM2.5', 'PM10', 'CO', 'NO2', 'hour', 'day', 'month', 'PM2.5_Change']
X = df[features]
y = df['Target_PM2.5_Next_Day']

# 80% data training ke liye, 20% testing ke liye
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train Machine Learning Model ---
print("Training Random Forest Model (This might take a few seconds)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluate Performance ---
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"RMSE (Error): {rmse:.2f}")
print(f"MAE (Absolute Error): {mae:.2f}")
print(f"R2 Score (Accuracy): {r2:.2f}")

# --- 5. Save the Trained Model ---
print("\nSaving the trained model to model.pkl...")
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("🎉 model.pkl generated successfully!")
