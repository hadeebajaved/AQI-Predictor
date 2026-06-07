import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime, timedelta
import random

# --- 1. Page Configuration ---
st.set_page_config(page_title="Lahore AQI AI", page_icon="🌍", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .forecast-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Smart Model Loading (Bypass Local Error) ---
@st.cache_resource
def load_model():
    try:
        import pickle
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        return None # Return None if Windows blocks it locally

model = load_model()

# --- 3. Fetch Data from MongoDB (Timezone Fixed) ---
@st.cache_data(ttl=60)
def get_recent_data():
    MONGO_URI = st.secrets["MONGO_URI"]
    client = MongoClient(MONGO_URI)
    db = client['AQI_Project']
    collection = db['Historical_Features']
    
    # 48 ghante ka data fetch kar rahe hain taake future predictions filter ho sakein
    records = list(collection.find({}, {'_id': 0}).sort([("datetime", -1)]).limit(48))
    
    if records:
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None) 
        
        # FIX: Explicitly Pakistan Time (Asia/Karachi) set kiya hai taake UTC server issue na ho
        current_time = pd.Timestamp.now(tz='Asia/Karachi').tz_localize(None)
        df = df[df['datetime'] <= current_time]
        
        if not df.empty:
            # Ab jo sab se upar row hogi, wo exactly "Abhi" (Current Hour) ki hogi
            return df.head(1)
            
    return pd.DataFrame()

df_data = get_recent_data()

# --- 4. Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3203/3203071.png", width=100)
    st.title("🌱 10Shine AQI")
    st.write("Real-time Lahore Air Quality and AI Forecast.")
    if model is None:
        st.warning("⚠️ Local Test Mode: ML Model is disabled to prevent Windows error. Visuals are simulated for design testing. Real predictions will work on Cloud.")

# --- 5. Main UI Header ---
st.title("🌍 Lahore Real-Time AQI & AI Forecast")
st.markdown("Automated 24-Hour & 3-Day Environmental Outlook")
st.divider()

if df_data.empty:
    st.error("⚠️ Loading Data... Please check database connection or wait a moment.")
else:
    latest_row = df_data.iloc[0]
    
    # --- AUTO-GENERATE FUTURE PREDICTIONS ---
    features = ['PM2.5', 'PM10', 'CO', 'NO2', 'hour', 'day', 'month', 'PM2.5_Change']
    
    # Yahan bhi Pakistan ka time laga diya gaya hai
    current_time_pkt = pd.Timestamp.now(tz='Asia/Karachi').tz_localize(None)
