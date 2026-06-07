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

# --- 3. Fetch Data from MongoDB ---
@st.cache_data(ttl=60)
def get_recent_data():
    MONGO_URI = "mongodb+srv://admin:Hadeeba.121@cluster0.eip7e7u.mongodb.net/?appName=Cluster0"
    client = MongoClient(MONGO_URI)
    db = client['AQI_Project']
    collection = db['Historical_Features']
    
    records = list(collection.find({}, {'_id': 0}).sort([("datetime", -1)]).limit(1))
    if records:
        return pd.DataFrame(records)
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
    st.error("⚠️ Loading Data...")
else:
    latest_row = df_data.iloc[0]
    
    # --- AUTO-GENERATE FUTURE PREDICTIONS ---
    features = ['PM2.5', 'PM10', 'CO', 'NO2', 'hour', 'day', 'month', 'PM2.5_Change']
    current_time = datetime.now()
    
    future_24h = []
    for i in range(1, 25):
        fut_time = current_time + timedelta(hours=i)
        input_data = latest_row.copy()
        input_data['hour'] = fut_time.hour
        input_data['day'] = fut_time.day
        input_data['month'] = fut_time.month
        
        if model is not None:
            pred = model.predict(pd.DataFrame([input_data])[features])[0]
        else:
            # Simulated dummy data for UI testing if model is blocked locally
            pred = latest_row['PM2.5'] + random.uniform(-15, 15) + (i * 0.5) 
            
        future_24h.append({"Time": fut_time, "Hour": fut_time.strftime("%I %p"), "Predicted PM2.5": round(pred, 2)})
    
    df_24h = pd.DataFrame(future_24h)

    future_3d = []
    for i in range(1, 4):
        fut_date = current_time + timedelta(days=i)
        input_data = latest_row.copy()
        input_data['day'] = fut_date.day
        input_data['month'] = fut_date.month
        
        if model is not None:
            pred = model.predict(pd.DataFrame([input_data])[features])[0]
        else:
            pred = latest_row['PM2.5'] + random.uniform(-25, 25)
            
        future_3d.append({"Date": fut_date.strftime("%A, %d %b"), "Predicted PM2.5": round(pred, 1)})
    
    # --- TABS FOR DASHBOARD ---
    tab1, tab2, tab3 = st.tabs(["📊 Current Status", "📈 Next 24 Hours", "📅 3-Day Forecast"])
    
    # TAB 1: CURRENT LIVE DATA
    with tab1:
        st.subheader("Live Environmental Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌫️ PM2.5 (AQI Proxy)", f"{latest_row['PM2.5']:.1f} µg/m³")
        col2.metric("😷 PM10 Level", f"{latest_row['PM10']:.1f} µg/m³")
        col3.metric("🚗 Carbon Monoxide (CO)", f"{latest_row['CO']:.2f} µg/m³")
        col4.metric("🏭 Nitrogen Dioxide (NO2)", f"{latest_row['NO2']:.1f} µg/m³")

    # TAB 2: NEXT 24 HOURS GRAPH
    with tab2:
        st.subheader("Hourly PM2.5 Prediction (Next 24 Hours)")
        fig = px.area(df_24h, x='Time', y='Predicted PM2.5', markers=True, line_shape="spline", title="AI Forecasted Hourly Trend")
        fig.update_traces(line_color='#2e86c1', fillcolor='rgba(46, 134, 193, 0.2)')
        fig.update_layout(xaxis_title="Time", yaxis_title="Predicted PM2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: NEXT 3 DAYS
    with tab3:
        st.subheader("Upcoming 3-Day Outlook")
        cols = st.columns(3)
        for idx, day_data in enumerate(future_3d):
            val = day_data['Predicted PM2.5']
            status = "🟢 Good" if val <= 50 else "🟡 Moderate" if val <= 100 else "🔴 Unhealthy"
            color = "#d4edda" if val <= 50 else "#fff3cd" if val <= 100 else "#f8d7da"
            
            with cols[idx]:
                st.markdown(f"""
                <div class="forecast-box" style="background-color: {color};">
                    <h3>{day_data['Date']}</h3>
                    <h2>{val} µg/m³</h2>
                    <p><b>{status}</b></p>
                </div>
                """, unsafe_allow_html=True)