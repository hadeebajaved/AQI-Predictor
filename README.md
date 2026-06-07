# 🌍 End-to-End Air Quality Index (AQI) Prediction and Real-Time Forecasting System for Lahore

[![View Live App](https://img.shields.io/badge/View_Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://aqi-predictor-u2yvyxwfzqtskvxkddpkv3.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF.svg)

**Developed By:** Hadeeba Javed  
**Program:** 10Shine Data Science Internship Program (Cohort 8) by 10Pearls  

## 📌 Project Overview
Air pollution, specifically high concentrations of Particulate Matter (PM2.5 and PM10), poses a significant environmental challenge in Lahore. This project presents a robust, automated, and scalable **Machine Learning Operations (MLOps) pipeline** to predict PM2.5 levels for the next 24 hours and forecast general air quality trends for the upcoming 3 days.

**🔗 Live Project Link:** [Click here to view the real-time AI forecast](https://aqi-predictor-u2yvyxwfzqtskvxkddpkv3.streamlit.app/)

The system autonomously fetches real-time environmental data, continuously trains the predictive model using a Random Forest Regressor, and displays actionable insights on an interactive web dashboard.

## ✨ Key Features
* **Real-Time Live Dashboard:** Displays current PM2.5, PM10, CO, and NO2 levels.
* **24-Hour AI Forecast:** Interactive area charts predicting hourly PM2.5 trends.
* **3-Day Outlook:** Color-coded AQI severity predictions (Good, Moderate, Unhealthy).
* **Fully Automated CI/CD Pipelines:** Zero manual intervention required for data updates or model training.

## 🏗️ System Architecture & MLOps Pipeline
This project utilizes a modern MLOps architecture rather than a conventional static Machine Learning model:

1. **Data Storage (Feature Store):** **MongoDB** is utilized as a centralized NoSQL database to securely store historical environmental features and continuous data streams.
2. **Continuous Automation (GitHub Actions):** 
   * `Hourly Feature Pipeline`: Fetches new environmental data via API and updates the MongoDB database every hour.
   * `Daily Model Training`: Retrains the Scikit-Learn **Random Forest** model every 24 hours to ensure adaptability to recent atmospheric changes.
3. **Frontend Deployment:** Deployed on **Streamlit Cloud** for an accessible and intuitive user interface.

## 🛠️ Technology Stack
* **Programming Language:** Python 3.11
* **Machine Learning:** Scikit-Learn, Pandas
* **Data Visualization:** Plotly
* **Database:** MongoDB (PyMongo)
* **Automation:** GitHub Actions (YAML)
* **Web Framework:** Streamlit
## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/hadeebajaved/AQI-Predictor.git
cd AQI-Predictor

**2. Install Dependencies:**
```bash
pip install -r requirements.txt

