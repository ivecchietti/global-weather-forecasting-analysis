# 🌦️ Weather Trend Forecasting

## 📌 Overview

This project focuses on forecasting weather trends using the **Global Weather Repository** dataset from Kaggle. The objective was to build a complete end-to-end machine learning pipeline including:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Forecasting with multiple models
- Spatial weather analysis
- Feature importance analysis

The project analyzes temporal and geographical weather patterns to predict future temperatures and identify the most influential variables affecting forecasting performance.

---

# 📂 Dataset

Dataset used:

- https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code

The dataset contains worldwide weather information including:

- Temperature
- Humidity
- Wind speed
- Pressure
- Air quality
- Precipitation
- Sunrise/Sunset
- Geographical coordinates
- Temporal information

and many additional meteorological variables.

---

# ⚙️ Features Implemented

## ✅ Data Preprocessing

Implemented a complete preprocessing pipeline including:

- Outlier detection using IQR
- Temporal sorting
- Feature scaling
- Frequency encoding
- One-hot encoding
- Cyclical feature engineering
- Lag feature generation
- Wind direction transformations
- Sunrise/Sunset numerical transformations

## Specification

Missing value handling turned out to be unnecesary, dince the dataset did not have any. 

### Feature Engineering

Key engineered features include:

- `temp_lag_1`
- `temp_lag_3`
- `temp_lag_24`
- `temp_roll_3`
- cyclical hour/month/day features
- wind direction sine/cosine encoding

---

## 📊 Exploratory Data Analysis (EDA)

Performed extensive EDA to understand weather behavior and temporal patterns:

- Temperature distributions
- Correlation heatmaps
- Global temperature evolution
- Country comparisons
- Seasonal analysis
- Outlier analysis
- Weather condition frequencies

---

## 🌍 Spatial Analysis

Implemented geographical weather analysis using:

- Latitude/Longitude distributions
- Spatial temperature heatmaps
- Global weather visualization
- Regional climate comparisons

This helped identify geographical temperature patterns and climate clusters worldwide.

---

## 🔍 Feature Importance Analysis

Feature importance was analyzed using tree-based models to determine which variables contributed most to forecasting performance.

Main findings showed that temporal and lag-based features dominate weather forecasting tasks.

---

# 🤖 Models Implemented

The following machine learning models were trained and evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

# 📈 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.618919 | 2.222449 | 0.940079 |
| Random Forest | 1.431488 | 2.051771 | 0.948929 |
| XGBoost | 1.325719 | 1.886768 | 0.956813 |



# 💡 Key Insights

- Lag features strongly dominate forecasting performance
- Weather forecasting shows high temporal dependency
- XGBoost achieved the best overall predictive performance
- Temperature evolution follows strong cyclical patterns
- Spatial analysis revealed clear regional climate differences
- Air quality variables showed moderate correlation with temperature changes

---

# 🛠️ Setup Instructions

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run The Project

Open the notebook:

```bash
jupyter notebook
```

Then run:

```bash
weather_forecasting.ipynb
```

---


# 📁 Project Structure

```text
├── notebooks/
│   └── weather_forecasting.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── metrics.py
│   └── utils.py
│
│
├── requirements.txt
└── README.md
```

---

# 🚀 Future Improvements

- Add ensemble forecasting models
- Deploy as an interactive dashboard
- Integrate real-time weather APIs
- Add advanced anomaly detection
- Improve spatial-temporal modeling

---

# 📄 Assessment Context

This project was developed as part of the PM Accelerator Data Scientist / Analyst Technical Assessment focused on weather forecasting and advanced data analysis.
