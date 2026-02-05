# Smart-Inventory-Demand-Forecasting-with-AI
This project is an AI-powered inventory management and demand forecasting tool built for small and medium-sized retailers. Using historical sales data and machine learning, it predicts short-term product demand and helps store owners make data-driven restocking decisions.

This project builds a forecasting system for retail sales using Machine Learning and Deep Learning. It helps small retailers:
- Forecast demand for the next 7 days
- Detect inventory shortages
- Visualize sales and predictions
- Get restock alerts via dashboard

## 🚀 Features
- LSTM-based deep learning time-series model (TensorFlow/Keras)
- Random Forest forecasting for baseline
- Real-world retail dataset (Walmart M5 Forecasting)
- Streamlit dashboard with alert system
- Feature engineering: lags, rolling windows, prices
- Train/test evaluation with RMSE

## 📊 Tech Stack
- Python, Pandas, NumPy
- TensorFlow / Keras, scikit-learn, Random Forest
- Matplotlib, Seaborn

## 📥 Download Dataset
This project uses the **M5 Forecasting - Accuracy** dataset from Kaggle.

👉 [Kaggle Dataset Link](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

- After downloading, place the following files in the `data/` directory:
  - `sales_train_validation.csv`
  - `calendar.csv`
  - `sell_prices.csv`


---

## 📊 Visuals

### 1. Random Forest – Actual vs Predicted

<img width="1205" alt="Random Forest – Actual vs Predicted
" src="https://github.com/Anmolpandey23/Smart-Inventory-Demand-Forecasting-with-AI/blob/main/Random%20Forest.png?raw=true" />

---

### 2. LSTM – Actual vs Predicted

<img width="1205" alt="LSTM – Actual vs Predicted" src="https://github.com/Anmolpandey23/Smart-Inventory-Demand-Forecasting-with-AI/blob/main/LSTM%20Model.png?raw=true"/>

---

### 3. LSTM – 7-Day Forecast

<img width="1205" alt="LSTM – 7-Day Forecast" src="https://github.com/Anmolpandey23/Smart-Inventory-Demand-Forecasting-with-AI/blob/main/LSTM%20Next%207%20day%20Forecast.png?raw=true" />

---

## ⚙️ How It Works

### 📌 Data Preparation
- Converts wide format to long using `melt`
- Merges calendar and pricing info
- Creates lag & rolling mean features

### 🧠 Model Training
- `RandomForestRegressor` on feature-engineered data
- `LSTM` with 30-day sequence sliding window

### 📦 Inventory Alert
- User inputs current stock
- System checks demand > inventory → ⚠️ alert

---

## 🚀 Running the App

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Smart-Inventory-Demand-Forecasting-with-AI.git
cd Smart-Inventory-Demand-Forecasting-with-AI

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

___

## 📘 License

This project is licensed under the [MIT License](https://github.com/Anmolpandey23/Smart-Inventory-Demand-Forecasting-with-AI/blob/169006428a6b099cb8abd9e8a8eb8bedb39087e5/LICENSE).
