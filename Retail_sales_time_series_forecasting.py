# Retail Sales Forecasting Dashboard

# Import libraries
import os
import requests
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from fredapi import Fred
import tensorflow as tf

# Fix TensorFlow threading for Streamlit compatibility
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load API key from environment variable
fred_api_key = os.getenv("FRED_API_KEY")
if not fred_api_key:
    raise ValueError("FRED_API_KEY environment variable not set")
fred = Fred(api_key=fred_api_key)

# Handle warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration dictionary
CONFIG = {
    "FRED_API_KEY": os.getenv("FRED_API_KEY", ""),
    "SERIES_ID": "RSXFS",
    "DEFAULT_START_DATE": "1992-01-01",
    "DEFAULT_END_DATE": "2025-04-01",
    "FORECAST_HORIZON": 12,
    "LSTM_N_STEPS": 12,
    "LSTM_EPOCHS": 10,  # Reduced for Streamlit Cloud
    "LSTM_BATCH_SIZE": 32,  # Increased for efficiency
    "LSTM_UNITS": 32,  # Reduced for lower resource usage
    "SEASONAL_PERIOD": 12,
    "SARIMA_ORDER": (1, 1, 1),
    "SARIMA_SEASONAL_ORDER": (1, 1, 1, 12)
}

# Data Fetching
@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.to_json()})
def fetch_fred_retail_sales(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from FRED (Advance Monthly Retail Sales)."""
    logger.info("Fetching FRED data")
    try:
        if not CONFIG["FRED_API_KEY"]:
            st.error("FRED API key not found. Set FRED_API_KEY environment variable.")
            return pd.DataFrame()

        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": CONFIG["SERIES_ID"],
            "observation_start": start_date,
            "observation_end": end_date,
            "file_type": "json",
            "api_key": CONFIG["FRED_API_KEY"]
        }
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()["observations"]
        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        logger.info("FRED data fetched successfully")
        return df.dropna()
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}", exc_info=True)
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Data Preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to time series: lags, rolling stats, month/year."""
    logger.info("Preprocessing data")
    try:
        df = df.dropna(subset=["value"]).copy()
        df.set_index("date", inplace=True)
        for lag in [1, 3, 12]:
            df[f"lag_{lag}"] = df["value"].shift(lag)
        df["roll_mean_3"] = df["value"].rolling(3).mean()
        df["roll_std_12"] = df["value"].rolling(12).std()
        df["month"] = df.index.month
        df["year"] = df.index.year
        logger.info("Data preprocessing completed")
        return df.dropna()
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}", exc_info=True)
        raise

def train_sarima(train: pd.Series):
    """Train SARIMA model and return forecast and confidence intervals."""
    logger.info("Starting SARIMA training")
    try:
        model = SARIMAX(train, order=CONFIG["SARIMA_ORDER"], seasonal_order=CONFIG["SARIMA_SEASONAL_ORDER"],
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        pred = fit.get_forecast(steps=CONFIG["FORECAST_HORIZON"])
        logger.info("SARIMA training completed")
        return pred.predicted_mean, pred.conf_int().iloc[:, 0], pred.conf_int().iloc[:, 1]
    except Exception as e:
        logger.error(f"Error in train_sarima: {str(e)}", exc_info=True)
        raise

def train_prophet(train: pd.Series):
    """Train Prophet model."""
    logger.info("Starting Prophet training")
    try:
        df = train.reset_index().rename(columns={"date": "ds", "value": "y"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=CONFIG["FORECAST_HORIZON"], freq="MS")
        forecast = model.predict(future).set_index("ds")
        logger.info("Prophet training completed")
        return (forecast["yhat"][-CONFIG["FORECAST_HORIZON"]:],
                forecast["yhat_lower"][-CONFIG["FORECAST_HORIZON"]:],
                forecast["yhat_upper"][-CONFIG["FORECAST_HORIZON"]:])
    except Exception as e:
        logger.error(f"Error in train_prophet: {str(e)}", exc_info=True)
        raise

def train_lstm(train: pd.Series, n_steps: int):
    """Train LSTM model."""
    logger.info("Starting LSTM training")
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train.values.reshape(-1, 1))
        logger.info(f"Scaled data shape: {scaled.shape}")

        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:i+n_steps])
                y.append(data[i+n_steps])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled, n_steps)
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        train_size = len(X) - CONFIG["FORECAST_HORIZON"]
        if train_size <= 0:
            raise ValueError("Not enough data for LSTM training after splitting")
        X_train, y_train = X[:train_size], y[:train_size]
        X_test = X[train_size:]
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        model = Sequential([
            LSTM(CONFIG["LSTM_UNITS"], activation='relu', input_shape=(n_steps, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        logger.info("LSTM model compiled")
        model.fit(X_train, y_train, epochs=CONFIG["LSTM_EPOCHS"], batch_size=CONFIG["LSTM_BATCH_SIZE"], verbose=0)
        logger.info("LSTM training completed")

        forecast = []
        current_seq = X_test[0].copy()
        for _ in range(CONFIG["FORECAST_HORIZON"]):
            pred = model.predict(current_seq.reshape(1, n_steps, 1), verbose=0)
            forecast.append(pred[0, 0])
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred[0, 0]

        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecast = pd.Series(forecast, index=pd.date_range(start=train.index[-1], periods=CONFIG["FORECAST_HORIZON"], freq='MS'))
        std = np.std(forecast)
        lower = forecast - 1.96 * std
        upper = forecast + 1.96 * std
        logger.info("LSTM forecast completed")
        return forecast, lower, upper
    except Exception as e:
        logger.error(f"Error in train_lstm: {str(e)}", exc_info=True)
        raise

# Evaluation
def evaluate_forecast(true: pd.Series, pred: pd.Series) -> dict:
    """Return evaluation metrics: MAE, RMSE, MAPE."""
    logger.info("Evaluating forecast")
    try:
        y_true = np.array(true)
        y_pred = np.array(pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan

        logger.info(f"MAPE: {mape:.2f}%")
        return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    except Exception as e:
        logger.error(f"Error in evaluate_forecast: {str(e)}", exc_info=True)
        raise

def plot_acf_pacf(series):
    """Plot ACF and PACF with Plotly."""
    logger.info("Plotting ACF and PACF")
    try:
        lags = 24
        acf_vals = acf(series, nlags=lags)
        pacf_vals = pacf(series, nlags=lags)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(lags+1)), y=acf_vals, name="ACF"))
        fig.add_trace(go.Bar(x=list(range(lags+1)), y=pacf_vals, name="PACF"))
        fig.update_layout(title="ACF and PACF", barmode='group')
        return fig
    except Exception as e:
        logger.error(f"Error in plot_acf_pacf: {str(e)}", exc_info=True)
        raise

# Streamlit app
def main():
    st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
    logger.info("Starting main function")

    # Define models dictionary at higher scope
    models = {
        "SARIMA": train_sarima,
        "Prophet": train_prophet,
        # "LSTM": lambda x: train_lstm(x, CONFIG["LSTM_N_STEPS"])  # Uncomment after testing
    }

    # Sidebar Inputs
    with st.sidebar:
        st.header("Settings")
        start_date = st.date_input("Start Date", pd.to_datetime(CONFIG["DEFAULT_START_DATE"]))
        end_date = st.date_input("End Date", pd.to_datetime(CONFIG["DEFAULT_END_DATE"]))
        CONFIG["FORECAST_HORIZON"] = st.slider("Forecast Horizon (months)", 1, 24, CONFIG["FORECAST_HORIZON"])

    # Fetch and preprocess data
    logger.info("Fetching and preprocessing data")
    df = fetch_fred_retail_sales(str(start_date), str(end_date))
    if df.empty:
        logger.error("No data fetched, exiting")
        return
    df = preprocess_data(df)
    train = df["value"].iloc[:-CONFIG["FORECAST_HORIZON"]]
    test = df["value"].iloc[-CONFIG["FORECAST_HORIZON"]:]

    # Tabs
    tab1, tab2, tab3 = st.tabs(["EDA", "Model Comparison", "Forecast"])

    with tab1:
        st.subheader("Exploratory Data Analysis")

        st.plotly_chart(px.line(df.reset_index(), x="date", y="value", title="Retail Sales Over Time"))

        decomp = STL(df["value"], period=12).fit()
        fig = decomp.plot()
        st.pyplot(fig)

        stat, pval, *_ = adfuller(df["value"])
        st.info(f"ADF Statistic = {stat:.4f}, p-value = {pval:.4f}")

        st.plotly_chart(plot_acf_pacf(df["value"]))

        heatmap_data = df.reset_index().pivot_table(index='year', columns='month', values='value')
        st.plotly_chart(px.imshow(heatmap_data, labels=dict(x="Month", y="Year", color="Sales"), title="Seasonality Heatmap"))

    with tab2:
        st.subheader("Model Performance Comparison")
        if st.button("Run Model Comparison"):
            logger.info("Running model comparison")
            rows = []
            for name, func in models.items():
                logger.info(f"Running model: {name}")
                forecast, *_ = func(train)
                metrics = evaluate_forecast(test, forecast)
                metrics["Model"] = name
                rows.append(metrics)
            st.table(pd.DataFrame(rows).set_index("Model").style.format("{:.2f}"))
            logger.info("Model comparison completed")

    with tab3:
        st.subheader("Forecast vs Actual")
        model_choice = st.selectbox("Select Model", ["SARIMA", "Prophet"])  # Update when LSTM is re-enabled
        forecast, lower, upper = models[model_choice](train)
        fc_df = pd.DataFrame({"date": forecast.index, "forecast": forecast, "lower": lower, "upper": upper})
        actual = df["value"].rename("actual")
        combined = pd.concat([actual, fc_df.set_index("date")], axis=1)

        fig = px.line(combined.reset_index(), x="date", y=["actual", "forecast"], title="Forecast vs Actual")
        fig.add_scatter(x=fc_df["date"], y=fc_df["lower"], mode='lines', name='Lower CI', line=dict(dash='dot'))
        fig.add_scatter(x=fc_df["date"], y=fc_df["upper"], mode='lines', name='Upper CI', line=dict(dash='dot'))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download Forecast", combined.to_csv(index=True), "forecast.csv")

if __name__ == "__main__":
    main()