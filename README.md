# 📈Time Series Forecasting: SARIMA vs Prophet vs LSTM

This project evaluates and compares three time series forecasting models — **SARIMA**, **Prophet**, and **LSTM** — for predicting economic indicators such as inflation. The aim is to identify the best-performing model based on standard evaluation metrics.

---

##  Overview

Forecasting future values of economic time series data is crucial for planning and policy-making. This project:

- Preprocesses historical data (e.g., inflation rates).
- Trains SARIMA, Prophet, and LSTM models.
- Compares the models using **MAE**, **RMSE**, and **MAPE**.
- Visualizes forecast vs actual trends.
- Outputs model performance in tabular form.

---

##  Model Performance Summary

| Model   | MAE       | RMSE      | MAPE (%) |
|---------|-----------|-----------|-----------|
| SARIMA  | 4564.71   | 6010.32   | 0.75      |
| Prophet | 16648.70  | 17130.43  | 2.72      |
| LSTM    | 55911.64  | 58440.47  | NaN       |

> **Note**: SARIMA performs best across all metrics.

---

##  Models Used

### 1. SARIMA
- Classical statistical model for seasonality + trend.
- Ideal for stable, linear time series.

### 2. Prophet (by Meta)
- Decomposable time series model (trend + seasonality + holidays).
- Robust to missing data and outliers.

### 3. LSTM (Long Short-Term Memory)
- Deep learning model for sequential data.
- Useful for capturing nonlinear patterns.
- Requires more data and tuning.

---




