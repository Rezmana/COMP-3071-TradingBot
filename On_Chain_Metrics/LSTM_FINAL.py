from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from coinmetrics.api_client import CoinMetricsClient

### TESTING ###

client = CoinMetricsClient()
metrics = client.get_asset_metrics(assets='btc', metrics=['PriceUSD','HashRate','TxTfrCnt','AdrActCnt', 'SplyCur'], start_time="2015-01-01", end_time="2024-01-02", frequency='1d')
metrics = pd.DataFrame(metrics)
metrics.to_csv(r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\On_Chain_Metrics\testing strategies\LSTM_File_.csv", index=False)
df = metrics.copy()

model    = load_model(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\trading_agent\model"
    r"\Bit_Coin_Price_Prediction_LSTM_Model_4chain.keras"
)
scaler_X = joblib.load(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\trading_agent\scalers"
    r"\btc_scaler_X_4ChainMetrics.pkl"
)
scaler_y = joblib.load(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\trading_agent\scalers"
    r"\btc_scaler_y_4ChainMetrics.pkl"
)

# Scale the features and target
df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

df['TxTfrCnt'] = pd.to_numeric(df['TxTfrCnt'], errors='coerce')
df['AdrActCnt'] = pd.to_numeric(df['AdrActCnt'], errors='coerce')
df['SplyCur'] = pd.to_numeric(df['SplyCur'], errors='coerce')
df['PriceUSD'] = pd.to_numeric(df['PriceUSD'], errors='coerce')

df['TxTfrCnt_per_address'] = df['TxTfrCnt'] / df['AdrActCnt']
df['SplyCur_per_address'] = df['SplyCur'] / df['AdrActCnt']

# Target: future price (e.g., 1-day ahead)
df['PriceUSD_target'] = df['PriceUSD'].shift(-1)
df['TxTfrCnt_per_address'] = df['TxTfrCnt'] / df['AdrActCnt']
df['SplyCur_per_address'] = df['SplyCur'] / df['AdrActCnt']


# Add technical indicators
# Moving averages
df['MA_7'] = df['PriceUSD_target'].rolling(window=7).mean()
df['MA_30'] = df['PriceUSD_target'].rolling(window=30).mean()
# Trend_Strength
df['MA_7_30_ratio'] = df['MA_7'] / df['MA_30']

# Volatility
# Added volatility in the last 30 days
df['Volatility_30'] = df['PriceUSD_target'].rolling(window=30).std()

 # Drop rows with NaNs from shifting
df.dropna(inplace=True)

X = df[['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD']].values
y = df['PriceUSD_target'].values

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

import joblib

joblib.dump(scaler_X, "btc_scaler_X_PriceUSD_v2.pkl")
joblib.dump(scaler_y, "btc_scaler_y_PriceUSD_v2.pkl")


def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Choose an appropriate sequence length (e.g., 60 days)
time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
n_features = X_train.shape[2]

# Evaluation and make predictions
# Predict on test data
y_pred_scaled = model.predict(X_test)

# Inverse transform to get actual price values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

rmse = math.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
print(f"RMSE for testing: {rmse}")
print(f"MAE for testing: {mae}")

def directional_accuracy(y_true, y_pred):
    y_true_direction = np.sign(np.diff(y_true.flatten()))
    y_pred_direction = np.sign(np.diff(y_pred.flatten()))
    return np.mean(y_true_direction == y_pred_direction)

dir_acc = directional_accuracy(y_test_actual, y_pred)

print(f"RMSE for testing: {rmse}")
print(f"MAE for testing: {mae}")
print(f"Directional Accuracy: {dir_acc:.4f}")

# # Visualizing the result


# Get the dates for the test set
# We need to find the dates that correspond to our test set
test_dates = df['time'].iloc[-(len(y_test_actual)):]

# Print daily predicted vs actual prices
print("\nDaily Predicted vs Actual Prices:")
print(f"{'Date':<12} {'Actual Price (USD)':<20} {'Predicted Price (USD)':<20}")
print("-" * 55)

for date, actual, predicted in zip(test_dates, y_test_actual.flatten(), y_pred.flatten()):
    print(f"{date.date()} {actual:<20.2f} {predicted:<20.2f}")


# Visualizing the result with dates
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test_actual, label='Actual Price', linewidth=2)
plt.plot(test_dates, y_pred, label='Predicted Price', linewidth=2, alpha=0.8)
plt.title(f'Bitcoin Price Prediction (RMSE: {rmse:.2f}, MAE: {mae:.2f})')
plt.xlabel('Date')
plt.ylabel('Price USD')
plt.legend()
plt.grid(alpha=0.3)



# Format x-axis to show dates properly
plt.gcf().autofmt_xdate()
plt.tight_layout()

# Add text annotation for metrics
plt.figtext(0.15, 0.02, f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | Dir. Accuracy: {dir_acc:.4f}",
            ha="left", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.show()