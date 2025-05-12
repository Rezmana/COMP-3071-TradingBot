import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

from coinmetrics.api_client import CoinMetricsClient
client = CoinMetricsClient()

model = load_model(r'trading_agent\model\Bit_Coin_Price_Prediction_LSTM_Model_4chain.keras')
scaler_X = joblib.load(r'trading_agent\scalers\btc_scaler_X_4ChainMetrics.pkl')
scaler_y = joblib.load(r'trading_agent\scalers\btc_scaler_y_4ChainMetrics.pkl')


# 'SplyCntCur' - this is for the supply count of the current coin
# 'CapMrktCurUSD' - this is for the market capitalization of the current coin
# 'PriceUSD' - this is for the price of the current coin
# 'TxTfrValAdjUSD' - this is for the adjusted transaction value in USD
# 'TxTfrValUSD' - this is for the transaction value in USD
# 'TxTfrCnt' - this is for the transaction count
# 'IssContNtv' - this is for the native issuance count
# 'IssContPctAnn' - this is for the annual percentage of the issuance count

metrics_eth = client.get_asset_metrics(assets='eth', metrics=['PriceUSD','HashRate','TxTfrCnt','AdrActCnt', 'SplyCur'], start_time="2017-01-01", end_time="2024-01-02", frequency='1d')

# Convert the DataCollection to a pandas DataFrame
metrics_eth = pd.DataFrame(metrics_eth)
metrics_eth['time'] = pd.to_datetime(metrics_eth['time'])
metrics_eth.sort_values('time', inplace=True)

# Create features
metrics_eth['PriceUSD_target'] = metrics_eth['PriceUSD'].shift(-1)
metrics_eth.dropna(inplace=True)

# Select same features used in training
# X = df[['HashRate','AdrActCnt', 'PriceUSD', 'TxTfrCnt_per_address', 'SplyCur_per_address']].values
X_eth = metrics_eth[['PriceUSD','HashRate','TxTfrCnt','AdrActCnt', 'SplyCur']].values
y_eth = metrics_eth['PriceUSD_target'].values

# Scale with the SAME scaler (important!)
X_eth_scaled = scaler_X.transform(X_eth)
y_eth_scaled = scaler_y.transform(y_eth.reshape(-1, 1))

time_steps = 60  # or whatever you used for Bitcoin

def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

X_eth_seq, y_eth_seq = create_sequences(X_eth_scaled, y_eth_scaled, time_steps)

# Predict ETH using Bitcoin-trained model
y_pred_eth_scaled = model.predict(X_eth_seq)

# Inverse transform to get actual prices
y_pred_eth = scaler_y.inverse_transform(y_pred_eth_scaled)
y_eth_actual = scaler_y.inverse_transform(y_eth_seq.reshape(-1, 1))

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

eth_rmse = np.sqrt(mean_squared_error(y_eth_actual, y_pred_eth))
eth_mae = mean_absolute_error(y_eth_actual, y_pred_eth)

print(f"ETH Testing RMSE: {eth_rmse:.2f}")
print(f"ETH Testing MAE: {eth_mae:.2f}")

# Predict ETH using Bitcoin-trained model
y_pred_eth_scaled = model.predict(X_eth_seq)

# Inverse transform to get actual prices
y_pred_eth = scaler_y.inverse_transform(y_pred_eth_scaled)
y_eth_actual = scaler_y.inverse_transform(y_eth_seq.reshape(-1, 1))

# Create a DataFrame with dates, actual prices, and predicted prices
results_df = pd.DataFrame({
    'Date': metrics_eth['time'].iloc[time_steps:].values,  # Skip first time_steps days
    'Actual_Price': y_eth_actual.flatten(),
    'Predicted_Price': y_pred_eth.flatten()
})

# Calculate percentage error and absolute error
results_df['Percentage_Error'] = ((results_df['Predicted_Price'] - results_df['Actual_Price']) / 
                                 results_df['Actual_Price']) * 100
results_df['Absolute_Error_USD'] = results_df['Predicted_Price'] - results_df['Actual_Price']

# Print header
print("\nDaily Ethereum Price Predictions:")
print("=" * 80)
print(f"{'Date':<12} | {'Actual Price ($)':>15} | {'Predicted Price ($)':>18} | {'Error ($)':>10} | {'Error (%)':>9}")
print("-" * 80)

# Print each row
for _, row in results_df.iterrows():
    print(f"{row['Date'].strftime('%Y-%m-%d')} | {row['Actual_Price']:15.2f} | {row['Predicted_Price']:18.2f} | "
          f"{row['Absolute_Error_USD']:10.2f} | {row['Percentage_Error']:8.2f}%")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average Absolute Error: ${results_df['Absolute_Error_USD'].abs().mean():.2f}")
print(f"Average Percentage Error: {results_df['Percentage_Error'].abs().mean():.2f}%")
print(f"Maximum Overestimation: {results_df['Percentage_Error'].max():.2f}%")
print(f"Maximum Underestimation: {results_df['Percentage_Error'].min():.2f}%")

# ... (rest of your existing code for metrics and plotting remains the same) ...
# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd

eth_rmse = np.sqrt(mean_squared_error(y_eth_actual, y_pred_eth))
eth_mae = mean_absolute_error(y_eth_actual, y_pred_eth)
print(f"ETH Testing RMSE: {eth_rmse:.2f}")
print(f"ETH Testing MAE: {eth_mae:.2f}")

# Create date range for x-axis (if you have actual dates, use those instead)
# This assumes your test data represents consecutive days
# If you have actual dates in your DataFrame, use those instead
try:
    # Attempt to get actual dates if available in your data
    dates = eth_data.iloc[-len(y_eth_actual):]['time'].values
except:
    # If not available, create a date range
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(y_eth_actual))

# Create a visualization
plt.figure(figsize=(14, 7))

# Plot the actual and predicted prices
plt.plot(dates, y_eth_actual, label='Actual ETH Price', color='blue', linewidth=2)
plt.plot(dates, y_pred_eth, label='Predicted ETH Price', color='orange', linewidth=2, linestyle='--')

# Add a vertical line to indicate the end of training data if applicable
# plt.axvline(x=train_end_date, color='red', linestyle='-', alpha=0.3, label='Train-Test Split')

# Add graph elements
plt.title(f'Ethereum Price Prediction using Bitcoin-Trained Model\nRMSE: ${eth_rmse:.2f}, MAE: ${eth_mae:.2f}', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Format the x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Add annotations showing percentage error at a few points
n_points = 5  # Number of points to annotate
step = len(y_eth_actual) // (n_points + 1)
for i in range(step, len(y_eth_actual), step):
    error_pct = (y_pred_eth[i][0] - y_eth_actual[i][0]) / y_eth_actual[i][0] * 100
    plt.annotate(f'{error_pct:.1f}%',
                 xy=(dates[i], y_pred_eth[i][0]),
                 xytext=(10, 0), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

# Show the max and min prediction errors
max_error_idx = np.abs(y_pred_eth - y_eth_actual).argmax()
plt.scatter(dates[max_error_idx], y_pred_eth[max_error_idx], color='red', s=100, zorder=5)
plt.annotate(f'Max Error: ${abs(y_pred_eth[max_error_idx][0] - y_eth_actual[max_error_idx][0]):.2f}',
             xy=(dates[max_error_idx], y_pred_eth[max_error_idx]),
             xytext=(20, 20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.savefig('eth_price_prediction.png', dpi=300)
plt.show()

# Additional plot: Error over time
plt.figure(figsize=(14, 5))
prediction_error = (y_pred_eth - y_eth_actual).reshape(-1)
plt.plot(dates, prediction_error, color='red', label='Prediction Error')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.fill_between(dates, prediction_error, 0, alpha=0.3, color='red' if np.mean(prediction_error) > 0 else 'green')
plt.title('Prediction Error Over Time (Actual - Predicted)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Error (USD)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eth_prediction_error.png', dpi=300)
plt.show()

