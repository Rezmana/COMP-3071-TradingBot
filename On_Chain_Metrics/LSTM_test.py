#!/usr/bin/env python
"""
onchain_test.py

This script loads the pre-trained LSTM on-chain metrics model (and scalers),
evaluates it on the test data, computes performance metrics, and visualizes
the predictions versus the actual prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# Set file paths and directories
output_dir = os.path.join(os.getcwd(), "onchain_output")
model_path = os.path.join(output_dir, "onchain_model_final.h5")
scaler_X_path = os.path.join(output_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(output_dir, "scaler_y.pkl")
    
# Load scalers
with open(scaler_X_path, "rb") as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, "rb") as f:
    scaler_y = pickle.load(f)
    
# Load the on-chain data CSV that was previously saved (or use your source if available)
df = pd.read_csv("LSTM.csv")
df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

# Recreate target and technical indicators as in training
df['PriceUSD_target'] = df['PriceUSD'].shift(-1)
df['PriceUSD_target'] = pd.to_numeric(df['PriceUSD_target'], errors='coerce')
df['PriceUSD_log'] = np.log1p(df['PriceUSD_target'])
df['MA_7'] = df['PriceUSD_target'].rolling(window=7).mean()
df['MA_30'] = df['PriceUSD_target'].rolling(window=30).mean()
df['MA_7_30_ratio'] = df['MA_7'] / df['MA_30']
df['Volatility_30'] = df['PriceUSD_target'].rolling(window=30).std()
df.dropna(inplace=True)

features = ['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur',
            'PriceUSD_target', 'MA_7', 'MA_30', 'MA_7_30_ratio', 'Volatility_30']
X = df[features].values
y = df['PriceUSD_log'].values

X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y.reshape(-1, 1))

# Create sequences; same function as used during training
def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split sequences into train, validation, and test sets (keep same proportion)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Load the trained model
model = load_model(model_path)
print("Model loaded from:", model_path)

# Evaluate on test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

rmse = math.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title('Bitcoin Price Prediction on Test Data')
plt.xlabel('Time Step')
plt.ylabel('Price USD')
plt.legend()
plot_path = os.path.join(output_dir, "bitcoin_price_prediction.png")
plt.savefig(plot_path)
plt.show()
print("Prediction plot saved to:", plot_path)
