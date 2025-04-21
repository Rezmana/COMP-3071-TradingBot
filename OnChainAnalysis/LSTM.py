import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from coinmetrics.api_client import CoinMetricsClient

client = CoinMetricsClient()

print(client)

# 'SplyCntCur' - this is for the supply count of the current coin
# 'CapMrktCurUSD' - this is for the market capitalization of the current coin
# 'PriceUSD' - this is for the price of the current coin
# 'TxTfrValAdjUSD' - this is for the adjusted transaction value in USD
# 'TxTfrValUSD' - this is for the transaction value in USD  
# 'TxTfrCnt' - this is for the transaction count
# 'IssContNtv' - this is for the native issuance count
# 'IssContPctAnn' - this is for the annual percentage of the issuance count
# 
metrics = client.get_asset_metrics(assets='btc', metrics=['PriceUSD','HashRate','TxTfrCnt','AdrActCnt', 'SplyCur'], start_time="2013-01-01", end_time="2023-01-02", frequency='1d')

metrics = pd.DataFrame(metrics)

metrics.head()

metrics.to_csv("LSTM.csv", index=False)

df = metrics.copy()

df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

# Target: future price (e.g., 1-day ahead)
df['PriceUSD_target'] = df['PriceUSD'].shift(-1)

df['PriceUSD_target'] = pd.to_numeric(df['PriceUSD_target'], errors='coerce')
df['PriceUSD_log'] = np.log1p(df['PriceUSD_target'])

# Add technical indicators
# Moving averages
df['MA_7'] = df['PriceUSD_target'].rolling(window=7).mean()
df['MA_30'] = df['PriceUSD_target'].rolling(window=30).mean()
df['MA_7_30_ratio'] = df['MA_7'] / df['MA_30']

# Volatility
df['Volatility_30'] = df['PriceUSD_target'].rolling(window=30).std()

 # Drop rows with NaNs from shifting
df.dropna(inplace=True)

# Base features
X = df[['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD_target', 'MA_7', 'MA_30', 'MA_7_30_ratio', 'Volatility_30']].values

# X = df[['HashRate']].values
y = df['PriceUSD_target'].values

from sklearn.preprocessing import MinMaxScaler

# Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Choose an appropriate sequence length (e.g., 60 days)
time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Reshape the input to be 3D [samples, time steps, features]
from sklearn.model_selection import train_test_split

# Split into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Building the LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam


# Get input dimensions
n_features = X_train.shape[2]

# Build model
# Base Sequential Model with LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, n_features)))
# # units = 50 is a common choice for LSTM layers, but you can experiment with this value adds 50 memory cells (neurons)
# model.add(Dropout(0.4))
# # Dropout layer to prevent overfitting, drops 20% of the neurons
# # Add another LSTM layer but returns only the last output 

# model.add(LSTM(units=50))
# model.add(Dropout(0.2))

# Bidirectional LSTM model
# Bidirectional LSTM model can capture dependencies in both directions (past and future)
# model = Sequential()
# model.add(Bidirectional(LSTM(units=50, return_sequences=True), 
#                         input_shape=(time_steps, n_features)))
# model.add(Dropout(0.4))
# model.add(Bidirectional(LSTM(units=50)))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# -----
# Predicting a single value (the next price)

# Compile model with a slightly lower learning rate for stability
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# CNN-LSTM model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
# from tensorflow.keras.optimizers import Adam

# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, n_features)))
# model.add(MaxPooling1D(pool_size=2))
# # Optionally add another Conv1D and pooling pair here if needed
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.4))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(1))

# model = Sequential()
# model.add(LSTM(units=50, 
#                return_sequences=True, 
#                input_shape=(time_steps, n_features),
#                kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
#                recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
# model.add(Dropout(0.2))

# model.add(LSTM(units=50,
#                kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
#                recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001)))
# model.add(Dropout(0.2))

# model.add(Dense(units=1,
#                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))

# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


# Train the model with a validation split
from tensorflow.keras.callbacks import EarlyStopping

# Add early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

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

# Visualizing the result
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price USD')
plt.legend()
plt.show()