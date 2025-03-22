import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
# API Configuration
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_TOKEN')  # Replace with your API key
ETHERSCAN_API_URL = "https://api.etherscan.io/api"
CRYPTOCOMPARE_API_URL = "https://min-api.cryptocompare.com/data/v2/histoday"

def fetch_price_data(days=180):
    """Fetch historical ETH price data from CryptoCompare"""
    params = {
        "fsym": "ETH",
        "tsym": "USD",
        "limit": days,
    }
    response = requests.get(CRYPTOCOMPARE_API_URL, params=params)
    data = response.json()
    
    if data["Response"] == "Success":
        df = pd.DataFrame(data["Data"]["Data"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'time': 'date', 'close': 'price'})
        return df[['date', 'price', 'volumeto', 'volumefrom']]
    else:
        raise Exception(f"Error fetching price data: {data['Message']}")

def fetch_etherscan_metrics():
    """Fetch key metrics from Etherscan API"""
    # Daily transaction count
    params = {
        "module": "stats",
        "action": "dailytx",
        "apikey": ETHERSCAN_API_KEY
    }
    response = requests.get(ETHERSCAN_API_URL, params=params)
    tx_data = response.json()
    
    if tx_data["status"] == "1":
        tx_df = pd.DataFrame(tx_data["result"])
        tx_df['date'] = pd.to_datetime(tx_df['UTCDate'])
        tx_df = tx_df.rename(columns={'value': 'daily_transactions'})
        tx_df['daily_transactions'] = tx_df['daily_transactions'].astype(float)
        
        # Keep only last 180 days for consistency with price data
        tx_df = tx_df.sort_values('date', ascending=False).head(180).sort_values('date')
        return tx_df[['date', 'daily_transactions']]
    else:
        raise Exception(f"Error fetching transaction data: {tx_data['message']}")

def fetch_gas_price_history():
    """Fetch historical gas price data"""
    params = {
        "module": "stats",
        "action": "dailyavggasprice",
        "apikey": ETHERSCAN_API_KEY
    }
    response = requests.get(ETHERSCAN_API_URL, params=params)
    gas_data = response.json()
    
    if gas_data["status"] == "1":
        gas_df = pd.DataFrame(gas_data["result"])
        gas_df['date'] = pd.to_datetime(gas_df['UTCDate'])
        gas_df = gas_df.rename(columns={'value': 'avg_gas_price'})
        gas_df['avg_gas_price'] = gas_df['avg_gas_price'].astype(float) / 1e9  # Convert to Gwei
        
        # Keep only last 180 days for consistency
        gas_df = gas_df.sort_values('date', ascending=False).head(180).sort_values('date')
        return gas_df[['date', 'avg_gas_price']]
    else:
        raise Exception(f"Error fetching gas price data: {gas_data['message']}")

def fetch_address_growth():
    """Fetch daily new address count"""
    params = {
        "module": "stats",
        "action": "dailynewaddress",
        "apikey": ETHERSCAN_API_KEY
    }
    response = requests.get(ETHERSCAN_API_URL, params=params)
    addr_data = response.json()
    
    if addr_data["status"] == "1":
        addr_df = pd.DataFrame(addr_data["result"])
        addr_df['date'] = pd.to_datetime(addr_df['UTCDate'])
        addr_df = addr_df.rename(columns={'value': 'new_addresses'})
        addr_df['new_addresses'] = addr_df['new_addresses'].astype(float)
        
        # Keep only last 180 days for consistency
        addr_df = addr_df.sort_values('date', ascending=False).head(180).sort_values('date')
        return addr_df[['date', 'new_addresses']]
    else:
        raise Exception(f"Error fetching address data: {addr_data['message']}")

def prepare_data():
    """Combine all data sources and prepare for modeling"""
    # Fetch all data
    price_df = fetch_price_data()
    tx_df = fetch_etherscan_metrics()
    gas_df = fetch_gas_price_history()
    addr_df = fetch_address_growth()
    
    # Merge all dataframes on date
    data = price_df.merge(tx_df, on='date', how='inner')
    data = data.merge(gas_df, on='date', how='inner')
    data = data.merge(addr_df, on='date', how='inner')
    
    # Feature engineering
    # 1. Add rolling averages for transactions
    data['tx_7d_avg'] = data['daily_transactions'].rolling(7).mean()
    # 2. Add gas price momentum
    data['gas_price_change'] = data['avg_gas_price'].pct_change()
    # 3. Calculate transaction to volume ratio
    data['tx_to_volume_ratio'] = data['daily_transactions'] / data['volumeto']
    # 4. Address growth momentum
    data['address_growth_7d'] = data['new_addresses'].rolling(7).sum().pct_change()
    
    # Target variable: next day's price
    data['next_day_price'] = data['price'].shift(-1)
    
    # Drop NaN values (due to rolling calculations and target shift)
    data = data.dropna()
    
    return data

def build_model(data, prediction_days=7):
    """Build and train the price prediction model"""
    # Prepare features and target
    features = [
        'daily_transactions', 'avg_gas_price', 'new_addresses', 
        'volumeto', 'volumefrom', 'tx_7d_avg', 'gas_price_change',
        'tx_to_volume_ratio', 'address_growth_7d'
    ]
    
    X = data[features]
    y = data['next_day_price']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data: use the last 30 days for testing
    train_size = len(data) - prediction_days
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, train_preds, test_preds, y_train, y_test, feature_importance

def evaluate_model(train_preds, test_preds, y_train, y_test):
    """Evaluate model performance"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)
    
    # Testing metrics
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"Training MAE: ${train_mae:.2f}")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print("\nTest metrics:")
    print(f"Test MAE: ${test_mae:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

def plot_results(data, train_preds, test_preds, y_train, y_test, feature_importance):
    """Visualize model predictions and feature importance"""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price prediction plot
    dates = data['date'].iloc[-(len(y_train) + len(y_test)):]
    train_dates = dates[:len(y_train)]
    test_dates = dates[-len(y_test):]
    
    ax1.plot(train_dates, y_train, label='Actual Price (Train)', color='blue')
    ax1.plot(train_dates, train_preds, label='Predicted Price (Train)', color='lightblue', linestyle='--')
    ax1.plot(test_dates, y_test, label='Actual Price (Test)', color='green')
    ax1.plot(test_dates, test_preds, label='Predicted Price (Test)', color='lightgreen', linestyle='--')
    
    ax1.set_title('ETH Price Prediction Using Etherscan Metrics')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature importance plot
    ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_title('Feature Importance')
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig('eth_price_prediction.png')
    plt.show()

def main():
    print("Fetching and preparing data...")
    data = prepare_data()
    
    print("Building model...")
    model, train_preds, test_preds, y_train, y_test, feature_importance = build_model(data)
    
    print("\nModel Evaluation:")
    metrics = evaluate_model(train_preds, test_preds, y_train, y_test)
    
    print("\nTop Features:")
    print(feature_importance.head())
    
    print("\nPlotting results...")
    plot_results(data, train_preds, test_preds, y_train, y_test, feature_importance)
    
    # Make future predictions
    print("\nPredicting prices for the next 7 days...")
    last_data = data[feature_importance['feature'].tolist()].iloc[-1].values.reshape(1, -1)
    scaler = MinMaxScaler().fit(data[feature_importance['feature'].tolist()])
    scaled_last = scaler.transform(last_data)
    next_day_price = model.predict(scaled_last)[0]
    print(f"Predicted ETH price for tomorrow: ${next_day_price:.2f}")

if __name__ == "__main__":
    main()