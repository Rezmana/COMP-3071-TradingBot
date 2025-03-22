import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()
# API Configuration - You'll need to register for these API keys
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_TOKEN')  # Replace with your API key
print(ETHERSCAN_API_KEY)
CRYPTOCOMPARE_API_URL = "https://min-api.cryptocompare.com/data/v2/histoday"

def fetch_price_data(days=90):
    """
    Fetch historical ETH price data from CryptoCompare
    """
    print("Fetching price data...")
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
        print(f"Successfully fetched {len(df)} days of price data")
        return df[['date', 'price', 'volumeto', 'volumefrom']]
    else:
        print(f"Error fetching price data: {data.get('Message', 'Unknown error')}")
        return None

def fetch_basic_etherscan_metrics(days=None):
    """
    Fetch basic metrics from Etherscan
    """
    print("Fetching Etherscan metrics...")
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    
    # If days is provided, fetch daily transaction data
    if days is not None:
        params = {
            "module": "stats",
            "action": "dailytx",
            "startdate": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "enddate": datetime.now().strftime("%Y-%m-%d"),
            "sort": "asc",
            "apikey": ETHERSCAN_API_KEY
        }
        
        try:
            response = requests.get(ETHERSCAN_API_URL, params=params)
            tx_data = response.json()
            
            if tx_data["status"] == "1":
                tx_df = pd.DataFrame(tx_data["result"])
                tx_df['date'] = pd.to_datetime(tx_df['UTCDate'])
                tx_df = tx_df.rename(columns={'value': 'daily_transactions'})
                tx_df['daily_transactions'] = tx_df['daily_transactions'].astype(float)
                print(f"Successfully fetched {len(tx_df)} days of transaction data")
                return tx_df[['date', 'daily_transactions']]
            else:
                print(f"Error fetching transaction data: {tx_data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Exception when fetching Etherscan metrics: {e}")
            return None
    # If days is not provided, fetch just the ETH supply as a test
    else:
        params = {
            "module": "stats",
            "action": "ethsupply",
            "apikey": ETHERSCAN_API_KEY
        }

        try:
            response = requests.get(ETHERSCAN_API_URL, params=params)
            data = response.json()
            print("Status:", response.status_code)
            print("Response:", data)
            
            if data["status"] == "1":
                supply = int(data["result"]) / 1e18  # Convert Wei to ETH
                print(f"Current ETH Supply: {supply:.2f} ETH")
                return supply
            else:
                print(f"Error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Exception when fetching Etherscan metrics: {e}")
            return None

def test_api_connections():
    """
    Test connections to APIs and return sample data
    """
    print("Testing API connections...")
    
    # Test CryptoCompare
    price_data = fetch_price_data(days=5)
    if price_data is not None:
        print("\nSample price data:")
        print(price_data.head())
    else:
        print("Could not connect to CryptoCompare API")
    
    # Allow time between API calls to avoid rate limiting
    time.sleep(1)
    
    # Test Etherscan with simple supply check
    eth_supply = fetch_basic_etherscan_metrics()
    if eth_supply is not None:
        print(f"\nCurrent ETH Supply: {eth_supply:.2f} ETH")
    else:
        print("Could not connect to Etherscan API")
    
    # Test Etherscan with daily transaction data
    tx_data = fetch_basic_etherscan_metrics(days=5)
    if tx_data is not None:
        print("\nSample transaction data:")
        print(tx_data.head())
    else:
        print("Could not connect to Etherscan API for transaction data")
    
    return price_data is not None and (eth_supply is not None or tx_data is not None)

def explore_data():
    """
    Fetch and explore a small sample of data
    """
    price_data = fetch_price_data(days=30)
    tx_data = fetch_basic_etherscan_metrics(days=30)
    
    if price_data is None or tx_data is None:
        print("Could not fetch required data for exploration")
        return
    
    # Merge data
    data = price_data.merge(tx_data, on='date', how='inner')
    
    # Print basic statistics
    print("\nData shape:", data.shape)
    print("\nData description:")
    print(data.describe())
    
    # Check correlations
    print("\nCorrelation between metrics:")
    print(data[['price', 'volumeto', 'daily_transactions']].corr())
    
    # Create a simple plot
    plt.figure(figsize=(12, 6))
    
    # Plot price
    ax1 = plt.subplot(211)
    ax1.plot(data['date'], data['price'], 'b-')
    ax1.set_title('ETH Price (USD)')
    ax1.grid(True)
    
    # Plot transactions
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(data['date'], data['daily_transactions'], 'r-')
    ax2.set_title('Daily Transactions')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('initial_data_exploration.png')
    print("\nSaved initial data exploration plot")

def initial_setup():
    """
    Perform initial setup and verify everything is working
    """
    print("=" * 50)
    print("ETHEREUM PRICE PREDICTION MODEL - INITIAL SETUP")
    print("=" * 50)
    
    if not test_api_connections():
        print("\nAPI connection test failed. Please check your API keys and internet connection.")
        return False
    
    print("\nAPI connections successful. Proceeding with data exploration...")
    explore_data()
    
    print("\nInitial setup complete. You can now extend this foundation to build your prediction model.")
    print("\nNext steps:")
    print("1. Fetch more on-chain metrics from Etherscan")
    print("2. Implement feature engineering")
    print("3. Split data for training/testing")
    print("4. Train and evaluate machine learning models")
    
    return True

if __name__ == "__main__":
    initial_setup()