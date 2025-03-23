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

# def fetch_basic_etherscan_metrics(days=None):
#     """
#     Fetch basic metrics from Etherscan
#     """
#     print("Fetching Etherscan metrics...")
#     ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    
#     # If days is provided, fetch daily transaction data
#     if days is not None:
#         params = {
#             "module": "stats",
#             "action": "dailytx",
#             "startdate": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
#             "enddate": datetime.now().strftime("%Y-%m-%d"),
#             "sort": "asc",
#             "apikey": ETHERSCAN_API_KEY
#         }
        
#         try:
#             response = requests.get(ETHERSCAN_API_URL, params=params)
#             tx_data = response.json()
            
#             if tx_data["status"] == "1":
#                 tx_df = pd.DataFrame(tx_data["result"])
#                 tx_df['date'] = pd.to_datetime(tx_df['UTCDate'])
#                 tx_df = tx_df.rename(columns={'value': 'daily_transactions'})
#                 tx_df['daily_transactions'] = tx_df['daily_transactions'].astype(float)
#                 print(f"Successfully fetched {len(tx_df)} days of transaction data")
#                 return tx_df[['date', 'daily_transactions']]
#             else:
#                 print(f"Error fetching transaction data: {tx_data.get('message', 'Unknown error')}")
#                 return None
#         except Exception as e:
#             print(f"Exception when fetching Etherscan metrics: {e}")
#             return None
#     # If days is not provided, fetch just the ETH supply as a test
#     else:
#         params = {
#             "module": "stats",
#             "action": "ethsupply",
#             "apikey": ETHERSCAN_API_KEY
#         }

#         try:
#             response = requests.get(ETHERSCAN_API_URL, params=params)
#             data = response.json()
#             print("Status:", response.status_code)
#             print("Response:", data)
            
#             if data["status"] == "1":
#                 supply = int(data["result"]) / 1e18  # Convert Wei to ETH
#                 print(f"Current ETH Supply: {supply:.2f} ETH")
#                 return supply
#             else:
#                 print(f"Error: {data.get('message', 'Unknown error')}")
#                 return None
#         except Exception as e:
#             print(f"Exception when fetching Etherscan metrics: {e}")
#             return None

def fetch_advanced_etherscan_metrics(days=30):
    """
    Fetch a comprehensive set of on-chain metrics from Etherscan
    
    Returns a DataFrame with multiple metrics aligned by date
    """
    print("Fetching advanced Etherscan metrics...")
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 1. Daily Transaction Count
    params = {
        "module": "stats",
        "action": "dailytx",
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            tx_df = pd.DataFrame(data["result"])
            tx_df['date'] = pd.to_datetime(tx_df['UTCDate'])
            tx_df = tx_df.rename(columns={'value': 'daily_transactions'})
            tx_df['daily_transactions'] = tx_df['daily_transactions'].astype(float)
            all_dfs.append(tx_df[['date', 'daily_transactions']])
            print(f"✓ Transaction count data: {len(tx_df)} days")
        else:
            print(f"✗ Error fetching transaction data: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Exception when fetching transaction data: {e}")
    
    time.sleep(0.2)  # Prevent rate limiting
    
    # 2. Daily Gas Used
    params = {
        "module": "stats",
        "action": "dailygasused",
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            gas_df = pd.DataFrame(data["result"])
            gas_df['date'] = pd.to_datetime(gas_df['UTCDate'])
            gas_df = gas_df.rename(columns={'value': 'gas_used'})
            gas_df['gas_used'] = gas_df['gas_used'].astype(float)
            all_dfs.append(gas_df[['date', 'gas_used']])
            print(f"✓ Gas used data: {len(gas_df)} days")
        else:
            print(f"✗ Error fetching gas used data: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Exception when fetching gas used data: {e}")
    
    time.sleep(0.2)  # Prevent rate limiting
    
    # 3. Daily Average Gas Price
    params = {
        "module": "stats",
        "action": "dailyavggasprice",
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            avg_gas_df = pd.DataFrame(data["result"])
            avg_gas_df['date'] = pd.to_datetime(avg_gas_df['UTCDate'])
            avg_gas_df = avg_gas_df.rename(columns={'value': 'avg_gas_price'})
            avg_gas_df['avg_gas_price'] = avg_gas_df['avg_gas_price'].astype(float) / 1e9  # Convert to Gwei
            all_dfs.append(avg_gas_df[['date', 'avg_gas_price']])
            print(f"✓ Average gas price data: {len(avg_gas_df)} days")
        else:
            print(f"✗ Error fetching average gas price data: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Exception when fetching average gas price data: {e}")
    
    time.sleep(0.2)  # Prevent rate limiting
    
    # 4. Daily New Address Count
    params = {
        "module": "stats",
        "action": "dailynewaddress",
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            addr_df = pd.DataFrame(data["result"])
            addr_df['date'] = pd.to_datetime(addr_df['UTCDate'])
            addr_df = addr_df.rename(columns={'value': 'new_addresses'})
            addr_df['new_addresses'] = addr_df['new_addresses'].astype(float)
            all_dfs.append(addr_df[['date', 'new_addresses']])
            print(f"✓ New address data: {len(addr_df)} days")
        else:
            print(f"✗ Error fetching new address data: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Exception when fetching new address data: {e}")
    
    time.sleep(0.2)  # Prevent rate limiting
    
    # 5. Daily Network Utilization (based on average block utilization)
    params = {
        "module": "stats",
        "action": "dailynetutilization",
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            util_df = pd.DataFrame(data["result"])
            util_df['date'] = pd.to_datetime(util_df['UTCDate'])
            util_df = util_df.rename(columns={'value': 'network_utilization'})
            util_df['network_utilization'] = util_df['network_utilization'].astype(float) * 100  # Convert to percentage
            all_dfs.append(util_df[['date', 'network_utilization']])
            print(f"✓ Network utilization data: {len(util_df)} days")
        else:
            print(f"✗ Error fetching network utilization data: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Exception when fetching network utilization data: {e}")
    
    time.sleep(0.2)  # Prevent rate limiting
    
    # 6. ETH Daily Market Cap
    params = {
        "module": "stats",
        "action": "ethdailymarketcap",
        "startdate": start_date.strftime("%Y-%m-%d"),
        "enddate": end_date.strftime("%Y-%m-%d"),
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            mcap_df = pd.DataFrame(data["result"])
            mcap_df['date'] = pd.to_datetime(mcap_df['UTCDate'])
            mcap_df = mcap_df.rename(columns={'value': 'market_cap'})
            mcap_df['market_cap'] = mcap_df['market_cap'].astype(float) / 1e9  # Convert to billions
            all_dfs.append(mcap_df[['date', 'market_cap']])
            print(f"✓ Market cap data: {len(mcap_df)} days")
        else:
            print(f"✗ Error fetching market cap data: {data.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"✗ Exception when fetching market cap data: {e}")
    
    # Combine all dataframes
    if not all_dfs:
        print("No data was successfully fetched!")
        return None
    
    # Start with the first dataframe
    combined_df = all_dfs[0]
    
    # Merge with the rest
    for df in all_dfs[1:]:
        combined_df = combined_df.merge(df, on='date', how='outer')
    
    # Sort by date
    combined_df = combined_df.sort_values('date')
    
    # Fill any missing values with forward fill then backward fill
    combined_df = combined_df.ffill().bfill()
    
    print(f"Combined {len(all_dfs)} metrics into a dataset with {len(combined_df)} days")
    return combined_df

def fetch_eth_balance_for_address(address):
    """
    Fetch the current ETH balance for a specific address
    Useful for tracking whale wallets or major exchanges
    """
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "balance",
        "address": address,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            balance = int(data["result"]) / 1e18  # Convert Wei to ETH
            return balance
        else:
            print(f"Error fetching address balance: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Exception when fetching address balance: {e}")
        return None

def track_whale_wallets(days=30, top_addresses=None):
    """
    Track the ETH balance of major wallets over time
    Requires manual tracking as historical balances require subscription
    """
    # You can replace these with actual addresses you want to track
    if top_addresses is None:
        top_addresses = {
            "Binance-1": "0x28C6c06298d514Db089934071355E5743bf21d60",
            "Binance-2": "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",
            "Kraken": "0xa1d8d972560c2f8144af871db508f0b0b10a3fbf"
        }
    
    whale_data = {}
    for name, address in top_addresses.items():
        balance = fetch_eth_balance_for_address(address)
        if balance is not None:
            print(f"{name} ({address[:8]}...): {balance:.2f} ETH")
            whale_data[name] = balance
        time.sleep(0.2)  # Prevent rate limiting
    
    return whale_data

def fetch_erc20_token_transfers(token_address, days=7):
    """
    Get ERC20 token transfer activity (useful for stablecoins like USDT, USDC)
    """
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    end_block = "latest"
    # Approximate number of blocks in the specified days (assuming ~13s block time)
    blocks_per_day = 24 * 60 * 60 / 13
    start_block = int(blocks_per_day * days)
    
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": token_address,
        "startblock": start_block,
        "endblock": end_block,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(ETHERSCAN_API_URL, params=params)
        data = response.json()
        
        if data["status"] == "1":
            transfers = pd.DataFrame(data["result"])
            transfers['timeStamp'] = pd.to_datetime(transfers['timeStamp'].astype(int), unit='s')
            transfers['value'] = transfers['value'].astype(float) / (10 ** int(transfers.iloc[0]['tokenDecimal']))
            
            # Group by day and calculate metrics
            transfers['date'] = transfers['timeStamp'].dt.date
            daily_stats = transfers.groupby('date').agg(
                transaction_count=('hash', 'count'),
                total_volume=('value', 'sum'),
                unique_addresses=('to', lambda x: len(set(x) | set(transfers.loc[transfers['date'] == x.name, 'from'])))
            ).reset_index()
            
            daily_stats['date'] = pd.to_datetime(daily_stats['date'])
            return daily_stats
        else:
            print(f"Error fetching token transfers: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Exception when fetching token transfers: {e}")
        return None

def fetch_defi_metrics():
    """
    Function stub for fetching DeFi metrics 
    Etherscan doesn't directly provide DeFi metrics, but you can track major protocol addresses
    """
    # This would require tracking specific DeFi protocol addresses or using a different API
    # For demonstration, we'll return a placeholder
    print("Note: Comprehensive DeFi metrics require additional APIs like The Graph")
    return None

def generate_advanced_features(data):
    """
    Generate advanced features from the basic metrics
    """
    # Ensure we have a copy to avoid modifying the original
    df = data.copy()
    
    # Technical indicators (similar to stock technical analysis)
    # 1. Moving averages
    df['tx_7d_ma'] = df['daily_transactions'].rolling(7).mean()
    df['gas_7d_ma'] = df['gas_used'].rolling(7).mean() if 'gas_used' in df.columns else None
    
    # 2. Rate of change indicators
    df['tx_growth'] = df['daily_transactions'].pct_change(7) * 100  # 7-day growth rate
    if 'new_addresses' in df.columns:
        df['address_growth'] = df['new_addresses'].pct_change(7) * 100
    
    # 3. Relative metrics
    if 'gas_used' in df.columns and 'daily_transactions' in df.columns:
        df['gas_per_tx'] = df['gas_used'] / df['daily_transactions']
    
    # 4. Volatility measures
    df['tx_volatility'] = df['daily_transactions'].rolling(7).std() / df['daily_transactions'].rolling(7).mean()
    
    # 5. Momentum indicators
    if 'network_utilization' in df.columns:
        df['util_momentum'] = df['network_utilization'] - df['network_utilization'].rolling(7).mean()
    
    # 6. Cyclical features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df.dropna()

def explore_advanced_metrics(days=30):
    """
    Fetch and explore advanced on-chain metrics
    """
    # Get price data
    price_data = fetch_price_data(days=days)
    
    # Get advanced on-chain metrics
    onchain_data = fetch_advanced_etherscan_metrics(days=days)
    
    if price_data is None or onchain_data is None:
        print("Could not fetch required data for exploration")
        return
    
    # Merge data
    data = price_data.merge(onchain_data, on='date', how='inner')
    
    # Generate advanced features
    advanced_data = generate_advanced_features(data)
    
    # Calculate correlations with price
    if 'price' in advanced_data.columns:
        correlations = advanced_data.corr()['price'].sort_values(ascending=False)
        print("\nMetrics correlation with ETH price:")
        print(correlations)
    
    # Basic visualization of key metrics
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Price vs Transactions
    ax1 = plt.subplot(311)
    ax1.set_title('ETH Price vs Daily Transactions')
    ax1.set_ylabel('ETH Price (USD)', color='b')
    ax1.plot(advanced_data['date'], advanced_data['price'], 'b-')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1b = ax1.twinx()
    ax1b.set_ylabel('Transactions', color='r')
    ax1b.plot(advanced_data['date'], advanced_data['daily_transactions'], 'r-')
    ax1b.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: Gas Used and Gas Price
    if 'gas_used' in advanced_data.columns and 'avg_gas_price' in advanced_data.columns:
        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_title('Gas Used vs Gas Price')
        ax2.set_ylabel('Gas Used', color='g')
        ax2.plot(advanced_data['date'], advanced_data['gas_used'], 'g-')
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax2b = ax2.twinx()
        ax2b.set_ylabel('Avg Gas Price (Gwei)', color='m')
        ax2b.plot(advanced_data['date'], advanced_data['avg_gas_price'], 'm-')
        ax2b.tick_params(axis='y', labelcolor='m')
    
    # Plot 3: New Addresses and Network Utilization
    if 'new_addresses' in advanced_data.columns and 'network_utilization' in advanced_data.columns:
        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title('New Addresses vs Network Utilization')
        ax3.set_ylabel('New Addresses', color='c')
        ax3.plot(advanced_data['date'], advanced_data['new_addresses'], 'c-')
        ax3.tick_params(axis='y', labelcolor='c')
        
        ax3b = ax3.twinx()
        ax3b.set_ylabel('Network Utilization (%)', color='y')
        ax3b.plot(advanced_data['date'], advanced_data['network_utilization'], 'y-')
        ax3b.tick_params(axis='y', labelcolor='y')
    
    plt.tight_layout()
    plt.savefig('advanced_metrics_exploration.png')
    print("\nSaved advanced metrics exploration plot")
    
    return advanced_data

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
    eth_supply = fetch_advanced_etherscan_metrics()
    if eth_supply is not None:
        print(f"\nCurrent ETH Supply: {eth_supply:.2f} ETH")
    else:
        print("Could not connect to Etherscan API")
    
    # Test Etherscan with daily transaction data
    tx_data = fetch_advanced_etherscan_metrics(days=5)
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
    tx_data = fetch_advanced_etherscan_metrics(days=30)
    
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