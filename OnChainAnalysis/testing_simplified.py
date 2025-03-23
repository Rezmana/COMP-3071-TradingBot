import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
# API Configuration - You'll need to register for these API keys
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_TOKEN')  # Replace with your API key
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

def fetch_historical_price_data(start_date, end_date=None, use_cache=True, cache_dir="cache"):
    """
    Fetch historical ETH price data from CryptoCompare for a specific date range.
    Uses CSV caching to avoid refetching.

    Parameters:
    - start_date (str): 'YYYY-MM-DD'
    - end_date (str): 'YYYY-MM-DD', defaults to today
    - use_cache (bool): Whether to load/save cached data
    - cache_dir (str): Folder to store cache files

    Returns:
    - pandas.DataFrame: Contains date, price, volumeto, volumefrom
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    cache_filename = f"{cache_dir}/price_data_{start_date}_to_{end_date}.csv"

    # Check cache
    if use_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if os.path.exists(cache_filename):
            print(f"📂 Loading cached price data: {cache_filename}")
            return pd.read_csv(cache_filename, parse_dates=["date"])

    print(f"⏳ Fetching price data from {start_date} to {end_date}...")

    # Convert dates to timestamps
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    # Calculate number of days
    days_diff = (end_timestamp - start_timestamp) // (24 * 3600)

    params = {
        "fsym": "ETH",
        "tsym": "USD",
        "limit": days_diff,
        "toTs": end_timestamp,
    }

    response = requests.get(CRYPTOCOMPARE_API_URL, params=params)
    data = response.json()

    if data.get("Response") == "Success":
        df = pd.DataFrame(data["Data"]["Data"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "date", "close": "price"})
        df = df[["date", "price", "volumeto", "volumefrom"]]
        
        if use_cache:
            df.to_csv(cache_filename, index=False)
            print(f"✔️ Saved price data to cache: {cache_filename}")

        return df
    else:
        print(f"❌ Error fetching price data: {data.get('Message', 'Unknown error')}")
        return None


def fetch_historical_price_data_chunked(start_date, end_date, chunk_days=2000):
    """
    Fetch historical price data in chunks to handle longer time periods
    """
    print(f"Fetching historical price data from {start_date} to {end_date or 'today'} in chunks...")
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.now()
    
    all_data = []
    current_start = start_dt
    
    print (current_start)
    print (end_dt)
    
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        
        chunk_start = current_start.strftime('%Y-%m-%d')
        chunk_end = current_end.strftime('%Y-%m-%d')
        
        print(f"  Fetching chunk: {chunk_start} to {chunk_end}")
        chunk_data = fetch_historical_price_data(chunk_start, chunk_end)
        if chunk_data is not None:
            all_data.append(chunk_data)
            # Avoid rate limits
            time.sleep(1)
        
        current_start = current_end + timedelta(days=1)
    
    if all_data:
        combined_data = pd.concat(all_data).drop_duplicates(subset='date').reset_index(drop=True)
        print(f"Successfully fetched {len(combined_data)} days of historical price data")
        return combined_data
    return None

def fetch_basic_etherscan_metrics(days=2, max_workers=10, use_cache=True, cache_file="cached_tx_data.csv"):
    """
    Fetch total daily Ethereum transaction counts using multithreading.
    If cache exists, load from CSV instead of calling Etherscan again.
    """
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"

    # ✅ Load from cache if available
    if use_cache and os.path.exists(cache_file):
        print(f"📂 Loading cached transaction data from {cache_file}...")
        return pd.read_csv(cache_file, parse_dates=["date"])

    results = []

    def get_block_by_timestamp(timestamp):
        params = {
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": str(timestamp),
            "closest": "before",
            "apikey": ETHERSCAN_API_KEY
        }
        res = requests.get(ETHERSCAN_API_URL, params=params)
        data = res.json()
        return int(data.get("result", "0"))

    def get_tx_count_for_block(block_number):
        block_hex = hex(block_number)
        params = {
            "module": "proxy",
            "action": "eth_getBlockTransactionCountByNumber",
            "tag": block_hex,
            "apikey": ETHERSCAN_API_KEY
        }
        try:
            res = requests.get(ETHERSCAN_API_URL, params=params)
            data = res.json()
            result = data.get("result")
            if result:
                return int(result, 16)
        except:
            pass
        return 0

    def fetch_tx_counts_parallel(start_block, end_block):
        tx_total = 0
        block_range = range(start_block, end_block + 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(get_tx_count_for_block, block): block
                for block in block_range
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="  Fetching tx counts"):
                try:
                    tx_total += future.result()
                except Exception as e:
                    print(f"Error fetching tx count: {e}")
        return tx_total

    print("⏳ Fetching daily Ethereum transaction counts...\n")
    today = datetime.utcnow()

    for i in range(days):
        date = today - timedelta(days=i)
        start_dt = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_dt = datetime(date.year, date.month, date.day, 23, 59, 59)

        print(f"{start_dt.date()}:")

        try:
            start_block = get_block_by_timestamp(int(start_dt.timestamp()))
            time.sleep(0.25)
            end_block = get_block_by_timestamp(int(end_dt.timestamp()))
            time.sleep(0.25)

            tx_total = fetch_tx_counts_parallel(start_block, end_block)
            results.append({"date": start_dt.date(), "daily_transactions": tx_total})
            print(f"  → Block range: {start_block}-{end_block}, Total transactions: {tx_total}\n")

        except Exception as e:
            print(f"  ✖ Error on {start_dt.date()}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(cache_file, index=False)
    print(f"✔️ Saved transaction data to {cache_file}")
    return df

def fetch_historical_etherscan_metrics(start_date, end_date=None, max_workers=10, use_cache=True, cache_file="cached_historical_tx_data.csv"):
    """
    Fetch historical Ethereum transaction counts for a specific date range
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
    max_workers (int): Maximum number of parallel workers
    use_cache (bool): Whether to use cached data if available
    cache_file (str): File to save/load cached data
    
    Returns:
    pandas.DataFrame: DataFrame with date and daily_transactions
    """
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    
    # Load from cache if available
    if use_cache and os.path.exists(cache_file):
        print(f"📂 Loading cached historical transaction data from {cache_file}...")
        historical_tx_data = pd.read_csv(cache_file, parse_dates=["date"])
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Check if we have all the dates we need
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        date_range = pd.date_range(start=start_dt, end=end_dt)
        existing_dates = pd.to_datetime(historical_tx_data['date']).dt.date
        
        # If we have all dates, return the filtered dataframe
        missing_dates = [d for d in date_range if d.date() not in existing_dates]
        if not missing_dates:
            mask = (historical_tx_data['date'].dt.date >= start_dt) & (historical_tx_data['date'].dt.date <= end_dt)
            return historical_tx_data[mask].reset_index(drop=True)
        
        print(f"Found {len(missing_dates)} missing dates. Fetching missing data...")
        
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.now()
    
    # Create date range
    date_range = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]
    
    results = []
    
    def get_block_by_timestamp(timestamp):
        params = {
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": str(timestamp),
            "closest": "before",
            "apikey": ETHERSCAN_API_KEY
        }
        res = requests.get(ETHERSCAN_API_URL, params=params)
        data = res.json()
        return int(data.get("result", "0"))

    def get_tx_count_for_block(block_number):
        block_hex = hex(block_number)
        params = {
            "module": "proxy",
            "action": "eth_getBlockTransactionCountByNumber",
            "tag": block_hex,
            "apikey": ETHERSCAN_API_KEY
        }
        try:
            res = requests.get(ETHERSCAN_API_URL, params=params)
            data = res.json()
            result = data.get("result")
            if result:
                return int(result, 16)
        except:
            pass
        return 0

    def fetch_tx_counts_parallel(start_block, end_block):
        tx_total = 0
        block_range = range(start_block, end_block + 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(get_tx_count_for_block, block): block
                for block in block_range
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="  Fetching tx counts"):
                try:
                    tx_total += future.result()
                except Exception as e:
                    print(f"Error fetching tx count: {e}")
        return tx_total

    print(f"⏳ Fetching historical Ethereum transaction counts from {start_date} to {end_date or 'today'}...\n")
    
    for date in tqdm(date_range, desc="Processing dates"):
        start_dt_day = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_dt_day = datetime(date.year, date.month, date.day, 23, 59, 59)

        print(f"{start_dt_day.date()}:")

        try:
            start_block = get_block_by_timestamp(int(start_dt_day.timestamp()))
            # time.sleep(0.25)  # Avoid rate limiting
            end_block = get_block_by_timestamp(int(end_dt_day.timestamp()))
            # time.sleep(0.25)  # Avoid rate limiting

            tx_total = sum(get_tx_count_for_block(block) for block in range(start_block, end_block + 1))
            results.append({"date": start_dt_day.date(), "daily_transactions": tx_total})
            print(f"  → Block range: {start_block}-{end_block}, Total transactions: {tx_total}\n")

        except Exception as e:
            print(f"  ✖ Error on {start_dt_day.date()}: {e}")
            results.append({"date": start_dt_day.date(), "daily_transactions": None})

    new_df = pd.DataFrame(results)
    
    # If we have existing data, merge with it
    if use_cache and os.path.exists(cache_file):
        historical_tx_data = pd.read_csv(cache_file, parse_dates=["date"])
        
        # Combine old and new data, keeping the newest version of any duplicate dates
        combined_df = pd.concat([historical_tx_data, new_df])
        combined_df = combined_df.drop_duplicates(subset='date', keep='last')
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Save updated cache
        combined_df.to_csv(cache_file, index=False)
        print(f"✔️ Updated cached historical transaction data in {cache_file}")
        
        # Filter to requested date range
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date_dt = datetime.now().date()
            
        mask = (combined_df['date'].dt.date >= start_date_dt) & (combined_df['date'].dt.date <= end_date_dt)
        return combined_df[mask].reset_index(drop=True)
    else:
        # Save new cache
        new_df.to_csv(cache_file, index=False)
        print(f"✔️ Saved historical transaction data to {cache_file}")
        return new_df

def fetch_eth_supply():
    url = "https://api.etherscan.io/api"
    params = {
        "module": "stats",
        "action": "ethsupply",
        "apikey": ETHERSCAN_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") == "1":
            supply = int(data["result"]) / 1e18  # Convert from Wei to ETH
            return supply
    except Exception as e:
        print(f"Error fetching ETH supply: {e}")
    return None
def fetch_gas_metrics(start_date, end_date=None, use_cache=True, cache_file="cached_gas_metrics.csv"):
    """
    Fetch daily gas used and average gas price metrics from Etherscan
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
    use_cache (bool): Whether to use cached data if available
    cache_file (str): File to save/load cached data
    
    Returns:
    pandas.DataFrame: DataFrame with date, gas_used, avg_gas_price
    """
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    
    # Load from cache if available
    if use_cache and os.path.exists(cache_file):
        print(f"📂 Loading cached gas metrics data from {cache_file}...")
        gas_data = pd.read_csv(cache_file, parse_dates=["date"])
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Check if we have all the dates we need
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        date_range = pd.date_range(start=start_dt, end=end_dt)
        existing_dates = pd.to_datetime(gas_data['date']).dt.date
        
        # If we have all dates, return the filtered dataframe
        missing_dates = [d for d in date_range if d.date() not in existing_dates]
        if not missing_dates:
            mask = (gas_data['date'].dt.date >= start_dt) & (gas_data['date'].dt.date <= end_dt)
            return gas_data[mask].reset_index(drop=True)
            
        print(f"Found {len(missing_dates)} missing dates. Fetching missing data...")
    
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.now()
    
    # Create date range
    date_range = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]
    
    results = []
    
    def get_block_by_timestamp(timestamp):
        params = {
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": str(timestamp),
            "closest": "before",
            "apikey": ETHERSCAN_API_KEY
        }
        res = requests.get(ETHERSCAN_API_URL, params=params)
        data = res.json()
        return int(data.get("result", "0"))

    def get_block_data(block_number):
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(block_number),
            "boolean": "true",
            "apikey": ETHERSCAN_API_KEY
        }
        try:
            res = requests.get(ETHERSCAN_API_URL, params=params)
            data = res.json()
            return data.get("result")
        except:
            return None

    print(f"⏳ Fetching Ethereum gas metrics from {start_date} to {end_date or 'today'}...\n")
    
    for date in tqdm(date_range, desc="Processing dates"):
        date_str = date.strftime('%Y-%m-%d')
        start_dt_day = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_dt_day = datetime(date.year, date.month, date.day, 23, 59, 59)

        print(f"{date_str}:")

        try:
            # Get start and end blocks for the day
            start_block = get_block_by_timestamp(int(start_dt_day.timestamp()))
            time.sleep(0.25)  # Avoid rate limiting
            end_block = get_block_by_timestamp(int(end_dt_day.timestamp()))
            time.sleep(0.25)  # Avoid rate limiting
            
            print(f"  → Block range: {start_block}-{end_block}")
            
            # Sample blocks throughout the day to get average metrics
            # We'll sample up to 24 blocks (roughly one per hour) to avoid rate limits
            num_blocks = min(24, end_block - start_block + 1)
            if num_blocks <= 0:
                continue
                
            block_step = (end_block - start_block) // num_blocks
            sample_blocks = [start_block + i * block_step for i in range(num_blocks)]
            
            total_gas_used = 0
            gas_prices = []
            
            for block in tqdm(sample_blocks, desc="  Sampling blocks"):
                block_data = get_block_data(block)
                if block_data:
                    # Gas used in the block
                    gas_used = int(block_data.get("gasUsed", "0x0"), 16)
                    total_gas_used += gas_used
                    
                    # Get gas prices from transactions
                    for tx in block_data.get("transactions", [])[:10]:  # Sample first 10 txs
                        gas_price = int(tx.get("gasPrice", "0x0"), 16) / 1e9  # Convert to Gwei
                        gas_prices.append(gas_price)
                
                time.sleep(0.25)  # Avoid rate limiting
            
            # Calculate averages
            daily_gas_used = total_gas_used * (end_block - start_block + 1) / len(sample_blocks)
            avg_gas_price = np.mean(gas_prices) if gas_prices else None
            
            results.append({
                "date": date.date(),
                "gas_used": daily_gas_used,
                "avg_gas_price": avg_gas_price,
                "blocks_processed": end_block - start_block + 1
            })
            
            print(f"  → Daily gas used (est.): {daily_gas_used:,.0f}, Avg gas price: {avg_gas_price:.2f} Gwei\n")

        except Exception as e:
            print(f"  ✖ Error on {date_str}: {e}")
            results.append({
                "date": date.date(),
                "gas_used": None,
                "avg_gas_price": None,
                "blocks_processed": None
            })
    
    new_df = pd.DataFrame(results)
    
    # If we have existing data, merge with it
    if use_cache and os.path.exists(cache_file):
        gas_data = pd.read_csv(cache_file, parse_dates=["date"])
        
        # Combine old and new data, keeping the newest version of any duplicate dates
        combined_df = pd.concat([gas_data, new_df])
        combined_df = combined_df.drop_duplicates(subset='date', keep='last')
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Save updated cache
        combined_df.to_csv(cache_file, index=False)
        print(f"✔️ Updated cached gas metrics data in {cache_file}")
        
        # Filter to requested date range
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date_dt = datetime.now().date()
            
        mask = (combined_df['date'].dt.date >= start_date_dt) & (combined_df['date'].dt.date <= end_date_dt)
        return combined_df[mask].reset_index(drop=True)
    else:
        # Save new cache
        new_df.to_csv(cache_file, index=False)
        print(f"✔️ Saved gas metrics data to {cache_file}")
        return new_df

def fetch_contract_interactions(start_date, end_date=None, top_contracts=10, use_cache=True, cache_file="cached_contract_interactions.csv"):
    """
    Fetch daily interactions with top Ethereum smart contracts
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
    top_contracts (int): Number of top contracts to track
    use_cache (bool): Whether to use cached data if available
    cache_file (str): File to save/load cached data
    
    Returns:
    pandas.DataFrame: DataFrame with date and contract interaction metrics
    """
    ETHERSCAN_API_URL = "https://api.etherscan.io/api"
    
    # Load from cache if available
    if use_cache and os.path.exists(cache_file):
        print(f"📂 Loading cached contract interactions data from {cache_file}...")
        contract_data = pd.read_csv(cache_file, parse_dates=["date"])
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Check if we have all the dates we need
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        date_range = pd.date_range(start=start_dt, end=end_dt)
        existing_dates = pd.to_datetime(contract_data['date']).dt.date
        
        # If we have all dates, return the filtered dataframe
        missing_dates = [d for d in date_range if d.date() not in existing_dates]
        if not missing_dates:
            mask = (contract_data['date'].dt.date >= start_dt) & (contract_data['date'].dt.date <= end_dt)
            return contract_data[mask].reset_index(drop=True)
            
        print(f"Found {len(missing_dates)} missing dates. Fetching missing data...")
    
    # First, find top contracts by transaction count
    # We'll look at the past week to identify currently active contracts
    print("Identifying top Ethereum contracts by transaction count...")
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get transaction count by address
    params = {
        "module": "stats",
        "action": "tokensupply",
        "contractaddress": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH as an example
        "apikey": ETHERSCAN_API_KEY
    }
    
    # Since Etherscan doesn't have a direct API for this,
    # we'll use a simplified approach and track some known important contracts
    important_contracts = {
        # Address: Name
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH",
        "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
        "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
        "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": "AAVE",
        "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "UNI",
        "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0": "MATIC",
        "0x514910771af9ca656af840dff83e8264ecf986ca": "LINK",
        "0x75231f58b43240c9718dd58b4967c5114342a86c": "OKB",
    }
    
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = datetime.now()
    
    # Create date range
    date_range = [start_dt + timedelta(days=x) for x in range((end_dt - start_dt).days + 1)]
    
    results = []
    
    print(f"⏳ Fetching contract interactions from {start_date} to {end_date or 'today'}...\n")
    
    for date in tqdm(date_range, desc="Processing dates"):
        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str}:")
        
        daily_data = {"date": date.date()}
        
        # For each important contract, get daily transaction count
        for contract_address, contract_name in tqdm(important_contracts.items(), desc="  Fetching contracts"):
            params = {
                "module": "account",
                "action": "txlist",
                "address": contract_address,
                "startblock": "0",
                "endblock": "99999999",
                "startdate": date_str,
                "enddate": date_str,
                "sort": "asc",
                "apikey": ETHERSCAN_API_KEY
            }
            
            try:
                response = requests.get(ETHERSCAN_API_URL, params=params)
                data = response.json()
                
                if data["status"] == "1" and data["message"] == "OK":
                    tx_count = len(data["result"])
                    daily_data[f"{contract_name}_interactions"] = tx_count
                    print(f"  → {contract_name}: {tx_count} interactions")
                else:
                    daily_data[f"{contract_name}_interactions"] = 0
                    print(f"  → {contract_name}: 0 interactions (API error: {data.get('message', 'Unknown error')})")
                
                # Calculate total ERC-20 transfers as a metric of DeFi activity
                # This would require additional API calls, which we'll skip for brevity
                
                # Wait to avoid rate limiting
                time.sleep(0.25)
                
            except Exception as e:
                daily_data[f"{contract_name}_interactions"] = None
                print(f"  ✖ Error fetching {contract_name} on {date_str}: {e}")
        
        # Add total DeFi interactions
        interaction_cols = [col for col in daily_data.keys() if col.endswith('_interactions') and daily_data[col] is not None]
        daily_data["total_contract_interactions"] = sum(daily_data[col] for col in interaction_cols)
        
        results.append(daily_data)
        print(f"  → Total interactions: {daily_data['total_contract_interactions']}\n")
    
    new_df = pd.DataFrame(results)
    
    # If we have existing data, merge with it
    if use_cache and os.path.exists(cache_file):
        contract_data = pd.read_csv(cache_file, parse_dates=["date"])
        
        # Combine old and new data, keeping the newest version of any duplicate dates
        combined_df = pd.concat([contract_data, new_df])
        combined_df = combined_df.drop_duplicates(subset='date', keep='last')
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Save updated cache
        combined_df.to_csv(cache_file, index=False)
        print(f"✔️ Updated cached contract interactions data in {cache_file}")
        
        # Filter to requested date range
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date_dt = datetime.now().date()
            
        mask = (combined_df['date'].dt.date >= start_date_dt) & (combined_df['date'].dt.date <= end_date_dt)
        return combined_df[mask].reset_index(drop=True)
    else:
        # Save new cache
        new_df.to_csv(cache_file, index=False)
        print(f"✔️ Saved contract interactions data to {cache_file}")
        return new_df

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
    
    # Test Etherscan with daily transaction data
    # Changed days to 7 and disabled cache to fetch fresh data
    tx_data = fetch_basic_etherscan_metrics(days=7, use_cache=True)
    if tx_data is not None:
        print("\nSample transaction data:")
        print(tx_data.head())
    else:
        print("Could not connect to Etherscan API for transaction data")
    
    return price_data is not None and tx_data is not None

def explore_data():
    """
    Fetch and explore a small sample of data
    """
    # Fetch data
    price_data = fetch_price_data(days=7)  # Increased days for better visualization
    tx_data = fetch_basic_etherscan_metrics(days=7)
    
    if price_data is None or tx_data is None:
        print("Error: Missing either price or transaction data")
        return
    
    # Ensure date column is in the right format for both dataframes
    if isinstance(price_data['date'].iloc[0], pd.Timestamp):
        price_data['date'] = price_data['date'].dt.date
    
    # Fix for transaction data dates
    try:
        tx_data['date'] = pd.to_datetime(tx_data['date'])
        tx_data['date'] = tx_data['date'].dt.date
    except Exception as e:
        print(f"Error converting transaction dates: {e}")
        print(f"Date type: {type(tx_data['date'].iloc[0])}")
    
    # Merge data correctly on date column
    print("Merging price and transaction data...")
    data = pd.merge(price_data, tx_data, on='date', how='inner')
    
    if data.empty:
        print("Error: No data after merge. Check date formats and ranges.")
        print("Price data dates:", price_data['date'].tolist())
        print("Transaction data dates:", tx_data['date'].tolist())
        return
    
    # Print basic statistics
    print("\nData shape:", data.shape)
    print("\nData columns:", data.columns.tolist())
    print("\nData description:")
    print(data.describe())
    
    # Check if required columns exist
    required_cols = ['price', 'volumeto', 'daily_transactions']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return
    
    # Check correlations
    print("\nCorrelation between metrics:")
    print(data[required_cols].corr())
    
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

def analyze_historical_data(start_date, end_date):
    """Enhanced version that includes on-chain metrics"""
    # Fetch various data sources
    price_data = fetch_historical_price_data_chunked(start_date, end_date)
    tx_data = fetch_historical_etherscan_metrics(start_date, end_date)
    # gas_data = fetch_gas_metrics(start_date, end_date)
    # contract_data = fetch_contract_interactions(start_date, end_date)
    
    # Merge all dataframes on date
    data = price_data.copy()
    for df in [tx_data, gas_data, contract_data]:
        if df is not None:
            # Ensure date is in the correct format
            if isinstance(df['date'].iloc[0], pd.Timestamp):
                df['date'] = df['date'].dt.date
            else:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Merge with main dataframe
            data = pd.merge(data, df, on='date', how='left')
    
    # Create correlation matrix with new metrics
    onchain_cols = ['price', 'daily_transactions', 'gas_used', 'avg_gas_price', 'total_contract_interactions']
    valid_cols = [col for col in onchain_cols if col in data.columns]
    
    print("\nCorrelation between on-chain metrics:")
    corr_matrix = data[valid_cols].corr()
    print(corr_matrix)
    
    # Visualize key metrics
    plt.figure(figsize=(15, 15))
    
    # Plot price
    ax1 = plt.subplot(411)
    ax1.plot(data['date'], data['price'], 'b-')
    ax1.set_title('ETH Price (USD)')
    ax1.grid(True)
    
    # Plot transactions
    ax2 = plt.subplot(412, sharex=ax1)
    ax2.plot(data['date'], data['daily_transactions'], 'r-')
    ax2.set_title('Daily Transactions')
    ax2.grid(True)
    
    # Plot gas used
    if 'gas_used' in data.columns:
        ax3 = plt.subplot(413, sharex=ax1)
        ax3.plot(data['date'], data['gas_used'], 'g-')
        ax3.set_title('Daily Gas Used')
        ax3.grid(True)
    
    # Plot contract interactions
    if 'total_contract_interactions' in data.columns:
        ax4 = plt.subplot(414, sharex=ax1)
        ax4.plot(data['date'], data['total_contract_interactions'], 'm-')
        ax4.set_title('Smart Contract Interactions')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'eth_onchain_metrics_{start_date}_to_{end_date or "today"}.png')
    
    return data

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
    # initial_setup()
    
    # Example of how to use historical data analysis
    # Uncomment to use
    # historical_data = analyze_historical_data('2025-03-20', '2025-03-21')
    
    # Testing each function
    # Uncomment to use
    # historical_price_data = fetch_historical_price_data('2025-03-20', '2025-03-21')
    # historical_tx_data = fetch_historical_etherscan_metrics('2025-03-20', '2025-03-21')
    gas_metrics = fetch_gas_metrics('2025-03-20', '2025-03-21')
    
