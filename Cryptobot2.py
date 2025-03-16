import ccxt
import time
import pandas as pd
import ta


# Configure Exchange (Example: Binance)
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET_KEY",
    "enableRateLimit": True
})


# Trading Configuration
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"  # 1-minute candles
ORDER_SIZE = 0.001  # Adjust based on your risk
SMA_PERIOD = 14  # Simple Moving Average period

def fetch_market_data(symbol, timeframe, limit=50):
    """Fetch OHLCV data from the exchange"""
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def apply_strategy(df):
    """Apply a simple SMA strategy"""
    df["sma"] = ta.trend.sma_indicator(df["close"], SMA_PERIOD)
    return df

def get_balance(asset):
    """Fetch available balance for a specific asset"""
    balance = exchange.fetch_balance()
    return balance.get("free", {}).get(asset, 0)

def place_order(side, amount, symbol):
    """Place a market order"""
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f"Order placed: {side} {amount} {symbol}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None

def trade_logic():
    """Main trading logic"""
    df = fetch_market_data(SYMBOL, TIMEFRAME)
    df = apply_strategy(df)
    
    last_close = df["close"].iloc[-1]
    last_sma = df["sma"].iloc[-1]

    print(f"Last Close: {last_close}, Last SMA: {last_sma}")

    if last_close > last_sma:
        print("Buy Signal Detected!")
        usdt_balance = get_balance("USDT")
        if usdt_balance > 10:  # Minimum balance condition
            place_order("buy", ORDER_SIZE, SYMBOL)
    elif last_close < last_sma:
        print("Sell Signal Detected!")
        btc_balance = get_balance("BTC")
        if btc_balance > ORDER_SIZE:
            place_order("sell", ORDER_SIZE, SYMBOL)

if __name__ == "__main__":
    while True:
        trade_logic()
        time.sleep(60)  # Run every minute
