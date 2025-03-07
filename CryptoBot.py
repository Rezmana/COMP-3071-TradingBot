import ccxt
import json
# Create an instance of the ccxt.binance class
exchange = ccxt.binance()

tickeer = exchange.fetch_ticker('BTC/USDT')

print(json.dumps(tickeer, indent=4))
# print(exchange.fetch_ticker('BTC/USDT'))

