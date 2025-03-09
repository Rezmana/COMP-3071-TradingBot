import pandas as pd
import re
import os
import ccxt
import json
from dotenv import load_dotenv
import os
import asyncio
from configparser import ConfigParser
from twikit import Client, TooManyRequests

load_dotenv()

MINIMUM_TWEETS = 10
query = 'BTC'


config = ConfigParser()
config.read('config.ini')
username = config['x']['username']
email = config['x']['email']
password = config['x']['password']

# Create an instance of the ccxt.binance class
# exchange = ccxt.binance()

# tickeer = exchange.fetch_ticker('BTC/USDT')
# print(json.dumps(tickeer, indent=4))

# Authenticate to Twitter
client = Client(language='en-US')
client.login(auth_info_1 = username, auth_info_2 = email, password=password) 
client.save_cookies('cookies.json')


