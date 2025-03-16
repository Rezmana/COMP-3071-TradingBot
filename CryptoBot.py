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

MINIMUM_TWEETS = 10
query = 'chatgpt'

config = ConfigParser()
config.read('config.ini')
username = config['x']['username']
email = config['x']['email']
password = config['x']['password']


client = Client(language='en-US')
async def main():
    # Login and save cookies
    await client.login(auth_info_1=username, auth_info_2=email, password=password)
    # await client.save_cookies('cookies.json')
    print("Logged in successfully!")

asyncio.run(main())
# Authenticate to Twitter


