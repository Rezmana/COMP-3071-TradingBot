# COMP-3071-TradingBot

---HOW TO RUN---

For testing the trading bot in its final version immediately, navigate to the 'trading_agent' folder.
There, the file 'trader_bot_BTC.py' runs the trading bot over 1 month's time (December 2023) and outputs a line chart showing each trading agent's
finances over the month. 
'trader_sentiment_bot_BTC.py' does the same, except it incorporates sentiment scores into the trading agents decision making.

The file used to train the LSTM can be found under 'On_Chain_Metrics>model training>LSTM_training.py.
The saved model can be found under 'trading_agent>model'
The saved scalers can be found under 'trading_agent>scalers'

To see the LSTM's price prediction for btc and eth over the period 2022-09-09 to 2024-01-01, navigate to the directory 'On_Chain_Metrics>testing strategies'.

To view the BERT scripts, navigate to 'Sentiment_Analysis>BERT runners>{model}.py', where {model} is the bert model used.
Under 'Sentiment_Analysis>BERT runners', you can also find all the CSV files containing the sentiment scored data.
Under 'Sentiment_Analysis>failed_attempts', you can view all previous scraping attempts from other sources. It is there purely as proof of prior attempts.


---IMPORTANT---

To install the requirements to run this code, please first install the requirements from 'requirements.txt' using the command:

pip install -r requirements.txt