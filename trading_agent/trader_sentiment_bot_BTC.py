import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# --- PARAMETERS ---
START_DATE       = '2023-12-01'
END_DATE         = '2023-12-31'
TIME_STEPS       = 60    # look-back window for LSTM
MODEL = "finbert"
SENTIMENT_FOLDER = fr"Sentiment_Analysis\BERT runners\sentiment_{MODEL}\bitcoin"
SENTIMENT_WEIGHT = 0.075
# --- 1) LOAD RAW ON-CHAIN DATA (no sl---
df_raw = pd.read_csv(
    r"On_Chain_Metrics\testing strategies\LSTM_File_.csv"
)
df_raw['time'] = pd.to_datetime(df_raw['time'])
df_raw['time'] = df_raw['time'].dt.tz_localize(None)
df_raw.sort_values('time', inplace=True)

# --- 2) SLICE WITH EXTENDED LOOK-BACK WINDOW ---
full_start = pd.to_datetime(START_DATE) - pd.Timedelta(days=TIME_STEPS)
full_end   = pd.to_datetime(END_DATE)

df = df_raw[(df_raw['time'] >= full_start) & (df_raw['time'] <= full_end)].copy()
df.reset_index(drop=True, inplace=True)

features = df[['PriceUSD', 'HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur']].values
prices   = df['PriceUSD'].values
dates    = df['time'].dt.date

# --- 3) LOAD & AGGREGATE ROLLING-WINDOW SENTIMENT JSONs ---
def load_sentiment_data(folder):
    sentiment = {}
    pattern = re.compile(r"bitcoin_\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})_FULL_scored\.json")
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if not m:
            continue
        end_str = m.group(1)
        path = os.path.join(folder, fname)
        with open(path, 'r') as f:
            data = json.load(f)
        pos = [e['probs']['pos'] for e in data if 'probs' in e]
        neg = [e['probs']['neg'] for e in data if 'probs' in e]
        sentiment[end_str] = ((sum(pos)/len(pos)) - (sum(neg)/len(neg))) if pos and neg else 0.0
    return sentiment

sentiment_data = load_sentiment_data(SENTIMENT_FOLDER)
print("Loaded sentiment for dates:", sorted(sentiment_data.keys()))

# --- 4) DEFINE SENTIMENT-AWARE AGENT CLASSES ---
class LSTMTradingAgent:
    def __init__(self, name, model, scaler_X, scaler_y, initial_cash=100, sentiment_k=SENTIMENT_WEIGHT):
        self.name = name
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.crypto = 0.0
        self.history = []
        self.sentiment_k = sentiment_k

    def predict_price(self, X_window):
        window_df = pd.DataFrame(X_window, columns=['PriceUSD', 'HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur'])
        window_df['TxTfrCnt_per_address'] = window_df['TxTfrCnt'] / window_df['AdrActCnt']
        window_df['SplyCur_per_address'] = window_df['SplyCur'] / window_df['AdrActCnt']
        window_df['PriceUSD_target'] = window_df['PriceUSD'].shift(-1)
        window_df['MA_7'] = window_df['PriceUSD'].rolling(window=7).mean()
        window_df['MA_30'] = window_df['PriceUSD'].rolling(window=30).mean()
        window_df['MA_7_30_ratio'] = window_df['MA_7'] / window_df['MA_30']
        window_df['Volatility_30'] = window_df['PriceUSD'].rolling(window=30).std()
        window_df.dropna(inplace=True)
        X_processed = window_df[['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD']].values
        Xs = self.scaler_X.transform(X_processed)
        Xs = Xs.reshape(1, *Xs.shape)
        y_pred_scaled = self.model.predict(Xs, verbose=0)
        p_hat = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        return p_hat

    def act(self, X_window, price_now, sentiment_score):
        raise NotImplementedError

    def get_final_balance(self, price_now):
        return self.cash + self.crypto * price_now


class SentimentAwareGreedyBuyer(LSTMTradingAgent):
    def act(self, X_window, price_now, sentiment_score):
        p_hat = self.predict_price(X_window)
        p_adj = p_hat * (1 + self.sentiment_k * sentiment_score)
        if p_adj > price_now * 1.001 and self.cash > 0:
            self.crypto += self.cash / price_now
            self.cash = 0
        elif p_adj < price_now * 0.999 and self.crypto > 0:
            self.cash += self.crypto * price_now
            self.crypto = 0
        self.history.append(self.get_final_balance(price_now))


class SentimentAwareCautiousHolder(LSTMTradingAgent):
    def act(self, X_window, price_now, sentiment_score):
        p_hat = self.predict_price(X_window)
        p_adj = p_hat * (1 + self.sentiment_k * sentiment_score)
        if p_adj > price_now * 1.05 and self.cash > 0:
            amt = 0.25 * self.cash
            self.crypto += amt / price_now
            self.cash -= amt
        elif p_adj < price_now * 0.98 and self.crypto > 0:
            amt = 0.5 * self.crypto
            self.cash += amt * price_now
            self.crypto -= amt
        self.history.append(self.get_final_balance(price_now))


class SentimentAwarePartialTrader(LSTMTradingAgent):
    def act(self, X_window, price_now, sentiment_score):
        p_hat = self.predict_price(X_window)
        p_adj = p_hat * (1 + self.sentiment_k * sentiment_score)
        if p_adj > price_now * 1.01 and self.cash > 0:
            amt = 0.1 * self.cash
            self.crypto += amt / price_now
            self.cash -= amt
        elif p_adj < price_now * 0.99 and self.crypto > 0:
            amt = 0.1 * self.crypto
            self.cash += amt * price_now
            self.crypto -= amt
        self.history.append(self.get_final_balance(price_now))


class SentimentAwareRandomTrader(LSTMTradingAgent):
    def __init__(self, name, model, scaler_X, scaler_y, initial_cash=100,
                 buy_prob=0.6, sell_prob=0.3, hold_prob=0.1, sentiment_k=0.02):
        super().__init__(name, model, scaler_X, scaler_y, initial_cash, sentiment_k)
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob
        self.hold_prob = hold_prob

    def act(self, X_window, price_now, sentiment_score):
        b = np.clip(self.buy_prob + self.sentiment_k * sentiment_score, 0, 1)
        s = np.clip(self.sell_prob - self.sentiment_k * sentiment_score, 0, 1)
        h = max(1 - b - s, 0)
        tot = b + s + h
        b, s, h = b / tot, s / tot, h / tot

        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[b, s, h])
        if decision == 'BUY' and self.cash > 0:
            pct = np.random.uniform(0.1, 0.7)
            amt = pct * self.cash
            self.crypto += amt / price_now
            self.cash -= amt
        elif decision == 'SELL' and self.crypto > 0:
            pct = np.random.uniform(0.1, 0.7)
            amt = pct * self.crypto
            self.cash += amt * price_now
            self.crypto -= amt

        self.history.append(self.get_final_balance(price_now))

class SentimentAwareMovingAverageTrader(LSTMTradingAgent):
    def __init__(self, name, model, scaler_X, scaler_y, initial_cash=100, 
                 short_window=7, long_window=30, sentiment_k=SENTIMENT_WEIGHT, 
                 historical_prices=None):
        super().__init__(name, model, scaler_X, scaler_y, initial_cash, sentiment_k)
        self.short_window = short_window
        self.long_window = long_window
        
        # Initialize with historical data if provided
        self.prices_seen = []
        if historical_prices is not None:
            self.prices_seen = historical_prices.copy()
            print(f"Pre-loaded {len(self.prices_seen)} historical prices for {name}")

    def act(self, X_window, price_now, sentiment_score):
        self.prices_seen.append(price_now)
        if len(self.prices_seen) >= self.long_window:
            sma = np.mean(self.prices_seen[-self.short_window:])
            lma = np.mean(self.prices_seen[-self.long_window:])
            thresh = lma * (1 - self.sentiment_k * sentiment_score)
            if sma > thresh and self.cash > 0:
                amt = 0.5 * self.cash
                self.crypto += amt / price_now
                self.cash -= amt
            elif sma < thresh and self.crypto > 0:
                amt = 0.5 * self.crypto
                self.cash += amt * price_now
                self.crypto -= amt
        self.history.append(self.get_final_balance(price_now))

# --- 5) LOAD MODEL & SCALERS ---
model = load_model(r"trading_agent\model\Bit_Coin_Price_Prediction_LSTM_Model_4chain.keras")
scaler_X = joblib.load(r"trading_agent\scalers\btc_scaler_X_4ChainMetrics.pkl")
scaler_y = joblib.load(r"trading_agent\scalers\btc_scaler_y_4ChainMetrics.pkl")

# Added for functioning of moving average trader
simulation_start_idx = np.where(df['time'].dt.date >= pd.to_datetime(START_DATE).date())[0][0]
historical_end_idx = simulation_start_idx - 1
historical_start_idx = max(0, historical_end_idx - 30)

historical_prices = []
if historical_start_idx < historical_end_idx:
    historical_prices = prices[historical_start_idx:historical_end_idx].tolist()


# --- 6) INSTANTIATE AGENTS ---
agents = [
    SentimentAwareGreedyBuyer("GreedyBuyer", model, scaler_X, scaler_y),
    SentimentAwareCautiousHolder("CautiousHolder", model, scaler_X, scaler_y),
    SentimentAwarePartialTrader("PartialTrader", model, scaler_X, scaler_y),
    SentimentAwareRandomTrader("RandomTrader", model, scaler_X, scaler_y),
SentimentAwareMovingAverageTrader("MovingAverageTrader", model, scaler_X, scaler_y,historical_prices=historical_prices)
]

# --- 7) RUN BACKTEST ---
print("Starting sentiment-aware simulation...")
for t in range(TIME_STEPS, len(prices)):
    curr_date = dates.iloc[t].isoformat()
    if curr_date < START_DATE:
        continue

    X_win = features[t - TIME_STEPS:t]
    price_now = prices[t]
    senti_score = sentiment_data.get(curr_date, 0.0)

    for agent in agents:
        p_hat = agent.predict_price(X_win)
        p_adj = p_hat * (1 + agent.sentiment_k * senti_score)

        print(f"{agent.name} | {curr_date} | price_now={price_now:.4f} | p_hat={p_hat:.4f} | p_adj={p_adj:.4f} | sentiment={senti_score:.4f}")
        agent.act(X_win, price_now, senti_score)

# --- 8) OUTPUT RESULTS & PLOT ---
print("\nFinal Tournament Results:")
for agent in agents:
    final_bal = agent.get_final_balance(prices[-1])
    growth = (final_bal - agent.initial_cash) / agent.initial_cash * 100
    print(f"{agent.name:20s}: ${final_bal:.2f} ({growth:.2f}% growth)")

plt.figure(figsize=(14, 7))
for agent in agents:
    plt.plot(agent.history, label=agent.name)
plt.title(f'{MODEL}: Sentiment-Aware Agent Portfolios (Dec 1–31)')
plt.xlabel('Trading Steps')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r"")
plt.show()
