import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

# --- PARAMETERS ---
start_date = '2023-12-01'
end_date   = '2023-12-31'
time_steps = 60    # how many days the LSTM looks back

# --- 1) LOAD RAW DATA (no slicing yet) ---
df_raw = pd.read_csv(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\On_Chain_Metrics\testing strategies\LSTM_File_.csv"
)
df_raw['time'] = pd.to_datetime(df_raw['time'])

# strip any timezone info so comparisons are tz-naive
df_raw['time'] = df_raw['time'].dt.tz_localize(None)
df_raw = df_raw.sort_values('time')

# --- 2) EXTEND SLICE BACKWARDS TO PRESERVE LOOK-BACK WINDOW ---
full_start = pd.to_datetime(start_date) - pd.Timedelta(days=time_steps)
full_end   = pd.to_datetime(end_date)

df = df_raw[(df_raw['time'] >= full_start) & (df_raw['time'] <= full_end)].copy()

# --- 3) PREPROCESS DATA ---
# Create target column
df['PriceUSD_target'] = df['PriceUSD'].shift(-1)

df['TxTfrCnt_per_address'] = df['TxTfrCnt'] / df['AdrActCnt']
df['SplyCur_per_address'] = df['SplyCur'] / df['AdrActCnt']

df.dropna(subset=['PriceUSD_target'], inplace=True)

# Prepare features (exclude technical indicators added later)
features = df[['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD']].values
prices = df['PriceUSD'].values

# --- 4) DEFINE AGENT CLASSES WITH CONSISTENT DEBUGGING ---
class LSTMTradingAgent:
    def __init__(self, name, model, scaler_X, scaler_y, initial_cash=100):
        self.name = name
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.crypto = 0.0
        self.history = []

    def predict_price(self, X_window):
        """Compute technical indicators per window to avoid truncation."""
        window_df = pd.DataFrame(
            X_window, 
            columns=['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD']
        )
        
        # Add derived features
        window_df['TxTfrCnt_per_address'] = window_df['TxTfrCnt'] / window_df['AdrActCnt']
        window_df['SplyCur_per_address'] = window_df['SplyCur'] / window_df['AdrActCnt']
        
        # Add technical indicators
        window_df['MA_7'] = window_df['PriceUSD'].rolling(window=7).mean()
        window_df['MA_30'] = window_df['PriceUSD'].rolling(window=30).mean()
        window_df['MA_7_30_ratio'] = window_df['MA_7'] / window_df['MA_30']
        window_df['Volatility_30'] = window_df['PriceUSD'].rolling(window=30).std()
        
        # Drop NaNs created by rolling windows
        window_df.dropna(inplace=True)
        
        # Check if window has enough data after dropping NaNs
        if len(window_df) == 0:
            return np.nan  # Handle edge cases
        
        # Select features and scale
        X_processed = window_df[['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD']].values
        Xs = self.scaler_X.transform(X_processed)
        Xs = Xs.reshape(1, *Xs.shape)
        
        # Predict and inverse transform
        y_pred_scaled = self.model.predict(Xs, verbose=0)
        p_hat = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        return p_hat

    def act(self, X_window, price_now, current_date):
        raise NotImplementedError("Subclasses must implement act()")

    def get_final_balance(self, price_now):
        return self.cash + self.crypto * price_now


class GreedyBuyer(LSTMTradingAgent):
    def act(self, X_window, price_now, current_date):
        p_hat = self.predict_price(X_window)
        
        # Debug output
        print(
            f"{self.name} | {current_date.date()} | "
            f"price_now={price_now:.2f} | "
            f"p_hat={p_hat:.2f} | "
            f"decision={'BUY' if p_hat > price_now * 1.001 and self.cash > 0 else 'SELL' if p_hat < price_now * 0.999 and self.crypto > 0 else 'HOLD'}"
        )
        
        if p_hat > price_now * 1.001 and self.cash > 0:
            self.crypto += self.cash / price_now
            self.cash = 0
        elif p_hat < price_now * 0.999 and self.crypto > 0:
            self.cash += self.crypto * price_now
            self.crypto = 0
        self.history.append(self.get_final_balance(price_now))


class CautiousHolder(LSTMTradingAgent):
    def act(self, X_window, price_now, current_date):
        p_hat = self.predict_price(X_window)
        
        # Debug output
        print(
            f"{self.name} | {current_date.date()} | "
            f"price_now={price_now:.2f} | "
            f"p_hat={p_hat:.2f} | "
            f"decision={'BUY' if p_hat > price_now * 1.05 and self.cash > 0 else 'SELL' if p_hat < price_now * 0.98 and self.crypto > 0 else 'HOLD'}"
        )
        
        if p_hat > price_now * 1.05 and self.cash > 0:
            amt = 0.25 * self.cash
            self.crypto += amt / price_now
            self.cash -= amt
        elif p_hat < price_now * 0.98 and self.crypto > 0:
            amt = 0.50 * self.crypto
            self.cash += amt * price_now
            self.crypto -= amt
        self.history.append(self.get_final_balance(price_now))


class PartialTrader(LSTMTradingAgent):
    def act(self, X_window, price_now, current_date):
        p_hat = self.predict_price(X_window)
        
        # Debug output
        print(
            f"{self.name} | {current_date.date()} | "
            f"price_now={price_now:.2f} | "
            f"p_hat={p_hat:.2f} | "
            f"decision={'BUY' if p_hat > price_now * 1.01 and self.cash > 0 else 'SELL' if p_hat < price_now * 0.99 and self.crypto > 0 else 'HOLD'}"
        )
        
        if p_hat > price_now * 1.01 and self.cash > 0:
            amt = 0.10 * self.cash
            self.crypto += amt / price_now
            self.cash -= amt
        elif p_hat < price_now * 0.99 and self.crypto > 0:
            amt = 0.10 * self.crypto
            self.cash += amt * price_now
            self.crypto -= amt
        self.history.append(self.get_final_balance(price_now))


class RandomTrader(LSTMTradingAgent):
    def __init__(self, name, model, scaler_X, scaler_y, initial_cash=100,
                 buy_prob=0.6, sell_prob=0.3, hold_prob=0.1):
        super().__init__(name, model, scaler_X, scaler_y, initial_cash)
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob
        self.hold_prob = hold_prob

    def act(self, X_window, price_now, current_date):
        p_hat = self.predict_price(X_window)
        decision = np.random.choice(
            ['BUY','SELL','HOLD'],
            p=[self.buy_prob, self.sell_prob, self.hold_prob]
        )
        
        # Debug output
        print(
            f"{self.name} | {current_date.date()} | "
            f"price_now={price_now:.2f} | "
            f"p_hat={p_hat:.2f} | "
            f"decision={decision}"
        )
        
        if decision == 'BUY' and self.cash > 0:
            buy_pct = np.random.uniform(0.1, 0.7)
            amt = self.cash * buy_pct
            self.crypto += amt / price_now
            self.cash -= amt
        elif decision == 'SELL' and self.crypto > 0:
            sell_pct = np.random.uniform(0.1, 0.7)
            amt = self.crypto * sell_pct
            self.cash += amt * price_now
            self.crypto -= amt
        self.history.append(self.get_final_balance(price_now))


class MovingAverageTrader(LSTMTradingAgent):
    def __init__(self, name, model, scaler_X, scaler_y, initial_cash=100,
                 short_window=7, long_window=30):
        super().__init__(name, model, scaler_X, scaler_y, initial_cash)
        self.short_window = short_window
        self.long_window = long_window
        self.prices_seen = []

    def act(self, X_window, price_now, current_date):
        p_hat = self.predict_price(X_window)
        self.prices_seen.append(price_now)
        decision = 'HOLD'
        
        if len(self.prices_seen) >= self.long_window:
            sma = np.mean(self.prices_seen[-self.short_window:])
            lma = np.mean(self.prices_seen[-self.long_window:])
            
            if sma > lma and self.cash > 0:
                decision = 'BUY'
                amt = 0.5 * self.cash
                self.crypto += amt / price_now
                self.cash -= amt
            elif sma < lma and self.crypto > 0:
                decision = 'SELL'
                amt = 0.5 * self.crypto
                self.cash += amt * price_now
                self.crypto -= amt
        
        # Debug output
        print(
            f"{self.name} | {current_date.date()} | "
            f"price_now={price_now:.2f} | "
            f"p_hat={p_hat:.2f} | "
            f"decision={decision}"
        )
        
        self.history.append(self.get_final_balance(price_now))

# --- 5) LOAD MODEL AND SCALERS ---
model = load_model(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\trading_agent\model"
    r"\Bit_Coin_Price_Prediction_LSTM_Model_4chain.keras"
)
scaler_X = joblib.load(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\trading_agent\scalers"
    r"\btc_scaler_X_4ChainMetrics.pkl"
)
scaler_y = joblib.load(
    r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code"
    r"\COMP-3071-TradingBot\trading_agent\scalers"
    r"\btc_scaler_y_4ChainMetrics.pkl"
)

# --- 6) INITIATE AGENTS ---
agents = [
    GreedyBuyer("GreedyBuyer", model, scaler_X, scaler_y),
    CautiousHolder("CautiousHolder", model, scaler_X, scaler_y),
    PartialTrader("PartialTrader", model, scaler_X, scaler_y),
    RandomTrader("RandomTrader", model, scaler_X, scaler_y),
    MovingAverageTrader("MovingAverageTrader", model, scaler_X, scaler_y)
]

# --- 7) RUN BACKTEST WITH DEBUGGING ---
print("Starting tournament simulation with look-back preserved...\n")

for t in range(time_steps, len(prices)):
    current_time = df['time'].iloc[t]
    if current_time < pd.to_datetime(start_date):
        continue

    X_window = features[t - time_steps : t]
    price_now = prices[t]

    # Daily header
    print(f"\n=== Day {t-time_steps+1}: {current_time.date()} ===")
    print(f"Actual Price: ${price_now:.2f}")
    print("-" * 50)
    
    for agent in agents:
        agent.act(X_window, price_now, current_time)

# --- 8) RESULTS & PLOT ---
print("\n\nFinal Tournament Results:")
for agent in agents:
    final_bal = agent.get_final_balance(prices[-1])
    growth = (final_bal - agent.initial_cash) / agent.initial_cash * 100
    print(f"{agent.name:20s}: ${final_bal:.2f} ({growth:.2f}% growth)")

plt.figure(figsize=(14, 7))
for agent in agents:
    plt.plot(agent.history, label=agent.name)
plt.title('Agent Portfolio Over Time')
plt.xlabel('Trading Steps')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()