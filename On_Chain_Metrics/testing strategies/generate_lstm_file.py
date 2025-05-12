import pandas as pd

from coinmetrics.api_client import CoinMetricsClient

client = CoinMetricsClient()

print(client)

metrics = client.get_asset_metrics(assets='btc', metrics=['PriceUSD','HashRate','TxTfrCnt','AdrActCnt', 'SplyCur'], start_time="2015-01-01", end_time="2024-01-02", frequency='1d')

metrics = pd.DataFrame(metrics)

metrics.head()

metrics.to_csv(r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\On_Chain_Metrics\testing strategies\LSTM_File_.csv", index=False)

df = metrics.copy()