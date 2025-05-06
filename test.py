import pandas as pd

# Define the path to the pickle file (adjust this as needed)
pickle_path = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\On_Chain_Metrics\data_pkl\integrated_price_data.pkl"

# Load the DataFrame
df = pd.read_pickle(pickle_path)

# Display the first few rows to inspect the content
print("First 5 rows of the DataFrame:")
print(df.head())

# Optionally, inspect specific columns, such as the embedding column
if "embedding" in df.columns:
    print("\nEmbedding column (first 5 entries):")
    print(df["embedding"].head())

# You can also print the DataFrame's info to see details about the columns
print("\nDataFrame Info:")
print(df.info())
