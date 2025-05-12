# hybrid_sentiment_pipeline.py

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# -------- SETTINGS -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    output_hidden_states=True
)
model.to(device)
model.eval()

# Coin tag dictionary for entity detection
COIN_TAGS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "ADA": ["cardano", "ada"],
    "SOL": ["solana", "sol"],
    # Add more coins here...
}
GENERAL_KEYWORDS = ["crypto", "blockchain", "altcoin", "stablecoin", "defi", "web3", "nft"]

# -------------- UTILITIES -------------- #
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy().flatten()
    # Convert to scalar: +1 for positive, 0 for neutral, -1 for negative
    return probs[2] - probs[0]

def get_cls_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(768)  # fallback for empty or bad input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]
        cls_embedding = last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().cpu()

def detect_coins(text):
    coins_found = set()
    clean = text.lower()
    for coin, aliases in COIN_TAGS.items():
        if any(alias in clean for alias in aliases):
            coins_found.add(coin)
    return list(coins_found)

def is_general_sentiment(text):
    return any(word in text.lower() for word in GENERAL_KEYWORDS)

# -------- SCRAPER PLACEHOLDER -------- #
def get_news_headlines():
    # Replace with real scraping from CoinDesk, CoinTelegraph, etc.
    return [
        {"title": "Bitcoin reaches new highs amid ETF optimism", "date": "2024-04-20"},
        {"title": "Ethereum gas fees surge as memecoin frenzy continues", "date": "2024-04-20"},
        {"title": "Crypto market uncertain after U.S. interest rate hike", "date": "2024-04-20"}
    ]

# -------- PIPELINE MAIN FUNCTION -------- #
def build_hybrid_sentiment_table():
    news_data = get_news_headlines()
    coin_sentiment = defaultdict(list)
    coin_embeddings = defaultdict(list)
    general_sentiment = []
    general_embeddings = []

    for item in tqdm(news_data):
        text = clean_text(item["title"])
        sentiment_score = get_sentiment_score(text)
        embedding = get_cls_embedding(text)
        coins = detect_coins(text)

        if coins:
            for coin in coins:
                coin_sentiment[coin].append(sentiment_score)
                coin_embeddings[coin].append(embedding)
        elif is_general_sentiment(text):
            general_sentiment.append(sentiment_score)
            general_embeddings.append(embedding)

    hybrid_data = {}
    for coin in coin_sentiment:
        specific_scores = coin_sentiment[coin]
        specific_embeddings = coin_embeddings[coin]

        if general_embeddings:
            # Combine embeddings: 70% coin-specific + 30% general
            coin_tensor = torch.stack(specific_embeddings).mean(dim=0)
            general_tensor = torch.stack(general_embeddings).mean(dim=0)
            hybrid_embedding = (0.7 * coin_tensor) + (0.3 * general_tensor)
        else:
            hybrid_embedding = torch.stack(specific_embeddings).mean(dim=0)

        hybrid_data[coin] = hybrid_embedding

    return hybrid_data


# -------- INTERACTION WITH FUSION MODEL -------- #
def generate_fusion_ready_dataframe(hybrid_embeddings, price_df):
    records = []

    for coin, embedding in hybrid_embeddings.items():
        coin_data = price_df[price_df["coin"] == coin]
        for _, row in coin_data.iterrows():
            label = assign_label(row["current_price"], row["future_price"])
            records.append({
                "coin": coin,
                "Timestamp": row["time"],
                "embedding": embedding,
                "target_label": label
            })

    return pd.DataFrame(records)


def assign_label(current_price, future_price, threshold_buy=0.02, threshold_sell=-0.02):
    ret = (future_price - current_price) / current_price
    if ret > threshold_buy: return 2  # Buy
    elif ret < threshold_sell: return 0  # Sell
    return 1  # Hold

# -------- Example Usage -------- #
if __name__ == "__main__":
    hybrid_embeddings = build_hybrid_sentiment_table()

    price_df = pd.read_pickle(r"C:\path\to\integrated_price_data.pkl")  # ensure 'coin' column exists
    fusion_df = generate_fusion_ready_dataframe(hybrid_embeddings, price_df)

    fusion_df.to_pickle("fusion_input_data_with_cls.pkl")
    print("Fusion-ready data with real RoBERTa embeddings saved.")

