import os
import snscrape.modules.twitter as sntwitter
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import re
import emoji
from datetime import datetime, timedelta
import numpy as np
import certifi

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# =========================
# CONFIGURATION
# =========================
KEYWORDS = "bitcoin OR btc"
START_DATE = "2017-01-01"
END_DATE = "2020-01-01"
MAX_TWEETS = 1000  # Per day
OUTPUT_DIR = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\twitter\data_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SETUP CARDIFFNLP MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
).to(device).eval()
LABELS = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

# =========================
# TEXT PREPROCESSING
# =========================
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^a-zA-Z0-9\s@#]", "", text)
    return text.strip().lower()

# =========================
# SENTIMENT PREDICTION
# =========================
def get_sentiment(text):
    if not text or len(text.strip()) == 0:
        return "NEUTRAL"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(scores).item()
        return LABELS[predicted_class]

# =========================
# SCRAPE TWEETS FOR A DAY
# =========================
def scrape_tweets_for_day(date):
    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    query = f'bitcoin OR btc since:{date} until:{next_day} lang:en'
    tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= MAX_TWEETS:
            break
        tweets.append({
            "id": tweet.id,
            "date": tweet.date.date(),
            "user": tweet.user.username,
            "text": tweet.content,
            "likes": tweet.likeCount or 0,
            "retweets": tweet.retweetCount or 0,
            "followers": tweet.user.followersCount or 0
        })

    return pd.DataFrame(tweets)

# =========================
# MAIN FUNCTION
# =========================
def run_pipeline(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_results = []

    for single_date in tqdm(pd.date_range(start, end), desc="Processing Days"):
        date_str = single_date.strftime("%Y-%m-%d")
        print(f"\n📆 Scraping tweets for: {date_str}")
        df = scrape_tweets_for_day(date_str)
        if df.empty:
            print("No tweets found.")
            continue

        df["clean_text"] = df["text"].apply(clean_text)
        df["sentiment"] = df["clean_text"].apply(get_sentiment)

        # For now, hardcode to Bitcoin
        df["coin"] = "BTC"

        # Sentiment scoring
        SENTIMENT_MAP = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
        df["sentiment_score"] = df["sentiment"].map(SENTIMENT_MAP)

        # Weight by followers + likes (with log1p to dampen outliers)
        df["weight"] = np.log1p(df["followers"] + df["likes"])
        df["weighted_sentiment"] = df["sentiment_score"] * df["weight"]

        all_results.append(df)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{date_str}_tweets.csv"), index=False)

    if all_results:
        full_df = pd.concat(all_results)
        full_df.to_csv(os.path.join(OUTPUT_DIR, "all_sentiment.csv"), index=False)
        print(f"\n✅ Full tweet dataset saved to {OUTPUT_DIR}/all_sentiment.csv")

        # Aggregate daily sentiment
        daily_sentiment = full_df.groupby("date").agg({
            "weighted_sentiment": "sum",
            "sentiment_score": "mean",
            "id": "count"
        }).rename(columns={"id": "tweet_volume"})

        daily_sentiment.to_csv(os.path.join(OUTPUT_DIR, "daily_sentiment_summary.csv"))
        print(f"📊 Daily summary saved to {OUTPUT_DIR}/daily_sentiment_summary.csv")
    else:
        print("\n⚠️ No results to save.")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_pipeline(START_DATE, END_DATE)
