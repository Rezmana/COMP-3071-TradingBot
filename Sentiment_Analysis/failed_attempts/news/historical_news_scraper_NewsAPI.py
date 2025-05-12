import requests
import requests
import json
from datetime import datetime, timedelta

API_KEY = "7e35e48e23584f5795f4eafd9b55d17a"
KEYWORDS = ["Bitcoin", "Ethereum"]
START_DATE = "2023-11-22"
END_DATE = "2024-01-02"
OUTPUT_FILE = "crypto_news_max.json"

def fetch_news(keyword, date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "from": date,
        "to": date,
        "apiKey": API_KEY,
        "pageSize": 100,  # Max articles per request (free tier allows this)
        "sortBy": "publishedAt",  # Most recent first
        "language": "en",
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

all_articles = []
current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    for keyword in KEYWORDS:
        articles = fetch_news(keyword, date_str)
        for article in articles:
            all_articles.append({
                "keyword": keyword,
                "title": article.get("title", ""),
                "snippet": article.get("description", ""),
                "date": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", ""),
                "url": article.get("url", ""),
            })
    current_date += timedelta(days=1)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_articles, f, indent=4)