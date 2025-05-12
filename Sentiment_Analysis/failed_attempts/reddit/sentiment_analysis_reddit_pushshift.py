# sentiment_pushshift.py

import os
import re
import pandas as pd
import datetime as dt
from psaw import PushshiftAPI
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Directories
graph_directory = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\sentiment graphs"
pkl_directory = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\data_pkl"
os.makedirs(graph_directory, exist_ok=True)
os.makedirs(pkl_directory, exist_ok=True)

# Pushshift setup
api = PushshiftAPI()

# Date range — change as needed
start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime(2023, 1, 2)
start_epoch = int(start_date.timestamp())
end_epoch = int(end_date.timestamp())

# Load NLTK assets
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load sentiment model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    output_hidden_states=False
).to(device).eval()
labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

# Sentiment classification
def get_sentiment_label(text):
    if not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(scores, dim=1).item()
        return labels[predicted_label]

# Query Reddit posts
print(f"Fetching Reddit posts from r/Cryptocurrency between {start_date.date()} and {end_date.date()}...")
submissions = list(api.search_submissions(
    after=start_epoch,
    before=end_epoch,
    subreddit='cryptocurrency',
    filter=['id', 'title', 'selftext', 'created_utc'],
    limit=5000  # You can increase or paginate this later
))

# Create DataFrame
df_posts = pd.DataFrame([{
    'id': s.id,
    'title': s.title,
    'text': s.selftext,
    'timestamp': dt.datetime.fromtimestamp(s.created_utc, tz=dt.timezone.utc).date()
} for s in submissions])

print(f"Total posts fetched: {len(df_posts)}")

# Preprocess and label
print("Preprocessing and classifying sentiment...")
df_posts["clean_text"] = df_posts["text"].apply(preprocess_text)
df_posts["sentiment_label"] = tqdm(df_posts["clean_text"].apply(get_sentiment_label))

# Save cleaned dataset
output_path = os.path.join(pkl_directory, "reddit_pushshift_data_with_sentiment.pkl")
df_posts.to_pickle(output_path)
print(f"Saved sentiment-labeled Reddit dataset to: {output_path}")

# output test
print(df_posts[["timestamp", "text", "clean_text", "sentiment_label"]].head())
