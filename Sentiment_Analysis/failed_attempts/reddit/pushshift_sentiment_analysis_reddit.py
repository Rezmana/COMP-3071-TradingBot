import os
import re
import time
import datetime
import pandas as pd
import nltk
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from psaw import PushshiftAPI
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA
import numpy as np
from dotenv import load_dotenv

# Set device and create directories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph_directory = r"C:\path\to\your\sentiment_graphs"
pkl_directory = r"C:\path\to\your\data_pkl"
os.makedirs(graph_directory, exist_ok=True) 
os.makedirs(pkl_directory, exist_ok=True)

# Load environment variables (if using any API credentials or similar settings)
load_dotenv()

# NLTK resources (assumed already downloaded)
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download('omw-1.4')

# Initialize Pushshift API
api = PushshiftAPI()

# Define your date range for historical posts (for example, all posts from 2022)
start_time = datetime.datetime(2022, 1, 1)
end_time = datetime.datetime(2022, 12, 31)
start_epoch = int(start_time.timestamp())
end_epoch = int(end_time.timestamp())

# Retrieve submissions using Pushshift
data = []
limit_posts = 3000  # Adjust as needed
subreddit = 'cryptocurrency'
gen = api.search_submissions(after=start_epoch,
                             before=end_epoch,
                             subreddit=subreddit,
                             limit=limit_posts)

# Loop through the submissions and store them in a list
for submission in gen:
    data.append({
        'Type': 'Post',
        'Post_id': submission.id,
        'Title': getattr(submission, 'title', ''),
        'Author': getattr(submission, 'author', 'Unknown'),
        'Timestamp': submission.created_utc,  # UNIX timestamp
        'Text': getattr(submission, 'selftext', ''),
        'Score': getattr(submission, 'score', 0),
        'Total_comments': getattr(submission, 'num_comments', 0),
        'Post_URL': getattr(submission, 'url', '')
    })

# You can similarly gather comments if needed, using api.search_comments()
# (Make sure to add a similar loop for comments if required.)

# Convert the scraped data into a DataFrame
fitness_data = pd.DataFrame(data)

# Save raw scraped data as pickle file
fitness_data.to_pickle(os.path.join(pkl_directory, 'reddit_data_psaw.pkl'))
print("Raw historical Reddit data saved.")

# Apply text preprocessing (same as before)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

fitness_data["clean_text"] = fitness_data["Text"].apply(preprocess_text)
fitness_data.to_pickle(os.path.join(pkl_directory, 'reddit_data_clean_psaw.pkl'))

# Load tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest",
                                                            output_hidden_states=True)
model.to(device)
model.eval()

# Define function to extract embeddings
def get_reddit_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        return torch.zeros(768)  # fallback vector of correct size
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]  # shape: [1, seq_len, 768]
        cls_embedding = last_hidden_state[:, 0, :]       # [CLS] token embedding
        return cls_embedding.squeeze().cpu()

def get_sentiment_label(text):
    if not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(scores, dim=1).item()
        return ['NEGATIVE', 'NEUTRAL', 'POSITIVE'][predicted_label]

# Compute and store embeddings and sentiment labels for posts
fitness_data["embedding"] = fitness_data["clean_text"].apply(get_reddit_embedding)
fitness_data["sentiment_label"] = fitness_data["clean_text"].apply(get_sentiment_label)

# Save the results with embeddings and sentiment as pickle files
fitness_data.to_pickle(os.path.join(pkl_directory, 'reddit_data_with_sentiment_psaw.pkl'))

# Optional: Plot sentiment distribution and visualize embeddings via PCA, if desired
def plot_sentiment_distribution(df, title="Sentiment Distribution", filename=None):
    sentiment_counts = df["sentiment_label"].value_counts()
    sentiment_counts = sentiment_counts.reindex(['NEGATIVE', 'NEUTRAL', 'POSITIVE'], fill_value=0)
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Entries")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

plot_sentiment_distribution(fitness_data, title="Post Sentiment Distribution (2022)",
                            filename=os.path.join(graph_directory, "post_sentiment_distribution_psaw.png"))

# PCA Visualization of embeddings
embedding_matrix = np.stack(fitness_data["embedding"].tolist())
pca = PCA(n_components=2)
reduced = pca.fit_transform(embedding_matrix)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                      c=fitness_data["sentiment_label"].map({'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}),
                      cmap='coolwarm')
plt.title("Reddit Sentiment Embeddings (PCA) - 2022")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(os.path.join(graph_directory, "scatter_plot_psaw.png"))
plt.show()
