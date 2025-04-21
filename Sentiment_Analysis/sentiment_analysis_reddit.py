import praw
import os
import pandas as pd
import re
import nltk
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
import time

# Set device and create graph directory if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph_directory = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\Sentiment_Analysis\sentiment graphs"
pkl_directory = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\Sentiment_Analysis\data_pkl"
os.makedirs(graph_directory, exist_ok=True) 
os.makedirs(pkl_directory, exist_ok=True)

# Load environment variables
load_dotenv()

# NLTK resources (assumed already downloaded)
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download('omw-1.4')

# Load Reddit API credentials
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

# Initialize Reddit API and define subreddit
reddit = praw.Reddit(client_id=reddit_client_id,
                     client_secret=reddit_client_secret,
                     user_agent=reddit_user_agent)
subreddit = reddit.subreddit('cryptocurrency')

# Lists to store posts and comments
data = []
data_comments = []

# Initialize NLTK utilities
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text Preprocessing Function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Scrape posts and comments from the subreddit
for post in subreddit.hot(limit=3000):
    data.append({
        'Type': 'Post',
        'Post_id': post.id,
        'Title': post.title,
        'Author': post.author.name if post.author else 'Unknown',
        'Timestamp': post.created_utc,
        'Text': post.selftext,
        'Score': post.score,
        'Total_comments': post.num_comments,
        'Post_URL': post.url
    })
    
    # Extract comments (replace_more to load more comments)
    post.comments.replace_more(limit=10)
    comments = post.comments.list()
    for comment in comments:
        data_comments.append({
            'Type': 'Comment',
            'Post_id': post.id,
            'Author': comment.author.name if comment.author else 'Unknown',
            'Timestamp': comment.created_utc,
            'Text': comment.body,
            'Score': comment.score,
            'Total_comments': 0,
            'Post_URL': post.url
        })
    time.sleep(0.5)  # Polite pause

# Convert scraped data to DataFrames
fitness_data = pd.DataFrame(data)
fitness_data_comments = pd.DataFrame(data_comments)

# Save raw scraped data as pickle files
fitness_data.to_pickle(os.path.join(pkl_directory, 'reddit_data.pkl'))
fitness_data_comments.to_pickle(os.path.join(pkl_directory, 'reddit_comments_data.pkl'))

# Apply text preprocessing
fitness_data["clean_text"] = fitness_data["Text"].apply(preprocess_text)
fitness_data_comments["clean_text"] = fitness_data_comments["Text"].apply(preprocess_text)

# Save preprocessed data as pickle
fitness_data.to_pickle(os.path.join(pkl_directory, 'reddit_data_clean.pkl'))
fitness_data_comments.to_pickle(os.path.join(pkl_directory, 'reddit_comments_data_clean.pkl'))

# Display preprocessed data
print(fitness_data[["Text", "clean_text"]].head())

# -------------------------------
# Define functions to extract embeddings and sentiment labels
# -------------------------------
def get_reddit_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        return torch.zeros(768)  # fallback vector
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
        return labels[predicted_label]

# Load tokenizer and model, then move model to device and set eval mode
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", 
                                                            output_hidden_states=True)
model.to(device)
model.eval()

# Define sentiment labels
labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

# Compute and store embeddings and sentiment labels for posts and comments
fitness_data["embedding"] = fitness_data["clean_text"].apply(get_reddit_embedding)
fitness_data_comments["embedding"] = fitness_data_comments["clean_text"].apply(get_reddit_embedding)

fitness_data["sentiment_label"] = fitness_data["clean_text"].apply(get_sentiment_label)
fitness_data_comments["sentiment_label"] = fitness_data_comments["clean_text"].apply(get_sentiment_label)

# Save the results with embeddings and sentiment as pickle files
fitness_data.to_pickle(os.path.join(pkl_directory, 'reddit_data_with_sentiment.pkl'))
fitness_data_comments.to_pickle(os.path.join(pkl_directory, 'reddit_comments_with_sentiment.pkl'))

# Optional preview
print(fitness_data[["clean_text", "sentiment_label"]].head())

# -------------------------------
# Plotting Functions: Save graphs to a specified directory
# -------------------------------
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

# Plot and save sentiment distribution for posts and comments
plot_sentiment_distribution(fitness_data, title="Post Sentiment Distribution", 
                            filename=os.path.join(graph_directory, "post_sentiment_distribution.png"))
plot_sentiment_distribution(fitness_data_comments, title="Comment Sentiment Distribution", 
                            filename=os.path.join(graph_directory, "comment_sentiment_distribution.png"))

# -------------------------------
# PCA Visualization of embeddings
# -------------------------------
# Convert torch tensors in the embedding column to numpy arrays
embedding_matrix = np.stack(fitness_data["embedding"].tolist())
pca = PCA(n_components=2)
reduced = pca.fit_transform(embedding_matrix)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                      c=fitness_data["sentiment_label"].map({'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}),
                      cmap='coolwarm')
plt.title("Reddit Sentiment Embeddings (PCA)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(os.path.join(graph_directory, "scatter_plot.png"))
plt.show()
