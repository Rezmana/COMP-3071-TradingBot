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
from transformers import pipeline
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Load environment variables
load_dotenv()

# Download NLTK resources
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')

# print(nltk.data.find('tokenizers/punkt_tab/english/'))

# Load Reddit API credentials
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

# Initialize Reddit API
reddit = praw.Reddit(client_id=reddit_client_id,
                     client_secret=reddit_client_secret,
                     user_agent=reddit_user_agent)

# Define subreddit
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
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Scrape posts from subreddit
for post in subreddit.hot(limit=10000):
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
    
    # Extract comments
    post.comments.replace_more(limit=10)  # Load all comments
    comments = post.comments.list()  # Convert comment forest into a list
    
    for comment in comments:
        data_comments.append({
            'Type': 'Comment',
            'Post_id': post.id,
            'Author': comment.author.name if comment.author else 'Unknown',
            'Timestamp': comment.created_utc,
            'Text': comment.body,
            'Score': comment.score,
            'Total_comments': 0,  # Comments do not have nested comments here
            'Post_URL': post.url
        })

# Convert to DataFrame
fitness_data = pd.DataFrame(data)
fitness_data_comments = pd.DataFrame(data_comments)

# Save to CSV
fitness_data.to_csv('reddit_data.csv', index=False)
fitness_data_comments.to_csv('reddit_comments_data.csv', index=False)

# Apply text preprocessing
fitness_data["clean_text"] = fitness_data["Text"].apply(preprocess_text)
fitness_data_comments["clean_text"] = fitness_data_comments["Text"].apply(preprocess_text)

fitness_data.to_csv('reddit_data_clean.csv', index=False)
fitness_data_comments.to_csv('reddit_comments_data_clean.csv', index=False)

# Display preprocessed data
print(fitness_data[["Text", "clean_text"]].head())



def get_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL"
    
    # Tokenize and truncate to max length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(scores, dim=1).item()
        return labels[predicted_label]

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model.eval()

# Sentiment labels for this model
labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

# Apply sentiment analysis
fitness_data["sentiment"] = fitness_data["clean_text"].apply(get_sentiment)
fitness_data_comments["sentiment"] = fitness_data_comments["clean_text"].apply(get_sentiment)

# Save results
fitness_data.to_csv('reddit_data_with_sentiment.csv', index=False)
fitness_data_comments.to_csv('reddit_comments_with_sentiment.csv', index=False)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Optional: preview
print(fitness_data[["clean_text", "sentiment"]].head())

# Function to plot sentiment distribution
def plot_sentiment_distribution(df, title="Sentiment Distribution"):
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_counts = sentiment_counts.reindex(['NEGATIVE', 'NEUTRAL', 'POSITIVE'], fill_value=0)  # Consistent order
    
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Entries")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot for posts
plot_sentiment_distribution(fitness_data, title="Post Sentiment Distribution")

# Plot for comments
plot_sentiment_distribution(fitness_data_comments, title="Comment Sentiment Distribution")
