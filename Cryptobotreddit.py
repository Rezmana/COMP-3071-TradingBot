import praw
import os
import pandas as pd
import re
import nltk
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load environment variables
load_dotenv()

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

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


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
for post in subreddit.hot(limit=10):
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
    post.comments.replace_more(limit=0)  # Load all comments
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

# Display preprocessed data
print(fitness_data[["Text", "clean_text"]].head())
