import praw
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

reddit = praw.Reddit(client_id=reddit_client_id,
                     client_secret=reddit_client_secret,
                     user_agent=reddit_user_agent)

subreddit = reddit.subreddit('cryptocurrency')

data = []
data_comments = []

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
    
    if post.num_comments > 0:
        comments = post.comments
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
            
fitness_data = pd.DataFrame(data)
fitness_data_comments = pd.DataFrame(data_comments)

fitness_data.to_csv('reddit_data.csv', index=False)
fitness_data_comments.to_csv('reddit_comments_data.csv', index=False)
