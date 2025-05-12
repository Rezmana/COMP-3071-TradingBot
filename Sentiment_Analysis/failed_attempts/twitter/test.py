import snscrape.modules.twitter as sntwitter

for tweet in sntwitter.TwitterSearchScraper('bitcoin since:2025-01-01 until:2025-01-02 lang:en').get_items():
    print(tweet.date, tweet.user.username, tweet.content)
    break
