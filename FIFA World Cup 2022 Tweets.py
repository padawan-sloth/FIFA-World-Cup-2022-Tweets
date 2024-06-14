import pandas as pd

# Load the dataset
file_path = 'C:/Users/Fierce/Desktop/Personal Projects/FIFA World Cup 2022 Tweets/fifa_world_cup_2022_tweets.csv'
world_cup_tweets = pd.read_csv(file_path)

# Display the first few rows of the dataframe and some summary statistics
world_cup_tweets.head(), world_cup_tweets.describe(include='all'), world_cup_tweets.info()

# Data Cleaning Steps:
# 1. Drop the 'Unnamed: 0' column as it's redundant.
# 2. Convert 'Date Created' from string to datetime format for easier analysis.

# Dropping the 'Unnamed: 0' column
world_cup_tweets.drop(columns=['Unnamed: 0'], inplace=True)

# Converting 'Date Created' to datetime
world_cup_tweets['Date Created'] = pd.to_datetime(world_cup_tweets['Date Created'])

# Verify the changes
world_cup_tweets.info(), world_cup_tweets.head()

import matplotlib.pyplot as plt

# Grouping the data by hour to see the distribution of tweets over time
world_cup_tweets['hour'] = world_cup_tweets['Date Created'].dt.hour
tweets_per_hour = world_cup_tweets.groupby('hour').size()

# Plotting the distribution of tweets per hour
plt.figure(figsize=(12, 6))
tweets_per_hour.plot(kind='bar', color='skyblue')
plt.title('Distribution of Tweets Over Time on the First Day of FIFA World Cup 2022')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Tweets')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Analyzing the distribution of sentiments in the dataset

sentiment_distribution = world_cup_tweets['Sentiment'].value_counts()

# Plotting the sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_distribution.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Distribution of Tweets on the First Day of FIFA World Cup 2022')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Analyzing the distribution of the number of likes to understand tweet popularity

# Descriptive statistics for the 'Number of Likes'
likes_stats = world_cup_tweets['Number of Likes'].describe()

# Histogram to show the distribution of likes
plt.figure(figsize=(12, 6))
plt.hist(world_cup_tweets['Number of Likes'], bins=50, color='purple', range=[0, 500])  # Limiting range to focus on majority
plt.title('Distribution of Likes on Tweets from the First Day of FIFA World Cup 2022')
plt.xlabel('Number of Likes')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')  # Log scale to better visualize the distribution
plt.tight_layout()
plt.show()

likes_stats

# Identifying high-engagement tweets
# Let's define high engagement as tweets that have likes greater than the 75th percentile (more than 2 likes)

high_engagement_tweets = world_cup_tweets[world_cup_tweets['Number of Likes'] > 2]

# Analyzing the content and sentiment of high-engagement tweets
high_engagement_content = high_engagement_tweets[['Tweet', 'Sentiment', 'Number of Likes']]
high_engagement_sentiment_distribution = high_engagement_tweets['Sentiment'].value_counts()

# Display some high-engagement tweets and their sentiment distribution
high_engagement_content.sample(5), high_engagement_sentiment_distribution

# Analyzing the sources of tweets
source_distribution = world_cup_tweets['Source of Tweet'].value_counts()

# Getting engagement by source by calculating the mean number of likes for each source
engagement_by_source = world_cup_tweets.groupby('Source of Tweet')['Number of Likes'].mean().sort_values(ascending=False)

# Displaying the source distribution and average engagement by source
source_distribution.head(10), engagement_by_source.head(10)

# Creating bar charts for the source distribution and engagement by source

# Top 10 Sources by Volume
plt.figure(figsize=(14, 7))
source_distribution.head(10).plot(kind='bar', color='lightblue')
plt.title('Top 10 Sources by Tweet Volume on the First Day of FIFA World Cup 2022')
plt.xlabel('Source of Tweet')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.figure(figsize=(14, 7))
# Top 10 Sources by Average Likes (for sources with more than 10 tweets to avoid outliers)
engagement_by_source_filtered = world_cup_tweets.groupby('Source of Tweet').filter(lambda x: len(x) > 10)
engagement_by_source_filtered = engagement_by_source_filtered.groupby('Source of Tweet')['Number of Likes'].mean().sort_values(ascending=False).head(10)
engagement_by_source_filtered.plot(kind='bar', color='lightgreen')
plt.title('Top 10 Sources by Average Likes on Tweets (Minimum 10 Tweets)')
plt.xlabel('Source of Tweet')
plt.ylabel('Average Number of Likes')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()

import seaborn as sns

# Prepare data for time series analysis of sentiments
sentiment_by_hour = world_cup_tweets.groupby(['hour', 'Sentiment']).size().unstack(fill_value=0)

# Normalize the sentiment counts by hour to compare them proportionally
sentiment_by_hour_normalized = sentiment_by_hour.div(sentiment_by_hour.sum(axis=1), axis=0)

# Plotting the distribution of sentiments over time
plt.figure(figsize=(14, 8))
sns.lineplot(data=sentiment_by_hour_normalized, linewidth=2.5)
plt.title('Normalized Sentiment Distribution Over Time on the First Day of FIFA World Cup 2022')
plt.xlabel('Hour of Day')
plt.ylabel('Proportion of Tweets')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
plt.show()

from wordcloud import WordCloud

# Function to generate a word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Combine all tweets into a single string for overall word cloud
overall_text = " ".join(tweet for tweet in world_cup_tweets['Tweet'])

# Generate word cloud for overall tweets
generate_word_cloud(overall_text, 'Word Cloud for All Tweets from the First Day of FIFA World Cup 2022')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to generate word cloud for specific sentiment
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Filter tweets by sentiment
positive_text = " ".join(tweet for tweet in world_cup_tweets[world_cup_tweets['Sentiment'] == 'positive']['Tweet'])
neutral_text = " ".join(tweet for tweet in world_cup_tweets[world_cup_tweets['Sentiment'] == 'neutral']['Tweet'])
negative_text = " ".join(tweet for tweet in world_cup_tweets[world_cup_tweets['Sentiment'] == 'negative']['Tweet'])

# Generate word clouds for each sentiment
generate_word_cloud(positive_text, 'Word Cloud for Positive Tweets')
generate_word_cloud(neutral_text, 'Word Cloud for Neutral Tweets')
generate_word_cloud(negative_text, 'Word Cloud for Negative Tweets')

# Analysis 1: User Engagement - Identifying top influencers or active accounts
# Assuming each tweet can be uniquely identified by its content and timestamp for this analysis
# We need username data which is not available directly, hence focusing on the tweets itself.

# Extract tweet counts by 'Source of Tweet' as a proxy for user activity (assuming more active sources)
top_sources_by_activity = world_cup_tweets['Source of Tweet'].value_counts().head(10)

# Analysis 2: Hashtag Analysis - Most popular hashtags
# Extracting all words that start with '#' from the tweets and count their frequency
import re
from collections import Counter

# Function to find hashtags in tweets
def find_hashtags(tweet):
    return re.findall(r'\#\w+', tweet.lower())  # Using lower to standardize hashtags

# Applying the function to each tweet and summing up the list of hashtags into a single list
hashtags = sum(world_cup_tweets['Tweet'].apply(find_hashtags), [])

# Counting the frequencies of each hashtag
hashtag_counts = Counter(hashtags)

# Most common hashtags
most_common_hashtags = hashtag_counts.most_common(10)

# Display results for both analyses
top_sources_by_activity, most_common_hashtags

# For the purpose of the example, let's assume a key match happened around midday (12 PM to 2 PM).
# We will filter tweets from that period and analyze sentiment changes.

# Filtering tweets around the assumed match time (12 PM to 2 PM UTC)
key_match_tweets = world_cup_tweets[(world_cup_tweets['hour'] >= 12) & (world_cup_tweets['hour'] <= 14)]

# Grouping by hour and sentiment to see the distribution
sentiment_during_match = key_match_tweets.groupby(['hour', 'Sentiment']).size().unstack(fill_value=0)

# Normalize the sentiment counts by hour to compare them proportionally during the match
sentiment_during_match_normalized = sentiment_during_match.div(sentiment_during_match.sum(axis=1), axis=0)

# Plotting the normalized sentiment distribution during the key match time
plt.figure(figsize=(10, 6))
sns.lineplot(data=sentiment_during_match_normalized, linewidth=2.5)
plt.title('Normalized Sentiment Distribution During Key Match Time (12 PM to 2 PM UTC)')
plt.xlabel('Hour of Day (During Match)')
plt.ylabel('Proportion of Tweets')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
plt.show()

# Assumption: Extracting mentions of popular team names or abbreviations from the tweets
# Common teams playing in World Cup 2022 might include "Brazil", "Germany", "Spain", "Argentina", etc.

# Teams to look for in tweets (as an example)
teams = ['brazil', 'germany', 'spain', 'argentina']

# Function to check if a tweet mentions a team
def mentions_team(tweet, team):
    return team in tweet.lower()

# Creating a new column for each team to mark if they are mentioned in the tweet
for team in teams:
    world_cup_tweets[team] = world_cup_tweets['Tweet'].apply(lambda x: mentions_team(x, team))

# Grouping data by team and sentiment
team_sentiment_counts = {}
for team in teams:
    team_sentiment_counts[team] = world_cup_tweets[world_cup_tweets[team] == True]['Sentiment'].value_counts(normalize=True)

# Display the sentiment distribution for each team
print(team_sentiment_counts)

# Re-importing necessary libraries and redoing the process

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Data Preparation
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(world_cup_tweets['Tweet'])
y = world_cup_tweets['Sentiment']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print(report)

# Let's check the dataset columns again to confirm the absence of explicit geographical data
print(world_cup_tweets.head())

# Additional countries to check in tweets based on their participation or interest in the World Cup
additional_countries = ['france', 'usa', 'england', 'mexico', 'japan']

# Add these countries to our list
teams.extend(additional_countries)

# Check for mentions of these additional countries in tweets
for country in additional_countries:
    world_cup_tweets[country] = world_cup_tweets['Tweet'].apply(lambda x: mentions_team(x, country))

# Aggregating sentiment by mentioned country
country_sentiment_counts = {}
for country in teams:
    if country in world_cup_tweets.columns:
        country_tweets = world_cup_tweets[world_cup_tweets[country] == True]
        country_sentiment_counts[country] = country_tweets['Sentiment'].value_counts(normalize=True)

print(country_sentiment_counts)
