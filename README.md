FIFA World Cup 2022 Tweets Analysis

Overview

This project involves analyzing a dataset of 30,000 tweets from the first day of the FIFA World Cup 2022, focusing on sentiment analysis, user engagement, predictive modeling, and geographical sentiment distribution.

Data Preparation

Data Cleaning: Removed redundant index columns and converted 'Date Created' to datetime format.
Text Preprocessing: Applied preprocessing to the tweet texts to clean and prepare for analysis.

Exploratory Data Analysis (EDA)
Tweet Distribution Over Time:

Plotted the number of tweets per hour, revealing peak times likely coinciding with key match events.
Sentiment Analysis:

Analyzed and visualized the distribution of sentiments (positive, neutral, negative). Positive sentiments were predominant.
Tweet Popularity Analysis:

Examined the distribution of likes per tweet, showing most tweets receive few or no likes, with a few exceptions receiving high engagement.
Source Analysis:

Analyzed the sources of tweets (e.g., iPhone, Android), showing diverse usage patterns across different platforms.

Detailed Analyses

High-Engagement Tweets:

Identified and analyzed tweets with high engagement, examining content and sentiment correlations.

Hashtag Analysis:

Extracted and counted hashtags, identifying the most popular ones which focused on the event and location (e.g., #Qatar2022).

Comparative Sentiment Analysis:

Compared sentiments associated with mentions of different teams and countries, revealing varying emotional responses.

Predictive Modeling:

Developed a logistic regression model to predict tweet sentiments based on text content. The model achieved an accuracy of 68%, with positive sentiments being the easiest to predict accurately.

Geographical Sentiment Distribution:

Inferred locations from tweet content and analyzed sentiment distributions associated with these inferred locations, providing insights into global emotional responses to the event.
Key Insights

Sentiment analysis highlighted generally positive reactions with nuanced responses based on specific events or mentions.

User engagement varied significantly with a small fraction of tweets capturing the majority of interactions.

The predictive model highlighted the potential of machine learning in analyzing social media data, with room for improvement in model complexity and feature engineering.

Conclusion

This analysis provided valuable insights into public sentiment and behavior during the FIFA World Cup 2022, using various data science techniques. These findings can assist in understanding social media dynamics and user engagement in large-scale events.
