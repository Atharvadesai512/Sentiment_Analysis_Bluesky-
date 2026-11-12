
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Load cleaned historical data
df = pd.read_csv("data/cleaned_posts.csv")

analyzer = SentimentIntensityAnalyzer()

# Classify each post
def classify(text):
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_sentiment"] = df["text"].apply(classify)

# Save results
output = Path("data/vader_results.csv")
df.to_csv(output, index=False)

print("Total posts:", len(df))
print("\nSentiment distribution:")
print(df["vader_sentiment"].value_counts())

print("\nSaved:", output)
