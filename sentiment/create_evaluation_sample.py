
import pandas as pd
import os

# Load our original cleaned Bluesky posts
df = pd.read_csv("data/cleaned_posts.csv")

# Randomly select 300 posts
sample = df.sample(n=300, random_state=42)

# Keep only required columns
sample = sample[["uri", "text"]].copy()

# Create empty column for manual labels
sample["human_sentiment"] = ""

# Save
os.makedirs("data", exist_ok=True)
sample.to_csv(
    "data/human_evaluation_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Sample created:", len(sample))
print("Saved: data/human_evaluation_sample.csv")
