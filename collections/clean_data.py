
import pandas as pd
from pathlib import Path

data_dir = Path("data")

df = pd.read_csv(data_dir / "historical_posts.csv")
original_count = len(df)

print("Original posts:", original_count)

# Remove missing or empty text
df = df.dropna(subset=["uri", "created_at", "text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Remove duplicate post IDs
df = df.drop_duplicates(subset=["uri"])

# Parse mixed ISO timestamps
df["created_at"] = pd.to_datetime(
    df["created_at"],
    format="ISO8601",
    utc=True,
    errors="coerce"
)

df = df.dropna(subset=["created_at"])

# Remove extremely short posts
df = df[df["text"].str.len() >= 10]

# Sort by original post time
df = df.sort_values("created_at")

output = data_dir / "cleaned_posts.csv"
df.to_csv(output, index=False)

print("Cleaned posts:", len(df))
print("Removed posts:", original_count - len(df))
print("First post:", df["created_at"].min())
print("Last post:", df["created_at"].max())
print("Saved:", output)
