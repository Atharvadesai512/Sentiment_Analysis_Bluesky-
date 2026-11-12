
import pandas as pd

df = pd.read_csv("data/historical_posts.csv")
coverage = pd.read_csv("data/collection_coverage.csv")

df["created_at"] = pd.to_datetime(
    df["created_at"],
    format="ISO8601",
    utc=True
)
print("Total posts:", len(df))
print("Duplicate URIs:", df["uri"].duplicated().sum())
print("First post:", df["created_at"].min())
print("Last post:", df["created_at"].max())

print("Failed windows:", (~coverage["success"]).sum())
print(
    "Possibly truncated windows:",
    coverage["possibly_truncated"].sum()
)

hourly = df.set_index("created_at").resample("h").size()

print("Hours represented:", (hourly > 0).sum())
print("Hours with zero sampled posts:", (hourly == 0).sum())
print("Average sampled posts/hour:", round(hourly.mean(), 2))
