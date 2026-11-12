from pathlib import Path

import pandas as pd

path = Path("data/human_evaluation_sample.csv")
posts = pd.read_csv(path, dtype=str, keep_default_na=False)
labels = {"p": "positive", "n": "negative", "u": "neutral"}

pending = posts.index[posts["human_sentiment"].str.strip().eq("")]
done = len(posts) - len(pending)

for index in pending[:20]:
    print(f"\nPost {done + 1} of {len(posts)}")
    print(posts.at[index, "text"])

    choice = input("p = positive, n = negative, u = neutral, q = quit: ").strip().lower()
    while choice not in labels and choice != "q":
        choice = input("Enter p, n, u, or q: ").strip().lower()

    if choice == "q":
        break

    posts.at[index, "human_sentiment"] = labels[choice]
    temp = path.with_suffix(".tmp")
    posts.to_csv(temp, index=False)
    temp.replace(path)
    done += 1

print(f"\nSaved progress: {done}/{len(posts)} posts labeled.")
