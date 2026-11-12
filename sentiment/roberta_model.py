
import pandas as pd
import torch
import time
from pathlib import Path
from transformers import pipeline

input_file = Path("data/vader_results.csv")
output_file = Path("data/roberta_results.csv")

df = pd.read_csv(input_file)

# Resume from previous progress if available
if output_file.exists():
    saved = pd.read_csv(output_file)
    done = set(saved["uri"])
    df = df[~df["uri"].isin(done)].copy()
    print("Already processed:", len(saved))
else:
    saved = pd.DataFrame()

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

print("Device:", "GPU" if device == 0 else "CPU")
print("Remaining posts:", len(df))

batch_size = 32
chunk_size = 1000

start = time.time()

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size].copy()

    texts = chunk["text"].fillna("").astype(str).tolist()

    results = classifier(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=128
    )

    chunk["roberta_sentiment"] = [
        result["label"].lower()
        for result in results
    ]

    chunk.to_csv(
        output_file,
        mode="a",
        header=not output_file.exists(),
        index=False
    )

    print(
        "Processed:",
        min(i + chunk_size, len(df)),
        "/",
        len(df),
        "Time:",
        round(time.time() - start, 1),
        "seconds"
    )

print("\nClassification completed!")
print("Saved:", output_file)
