"""Repeated local inference measurements; excludes collection, queues and storage."""

import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("USE_TF", "0")
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    test = pd.read_csv(ROOT / "data/bilstm_test.csv")
    sample = test.sample(n=96, random_state=42)
    texts = sample.text.tolist()
    vader = SentimentIntensityAnalyzer()
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    roberta = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True).eval()

    def classify(model, batch):
        if model == "VADER":
            return [vader.polarity_scores(text) for text in batch]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return roberta(**inputs).logits.argmax(dim=1).tolist()

    raw = []
    summaries = []
    with torch.inference_mode():
        for model in ["VADER", "RoBERTa"]:
            for batch_size in [1, 8, 32]:
                for _ in range(3):
                    classify(model, texts[:batch_size])
                for repeat in range(3):
                    durations = []
                    for offset in range(0, len(texts), batch_size):
                        start = time.perf_counter_ns()
                        classify(model, texts[offset:offset + batch_size])
                        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                        durations.append(elapsed_ms)
                        raw.append({"model": model, "batch_size": batch_size, "repeat": repeat,
                                    "offset": offset, "elapsed_ms": elapsed_ms})
                    summaries.append({
                        "model": model, "batch_size": batch_size, "repeat": repeat,
                        "posts_per_second": len(texts) / (sum(durations) / 1000),
                        "p50_batch_ms": float(np.quantile(durations, .5)),
                        "p95_batch_ms": float(np.quantile(durations, .95)),
                        "p99_batch_ms": float(np.quantile(durations, .99)),
                    })
                print(f"Measured {model}, batch {batch_size}", flush=True)
    output = ROOT / "research/runtime"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw).to_csv(output / "batch_timings.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    metadata = {
        "scope": "Warm local tokenization plus inference only; excludes initialization, arrival, queueing and storage",
        "hardware": platform.processor(), "platform": platform.platform(), "torch_threads": 4,
        "device": "CPU", "sample_rows": sample.index.tolist(), "repeats": 3,
        "roberta_revision": roberta.config._commit_hash, "max_tokens": 128,
        "limitations": "96 fixed test posts; batch 32 has only three timings per repeat. Tail quantiles are descriptive and unstable. Single-post latency is the batch-1 result.",
        "bilstm": "Unavailable because TensorFlow import is blocked by application-control policy",
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
