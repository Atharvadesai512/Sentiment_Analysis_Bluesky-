"""Evaluate models on the same local TweetEval test split."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("USE_TF", "0")
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vader", "roberta", "linear", "bilstm_saved"], required=True)
    args = parser.parse_args()
    test_path = ROOT / "data/bilstm_test.csv"
    test = pd.read_csv(test_path)
    train = pd.read_csv(ROOT / "data/bilstm_train.csv")
    texts = test.text.tolist()
    timing = None
    metadata = {}
    if args.model == "vader":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        model = SentimentIntensityAnalyzer()
        model.polarity_scores(texts[0])
        start = time.perf_counter()
        scores = [model.polarity_scores(text)["compound"] for text in texts]
        elapsed = time.perf_counter() - start
        predicted = [2 if score >= .05 else 0 if score <= -.05 else 1 for score in scores]
        timing = {"elapsed_seconds": elapsed, "posts_per_second": len(texts) / elapsed}
    elif args.model == "linear":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(
            TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        model.fit(train.text, train.label)
        start = time.perf_counter()
        predicted = model.predict(texts).tolist()
        elapsed = time.perf_counter() - start
        timing = {"elapsed_seconds": elapsed, "posts_per_second": len(texts) / elapsed}
        metadata["training"] = "Local training split only; fixed baseline settings"
    elif args.model == "bilstm_saved":
        saved = pd.read_csv(ROOT / "data/bilstm_test_results.csv")
        if not saved[["text", "label"]].equals(test[["text", "label"]]):
            raise ValueError("Saved predictions do not match test rows")
        predicted = saved.bilstm_sentiment.map({label: i for i, label in enumerate(LABELS)}).tolist()
        metadata["prediction_source"] = "Existing saved predictions; fresh inference unavailable"
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        model.eval()
        metadata = {"checkpoint": model_name, "revision": model.config._commit_hash,
                    "threads": 4, "batch_size": 32, "max_length": 128,
                    "preprocessing": "Original local text; same truncation as historical script"}
        mapping = {int(i): LABELS.index(label.lower()) for i, label in model.config.id2label.items()}
        predicted = []
        with torch.inference_mode():
            model(**tokenizer(texts[:2], padding=True, truncation=True, max_length=128, return_tensors="pt"))
            start = time.perf_counter()
            for offset in range(0, len(texts), 32):
                inputs = tokenizer(texts[offset:offset+32], padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
                ids = model(**inputs).logits.argmax(dim=1).tolist()
                predicted.extend(mapping[i] for i in ids)
                if offset % 1024 == 0:
                    print(f"RoBERTa {len(predicted)}/{len(texts)}", flush=True)
        elapsed = time.perf_counter() - start
        timing = {"elapsed_seconds": elapsed, "posts_per_second": len(texts) / elapsed}
    if len(predicted) != len(test) or not set(predicted).issubset({0, 1, 2}):
        raise ValueError("Invalid predictions")
    majority = int(train.label.mode().iloc[0])
    result = {
        "model": args.model, "rows": len(test), "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "metadata": metadata, "timing": timing,
        "timing_scope": "Single offline pass, excluding initialization; not streaming latency or a controlled speed comparison",
        "report": classification_report(test.label, predicted, labels=[0, 1, 2], target_names=LABELS,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test.label, predicted, labels=[0, 1, 2]).tolist(),
        "train_majority_baseline_accuracy": float(test.label.eq(majority).mean()),
    }
    output = ROOT / "research/benchmark"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": test.index, "label": test.label, "prediction": predicted}).to_csv(
        output / f"{args.model}_predictions.csv", index=False)
    (output / f"{args.model}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
