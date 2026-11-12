"""Recompute the research results from saved predictions without loading models."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import pandas as pd


LABELS = ["negative", "neutral", "positive"]
MODELS = {
    "VADER": "vader",
    "RoBERTa": "roberta",
    "BiLSTM": "bilstm",
}


def load_predictions(data, corpus):
    predictions = pd.DataFrame(index=corpus.index)
    for model, prefix in MODELS.items():
        saved = pd.read_csv(data / f"{prefix}_results.csv").set_index(
            "uri", verify_integrity=True
        )
        if not saved.index.sort_values().equals(corpus.index.sort_values()):
            raise ValueError(f"{model} predictions do not cover the cleaned corpus")
        saved = saved.reindex(corpus.index)
        if not saved["text"].equals(corpus["text"]):
            raise ValueError(f"{model} prediction texts differ from the cleaned corpus")
        labels = saved[f"{prefix}_sentiment"]
        if not labels.isin(LABELS).all():
            raise ValueError(f"{model} has missing or unrecognized predictions")
        predictions[model] = labels
    return predictions


def cross_tab(left, right):
    return pd.crosstab(left, right).reindex(index=LABELS, columns=LABELS, fill_value=0)


def evaluate(gold, predicted):
    matrix = cross_tab(gold, predicted)
    metrics = {}
    for label in LABELS:
        correct = int(matrix.loc[label, label])
        support = int(matrix.loc[label].sum())
        predicted_count = int(matrix[label].sum())
        metrics[label] = {
            "precision": correct / predicted_count if predicted_count else 0,
            "recall": correct / support if support else 0,
            "f1-score": 2 * correct / (support + predicted_count) if support + predicted_count else 0,
            "support": support,
        }
    metrics["accuracy"] = sum(int(matrix.loc[label, label]) for label in LABELS) / len(gold)
    metrics["macro avg"] = {
        name: sum(metrics[label][name] for label in LABELS) / len(LABELS)
        for name in ("precision", "recall", "f1-score")
    }
    return {"metrics": metrics, "confusion_matrix": matrix.to_numpy().tolist()}


def compare(predictions, left, right):
    matrix = cross_tab(predictions[left], predictions[right])
    size = len(predictions)
    observed = sum(int(matrix.loc[label, label]) for label in LABELS) / size
    expected = float(matrix.sum(axis=0).dot(matrix.sum(axis=1))) / size**2
    return {
        "models": [left, right],
        "agreement": observed,
        "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
    }


def summarize(data, output):
    corpus = pd.read_csv(data / "cleaned_posts.csv").set_index("uri", verify_integrity=True)
    predictions = load_predictions(data, corpus)
    historical = pd.read_csv(data / "historical_posts.csv", usecols=["uri"])
    coverage = pd.read_csv(data / "collection_coverage.csv")
    dates = pd.to_datetime(corpus["created_at"], format="ISO8601", utc=True)

    test = pd.read_csv(data / "bilstm_test.csv").dropna(subset=["text", "label"])
    saved = pd.read_csv(data / "bilstm_test_results.csv")
    if not test[["text", "label"]].reset_index(drop=True).equals(saved[["text", "label"]]):
        raise ValueError("Saved BiLSTM predictions do not match the labeled test rows")
    gold = saved["label"].map(dict(enumerate(LABELS)))
    if gold.isna().any() or not saved["bilstm_sentiment"].isin(LABELS).all():
        raise ValueError("Invalid label in the BiLSTM test results")

    sample = pd.read_csv(data / "human_evaluation_sample.csv").set_index(
        "uri", verify_integrity=True
    )
    if not sample.index.isin(corpus.index).all():
        raise ValueError("Human evaluation sample contains unknown post IDs")
    if not sample["text"].equals(corpus.loc[sample.index, "text"]):
        raise ValueError("Human evaluation sample texts differ from the cleaned corpus")
    human = sample["human_sentiment"].dropna().astype(str).str.strip().str.lower()
    human = human[human.ne("")]
    if not human.isin(LABELS).all():
        raise ValueError("Human labels must be negative, neutral, positive or blank")

    summary = {
        "label_order": LABELS,
        "historical": {
            "raw_rows": len(historical),
            "cleaned_rows": len(corpus),
            "first_post_utc": dates.min().isoformat(),
            "last_post_utc": dates.max().isoformat(),
            "window_start_utc": coverage["window_start"].min(),
            "window_end_exclusive_utc": coverage["window_end"].max(),
            "windows": len(coverage),
            "successful_windows": int(coverage["success"].sum()),
            "possibly_truncated_windows": int(coverage["possibly_truncated"].sum()),
            "class_counts": {
                model: predictions[model].value_counts().reindex(LABELS, fill_value=0).to_dict()
                for model in MODELS
            },
            "pairwise_agreement": [
                compare(predictions, left, right) for left, right in combinations(MODELS, 2)
            ],
            "all_three_agree": int(predictions.nunique(axis=1).eq(1).sum()),
        },
        "bilstm_labeled_test": {
            "rows": len(test),
            "majority_baseline_accuracy": float(gold.value_counts(normalize=True).max()),
            **evaluate(gold, saved["bilstm_sentiment"]),
        },
        "human_evaluation": {
            "sample_size": len(sample),
            "labeled_rows": len(human),
            "models": {
                model: evaluate(human, predictions.loc[human.index, model]) for model in MODELS
            } if len(human) else {},
        },
    }

    hourly = dates.dt.floor("h")
    index = pd.date_range(hourly.min(), hourly.max(), freq="h")
    blocks = []
    for model in MODELS:
        counts = (
            pd.crosstab(hourly, predictions[model])
            .reindex(index=index, columns=LABELS, fill_value=0)
            .rename_axis("hour_utc")
        )
        counts.insert(0, "model", model)
        counts["total"] = counts[LABELS].sum(axis=1)
        denominator = counts["total"].where(counts["total"].ne(0))
        for label in LABELS:
            counts[f"{label}_share"] = counts[label] / denominator
        counts["net_sentiment"] = (counts["positive"] - counts["negative"]) / denominator
        blocks.append(counts)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.concat(blocks).to_csv(output / "historical_hourly_sentiment.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "research")
    args = parser.parse_args()
    summarize(args.data_dir, args.output_dir)
