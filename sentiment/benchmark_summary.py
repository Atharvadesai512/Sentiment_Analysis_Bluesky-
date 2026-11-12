"""Summarize benchmark scores with paired, row-bootstrap uncertainty."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def scores(gold, predicted):
    matrix = np.bincount(3 * gold + predicted, minlength=9).reshape(3, 3)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * matrix.diagonal(), denominator,
                   out=np.zeros(3, dtype=float), where=denominator != 0)
    return np.array([np.trace(matrix) / len(gold), f1.mean()])


def main():
    folder = ROOT / "research/benchmark"
    names = ["vader", "linear", "bilstm_saved", "roberta"]
    frames = {name: pd.read_csv(folder / f"{name}_predictions.csv") for name in names}
    reference = frames[names[0]][["row_id", "label"]]
    if not all(frame[["row_id", "label"]].equals(reference) for frame in frames.values()):
        raise ValueError("Benchmark rows or labels differ across models")
    gold = reference.label.to_numpy(dtype=int)
    predictions = {name: frame.prediction.to_numpy(dtype=int) for name, frame in frames.items()}
    rng = np.random.default_rng(42)
    iterations = 1000
    bootstrap = np.empty((iterations, len(names), 2))
    for iteration in range(iterations):
        sample = rng.integers(0, len(gold), size=len(gold))
        for model_index, name in enumerate(names):
            bootstrap[iteration, model_index] = scores(gold[sample], predictions[name][sample])
    result = {
        "method": "1000 paired row-bootstrap resamples; percentile 95% intervals; seed 42",
        "limitation": "Assumes independent rows; author and temporal clustering are not represented",
        "models": {},
        "paired_differences": [],
    }
    for model_index, name in enumerate(names):
        values = scores(gold, predictions[name])
        bounds = np.quantile(bootstrap[:, model_index], [.025, .975], axis=0)
        result["models"][name] = {
            metric: {"score": float(values[i]), "lower": float(bounds[0, i]), "upper": float(bounds[1, i])}
            for i, metric in enumerate(["accuracy", "macro_f1"])
        }
    for left, right in combinations(range(len(names)), 2):
        delta = bootstrap[:, left, :] - bootstrap[:, right, :]
        bounds = np.quantile(delta, [.025, .975], axis=0)
        result["paired_differences"].append({
            "left_minus_right": [names[left], names[right]],
            "accuracy_95_interval": bounds[:, 0].tolist(),
            "macro_f1_95_interval": bounds[:, 1].tolist(),
        })
    (folder / "uncertainty.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    splits = {}
    for split in ["train", "validation", "test"]:
        frame = pd.read_csv(ROOT / f"data/bilstm_{split}.csv")
        normalized = frame.text.str.lower().str.split().str.join(" ")
        splits[split] = set(normalized)
    overlaps = {f"{left}_{right}": len(splits[left] & splits[right])
                for left, right in combinations(splits, 2)}
    (folder / "normalized_text_overlap.json").write_text(json.dumps(overlaps, indent=2), encoding="utf-8")
    print(json.dumps({"models": result["models"], "split_overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()
