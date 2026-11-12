"""Coverage-aware sentiment series and trailing rolling means for the historical corpus."""

from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from research_results import LABELS, MODELS, load_predictions

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "data"
    corpus = pd.read_csv(data / "cleaned_posts.csv").set_index("uri", verify_integrity=True)
    predictions = load_predictions(data, corpus)
    created = pd.to_datetime(corpus.created_at, utc=True, format="ISO8601")
    coverage = pd.read_csv(data / "collection_coverage.csv")
    coverage["start"] = pd.to_datetime(coverage.window_start, utc=True)
    coverage["end"] = pd.to_datetime(coverage.window_end, utc=True)
    if coverage.start.duplicated().any() or not coverage.end.sub(coverage.start).eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Coverage must contain unique 15-minute windows")
    output = ROOT / "research/temporal"
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for frequency in ("15min", "h", "D"):
        step = pd.Timedelta(frequency if frequency == "15min" else "1" + frequency)
        index = pd.date_range(coverage.start.min().floor(frequency),
                              coverage.end.max().ceil(frequency) - step, freq=frequency)
        window_group = coverage.groupby(coverage.start.dt.floor(frequency))
        flags = window_group.agg(
            logged_windows=("success", "size"), successful_windows=("success", "sum"),
            cap_hit_windows=("possibly_truncated", "sum"),
        ).reindex(index, fill_value=0)
        flags["expected_windows"] = int(step / pd.Timedelta(minutes=15))
        flags["fully_observed_uncapped"] = (
            flags.successful_windows.eq(flags.expected_windows) & flags.cap_hit_windows.eq(0)
        )
        blocks = []
        for model in MODELS:
            counts = pd.crosstab(created.dt.floor(frequency), predictions[model]).reindex(
                index=index, columns=LABELS, fill_value=0)
            counts["total"] = counts[LABELS].sum(axis=1)
            denominator = counts.total.where(counts.total.gt(0))
            counts["net_sentiment"] = (counts.positive - counts.negative) / denominator
            for label in LABELS:
                counts[label + "_share"] = counts[label] / denominator
            counts = counts.join(flags)
            counts["net_sentiment_uncapped"] = counts.net_sentiment.where(counts.fully_observed_uncapped)
            if frequency == "h":
                for hours in (3, 6):
                    counts[f"trailing_{hours}h_mean"] = counts.net_sentiment.rolling(hours, min_periods=hours).mean()
                    counts[f"trailing_{hours}h_uncapped_mean"] = counts.net_sentiment_uncapped.rolling(
                        hours, min_periods=hours).mean()
                counts["net_change_from_previous_hour"] = counts.net_sentiment.diff()
            counts.insert(0, "model", model)
            blocks.append(counts.rename_axis("period_start_utc"))
        combined = pd.concat(blocks)
        combined.to_csv(output / f"sentiment_{frequency}.csv")
        wide = combined.reset_index().pivot(index="period_start_utc", columns="model", values="net_sentiment")
        safe = flags.fully_observed_uncapped
        comparisons = []
        for left, right in combinations(MODELS, 2):
            pair = wide[[left, right]].dropna()
            uncapped = pair.loc[safe.reindex(pair.index)]
            comparisons.append({
                "models": [left, right], "paired_periods": len(pair),
                "pearson_correlation": float(pair[left].corr(pair[right])),
                "mean_absolute_difference": float(pair[left].sub(pair[right]).abs().mean()),
                "opposite_sign_periods": int(pair[left].mul(pair[right]).lt(0).sum()),
                "uncapped_paired_periods": len(uncapped),
                "uncapped_mean_absolute_difference": float(uncapped[left].sub(uncapped[right]).abs().mean())
                if len(uncapped) else None,
            })
        summaries[frequency] = {
            "periods": len(index), "periods_with_cap_hits": int(flags.cap_hit_windows.gt(0).sum()),
            "fully_observed_uncapped_periods": int(safe.sum()), "comparisons": comparisons,
        }
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
