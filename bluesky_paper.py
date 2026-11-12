"""Validate the hourly Bluesky series and reproduce the paper's results."""
import argparse
from pathlib import Path

import pandas as pd

LABELS = ["negative", "neutral", "positive"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_columns(data, filename, columns):
    path = data / filename
    require(path.is_file(), f"Missing file: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(columns) - set(header))
    require(not missing, f"{filename} is missing {missing}. Found: {list(header)}")
    return pd.read_csv(path, usecols=columns)


def read_posts(data, filename, columns):
    posts = read_columns(data, filename, ["uri", *columns])
    require(posts["uri"].notna().all() and posts["uri"].is_unique,
            f"{filename} contains blank or duplicate post URIs.")
    return posts


def check_labels(labels, name):
    labels = labels.astype("string").str.strip().str.lower()
    invalid = ~labels.isin(LABELS)
    require(not invalid.any(),
            f"{name}: {int(invalid.sum())} blank or invalid labels. Use {', '.join(LABELS)}.")
    return labels


def make_hourly(data):
    posts = read_posts(data, "roberta_results.csv", ["created_at", "roberta_sentiment"])
    posts["hour_utc"] = pd.to_datetime(
        posts["created_at"], utc=True, format="mixed"
    ).dt.floor("h")
    require(posts["hour_utc"].notna().all(), "One or more post timestamps are blank.")
    posts["roberta_sentiment"] = check_labels(posts["roberta_sentiment"], "RoBERTa")
    hours = pd.date_range(posts["hour_utc"].min().normalize(), periods=336, freq="h",
                          name="hour_utc")
    counts = posts.groupby(["hour_utc", "roberta_sentiment"]).size().unstack(fill_value=0)
    require(counts.index.equals(hours), "Expected 336 consecutive UTC hours with sampled posts.")
    hourly = counts.reindex(columns=LABELS, fill_value=0)
    hourly["total_posts"] = hourly[LABELS].sum(axis=1)
    require(hourly["total_posts"].sum() == len(posts), "Hourly counts do not sum to post count.")
    for label in LABELS:
        hourly[f"{label}_share"] = hourly[label] / hourly["total_posts"]
    hourly["net_sentiment_pp"] = 100 * (hourly["positive_share"] - hourly["negative_share"])

    coverage = read_columns(data, "collection_coverage.csv",
                            ["window_start", "window_end", "success", "possibly_truncated"])
    starts = pd.to_datetime(coverage["window_start"], utc=True, format="mixed")
    ends = pd.to_datetime(coverage["window_end"], utc=True, format="mixed")
    quarters = pd.date_range(hours[0], periods=336 * 4, freq="15min")
    require(pd.DatetimeIndex(starts.sort_values()).equals(quarters),
            "Coverage must contain one record for each 15-minute UTC window.")
    require((ends - starts).eq(pd.Timedelta(minutes=15)).all(),
            "Expected 15-minute coverage windows.")
    require(coverage["success"].eq(True).all(), "At least one collection window failed.")
    require(coverage["possibly_truncated"].isin([True, False]).all(),
            "Coverage has missing or invalid truncation flags.")
    hourly["flagged_15min_windows"] = (
        coverage["possibly_truncated"].astype(int).groupby(starts.dt.floor("h")).sum()
    )
    return hourly


def forecast(data, output):
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hourly = make_hourly(data)
    y = hourly["net_sentiment_pp"].asfreq("h")
    horizon = 24
    folds = []
    for start in range(len(y) - 5 * horizon, len(y), horizon):
        train = y.iloc[:start]
        test = y.iloc[start:start + horizon]
        require(test.index[0] - train.index[-1] == pd.Timedelta(hours=1),
                "Unexpected gap or overlap between training and test.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            arima = ARIMA(train, order=(1, 0, 0)).fit()
            sarima = SARIMAX(train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24),
                              trend="c").fit(disp=False, maxiter=250)
            hw = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=24
            ).fit()

        fold = pd.DataFrame({
            "actual": test,
            "Naive": float(train.iloc[-1]),
            "Seasonal naive": train.iloc[-horizon:].to_numpy(),
            "ARIMA(1,0,0)": arima.forecast(horizon),
            "SARIMA(1,0,0)(1,0,0)[24]": sarima.forecast(horizon),
            "Holt-Winters (additive)": hw.forecast(horizon),
        }, index=test.index)
        require(np.isfinite(fold.to_numpy()).all(), "Missing or non-finite forecast values.")
        fold["fold"] = len(folds) + 1
        folds.append(fold)

    predictions = pd.concat(folds)
    require(predictions.index.equals(y.iloc[-120:].index), "Forecasts must cover the last 120 hours.")
    metrics = []
    for model in predictions.columns.drop(["actual", "fold"]):
        error = predictions[model] - predictions["actual"]
        metrics.append({"model": model, "mae_pp": error.abs().mean(),
                        "rmse_pp": np.sqrt(error.pow(2).mean())})
    scores = pd.DataFrame(metrics).sort_values("mae_pp")
    best = scores.iloc[0]
    hourly.to_csv(output / "hourly_sentiment_validated.csv")
    predictions.to_csv(output / "forecast_predictions.csv")
    scores.to_csv(output / "forecast_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    predictions[["actual", best["model"]]].plot(ax=ax, linewidth=1.5)
    ax.set(title="Hourly net sentiment across five 24-hour forecast windows",
           xlabel="Date (UTC)", ylabel="Net sentiment (percentage points)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "forecast_plot.png", dpi=200)
    plt.close(fig)

    flags = int(hourly["flagged_15min_windows"].sum())
    flagged_hours = int(hourly["flagged_15min_windows"].gt(0).sum())
    table = ["| Model | MAE (pp) | RMSE (pp) |", "|---|---:|---:|"]
    table.extend(f"| {r.model} | {r.mae_pp:.2f} | {r.rmse_pp:.2f} |"
                 for r in scores.itertuples(index=False))
    paper = (
        "# Exploratory hourly sentiment forecasting\n\n"
        f"We aggregated {int(hourly['total_posts'].sum()):,} unique sampled Bluesky posts "
        f"into {len(hourly)} consecutive UTC hours from {hourly.index[0]:%Y-%m-%d %H:%M} "
        f"through {hourly.index[-1]:%Y-%m-%d %H:%M}. "
        "Hourly net sentiment was the RoBERTa positive share minus the negative share, "
        "expressed in percentage points. "
        f"The collection log flagged {flags} of 1,344 15-minute windows as possibly truncated "
        f"across {flagged_hours} hours.\n\n"
        "We compared five methods over five expanding-window forecasts, each covering the "
        "next 24 hours. The first forecast used nine days of training data. "
        "All models were scored on the same 120 observations. The seasonal naive method "
        "repeated the previous day's observations at the corresponding hours. "
        "Model orders were fixed for this exploratory comparison.\n\n"
        + "\n".join(table) + "\n\n"
        f"{best['model']} had the lowest observed MAE "
        f"({best['mae_pp']:.2f} percentage points) over these forecast windows. "
        "These results describe the sampled posts in this observation window. "
        "Classifier performance is evaluated separately against independent human labels.\n\n"
        "**Figure caption.** Observed and predicted hourly net sentiment across five "
        f"consecutive forecast windows. The prediction line shows {best['model']}.\n"
    )
    (output / "paper_results.md").write_text(paper, encoding="utf-8")
    print(f"PASS: {len(hourly)} UTC hours, {int(hourly['total_posts'].sum()):,} posts, "
          f"five forecast windows, {len(predictions)} scored hours.")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def normalized_text(path):
    text = pd.read_csv(
        path, usecols=["text"], dtype="string", keep_default_na=False
    )["text"]

    text = (
        text.str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    require(text.ne("").all(), f"{path.name} contains blank post text.")
    return text


def check_sample(data):
    sample_file = "human_evaluation_sample.csv"
    sample = read_posts(data, sample_file, ["human_sentiment"])
    require(len(sample) > 0, "The evaluation sample is empty.")

    blank_labels = sample["human_sentiment"].isna().sum()
    print(f"Evaluation posts: {len(sample)}. Blank labels: {blank_labels}.")

    sample_text = None
    has_overlap = False

    for filename in ["bilstm_train.csv", "bilstm_validation.csv"]:
        path = data / filename
        require(path.is_file(), f"Missing file: {path}")
        columns = pd.read_csv(path, nrows=0).columns
        require("uri" in columns or "text" in columns,
                f"{filename} needs a uri or text column.")

        if "uri" in columns:
            split = read_posts(data, filename, [])
            count = int(sample["uri"].isin(split["uri"]).sum())
            print(f"URI overlap with {filename}: {count}")
            has_overlap = has_overlap or count > 0

        if "text" in columns:
            if sample_text is None:
                sample_text = normalized_text(data / sample_file)

            count = int(sample_text.isin(normalized_text(path)).sum())
            print(f"Text overlap with {filename}: {count}")
            has_overlap = has_overlap or count > 0

    require(not has_overlap, "Use held-out test posts with zero overlap.")
    print("PASS: zero matches with BiLSTM training and validation.")
    return sample

def benchmark(data, output, bilstm_column):
    from sklearn.metrics import classification_report, f1_score

    sample = check_sample(data)
    sample["human_sentiment"] = check_labels(sample["human_sentiment"], "Human evaluation")
    require(set(sample["human_sentiment"]) == set(LABELS),
            "The human evaluation sample needs examples of all three classes.")

    roberta = read_posts(data, "roberta_results.csv", ["roberta_sentiment", "vader_sentiment"])
    bilstm = read_posts(data, "bilstm_results.csv", [bilstm_column])
    matched = sample.merge(roberta, on="uri", how="left", validate="one_to_one")
    matched = matched.merge(bilstm, on="uri", how="left", validate="one_to_one")
    results = []
    for name, column in [("VADER", "vader_sentiment"), ("RoBERTa", "roberta_sentiment"),
                         ("BiLSTM", bilstm_column)]:
        pred = check_labels(matched[column], name)
        true = matched["human_sentiment"]
        detail = classification_report(
            true, pred, labels=LABELS, output_dict=True, zero_division=0
        )
        row = {"model": name, "posts": len(matched),
               "macro_f1": f1_score(true, pred, labels=LABELS, average="macro")}
        for label in LABELS:
            for metric in ("precision", "recall", "f1-score", "support"):
                row[f"{label}_{metric}"] = detail[label][metric]
        results.append(row)
    scores = pd.DataFrame(results)
    scores.to_csv(output / "classifier_benchmark.csv", index=False)
    print(f"PASS: {len(matched)} matched posts and zero URI overlap with BiLSTM train/validation.")
    print(scores[["model", "posts", "macro_f1"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["forecast", "audit", "benchmark"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("research/bluesky_results"))
    parser.add_argument("--bilstm-column", default="bilstm_sentiment")
    args = parser.parse_args()
    try:
        if args.step == "audit":
            check_sample(args.data)
            return
        args.output.mkdir(parents=True, exist_ok=True)
        if args.step == "benchmark":
            benchmark(args.data, args.output, args.bilstm_column)
        else:
            forecast(args.data, args.output)
    except (ValueError, ImportError) as exc:
        parser.exit(1, f"CHECK FAILED: {exc}\n")
    print(f"Results folder: {args.output.resolve()}")


if __name__ == "__main__":
    main()
