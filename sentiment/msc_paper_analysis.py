"""M.Sc. paper diagnostics that can be computed from saved predictions.

Produces four evidence tables:
1. How labeled-test errors shift mean net sentiment.
2. Heuristic disagreement strata and short examples.
3. Historical robustness under duplicate, coverage and author-cap filters.
4. An audit of the unfinished Bluesky human sample.

No new labels are created and no Bluesky accuracy is estimated.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/publication/evidence"
LABELS = ["negative", "neutral", "positive"]
SCORE = {"negative": -1, "neutral": 0, "positive": 1, 0: -1, 1: 0, 2: 1}
MODELS = ["vader", "bilstm_saved", "roberta"]
NAMES = {"vader": "VADER", "bilstm_saved": "BiLSTM", "roberta": "RoBERTa", "linear": "TF-IDF + LR"}


def score_of(values):
    return pd.Series(values).map(SCORE).astype(float)


def strata(text):
    words = text.fillna("").str.split().str.len()
    body = text.fillna("")
    return {
        "short_post": words.le(6),
        "contains_url": body.str.contains(r"https?://|www\.", case=False, regex=True),
        "sarcasm_cue": body.str.contains(
            r"(?:/s\b|\bsarcas|\byeah right\b|\bas if\b|\boh sure\b|\bsure jan\b)",
            case=False, regex=True),
        "news_cue": body.str.contains(
            r"\b(?:breaking|according to|reports that|announced|headline|press release)\b",
            case=False, regex=True),
        "ai_term": body.str.contains(
            r"\b(?:AI|ChatGPT|GPT|LLM|artificial intelligence|machine learning)\b",
            regex=True),
    }


EXPLICIT = re.compile(r"\b(?:penis|naked|porn|fuck|sex)\b", re.I)


def suitable_excerpt(text):
    return EXPLICIT.search(str(text)) is None


def clip(text, limit=180):
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def labeled_error_analysis():
    test = pd.read_csv(ROOT / "data/bilstm_test.csv")
    frames = {}
    for model in ["vader", "linear", "bilstm_saved", "roberta"]:
        frame = pd.read_csv(ROOT / f"research/benchmark/{model}_predictions.csv")
        if not frame.row_id.equals(pd.Series(test.index, name="row_id")):
            raise ValueError(f"{model} predictions are not aligned to the labeled test rows")
        if not frame.label.equals(test.label):
            raise ValueError(f"{model} gold labels differ from the labeled test file")
        frames[model] = frame.prediction.to_numpy()
    gold = test.label.to_numpy()
    gold_score = score_of(gold).to_numpy()
    text = test.text
    flags = strata(text)
    rows = []
    for model, prediction in frames.items():
        pred_score = score_of(prediction).to_numpy()
        signed = pred_score - gold_score
        rows.append({
            "scope": "all_labeled_test", "model": NAMES[model], "posts": len(test),
            "accuracy": float(np.mean(prediction == gold)),
            "mean_net_error": float(signed.mean()),
            "positive_overcall_share": float(np.mean((prediction == 2) & (gold != 2))),
        })
        for name, mask in flags.items():
            chosen = mask.to_numpy()
            rows.append({
                "scope": name, "model": NAMES[model], "posts": int(chosen.sum()),
                "accuracy": float(np.mean(prediction[chosen] == gold[chosen])),
                "mean_net_error": float(signed[chosen].mean()),
                "positive_overcall_share": float(np.mean((prediction[chosen] == 2) & (gold[chosen] != 2))),
            })
    pd.DataFrame(rows).to_csv(OUT / "labeled_error_strata.csv", index=False)

    contributions = []
    for model, prediction in frames.items():
        pred_name = pd.Series(prediction).map({0: "negative", 1: "neutral", 2: "positive"})
        gold_name = pd.Series(gold).map({0: "negative", 1: "neutral", 2: "positive"})
        pred_score = score_of(prediction).to_numpy()
        for true_label in LABELS:
            for pred_label in LABELS:
                chosen = (gold_name == true_label) & (pred_name == pred_label)
                count = int(chosen.sum())
                delta = SCORE[pred_label] - SCORE[true_label]
                contributions.append({
                    "model": NAMES[model], "true_label": true_label, "predicted_label": pred_label,
                    "posts": count, "score_delta": delta,
                    "contribution_to_mean_net_error": count * delta / len(test),
                })
    contrib = pd.DataFrame(contributions)
    contrib.to_csv(OUT / "labeled_error_contribution.csv", index=False)

    pred_frame = pd.DataFrame({NAMES[m]: frames[m] for m in MODELS}, index=test.index)
    pred_frame["gold"] = gold
    pred_frame["text"] = text
    pred_frame["words"] = text.fillna("").str.split().str.len()
    examples = []
    targets = {
        "positive_overcall": (pred_frame.gold != 2) & (pred_frame.VADER == 2) & (pred_frame.RoBERTa != 2),
        "short_post": flags["short_post"] & (pred_frame.VADER != pred_frame.RoBERTa),
        "contains_url": flags["contains_url"] & (pred_frame.VADER != pred_frame.RoBERTa),
        "sarcasm_cue": flags["sarcasm_cue"] & (pred_frame.VADER != pred_frame.gold),
        "three_way_disagreement": pred_frame[["VADER", "BiLSTM", "RoBERTa"]].nunique(axis=1).eq(3),
    }
    for kind, mask in targets.items():
        chosen = pred_frame[mask & pred_frame.text.map(suitable_excerpt)].sort_index().head(2)
        for row_id, row in chosen.iterrows():
            examples.append({
                "source": "labeled_test", "kind": kind, "row_id": int(row_id),
                "gold": LABELS[int(row.gold)],
                "vader": LABELS[int(row.VADER)], "bilstm": LABELS[int(row.BiLSTM)],
                "roberta": LABELS[int(row.RoBERTa)], "text": clip(row.text),
            })
    return pred_frame, examples


def historical_analysis(labeled_examples):
    from research_results import load_predictions
    corpus = pd.read_csv(ROOT / "data/cleaned_posts.csv").set_index("uri", verify_integrity=True)
    labels = load_predictions(ROOT / "data", corpus)
    scores = labels.apply(lambda column: column.map(SCORE)).astype(float)
    text = corpus.text
    dates = pd.to_datetime(corpus.created_at, format="ISO8601", utc=True)
    author = corpus.index.to_series().str.extract(r"^at://([^/]+)/", expand=False)
    if author.isna().any():
        raise ValueError("Every historical URI must contain an author identifier")
    hour = dates.dt.floor("h")
    normalized = text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    order = pd.DataFrame({"time": dates, "post_uri": corpus.index}, index=corpus.index).sort_values(["time", "post_uri"])
    unique_text = pd.Series(corpus.index.isin(normalized.loc[order.index].drop_duplicates().index), index=corpus.index)
    coverage = pd.read_csv(ROOT / "research/temporal/sentiment_h.csv")
    uncapped_hours = pd.to_datetime(
        coverage[(coverage.model == "VADER") & coverage.fully_observed_uncapped.astype(bool)].period_start_utc,
        utc=True)
    uncapped = hour.isin(uncapped_hours)
    author_rank = order.groupby(author.loc[order.index]).cumcount()
    cap1 = pd.Series(author_rank.eq(0), index=order.index).reindex(corpus.index)
    cap5 = pd.Series(author_rank.lt(5), index=order.index).reindex(corpus.index)
    masks = {
        "all_posts": pd.Series(True, index=corpus.index),
        "unique_normalized_text": unique_text,
        "uncapped_hours": uncapped,
        "one_post_per_author": cap1,
        "at_most_five_posts_per_author": cap5,
        "unique_uncapped_and_capped": unique_text & uncapped & cap5,
    }
    rows = []
    for name, mask in masks.items():
        part = scores.loc[mask]
        for model in part:
            rows.append({"analysis": name, "model": model, "units": int(mask.sum()),
                         "net_sentiment": float(part[model].mean())})
        gap = float((part.VADER - part.RoBERTa).mean())
        rows.append({"analysis": name, "model": "VADER_minus_RoBERTa", "units": int(mask.sum()),
                     "net_sentiment": gap})
    pd.DataFrame(rows).to_csv(OUT / "robustness_extended.csv", index=False)

    flags = strata(text)
    strata_rows = []
    disagree = labels[["VADER", "RoBERTa", "BiLSTM"]].nunique(axis=1).gt(1)
    for name, mask in {**flags, "all_posts": pd.Series(True, index=corpus.index)}.items():
        chosen = mask if name != "all_posts" else mask
        part = scores.loc[chosen]
        strata_rows.append({
            "stratum": name, "posts": int(chosen.sum()),
            "share_of_corpus": float(chosen.mean()),
            "disagreement_rate": float(disagree[chosen].mean()),
            "vader_net": float(part.VADER.mean()),
            "roberta_net": float(part.RoBERTa.mean()),
            "bilstm_net": float(part.BiLSTM.mean()),
            "vader_minus_roberta": float((part.VADER - part.RoBERTa).mean()),
        })
    pd.DataFrame(strata_rows).to_csv(OUT / "bluesky_disagreement_strata.csv", index=False)

    examples = list(labeled_examples)
    view = labels.copy()
    view["text"] = text
    targets = {
        "vader_positive_roberta_negative": (view.VADER == "positive") & (view.RoBERTa == "negative"),
        "short_post": flags["short_post"] & disagree,
        "contains_url": flags["contains_url"] & (view.VADER != view.RoBERTa),
        "sarcasm_cue": flags["sarcasm_cue"] & (view.VADER != view.RoBERTa),
        "news_cue": flags["news_cue"] & (view.VADER != view.RoBERTa),
        "ai_term": flags["ai_term"] & (view.VADER == "positive") & (view.RoBERTa == "negative"),
        "three_way_disagreement": view[["VADER", "RoBERTa", "BiLSTM"]].nunique(axis=1).eq(3),
    }
    for kind, mask in targets.items():
        chosen = view[mask & view.text.map(suitable_excerpt)].sort_index().head(1)
        for uri, row in chosen.iterrows():
            examples.append({
                "source": "bluesky_model_output", "kind": kind, "row_id": uri,
                "gold": "", "vader": row.VADER, "bilstm": row.BiLSTM,
                "roberta": row.RoBERTa, "text": clip(row.text),
            })
    pd.DataFrame(examples).to_csv(OUT / "error_examples.csv", index=False)

    sample = pd.read_csv(ROOT / "data/human_evaluation_sample.csv", dtype=str, keep_default_na=False)
    filled = sample.human_sentiment.str.strip()
    filled = filled[filled.ne("")]
    audit = {
        "sample_size": int(len(sample)),
        "labeled_posts": int(len(filled)),
        "unlabeled_posts": int(len(sample) - len(filled)),
        "label_counts": filled.str.lower().value_counts().to_dict(),
        "status": "incomplete",
        "claim_boundary": (
            "Bluesky aggregate trends in this study are classifier outputs. "
            "They are not verified measurements of public opinion, because the "
            "target corpus has no completed independent human ground truth."
        ),
        "minimum_useful_sample": "100 to 200 independently labeled posts, double-annotated on a subset, with texts withheld from the annotator’s view of model outputs.",
    }
    if not set(filled.str.lower()).issubset(set(LABELS) | {""}):
        audit["warning"] = "Unexpected human label values are present; they are not used as accuracy."
    (OUT / "validation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, examples = labeled_error_analysis()
    audit = historical_analysis(examples)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
