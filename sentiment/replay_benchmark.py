"""Controlled arrivals through a bounded queue, inference and project SQLite storage."""
import json
import os
from pathlib import Path
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from contextlib import closing

os.environ.setdefault("USE_TF", "0")
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import db


def main():
    torch.set_num_threads(4)
    sample = pd.read_csv(ROOT / "data/cleaned_posts.csv").sample(n=96, random_state=42)
    texts = sample.text.tolist()
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True).eval()
    vader = SentimentIntensityAnalyzer()

    def classify(method, text):
        if method == "VADER":
            result = vader.polarity_scores(text)
            label = "Positive" if result["compound"] >= .05 else "Negative" if result["compound"] <= -.05 else "Neutral"
            return label, abs(result["compound"])
        with torch.inference_mode():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            probabilities = model(**inputs).logits.softmax(dim=1)[0]
            index = int(probabilities.argmax())
            return model.config.id2label[index].title(), float(probabilities[index])

    rows = []
    summaries = []
    output = ROOT / "research/replay"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentiment_replay_") as temporary:
        db.DATA_DIR = Path(temporary)
        for method in ["VADER", "RoBERTa"]:
            for _ in range(3):
                classify(method, texts[0])
            for rate in [10, 40, 80]:
                for repeat in range(3):
                    db.DB_PATH = db.DATA_DIR / f"{method}_{rate}_{repeat}.db"
                    db.init_db()
                    pending = queue.Queue(maxsize=16)
                    records = []
                    dropped = []
                    depths = []
                    run_start = time.perf_counter()

                    def consume():
                        while True:
                            item = pending.get()
                            try:
                                if item is None:
                                    return
                                index, arrival, scheduled = item
                                start = time.perf_counter()
                                error = ""
                                inference_end = start
                                try:
                                    label, confidence = classify(method, texts[index])
                                    inference_end = time.perf_counter()
                                    db.insert_record(datetime.now(timezone.utc).isoformat(),
                                                     texts[index], label, confidence)
                                except Exception as exc:
                                    error = repr(exc)
                                finish = time.perf_counter()
                                records.append({
                                    "model": method, "offered_rate": rate, "repeat": repeat, "sample_index": index,
                                    "arrival_lateness_ms": (arrival - scheduled) * 1000,
                                    "queue_ms": (start - arrival) * 1000,
                                    "inference_ms": (inference_end - start) * 1000,
                                    "storage_ms": (finish - inference_end) * 1000,
                                    "arrival_to_commit_ms": (finish - arrival) * 1000,
                                    "error": error,
                                })
                            finally:
                                pending.task_done()

                    worker = threading.Thread(target=consume, daemon=True)
                    worker.start()
                    for index in range(len(texts)):
                        scheduled = run_start + index / rate
                        time.sleep(max(0, scheduled - time.perf_counter()))
                        arrival = time.perf_counter()
                        try:
                            pending.put_nowait((index, arrival, scheduled))
                            depths.append(pending.qsize())
                        except queue.Full:
                            dropped.append(index)
                    pending.put(None)
                    pending.join()
                    worker.join()
                    elapsed = time.perf_counter() - run_start
                    successful = [row for row in records if not row["error"]]
                    latencies = [row["arrival_to_commit_ms"] for row in successful]
                    with closing(sqlite3.connect(db.DB_PATH)) as connection:
                        stored = connection.execute("SELECT count(*) FROM sentiment_records").fetchone()[0]
                    assert stored == len(successful)
                    assert len(records) + len(dropped) == len(texts)
                    summaries.append({
                        "model": method, "offered_rate": rate, "repeat": repeat,
                        "offered": len(texts), "stored": stored, "dropped": len(dropped),
                        "errors": len(records) - stored, "max_queue_depth": max(depths, default=0),
                        "completed_posts_per_second_including_drain": stored / elapsed,
                        "p50_ms": float(np.quantile(latencies, .5)) if latencies else None,
                        "p95_ms": float(np.quantile(latencies, .95)) if latencies else None,
                        "p99_ms": float(np.quantile(latencies, .99)) if latencies else None,
                        "over_1000ms": sum(value > 1000 for value in latencies),
                        "dropped_sample_indices": dropped,
                    })
                    rows.extend(records)
                    pd.DataFrame(rows).to_csv(output / "post_timings.csv", index=False)
                    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
                    print(f"{method} rate={rate} repeat={repeat} stored={stored} dropped={len(dropped)}", flush=True)
    pd.DataFrame(rows).to_csv(output / "post_timings.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    protocol = {
        "sample_uris": sample.uri.tolist(), "sample_seed": 42, "posts_per_run": 96,
        "queue_capacity": 16, "batch_size": 1, "threads": 4, "repeats": 3,
        "offered_rates": [10, 40, 80], "device": "CPU",
        "roberta_revision": model.config._commit_hash,
        "scope": "Controlled replay of cleaned posts; actual project db.insert_record commits to isolated temporary databases",
        "excluded": "Network delivery, filtering, reconnects, live collector event-loop scheduling and backup overhead",
        "vader_confidence_field": "Absolute compound used only as a storage-compatible value, not a probability",
        "deadline_threshold_ms": 1000,
        "limitation": "Short fixed workload; order is VADER then RoBERTa; not a long-run capacity or production availability test",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
