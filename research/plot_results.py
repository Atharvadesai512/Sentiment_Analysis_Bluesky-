"""Create research figures from generated benchmark and temporal results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

hourly = pd.read_csv(ROOT / "temporal/sentiment_h.csv")
hourly["period_start_utc"] = pd.to_datetime(hourly.period_start_utc, utc=True)
colors = {"VADER": "#9c541c", "RoBERTa": "#275d9f", "BiLSTM": "#555555"}
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, layout="constrained")
for model, rows in hourly.groupby("model", sort=False):
    axes[0].plot(rows.period_start_utc, rows.trailing_6h_mean, label=model, color=colors[model])
axes[0].axhline(0, color="black", linewidth=.6)
axes[0].set(title="Historical sentiment by model", ylabel="Net sentiment\n6-hour trailing mean")
axes[0].legend(ncol=3)
coverage = hourly[hourly.model.eq("VADER")]
axes[1].bar(coverage.period_start_utc, coverage.cap_hit_windows, width=1/24, color="#777777")
axes[1].set(ylabel="Cap-hit requests per hour", xlabel="Post creation date (UTC)", ylim=(0, 4.5))
axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.suptitle("AI search corpus · 19 October–1 November 2026 · 127,741 posts", fontsize=11)
fig.savefig(FIGURES / "sentiment_trends.png", dpi=220)
plt.close(fig)

records = []
names = {"vader": "VADER", "linear": "TF-IDF + LR", "bilstm_saved": "BiLSTM (saved)", "roberta": "RoBERTa"}
for key, name in names.items():
    result = json.loads((ROOT / f"benchmark/{key}_metrics.json").read_text())
    records.append({"Model": name, "Accuracy": result["report"]["accuracy"],
                    "Macro F1": result["report"]["macro avg"]["f1-score"],
                    "Macro recall": result["report"]["macro avg"]["recall"]})
frame = pd.DataFrame(records).set_index("Model")
frame.to_csv(ROOT / "benchmark/comparison.csv")
fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
frame[["Accuracy", "Macro F1"]].plot.bar(ax=ax, color=["#275d9f", "#888888"], rot=0)
ax.set(ylim=(0, 1), ylabel="Score (0–1)", xlabel="Model",
       title="Same labeled Twitter test split · 12,284 examples")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)
fig.savefig(FIGURES / "benchmark_comparison.png", dpi=220)
plt.close(fig)
