"""Create the Step 1 quantification figure from saved, text-free evidence."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/publication/manuscript"
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(ROOT / "research/publication/evidence/quantification_summary.csv")
primary = summary[(summary.direction == 0) & (summary.bag_size == 500)].copy()
order = ["vader", "linear", "bilstm_saved", "roberta"]
names = ["VADER", "TF-IDF + LR", "BiLSTM", "RoBERTa"]
cc = primary[primary.method == "CC"].set_index("model").loc[order]
acc = primary[primary.method == "ACC"].set_index("model").loc[order]

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=True)
x = range(len(order))
width = 0.35
for axis, key, ylabel, title in [
    (axes[0], "prevalence_mae", "Mean absolute error", "Class-prevalence estimation"),
    (axes[1], "strict_sign_error_rate", "Strict sign-error rate", "Aggregate net-sentiment direction"),
]:
    axis.bar([i - width / 2 for i in x], cc[key], width, color="#8B8B8B", label="Classify and count")
    axis.bar([i + width / 2 for i in x], acc[key], width, color="#2166AC", label="Adjusted classify and count")
    axis.set_xticks(list(x), names, rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title, weight="bold", fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=.6)
    axis.set_axisbelow(True)
axes[1].set_ylim(0, .26)
axes[0].legend(loc="upper left", frameon=False, fontsize=7)
fig.savefig(OUT / "quantification_diagnostic.png", dpi=300, bbox_inches="tight")

robust = pd.read_csv(ROOT / "research/publication/evidence/robustness_extended.csv")
order_variants = [
    "all_posts", "unique_normalized_text", "uncapped_hours",
    "at_most_five_posts_per_author", "unique_uncapped_and_capped",
]
labels = ["All posts", "Unique text", "Uncapped hours", "≤5 posts/author", "Combined filter"]
part = robust[robust.model.isin(["VADER", "RoBERTa"]) & robust.analysis.isin(order_variants)]
fig, axis = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
x = range(len(order_variants))
width = 0.36
for offset, model, color in [(-width / 2, "VADER", "#B2182B"), (width / 2, "RoBERTa", "#2166AC")]:
    values = part[part.model == model].set_index("analysis").loc[order_variants].net_sentiment
    axis.bar([i + offset for i in x], values, width, color=color, label=model)
axis.axhline(0, color="#444444", linewidth=.8)
axis.set_xticks(list(x), labels, rotation=12, ha="right")
axis.set_ylabel("Mean net sentiment")
axis.set_title("Historical trend under processing checks", weight="bold", fontsize=9)
axis.spines[["top", "right"]].set_visible(False)
axis.grid(axis="y", color="#DDDDDD", linewidth=.6)
axis.set_axisbelow(True)
axis.legend(frameon=False, fontsize=8)
fig.savefig(OUT / "robustness_diagnostic.png", dpi=300, bbox_inches="tight")
