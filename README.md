
# Sentiment Classification Under Processing Constraints

This repository contains a reproducible M.Sc. research project on three-class
sentiment analysis for social-media text. It compares VADER, a TF-IDF logistic
regression baseline, an existing BiLSTM and a Twitter RoBERTa checkpoint on the
same labeled TweetEval test split. It also studies how model choice changes
aggregate outputs over a historical Bluesky search corpus and measures local
processing behaviour through inference and queue-to-SQLite replay experiments.

## Main findings

- RoBERTa achieved 72.18% accuracy and 0.724 macro F1 on 12,284 labeled
  TweetEval test posts. The result concerns this benchmark, not Bluesky.
- The 127,741-post historical corpus produces substantially different aggregate
  sentiment outputs across the three classifiers.
- A bounded replay with real SQLite inserts shows that completed throughput and
  dropped records depend on the model and offered arrival rate.

## Data and privacy

Raw post text, databases, model files and local virtual environments are
intentionally excluded from version control. The repository keeps aggregate
results, row-level benchmark labels/predictions without text, source hashes,
protocols and figures so that claims can be audited without publishing the
historical corpus. Follow relevant platform terms, data licences and research
ethics requirements before obtaining or sharing data.

## Publication materials

The current dissertation, manuscript, presentation and assessment material are
in [`research/submission`](research/submission). The manuscript is a draft and
has not been submitted to or accepted by any venue.

## Reproduce the research results

From the project directory, run the analysis against the saved predictions:

```powershell
.\venv\Scripts\python.exe sentiment\research_results.py
```

The analysis requires pandas 2.2 and uses the same historical Bluesky posts for
VADER, RoBERTa, and BiLSTM. It checks that the saved predictions align with the
cleaned corpus by post URI and text. No model downloads are needed.

- `research/metrics.json` contains collection coverage, class counts, pairwise
  agreement, and the BiLSTM scores and confusion matrix on its labeled test set.
- `research/historical_hourly_sentiment.csv` contains counts, shares, and net
  sentiment by UTC creation hour and classifier for the historical AI search.
  This is separate from the live-monitor exports in `data/`.
- `research/submission/Dissertation.docx` presents the final study and results.
- `research/Submission Package.zip` contains the documents, presentation, source and evidence.

The current study uses the existing labeled Twitter benchmark for accuracy,
and the historical Bluesky corpus for agreement and temporal analysis.
Manual annotation is optional. Bluesky accuracy remains unmeasured.
The manuscript states this boundary directly: Bluesky trends are model
outputs, not verified public opinion. The human sample in
`data/human_evaluation_sample.csv` has 300 posts and only 2 completed labels.

The M.Sc. diagnostics below recompute how labeled errors shift mean net
sentiment, list disagreement examples, and repeat the historical trend after
duplicate-text, uncapped-hour and author-cap filters:

```powershell
.\venv\Scripts\python.exe sentiment\msc_paper_analysis.py
.\venv\Scripts\python.exe research\publication\plot_step1.py
.\venv\Scripts\python.exe research\publication\build_step1_paper.py
```

Evidence is written to `research/publication/evidence/`. The manuscript is
`research/publication/manuscript/Step 1 Quantification Manuscript.docx`.

### Benchmark and temporal experiments

Run from the project directory using Python 3.12. Additional analysis packages
are listed in `requirements-research.txt`.

```powershell
.\venv\Scripts\python.exe sentiment\benchmark.py --model vader
.\venv\Scripts\python.exe sentiment\benchmark.py --model linear
.\venv\Scripts\python.exe sentiment\benchmark.py --model bilstm_saved
.\venv\Scripts\python.exe sentiment\benchmark.py --model roberta
.\venv\Scripts\python.exe sentiment\benchmark_summary.py
.\venv\Scripts\python.exe sentiment\temporal_analysis.py
.\venv\Scripts\python.exe sentiment\runtime_benchmark.py
.\venv\Scripts\python.exe sentiment\replay_benchmark.py
.\venv\Scripts\python.exe research\plot_results.py
```

Run the runtime benchmark after other inference jobs finish. It measures local
tokenization and inference on a fixed sample, excluding collection, queueing,
and storage. Batch latency is distinct from single-post latency. The BiLSTM
comparison uses saved predictions because Windows application control blocks
TensorFlow import on this host.

Outputs are under `research/benchmark`, `research/temporal`,
`research/runtime`, `research/replay`, and `research/figures`. The uncertainty calculation uses
paired row bootstrap samples and assumes independent test rows.

Controlled replay measures arrivals through a bounded queue to actual SQLite commits. It excludes network delivery and collector orchestration. Run it separately from other inference jobs.

Pipeline: **Bluesky posts → filtering → RoBERTa sentiment classification →
timestamped SQLite storage → hourly/daily aggregation → time-series dataset**

The research extension adds trailing moving averages, changes in hourly net
sentiment, and comparisons across aggregation intervals. Forecasting and
anomaly detection remain future work.

## Project structure

```
sentiment-monitor/
├── config.py             # ALL settings in one place (thresholds, paths, patterns)
├── stream_monitor.py     # real-time pipeline: connect, filter, classify, store, back up
├── filters.py            # bot / duplicate / per-author-rate filters
├── db.py                 # SQLite storage + backup — the only file that touches sqlite3
├── aggregate.py          # SQLite -> hourly/daily CSVs (gap-free) + raw export
├── backup.py             # manual one-off database backup
├── scripts/
│   ├── run_monitor.ps1               # keeps stream_monitor.py running, auto-restarts on crash
│   ├── start_monitor_background.ps1  # launches run_monitor.ps1 hidden/detached
│   ├── stop_monitor.ps1              # stops the background monitor
│   ├── setup_scheduled_task.ps1      # boot-time Scheduled Task (survives reboot/logout)
│   └── remove_scheduled_task.ps1     # removes that task
├── data/                 # created automatically
│   ├── sentiment.db      #   every classified post
│   ├── backups/          #   automatic database copies
│   └── *.csv             #   outputs of aggregate.py
├── logs/                 # created when running in the background
├── requirements.txt
└── README.md
```

## 1. Set up

```powershell
cd sentiment-monitor
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`torch` and `transformers` are large; the first install and the first model
download can take several minutes.

## 2. Run the collector (foreground, for testing)

```powershell
python stream_monitor.py
```

`Ctrl+C` stops it; everything already saved stays in the database. Every
5 minutes it logs a summary line such as:

```
STATS (last 300s): classify_failed=1, received=1820, skipped_bot_pattern=44, skipped_bot_rate=12,
skipped_duplicate=57, skipped_junk=31, skipped_non_english=1105, stored=570
```

This shows exactly how many posts each filter removed — useful for tuning.

## 3. Run it in the background long-term

```powershell
.\scripts\start_monitor_background.ps1      # start (hidden, survives closing the window)
Get-Content -Wait .\logs\stream_monitor.log # watch it (Ctrl+C stops watching only)
.\scripts\stop_monitor.ps1                  # stop it
```

It restarts `stream_monitor.py` automatically if the process ever exits.
If PowerShell blocks the scripts, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Surviving reboots and logout (Scheduled Task)

From an **Administrator** PowerShell:

```powershell
cd sentiment-monitor\scripts
.\setup_scheduled_task.ps1
Start-ScheduledTask -TaskName "SentimentMonitor"   # start now, no reboot needed
```

The task starts at every boot (before any login), keeps running after
logout, restarts itself on failure, and has Task Scheduler's default 72-hour
limit disabled. Remove it with `.\remove_scheduled_task.ps1` (then also
`.\stop_monitor.ps1` if it is running). The model cache is pinned to
`.cache\huggingface` inside the project so it behaves the same whether you
or the `SYSTEM` account start it.

## 4. Generate the time-series dataset

Run any time, even while collecting:

```powershell
python aggregate.py
```

Outputs in `data/`:

| file | contents |
|---|---|
| `hourly_sentiment.csv` | one row per hour — the time-series dataset |
| `daily_sentiment.csv` | one row per day — the time-series dataset |
| `all_classified_posts.csv` | every individual post (timestamp, text, sentiment, confidence), newest first |

Columns of the hourly/daily files:

| column | meaning |
|---|---|
| `Time` | start of the hour/day (UTC) |
| `Positive`, `Negative`, `Neutral` | post counts |
| `Total` | sum of the three |
| `Positive_Share`, `Negative_Share`, `Neutral_Share` | count ÷ Total (0–1) |
| `Net_Sentiment` | (Positive − Negative) ÷ Total; −1 all negative, +1 all positive |

**Gap-free index.** Every hour/day between the first and last record gets a
row. Periods with no posts show `0` counts and **blank** shares (a share of
zero posts is undefined — blank means "no data", not "0% positive"). Time-series
models need evenly spaced time steps, so gaps must be visible, not missing.
A run of zero rows usually means the collector was off.

**Why shares and Net_Sentiment.** Raw counts change with how many people are
posting (busy at midday, quiet at 3 a.m.), not only with how they feel. The
share columns remove that volume effect and are usually the better series for
modeling *sentiment* change.

**Last row is partial.** The final row is the period still in progress. Drop it
(or wait for the period to end) before modeling.

## Filtering

Applied to every post before classification, cheapest first. All thresholds
are in `config.py`.

1. **Junk** — after removing URLs, fewer than `MIN_MEANINGFUL_CHARS` characters remain.
2. **Known bot templates** — `BOT_TEXT_PATTERNS`: flight (ACARS) bots, ship
   alerts, "now playing" radio posts, weather-warning feeds, earthquake feeds,
   "Frame N of M" bots. Add your own regex when you spot a new bot in the log.
3. **English only** — a post's declared `langs` is trusted; if missing,
   `langdetect` must be ≥ `LANGDETECT_MIN_CONFIDENCE` sure it is English.
   Strict on purpose: very short posts ("gm", "lol") can't be identified by any
   detector and are dropped.
4. **Per-author posting rate** — an account (Bluesky DID) posting more than
   `MAX_POSTS_PER_AUTHOR_PER_HOUR` (default 20) posts in an hour is treated as
   a bot; its extra posts that hour are skipped.
5. **Duplicate text** — a post whose normalized text (lowercase, with URLs,
   @mentions, #hashtags, digits and punctuation removed) was already seen in
   the last `DEDUP_WINDOW_HOURS` (default 24) is skipped. Posts whose
   normalized text is shorter than `DEDUP_MIN_CHARS` (default 25) are **never**
   treated as duplicates, because short replies ("thank you") are written
   independently by many real people.

After classification: **confidence gate** — posts below `MIN_CONFIDENCE_TO_STORE`
(default 0.0 = off) are not saved. Raise it (e.g. 0.6) to trade volume for
average reliability.

Why bots matter: they post huge volumes of near-identical, mostly-Neutral text
that inflates the Neutral count and hides real shifts in human sentiment.

**Limits to be aware of.** Pattern and rate rules catch the common bots, not
all of them; duplicate memory resets when the collector restarts; and a few
real people will occasionally be filtered. Check the STATS lines and
`all_classified_posts.csv` periodically and adjust.

## Automatic backups

While `stream_monitor.py` runs it copies the database to `data/backups/`
every `BACKUP_INTERVAL_HOURS` (default 6) and keeps the newest
`BACKUP_KEEP_COUNT` (default 28 = 7 days). It uses SQLite's online-backup
API, so it is safe while data is being written, and writes to a temporary file
first so an interrupted backup never leaves a corrupt "good" file. A failed
backup is logged and never stops collection.

Manual backup at any time: `python backup.py`

**Restore:** stop the collector, then copy the chosen file from `data/backups/`
over `data/sentiment.db`, and start the collector again.

Note: backups protect against file corruption or accidental deletion, not
against the whole disk failing — copy `data/backups/` somewhere else
occasionally for that.

## Interpreting accuracy

Use the locally generated benchmark metrics for quantitative claims. Scores
on the labeled Twitter test set do not establish accuracy on Bluesky.
Confidence thresholds change which posts are retained and may bias aggregate
sentiment. Classification errors can be systematic and may persist after
aggregation.

## Where the data lives, and what one record looks like

Table `sentiment_records` in `data/sentiment.db`:

| column | type | meaning |
|---|---|---|
| id | integer | auto-incrementing row id |
| timestamp | text | ISO 8601 UTC, when the post was processed |
| text | text | the original post text |
| sentiment | text | `Positive`, `Negative` or `Neutral` |
| confidence_score | real | model confidence, 0–1 |

Example: `(1, "2026-08-20T09:02:03+00:00", "I really love this place", "Positive", 0.9421)`

## How this dataset will be used later

Each row of the hourly/daily files is one time step. A model for the later
stage would typically use `Net_Sentiment` or a share column as the series, and
the counts/`Total` as volume context. That modeling stage is **not** part of
this project yet.

## Notes

- Data collected before a filter change was filtered under the old rules. For a
  consistent dataset, delete `data/sentiment.db` (and the CSVs) when you change
  filter settings significantly and start collecting fresh.
- Bluesky Jetstream is public; no login or API key is needed.
