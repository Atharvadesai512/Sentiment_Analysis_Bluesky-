# Current experiment scope

The revised study uses the existing labeled TweetEval sentiment test split
for accuracy evaluation. Manual annotation is outside the current scope.
The historical Bluesky corpus supports model agreement, aggregation,
moving averages and collection-coverage sensitivity analysis.

## Reproduction

Commands are in the project README under Benchmark and temporal experiments.
The source data, package versions and model artifact are fingerprinted in
`experiment_manifest.json`. All models use the same 12,284 local test rows.
The linear baseline uses only the local training split. RoBERTa is an
off-the-shelf sentiment checkpoint. Its pretraining and training budget differ
from those of the local models.

## Outputs

- `benchmark/*_metrics.json` contains class-level scores and confusion matrices.
- `benchmark/*_predictions.csv` contains row-aligned test predictions.
- `benchmark/uncertainty.json` contains paired bootstrap intervals.
- `temporal/sentiment_h.csv` contains hourly counts, trailing averages and coverage flags.
- `temporal/summary.json` compares series across 15-minute, hourly and daily intervals.
- `runtime/protocol.json` records the timing conditions and sample IDs.
- `runtime/batch_timings.csv` contains every repeated timing observation.
- `figures/` contains research figures generated from these outputs.

## Interpretation

Accuracy on Bluesky remains unverified. The BiLSTM scores are computed from
saved predictions because Windows application control blocks TensorFlow import,
including outside the sandbox. BiLSTM timing is unavailable.

The runtime experiment measures warm tokenization and inference on 96 fixed
test posts, over three repeats and batch sizes 1, 8 and 32. It excludes arrival,
queueing and storage. Larger batches have few timing observations, so tail
quantiles are descriptive. Controlled arrival-to-commit replay is now available in `replay/`. It measures a bounded queue and real SQLite commits at offered rates of 10, 40 and 80 posts/s, three repeats each. It excludes network and collector orchestration, and does not establish sustained live-service capacity.

Bootstrap intervals assume independent test rows. No cross-split normalized
text overlap was found in the local files. This check does not rule out other
forms of leakage or overlap with pretrained-model corpora.

The coverage field named fully_observed_uncapped means all scheduled requests
were successful and below the response cap. It does not establish that the API
returned every eligible post. Excluding capped periods changes the observed
sample and is a sensitivity analysis.

The earlier manuscript and completion plan predate this revised scope.
Use the final documents in `submission/`, which incorporate these experiment outputs and their limitations.
