# Viva preparation

## A short opening

This project compares sentiment classifiers on a common labeled Twitter test
set and studies how their predictions change a Bluesky sentiment time series.
It also measures local inference and controlled replay through a queue and
SQLite storage. RoBERTa achieves the highest benchmark scores, while the
historical models often disagree about sentiment direction.

## Questions and answers

**1. What is the contribution?**

A reproducible empirical comparison linking labeled performance, temporal
interpretation and processing behaviour. The work evaluates existing model
instances and documents their trade-offs.

**2. Why did you use Bluesky?**

Its public interface made data collection practical. Accessibility motivated
the platform choice. The research focus is sentiment classification and
processing constraints. The change from the proposal should be discussed with
the guide.

**3. Why are Twitter labels used?**

The project already contains labeled TweetEval sentiment splits. They provide
a shared independent test target. The Bluesky sample has no independent labels.

**4. What is the best model?**

The evaluated RoBERTa checkpoint scores highest on this Twitter test split.
That result does not establish the highest accuracy on Bluesky or every dataset.

**5. What are the main scores?**

RoBERTa accuracy is 72.18% and macro F1 is 0.724. BiLSTM scores 59.47% and 0.558
from saved predictions. TF-IDF plus logistic regression scores 59.22% and 0.560.
VADER scores 53.00% and 0.529. The neutral majority baseline is 48.33% accurate.

**6. Why include a linear baseline?**

It tests whether the additional complexity of a neural model improves results
over a simpler learned approach. Its performance is close to the saved BiLSTM.

**7. What does macro F1 add?**

It averages the F1 score of each class equally. Accuracy can obscure weak
performance on less frequent classes. BiLSTM negative recall is only 0.325.

**8. Is BiLSTM significantly better than the linear baseline?**

The paired bootstrap intervals for the accuracy and macro-F1 differences
include zero. The current evidence does not establish a clear ranking.

**9. How did you check uncertainty?**

We used 1,000 paired bootstrap resamples of test rows with seed 42. The same
sample indices were used for each model. The intervals assume independent
rows and do not cover retraining-seed variation.

**10. What is net sentiment?**

It is the positive count minus the negative count, divided by the total count.
It ranges from −1 to +1 when observations exist. Empty intervals have undefined
shares, so the analysis leaves them blank.

**11. What did the temporal analysis show?**

VADER and RoBERTa have opposite signs in 327 of 336 hourly periods. Their
hourly mean absolute difference is approximately 0.346. They can move together
while having different baseline levels.

**12. What does the six-hour moving average mean?**

It is the mean of the current and previous five hourly net-sentiment values.
It uses no future observations and requires all six values. It is smoothing.
It does not forecast the next period.

**13. How complete is the collection?**

All 1,344 logged requests succeeded, but 194 reached the 100-result cap.
Successful responses do not guarantee complete API recall. The search is
query-defined and covers only two weeks.

**14. What is the difference between inference timing and replay latency?**

Inference timing measures warm text processing and classification. Replay
arrival-to-commit latency also includes waiting in the queue and SQLite
storage. Neither experiment measures the live network delivery delay.

**15. How does replay work?**

It offers 96 sampled cleaned posts at fixed rates, uses one classification
worker and a queue of capacity 16, and commits admitted posts using db.py.
Each rate/model condition is repeated three times. Full queues drop new
arrivals. Consult the final presentation for the measured drop counts.

**16. Why can a latency percentile hide overload?**

The percentile only describes successfully completed posts. Dropped arrivals
must be reported separately. A system can drop many posts while keeping
completed-post latency bounded.

**17. Why is BiLSTM timing missing?**

Windows application control blocks a TensorFlow component during import.
The saved predictions align with the test rows and support rescoring.
They cannot provide fresh runtime measurements.

**18. What did you do about leakage?**

The linear baseline fits only on training rows. The local splits have no
overlap after lowercasing and whitespace normalization. This limited check
does not rule out semantic duplicates, author overlap or pretraining exposure.

**19. What does the AI query tell you?**

It defines the collected search sample. Whole-post polarity is not necessarily
the author's attitude toward AI. The report avoids a population-level claim.

**20. Is this an IEEE publication?**

It is a research manuscript prepared for further venue-specific development.
It has not been submitted or accepted. Peer review and venue requirements
determine publication.

## Demonstration sequence

1. Open the common benchmark comparison figure and explain the shared test set.
2. Open one metrics JSON to show class support and its confusion matrix.
3. Open the temporal figure and explain trailing smoothing and cap-hit periods.
4. Open replay/summary.json to show stored and dropped counts.
5. Show the reproduction commands in the project README.

Use saved outputs for a short viva demonstration. A full RoBERTa test run takes
several minutes on the inspected CPU, and TensorFlow remains blocked.
