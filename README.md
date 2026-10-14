# DriftWatch

An interactive classification-monitoring sandbox for exploring feature drift, prediction quality and calibration. Upload reference/current batches, change alert and decision thresholds, compare segments and export a JSON report. Built as an AI engineering portfolio prototype with AI coding assistance.

## Quick start

Python 3.10+. Standard library only; no accounts or package installation.

```sh
python app.py
# Open http://127.0.0.1:8766
python -m unittest discover -v
```

## Features

- Reference-derived quantile buckets with explicit underflow/overflow coverage.
- Population stability index with Laplace-smoothed normalized probabilities.
- Confusion matrix, accuracy, precision, recall, F1 and Brier score.
- Ten-bin expected calibration error and an interactive SVG calibration plot.
- Segment diagnostics with configurable minimum-sample warnings.
- Seeded synthetic traffic with independently controlled input/concept shift.
- User batch import and downloadable reports.
- Input validation for finite probabilities, binary labels and consistent feature schemas.
- Local-only API with 1 MB request limit and cross-origin write rejection.
- CI test matrix for Python 3.10, 3.12 and 3.13.

## Architecture

```text
Browser controls / JSON -> HTTP validation -> monitoring functions -> JSON report
                                  |          |-- reference quantiles / PSI
                                  |          |-- confusion / calibration / Brier
                                  |          |-- sample-aware segment summaries
                                  +-> seeded synthetic generator (optional)
```

No trained model is shipped. The demo simulates scores and outcomes so the monitoring behavior is reproducible. It does not claim production usage, real customers, predictive performance on real datasets or statistical fairness guarantees.

## Import format

```json
{"reference":[{"label":1,"score":0.8,"segment":"desktop","features":{"signal":1.2}}],"current":[{"label":0,"score":0.6,"segment":"mobile","features":{"signal":2.4}}]}
```

Use representative batches, not the one-row illustration, for useful diagnostics. POST `/api/report` accepts this object plus `threshold` and `alert`. Omitting arrays generates two synthetic 400-row batches; `shift` controls distribution/concept shift. GET `/api/health` checks readiness.

## Tests and interpretation

Tests verify hand-calculated confusion/Brier values, perfect calibration, undefined precision, constant/shifted feature distributions, finite input validation, threshold behavior, reproducibility and small-sample warnings.

PSI thresholds are heuristics, not p-values. Feature drift does not establish concept drift or explain causality. Precision/recall/F1 return null where undefined; this is not silently treated as zero. Brier and calibration metrics require reliable labels. Segment comparisons are descriptive, not a fairness certification. Quantile bucket quality depends on the reference sample.

This development server is unauthenticated and binds loopback. No data is persisted. Do not expose it to the internet. Future work: temporal batch storage, delayed-label support, alert deduplication, confidence intervals and monitoring an actual deployed model.

## Interview walkthrough

Show why shifting the input distribution can trigger PSI without proving an accuracy drop. Increase the decision threshold to discuss precision/recall tradeoffs. Explain why a calibrated model can still discriminate poorly and why small segment samples need caution.

MIT licensed. Author: Ali Farhat.
