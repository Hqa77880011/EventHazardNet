# EventHazardNet

Implementation of **Action-Conditioned Continuous-Time Event–Survival Learning for Maintenance Risk**.

## Installation

Use Python 3.11 or 3.12. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) for compressor telemetry, NASA C-MAPSS through the [NASA prognostics repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), or a maintenance simulator that records factual actions and true action-specific risk curves. Public degradation records do not supply counterfactual truth.

Export one JSON object per asset into `data/assets.jsonl`. Each object has these fields:

| Field | Meaning |
|---|---|
| `asset`, `split` | Unique asset ID and `train`, `val` or `test`; one asset appears once |
| `end`, `cause` | Follow-up time and terminal cause: `0` censored, `1..K` failure causes |
| `events` | Time-sorted objects containing `time`, `mark`, numeric vector `x`, and optional binary `action` |
| `decisions` | Objects containing decision `time`, factual binary `action`, optional `downtime_days`, optional `true_risk` |

Ordinary event `mark` is in `0..M-1`; its `x` vector contains the same sensor fields for all marks, with unavailable readings represented by JSON `null`. A maintenance event has an `action` of `0` or `1` and the same-length `x` vector. Ordinary events omit `action`. An explicit action zero is still a decision event and is distinct from an observation. Events occur strictly before `end`.

Decision snapshots describe a choice immediately before events at that timestamp. Their histories contain only strictly earlier events. Decisions must precede terminal follow-up. Use a common time unit, such as hours, for events, horizons and solver steps. Preserve irregular timestamps; do not forward-fill observations into new events. Split by asset before fitting; later cohorts can form the test split. Sensor standardization uses training observations only.

For simulator evaluation, `true_risk` is a 2×`--grid` array, with action rows `[0,1]` and horizons equally spaced from `horizon/grid` through `horizon`. Supply the simulator's analytic or independently Monte Carlo estimated failure probabilities, never factual outcome duplicates. `downtime_days` supplies the cost of action one.

## Run and evaluate

```bash
eventhazardnet data/assets.jsonl --marks 4 --causes 2 --horizon 24 --step 1 --output outputs/full --seed 0
```

The command fits and evaluates. Enable `--dr-weight 0.5 --arc-weight 0.1` only with supported action histories and at least three training assets carrying decisions.

## Ablations

Supported controls: `joint_ct`, `no_events`, `no_dr`, `no_arc`, `no_monotone`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
eventhazardnet data/assets.jsonl --marks 4 --causes 2 --horizon 24 --step 1 --output outputs/joint_ct --seed 0 --variant joint_ct
```

## Outputs

`model.pt` holds parameters/standardization/arguments; `history.csv` contains losses; `predictions.csv` and `curves.npz` contain test decisions and risk curves; `metrics.json` and `brier.png` contain computed evaluation summaries.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
