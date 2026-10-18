# EventHazardNet

Implementation of **Action-Conditioned Continuous-Time Event–Survival Learning for Maintenance Risk**.

EventHazardNet separates continuous condition evolution, ordinary observation updates, maintenance jumps, nonterminal event intensity and terminal competing risks. Maintenance changes both the latent condition and decaying action memory before future risks are generated. The training objective combines survival and point-process likelihoods, censoring-weighted calibration, cross-fitted action-risk targets and observed-continuation consistency.

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12. Activate the environment using your shell's activation command. CPU execution is supported; cost grows with asset count, event count and elapsed-time integration steps.

## Data preparation

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

## Training and execution

```bash
eventhazardnet data/assets.jsonl --marks 4 --causes 2 --horizon 24 --step 1 --output outputs/full
```

`--hidden 32` sets latent and action-memory width. `--step` limits the RK4 integration interval, `--grid 12` sets evaluation horizons, and `--epochs`/`--lr` control optimization. A smaller step improves numerical resolution at additional cost. Event likelihood uses eight-node Gauss–Legendre quadrature inside recorded event intervals. Terminal events are excluded from observation-mark likelihood.

When actions have recorded histories and defensible outcome support, enable the adjustment and continuation objectives:

```bash
eventhazardnet data/assets.jsonl --marks 4 --causes 2 --dr-weight 0.5 --arc-weight 0.1 --output outputs/action_adjusted
```

The doubly robust branch cross-fits propensity, outcome and discrete-time conditional censoring models across three training-asset folds. Propensities and censor survival are clipped at 0.05. Pseudo-outcomes are retained without clipping. At least three training assets with decision records are required. The continuation term compares actual future observation likelihood and horizon risk under factual versus contradicted actions. Leave these weights zero when public data do not support intervention adjustment.

## Comparisons and ablations

```bash
eventhazardnet data/assets.jsonl --marks 4 --variant joint_ct --output outputs/joint_ct
eventhazardnet data/assets.jsonl --marks 4 --variant no_events --output outputs/no_events
eventhazardnet data/assets.jsonl --marks 4 --variant no_monotone --output outputs/no_monotone
eventhazardnet data/assets.jsonl --marks 4 --dr-weight 0.5 --arc-weight 0.1 --variant no_dr --output outputs/no_dr
eventhazardnet data/assets.jsonl --marks 4 --dr-weight 0.5 --arc-weight 0.1 --variant no_arc --output outputs/no_arc
```

`joint_ct` retains the event encoder, continuous dynamics and hazard heads but replaces action jumps by an action indicator at hazard readout. `no_events` removes point-process supervision. `no_monotone` removes the nonnegative within-interval hazard-growth term and uses piecewise constant hazards. Match `--causes`, seeds, horizons and optimization budgets across runs.

## Results

`model.pt` contains parameters, training standardization and arguments. `history.csv` records actual losses. `predictions.csv` records each test decision, factual risk, selected action and remaining-life quantiles; a quantile beyond the evaluated horizon is empty. `curves.npz` contains horizon risks and cause-specific cumulative incidence. `brier.png` shows censoring-weighted Brier score by horizon.

`metrics.json` contains IBS, horizon-specific dynamic AUC, cause-specific concordance, survival NLL per asset, ten equal-mass-bin censoring-weighted ECE, normalized persistent-warning lead time, test false-alarm fraction, parameter count, timing and process memory. Warning thresholds use the 90th percentile of validation failure-free horizon risks. A persistent warning requires two consecutive decision points; missed failures contribute zero lead time. Lead times are normalized by median uncensored training lifetime. Cause-specific concordance compares each cause's observed failure with assets still at risk at that time; competing earlier failures are excluded.

CF-IAE and policy value are computed only for simulator decisions carrying `true_risk`; otherwise they are `null`. Policy value compares the selected action with the logged action using failure cost 1, maintenance cost 0.08 and downtime cost 0.02 per day. It measures simulator risk utility rather than observed treatment benefit. Undefined AUC/concordance populations are `null`.

## Modeling conventions

Mark-specific projections retain missingness and sinusoidal gap features. Static covariates and operating regimes should be included as explicitly observed sensor fields; the compact encoder does not fit separate regime embeddings. RK4 integrates latent dynamics and competing-risk incidence on the declared step grid. No future observation is consumed during a decision rollout. Conditional censoring adjustment uses ten person-period bins; metric IPCW uses training-landmark censor survival. Recent-history nuisance features contain last observed values, mark counts and decision time. These choices replace unspecified architectures while preserving the separation between observation, maintenance and terminal outcomes.

## Reproduction logic conventions

A persistent warning is emitted at the second qualifying decision snapshot. Warning lead time is measured from that confirmation time; the first crossing alone cannot be used to claim an earlier warning.
