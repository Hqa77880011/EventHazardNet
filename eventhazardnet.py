import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import psutil
from numpy.polynomial.legendre import leggauss
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def classifier(x, y):
    return (LogisticRegression(max_iter=1000) if len(np.unique(y)) > 1 else DummyClassifier(strategy="prior")).fit(x, y)


def probability(model, x):
    predicted = model.predict_proba(x)
    return predicted[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(x))


def censor_survival(times, causes, query):
    survival = np.ones_like(np.asarray(query), dtype=float)
    for t in np.unique(times):
        at_risk = np.sum(times >= t)
        failures = np.sum((times == t) & (causes == 0))
        survival[np.asarray(query) > t] *= 1 - failures / max(1, at_risk)
    return survival.clip(.05)


class EventHazardNet(torch.nn.Module):
    def __init__(self, features, marks, causes, hidden, step, variant):
        super().__init__()
        self.hidden, self.step, self.variant = hidden, step, variant
        self.mark = torch.nn.Embedding(marks, hidden)
        self.inputs = torch.nn.ModuleList([torch.nn.Linear(features * 2, hidden) for _ in range(marks)])
        self.gap = torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
        self.norm = torch.nn.LayerNorm(hidden)
        self.update = torch.nn.GRUCell(hidden, hidden)
        self.flow = torch.nn.Sequential(torch.nn.Linear(hidden * 2, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
        self.damping = torch.nn.Linear(hidden, hidden)
        self.decay = torch.nn.Parameter(torch.zeros(hidden))
        self.actions = torch.nn.Embedding(1, hidden)
        self.action_low = torch.nn.Linear(hidden * 2, 8)
        self.action_basis = torch.nn.Parameter(torch.randn(hidden, 8) * .01)
        self.intensity = torch.nn.Linear(hidden * 2, marks)
        self.base = torch.nn.Linear(hidden * 2, causes)
        self.growth = torch.nn.Linear(hidden * 2, causes)
        self.readout_action = torch.nn.Linear(2, causes, bias=False)

    def field(self, h, q):
        state = torch.cat([h, q])
        dh = self.flow(state) - torch.nn.functional.softplus(self.damping(h)) * h
        dq = -torch.nn.functional.softplus(self.decay) * q
        growth = torch.nn.functional.softplus(self.growth(state))
        if self.variant == "no_monotone":
            growth = torch.zeros_like(growth)
        return dh, dq, growth

    def evolve(self, h, q, dt):
        steps = max(1, int(math.ceil(float(dt) / self.step)))
        delta = float(dt) / steps
        accumulated = torch.zeros(self.base.out_features)
        for _ in range(steps):
            k1 = self.field(h, q)
            k2 = self.field(h + delta * k1[0] / 2, q + delta * k1[1] / 2)
            k3 = self.field(h + delta * k2[0] / 2, q + delta * k2[1] / 2)
            k4 = self.field(h + delta * k3[0], q + delta * k3[1])
            h = h + delta * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            q = q + delta * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            accumulated = accumulated + delta * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        return h, q, accumulated

    def jump(self, h, q, action):
        if self.variant == "joint_ct" or action == 0:
            return h, q
        embedding = self.actions(torch.tensor(0))
        delta = self.action_basis @ torch.tanh(self.action_low(torch.cat([h, embedding])))
        return h + delta, q + embedding

    def interval(self, h, q, dt, action=0):
        offset = self.readout_action(torch.nn.functional.one_hot(torch.tensor(action), 2).float()) if self.variant == "joint_ct" else 0
        base = torch.nn.functional.softplus(self.base(torch.cat([h, q])) + offset)
        nodes, weights = leggauss(8)
        survival_integral = base.sum() * 0
        event_integral = base.sum() * 0
        for node, weight in zip(nodes, weights):
            elapsed = (node + 1) * dt / 2
            nh, nq, growth = self.evolve(h, q, elapsed)
            survival_integral = survival_integral + weight * (base + growth).sum() * dt / 2
            event_integral = event_integral + weight * torch.nn.functional.softplus(self.intensity(torch.cat([nh, nq]))).sum() * dt / 2
        final_h, final_q, growth = self.evolve(h, q, dt)
        return final_h, final_q, base + growth, survival_integral, event_integral

    def history(self, record, cutoff=None, likelihood=False):
        h, q = torch.zeros(self.hidden), torch.zeros(self.hidden)
        previous = 0.
        survival_loss = h.sum()
        event_loss = h.sum()
        action = 0
        end = record["end"] if cutoff is None else cutoff
        for event in record["events"]:
            if event["time"] > end or (cutoff is not None and event["time"] == end):
                break
            dt = event["time"] - previous
            h, q, hazard, integral, intensity_integral = self.interval(h, q, dt, action)
            survival_loss = survival_loss + integral
            event_loss = event_loss + intensity_integral
            if event.get("action") is not None:
                action = int(event["action"])
                h, q = self.jump(h, q, action)
            else:
                mark = int(event["mark"])
                event_loss = event_loss - torch.log(torch.nn.functional.softplus(self.intensity(torch.cat([h, q])))[mark].clamp_min(1e-8))
                values = np.asarray(event["x"], dtype=float)
                available = np.isfinite(values)
                vector = torch.tensor(np.r_[np.nan_to_num(values), available], dtype=torch.float32)
                gap = torch.tensor([np.log1p(dt), np.sin(dt), np.cos(dt)], dtype=torch.float32)
                embedding = self.norm(self.mark(torch.tensor(mark)) + self.inputs[mark](vector) + self.gap(gap))
                h = self.update(embedding, h)
            previous = event["time"]
        h, q, hazard, integral, intensity_integral = self.interval(h, q, end - previous, action)
        survival_loss += integral
        event_loss += intensity_integral
        if likelihood and record["cause"] > 0:
            survival_loss -= torch.log(hazard[record["cause"] - 1].clamp_min(1e-8))
        return h, q, survival_loss, event_loss

    def risk_curve(self, h, q, action, grid):
        h, q = self.jump(h, q, action)
        offset = self.readout_action(torch.nn.functional.one_hot(torch.tensor(action), 2).float()) if self.variant == "joint_ct" else 0
        base = torch.nn.functional.softplus(self.base(torch.cat([h, q])) + offset)
        ramp = torch.zeros_like(base)
        cumulative_hazard = h.sum() * 0
        cumulative_incidence = torch.zeros_like(base)
        risks, incidence = [], []
        previous = 0.
        def derivative(state):
            sh, sq, sramp, shazard, _ = state
            dh, dq, dg = self.field(sh, sq)
            hazard = base + sramp
            return dh, dq, dg, hazard.sum(), torch.exp(-shazard) * hazard
        for horizon in grid:
            steps = max(1, int(math.ceil(float(horizon - previous) / self.step)))
            delta = float(horizon - previous) / steps
            state = (h, q, ramp, cumulative_hazard, cumulative_incidence)
            for _ in range(steps):
                k1 = derivative(state)
                k2 = derivative(tuple(s + delta * k / 2 for s, k in zip(state, k1)))
                k3 = derivative(tuple(s + delta * k / 2 for s, k in zip(state, k2)))
                k4 = derivative(tuple(s + delta * k for s, k in zip(state, k3)))
                state = tuple(s + delta * (a + 2 * b + 2 * c + d) / 6 for s, a, b, c, d in zip(state, k1, k2, k3, k4))
            h, q, ramp, cumulative_hazard, cumulative_incidence = state
            risks.append(1 - torch.exp(-cumulative_hazard))
            incidence.append(cumulative_incidence)
            previous = horizon
        return torch.stack(risks), torch.stack(incidence)

    def continuation(self, h, q, record, decision, action, horizon):
        h, q = self.jump(h, q, action)
        previous = decision
        score = h.sum() * 0
        end = min(record["end"], decision + horizon)
        count = 0
        for event in record["events"]:
            if event["time"] <= decision:
                continue
            if event["time"] > end or event.get("action") is not None:
                break
            gap = event["time"] - previous
            h, q, _, _, integral = self.interval(h, q, gap, action)
            mark = int(event["mark"])
            intensity = torch.nn.functional.softplus(self.intensity(torch.cat([h, q])))[mark]
            score = score + integral - torch.log(intensity.clamp_min(1e-8))
            values = np.asarray(event["x"], dtype=float)
            vector = torch.tensor(np.r_[np.nan_to_num(values), np.isfinite(values)], dtype=torch.float32)
            time_embedding = self.gap(torch.tensor([np.log1p(gap), np.sin(gap), np.cos(gap)], dtype=torch.float32))
            embedding = self.norm(self.mark(torch.tensor(mark)) + self.inputs[mark](vector) + time_embedding)
            h = self.update(embedding, h)
            previous = event["time"]
            count += 1
        if previous < end:
            score = score + self.interval(h, q, end - previous, action)[4]
        return score / max(1, count)


def features_at(record, decision, marks):
    last = np.zeros(len(record["events"][0]["x"]))
    counts = np.zeros(marks)
    for event in record["events"]:
        if event["time"] >= decision:
            break
        if event.get("action") is None:
            values = np.array(event["x"], float)
            last[np.isfinite(values)] = values[np.isfinite(values)]
            counts[int(event["mark"])] += 1
    return np.r_[last, np.log1p(counts), decision]


def decision_rows(records, args):
    rows = []
    for i, record in enumerate(records):
        for d in record.get("decisions", []):
            remaining = record["end"] - d["time"]
            outcome = int(record["cause"] > 0 and remaining <= args.horizon)
            observed = int(outcome or remaining >= args.horizon)
            rows.append(dict(asset=i, decision=d, x=features_at(record, d["time"], args.marks), action=d["action"], remaining=remaining, cause=record["cause"], observed=observed, outcome=outcome))
    return rows


def nuisance_targets(rows, args):
    if len(set(r["asset"] for r in rows)) < 3:
        raise ValueError("Doubly robust training requires decisions from at least three training assets")
    x = np.stack([r["x"] for r in rows])
    actions = np.array([r["action"] for r in rows])
    groups = np.array([r["asset"] for r in rows])
    targets = np.zeros((len(rows), 2))
    for train, held in GroupKFold(3).split(x, groups=groups):
        scaler = StandardScaler().fit(x[train])
        tx, hx = scaler.transform(x[train]), scaler.transform(x[held])
        propensity = probability(classifier(tx, actions[train]), hx).clip(.05, .95)
        observed_ids = train[np.array([rows[i]["observed"] for i in train]).astype(bool)]
        out_design = np.c_[scaler.transform(x[observed_ids]), actions[observed_ids]]
        outcome_model = classifier(out_design, np.array([rows[i]["outcome"] for i in observed_ids]))
        censor_x, censor_y = [], []
        boundaries = np.linspace(0, args.horizon, 11)
        for i in train:
            history = scaler.transform(x[i:i + 1])[0]
            row = rows[i]
            for k in range(10):
                if row["remaining"] <= boundaries[k]:
                    break
                censor_x.append(np.r_[history, row["action"], k / 10])
                censor_y.append(int(row["cause"] == 0 and row["remaining"] <= boundaries[k + 1]))
        censor_model = classifier(np.asarray(censor_x), np.asarray(censor_y))
        for slot, i in enumerate(held):
            row = rows[i]
            survival = 1.
            for k in range(10):
                if boundaries[k] >= min(row["remaining"], args.horizon):
                    break
                hazard = probability(censor_model, np.r_[hx[slot], row["action"], k / 10][None])[0]
                survival *= 1 - hazard
            for action in (0, 1):
                q = probability(outcome_model, np.r_[hx[slot], action][None])[0]
                e = propensity[slot] if action else 1 - propensity[slot]
                weight = int(row["action"] == action) * row["observed"] / (e * max(.05, survival))
                targets[i, action] = q + weight * (row["outcome"] - q)
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--output", default="outputs/full")
    p.add_argument("--variant", choices=["full", "joint_ct", "no_events", "no_dr", "no_arc", "no_monotone"], default="full")
    p.add_argument("--marks", type=int, required=True)
    p.add_argument("--causes", type=int, default=1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--step", type=float, default=1.)
    p.add_argument("--horizon", type=float, default=24)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--event-weight", type=float, default=.1)
    p.add_argument("--dr-weight", type=float, default=0.)
    p.add_argument("--arc-weight", type=float, default=0.)
    p.add_argument("--calibration-weight", type=float, default=.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    records = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if args.grid < 2 or args.step <= 0 or args.horizon <= 0 or not all(any(r["split"] == s for r in records) for s in ("train", "val", "test")):
        raise ValueError("Supply train/val/test assets, positive integration scales and at least two horizons")
    if len({r["asset"] for r in records}) != len(records):
        raise ValueError("One complete record per asset; no asset may cross splits")
    for r in records:
        times = [e["time"] for e in r["events"]]
        if not times or times != sorted(times) or times[0] < 0 or times[-1] >= r["end"]:
            raise ValueError("Event times must be sorted, nonnegative, and strictly before terminal time")
        if any(not 0 <= d["time"] < r["end"] or d["action"] not in (0, 1) for d in r.get("decisions", [])):
            raise ValueError("Decision times must precede terminal follow-up and actions must be binary")
    train = [r for r in records if r["split"] == "train"]
    if not any(r["cause"] > 0 for r in train):
        raise ValueError("Training needs uncensored failures for survival and normalized warning evaluation")
    values = np.array([e["x"] for r in train for e in r["events"] if e.get("action") is None], float)
    mean, scale = np.nanmean(values, 0), np.nanstd(values, 0).clip(1e-6)
    for r in records:
        for e in r["events"]:
            e["x"] = ((np.array(e["x"], float) - mean) / scale).tolist()
    torch.manual_seed(args.seed)
    model = EventHazardNet(len(mean), args.marks, args.causes, args.hidden, args.step, args.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    grid = np.linspace(args.horizon / args.grid, args.horizon, args.grid)
    decisions = decision_rows(train, args)
    if not decisions:
        raise ValueError("Training assets require declared decision snapshots")
    targets = nuisance_targets(decisions, args) if args.dr_weight > 0 and args.variant != "no_dr" else None
    rng = np.random.default_rng(args.seed)
    history = []
    training_times = np.array([r["end"] for r in train])
    training_causes = np.array([r["cause"] for r in train])
    censor_times = np.array([r["remaining"] for r in decisions])
    censor_causes = np.array([r["cause"] for r in decisions])
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        for i in rng.permutation(len(train)):
            record = train[i]
            _, _, survival_loss, event_loss = model.history(record, likelihood=True)
            loss = survival_loss + (0 if args.variant == "no_events" else args.event_weight * event_loss)
            for j, row in enumerate(decisions):
                if row["asset"] != i:
                    continue
                h, q, _, _ = model.history(record, row["decision"]["time"])
                curves = [model.risk_curve(h, q, a, grid)[0] for a in (0, 1)]
                if targets is not None:
                    loss += args.dr_weight * sum((curves[a][-1] - targets[j, a]) ** 2 for a in (0, 1))
                observable = (row["remaining"] >= grid) | ((row["cause"] > 0) & (row["remaining"] <= grid))
                weights = observable / censor_survival(censor_times, censor_causes, np.minimum(grid, row["remaining"]))
                truth = ((row["remaining"] <= grid) & (row["cause"] > 0)).astype(float)
                loss += args.calibration_weight * (torch.tensor(weights, dtype=torch.float32) * (curves[row["action"]] - torch.tensor(truth, dtype=torch.float32)).square()).mean()
                if args.arc_weight > 0 and args.variant != "no_arc" and row["observed"]:
                    factual = model.continuation(h, q, record, row["decision"]["time"], row["action"], args.horizon) + .5 * (curves[row["action"]][-1] - row["outcome"]) ** 2
                    contradicted = model.continuation(h, q, record, row["decision"]["time"], 1 - row["action"], args.horizon) + .5 * (curves[1 - row["action"]][-1] - row["outcome"]) ** 2
                    loss += args.arc_weight * torch.relu(.03 + factual - contradicted)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, loss=float(np.mean(losses))))
    training_seconds = time.perf_counter() - training_started
    rows = []
    curves = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for split in ("val", "test"):
            assets = [r for r in records if r["split"] == split]
            for row in decision_rows(assets, args):
                record = assets[row["asset"]]
                h, q, _, _ = model.history(record, row["decision"]["time"])
                action_curves, incidence = zip(*(model.risk_curve(h, q, a, grid) for a in (0, 1)))
                risks = np.stack([c.numpy() for c in action_curves])
                cf = row["decision"].get("true_risk")
                cf_error = float(np.trapezoid(np.c_[np.zeros(2), np.abs(risks - np.asarray(cf))], np.r_[0, grid], axis=1).mean() / args.horizon) if cf is not None else None
                selected = int(risks[1, -1] + .08 + .02 * row["decision"].get("downtime_days", 0) < risks[0, -1])
                policy_value = None
                if cf is not None:
                    truth = np.asarray(cf)
                    cost = lambda a: truth[a, -1] + a * (.08 + .02 * row["decision"].get("downtime_days", 0))
                    policy_value = float(cost(row["action"]) - cost(selected))
                remaining_quantiles = {f"rul_q{int(level * 100)}": float(np.interp(level, np.r_[0, risks[row["action"]]], np.r_[0, grid])) if risks[row["action"], -1] >= level else None for level in (.1, .5, .9)}
                rows.append(dict(asset=str(record["asset"]), split=split, decision=float(row["decision"]["time"]), remaining=row["remaining"], cause=row["cause"], observed=row["observed"], outcome=row["outcome"], action=row["action"], risk=float(risks[row["action"], -1]), selected_action=selected, cf_iae=cf_error, policy_value=policy_value, **remaining_quantiles))
                curves.append(dict(risk=risks[row["action"]], incidence=incidence[row["action"]].numpy(), split=split))
    evaluation = pd.DataFrame(rows)
    val = evaluation[evaluation.split == "val"]
    if not len(val) or not (evaluation.split == "test").any():
        raise ValueError("Supply decision snapshots on validation and test assets")
    controls = val[(val.observed == 1) & (val.outcome == 0)]
    if controls.empty:
        raise ValueError("Validation needs observed failure-free horizons to lock the warning threshold")
    alert_threshold = float(np.quantile(controls.risk, .9))
    test_ids = np.where(evaluation.split.to_numpy() == "test")[0]
    test = evaluation.iloc[test_ids].copy()
    risk = np.stack([curves[i]["risk"] for i in test_ids])
    incidence = np.stack([curves[i]["incidence"] for i in test_ids])
    durations = test.remaining.to_numpy()
    causes = test.cause.to_numpy()
    brier, auc = [], []
    for k, horizon in enumerate(grid):
        event = (durations <= horizon) & (causes > 0)
        alive = durations > horizon
        weight = event / censor_survival(censor_times, censor_causes, durations) + alive / censor_survival(censor_times, censor_causes, np.full(len(durations), horizon))
        brier.append(float(np.mean(weight * (event - risk[:, k]) ** 2)))
        if event.any() and alive.any():
            case_weights = weight[event]
            concordant = (risk[event, k, None] > risk[alive, k][None]).astype(float) + .5 * (risk[event, k, None] == risk[alive, k][None])
            auc.append(float(np.sum(concordant * case_weights[:, None]) / (case_weights.sum() * alive.sum())))
        else:
            auc.append(None)
    concordance = {}
    for cause in range(1, args.causes + 1):
        wins, pairs = 0., 0
        for i in range(len(test)):
            if causes[i] != cause:
                continue
            controls = durations > durations[i]
            ci = incidence[i, -1, cause - 1]
            cj = incidence[controls, -1, cause - 1]
            wins += np.sum(ci > cj) + .5 * np.sum(ci == cj)
            pairs += len(cj)
        concordance[str(cause)] = float(wins / pairs) if pairs else None
    observed = test.observed.to_numpy().astype(bool)
    ece = 0.
    if observed.any():
        order = np.argsort(test.risk.to_numpy()[observed])
        pcal = test.risk.to_numpy()[observed]
        ycal = test.outcome.to_numpy()[observed]
        ece_weights = 1 / censor_survival(censor_times, censor_causes, np.minimum(durations[observed], args.horizon))
        for ids in np.array_split(order, min(10, len(order))):
            ece += ece_weights[ids].sum() / ece_weights.sum() * abs(np.average(pcal[ids], weights=ece_weights[ids]) - np.average(ycal[ids], weights=ece_weights[ids]))
    leads = []
    for _, group in test.groupby("asset"):
        group = group.sort_values("decision")
        active = group.risk.to_numpy() >= alert_threshold
        persistent = np.where(active[:-1] & active[1:])[0]
        if (group.cause > 0).any():
            leads.append(float(group.remaining.iloc[persistent[0]]) if len(persistent) else 0.)
    nll = []
    with torch.no_grad():
        for r in records:
            if r["split"] == "test":
                nll.append(float(model.history(r, likelihood=True)[2]))
    median_lifetime = float(np.median(training_times[training_causes > 0]))
    test_controls = test[(test.observed == 1) & (test.outcome == 0)]
    metrics = dict(ibs=float(np.trapezoid(brier, grid) / (grid[-1] - grid[0])), brier_by_horizon=brier, dynamic_auc=auc, cause_specific_concordance=concordance, survival_nll_per_asset=float(np.mean(nll)), ece_equal_mass_10=float(ece), warning_lead_normalized=float(np.mean(leads) / median_lifetime) if leads else None, warning_threshold=alert_threshold, test_horizon_false_alarm_fraction=float((test_controls.risk >= alert_threshold).mean()) if len(test_controls) else None, cf_iae=float(test.cf_iae.dropna().mean()) if test.cf_iae.notna().any() else None, policy_value=float(test.policy_value.dropna().mean()) if test.policy_value.notna().any() else None, inference_seconds=time.perf_counter() - started, training_seconds=training_seconds, process_rss_mb=psutil.Process().memory_info().rss / 1024 ** 2, parameter_count=sum(p.numel() for p in model.parameters()))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), mean=mean, scale=scale, arguments=vars(args)), out / "model.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    test.to_csv(out / "predictions.csv", index=False)
    np.savez_compressed(out / "curves.npz", horizon=grid, risk=risk, incidence=incidence)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    plt.plot(grid, brier, color="#4d8796")
    plt.xlabel("Time after decision")
    plt.ylabel("IPCW Brier score")
    plt.tight_layout()
    plt.savefig(out / "brier.png", dpi=180)


if __name__ == "__main__":
    main()
