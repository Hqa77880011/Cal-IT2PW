import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor


def reduce_interval(lower, upper, values, left):
    order = np.argsort(values, axis=1)
    y = np.take_along_axis(values, order, axis=1)
    lo = np.take_along_axis(lower, order, axis=1)
    hi = np.take_along_axis(upper, order, axis=1)
    w = (lo + hi) / 2
    estimate = (w * y).sum(1) / w.sum(1).clip(1e-12)
    for _ in range(100):
        switch = y <= estimate[:, None]
        w = np.where(switch, hi if left else lo, lo if left else hi)
        updated = (w * y).sum(1) / w.sum(1).clip(1e-12)
        if np.max(np.abs(updated - estimate)) < 1e-8:
            return updated
        estimate = updated
    return estimate


class FuzzyWindow(torch.nn.Module):
    def __init__(self, centers, noise, batch_width, variant):
        super().__init__()
        self.centers = torch.nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.log_sigma = torch.nn.Parameter(torch.zeros_like(self.centers))
        self.coef = torch.nn.Parameter(torch.zeros(len(centers), centers.shape[1] + 1))
        self.log_effect = torch.nn.Parameter(torch.full((len(centers),), -2.0))
        self.beta = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("measurement", torch.tensor(noise, dtype=torch.float32))
        self.register_buffer("batch_width", torch.tensor(batch_width, dtype=torch.float32))
        self.variant = variant

    def forward(self, x):
        delta = x[:, None] - self.centers
        sigma = self.log_sigma.exp().clamp(0.05, 20)
        ambiguity = torch.nn.functional.softplus(self.beta)
        width = ambiguity[0] * self.measurement + ambiguity[1] * self.batch_width
        if self.variant in ("fixed", "anfis"):
            width = torch.zeros_like(width) + (0.15 if self.variant == "fixed" else 0)
        lo = torch.exp(-0.5 * (delta / (sigma * torch.exp(-width))).square().sum(-1)).clamp_min(1e-20)
        hi = torch.exp(-0.5 * (delta / (sigma * torch.exp(width))).square().sum(-1)).clamp_min(1e-20)
        value = torch.cat([x, torch.ones(len(x), 1)], 1) @ self.coef.T
        spread = torch.nn.functional.softplus(self.log_effect)
        weights = (lo + hi) / 2
        mid = (weights * value).sum(1) / weights.sum(1)
        return mid, lo, hi, value - spread, value + spread

    def interval(self, x):
        with torch.no_grad():
            _, lo, hi, left, right = self(torch.tensor(x, dtype=torch.float32))
        return reduce_interval(lo.numpy(), hi.numpy(), left.numpy(), True), reduce_interval(lo.numpy(), hi.numpy(), right.numpy(), False)


def quantile(scores, alpha):
    level = min(1, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def fit(args):
    frame = pd.read_csv(args.data)
    spec = json.loads(Path(args.spec).read_text())
    features = spec["features"]
    if frame.groupby("batch")["split"].nunique().max() != 1:
        raise ValueError("Each batch must occur in one split only")
    splits = {s: frame[frame.split == s] for s in ("train", "interval", "risk", "decision", "test")}
    if any(len(v) < 2 for v in splits.values()):
        raise ValueError("Supply nonempty batch-disjoint train, interval, risk, decision and test splits")
    train = splits["train"]
    center = train[features].median().to_numpy()
    scale = (train[features].quantile(.75) - train[features].quantile(.25)).to_numpy().clip(1e-6)
    x = {s: (v[features].to_numpy() - center) / scale for s, v in splits.items()}
    y = {s: v.quality.to_numpy() for s, v in splits.items()}
    if not all(np.isfinite(v).all() for v in [*x.values(), *y.values()]):
        raise ValueError("Quality and process features must be finite")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    models = {}
    if args.variant in ("gp", "quantile"):
        if args.variant == "gp":
            predictor = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), normalize_y=True).fit(x["train"], y["train"])
            def raw(z):
                mu, sd = predictor.predict(z, return_std=True)
                return mu - 1.645 * sd, mu + 1.645 * sd
        else:
            predictor = [QuantileRegressor(quantile=q, alpha=.01, solver="highs").fit(x["train"], y["train"]) for q in (.05, .95)]
            def raw(z):
                a, b = [m.predict(z) for m in predictor]
                return np.minimum(a, b), np.maximum(a, b)
        models["predictor"] = predictor
        rule_count = 0
    else:
        centers = KMeans(n_clusters=min(args.rules, len(train)), random_state=args.seed, n_init=10).fit(x["train"]).cluster_centers_
        noise = np.asarray(spec.get("measurement_sd", [0.] * len(features))) / scale
        batch_mean = train.groupby("batch")[features].mean().to_numpy()
        batch_width = np.broadcast_to(np.std(batch_mean / scale, axis=0), centers.shape).copy()
        model = FuzzyWindow(centers, noise, batch_width, args.variant)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        z = torch.tensor(x["train"], dtype=torch.float32)
        target = torch.tensor(y["train"], dtype=torch.float32)
        for _ in range(args.epochs):
            for ids in np.array_split(rng.permutation(len(z)), max(1, int(np.ceil(len(z) / args.batch_size)))):
                prediction, lo, hi, left, right = model(z[ids])
                lower = (lo * left).sum(1) / lo.sum(1)
                upper = (hi * right).sum(1) / hi.sum(1)
                loss = (prediction - target[ids]).square().mean() + .1 * (torch.relu(lower - target[ids]) + torch.relu(target[ids] - upper)).mean() + .001 * torch.nn.functional.softplus(model.log_effect).mean()
                if args.variant != "no_monotonicity":
                    for name, direction in spec.get("monotonic", {}).items():
                        j = features.index(name)
                        shifted = z[ids].clone()
                        shifted[:, j] += .05
                        loss = loss + .5 * torch.relu(-direction * (model(shifted)[0] - prediction)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        raw = model.interval
        models["model"] = model
        rule_count = len(centers)
    a, b = raw(x["interval"])
    correction = 0 if args.variant == "no_calibration" else max(0, quantile(np.maximum(a - y["interval"], y["interval"] - b), args.alpha))
    regime_correction = {}
    for regime, group in splits["interval"].groupby("regime"):
        if len(group) >= args.minimum_regime:
            ids = splits["interval"].index.get_indexer(group.index)
            regime_correction[str(regime)] = max(0, quantile(np.maximum(a[ids] - y["interval"][ids], y["interval"][ids] - b[ids]), args.alpha)) if args.variant != "no_calibration" else 0
    def bounds(s):
        lo, hi = raw(x[s])
        c = np.array([regime_correction.get(str(v), correction) for v in splits[s].regime])
        return lo - c, hi + c
    rlo, rhi = bounds("risk")
    risk_target = (y["risk"] < args.minimum_quality).astype(float)
    global_cal = IsotonicRegression(out_of_bounds="clip").fit(args.minimum_quality - rlo, risk_target)
    calibrators = {}
    for regime, group in splits["risk"].groupby("regime"):
        if len(group) >= args.minimum_regime:
            ids = splits["risk"].index.get_indexer(group.index)
            calibrators[str(regime)] = IsotonicRegression(out_of_bounds="clip").fit(args.minimum_quality - rlo[ids], risk_target[ids])
    def probabilities(s, lo):
        out = global_cal.predict(args.minimum_quality - lo)
        for regime, cal in calibrators.items():
            ids = np.asarray(splits[s].regime.astype(str) == regime)
            if ids.any():
                out[ids] = cal.predict(args.minimum_quality - lo[ids])
        return out
    dlo, dhi = bounds("decision")
    dp = probabilities("decision", dlo)
    margins = np.linspace(0, max(np.std(y["train"]), 1e-6), 31)
    def decide(lo, p, margin):
        return np.where((lo >= args.minimum_quality + margin) & (p <= args.risk), 0, np.where(lo <= args.minimum_quality - margin, 2, 1))
    costs = np.asarray(spec.get("decision_cost", [[0, 10], [1, 1], [3, 0]]))
    margin = min(margins, key=lambda m: costs[decide(dlo, dp, m), (y["decision"] < args.minimum_quality).astype(int)].mean())
    started = time.perf_counter()
    lo, hi = bounds("test")
    p = probabilities("test", lo)
    decisions = decide(lo, p, margin)
    latency = (time.perf_counter() - started) * 1000 / len(lo)
    fail = y["test"] < args.minimum_quality
    ece = sum(np.mean((p >= t) & (p < t + .1 + 1e-12)) * abs(p[(p >= t) & (p < t + .1 + 1e-12)].mean() - fail[(p >= t) & (p < t + .1 + 1e-12)].mean()) for t in np.arange(0, 1, .1) if np.any((p >= t) & (p < t + .1 + 1e-12)))
    violations = []
    for name, direction in spec.get("monotonic", {}).items():
        shifted = x["test"].copy()
        shifted[:, features.index(name)] += .05
        qlo, qhi = raw(shifted)
        blo, bhi = raw(x["test"])
        violations.extend((direction * ((qlo + qhi) - (blo + bhi)) < -1e-6).tolist())
    accepted = decisions == 0
    metrics = dict(coverage=float(np.mean((lo <= y["test"]) & (y["test"] <= hi))), interval_width=float(np.mean(hi - lo)), false_acceptance=float(fail[accepted].mean()) if accepted.any() else None, false_rejection=float(np.mean(decisions[~fail] == 2)) if (~fail).any() else None, ece=float(ece), brier=float(np.mean((p - fail) ** 2)), decision_cost=float(costs[decisions, fail.astype(int)].mean()), release_fraction=float(accepted.mean()), rule_count=rule_count, monotonicity_violation=float(np.mean(violations)) if violations else None, latency_ms=latency, margin=float(margin))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    models.update(center=center, scale=scale, correction=correction, regime_correction=regime_correction, calibrators=calibrators, global_cal=global_cal, margin=margin, spec=spec, arguments=vars(args))
    with (out / "model.pkl").open("wb") as handle:
        pickle.dump(models, handle)
    pd.DataFrame(dict(quality=y["test"], lower=lo, upper=hi, failure_probability=p, decision=decisions)).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    order = np.argsort(y["test"])
    plt.figure(figsize=(9, 4))
    plt.fill_between(np.arange(len(lo)), lo[order], hi[order], alpha=.3)
    plt.plot(y["test"][order], color="black", linewidth=.8)
    plt.axhline(args.minimum_quality, color="red", linestyle="--")
    plt.xlabel("Test settings ordered by measured quality")
    plt.ylabel("Quality")
    plt.tight_layout()
    plt.savefig(out / "process_window.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("spec")
    parser.add_argument("--output", default="outputs/full")
    parser.add_argument("--variant", choices=["full", "fixed", "anfis", "no_monotonicity", "no_calibration", "gp", "quantile"], default="full")
    parser.add_argument("--rules", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--risk", type=float, default=.05)
    parser.add_argument("--minimum-quality", type=float, required=True)
    parser.add_argument("--minimum-regime", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
