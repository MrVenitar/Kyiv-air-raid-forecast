"""Train, select and evaluate the alert-onset forecaster for one alert source.

    python src/run.py --source volunteer        # latest period (eTryvoga, oblast level)
    python src/run.py --source official         # official channel, main-city level, ends 2026-09-07

Splits (by time, with a 24 h gap so no label window crosses a boundary):
    train  [train_start, val_start - 24h)
    val    [val_start, test_start - 24h)        4 weeks: model choice, training window, warning thresholds
    test   [test_start, last_t]                 last 4 weeks with fully observed 24 h labels; used once
All models are fitted on train only; val makes every choice; test is scored at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data import load_alerts
from features import FEATURE_GROUPS, HORIZONS, LABEL_HISTORY, build_table, hourly_grid, label_history_features, recalibrate_online, tag_messages

ROOT = Path(__file__).resolve().parents[1]
DAY = pd.Timedelta("1D")
CAPS = (1, 3, 7)                     # allowed false warning-hours per location per week
SETS = {                             # ablation: feature groups added one at a time
    "calendar": ["calendar"],
    "+recent rates": ["calendar", "recent rates"],
    "+own history": ["calendar", "recent rates", "own"],
    "+spatial": ["calendar", "recent rates", "own", "spatial"],
    "+Air Force text": ["calendar", "recent rates", "own", "spatial", "text"],
}
FULL = "+Air Force text"
CATEGORICAL = ["loc_id", "dow"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ splits and baselines
def make_splits(last_t: pd.Timestamp, train_start: str, weeks: int = 4) -> dict:
    test_start = last_t - pd.Timedelta(weeks=weeks) + pd.Timedelta("1h")
    val_end = test_start - DAY                      # exclusive
    val_start = val_end - pd.Timedelta(weeks=weeks)
    train_end = val_start - DAY
    return {"train": (pd.Timestamp(train_start), train_end), "val": (val_start, val_end),
            "test": (test_start, last_t + pd.Timedelta("1h"))}


def in_split(df: pd.DataFrame, rng: tuple) -> pd.Series:
    return (df.t >= rng[0]) & (df.t < rng[1])


def seasonal_baseline(df: pd.DataFrame, h: int, days: int = 28) -> pd.Series:
    """P = share of the previous `days` days on which an onset followed the same local hour (same location).
    Uses y_H at t - k days (k >= 1); those labels are complete by t because H <= 24 h."""
    y = df[f"y_{h}h"]
    key = [df.location, df.hour_local]
    g = y.groupby(key)
    mean = g.transform(lambda s: s.shift(1).rolling(days, min_periods=7).mean())
    clim = y.groupby(df.location).transform(lambda s: s.shift(24 * 7).expanding(min_periods=24).mean())
    return mean.fillna(clim).fillna(y.mean()).clip(1e-4, 1 - 1e-4)


# ------------------------------------------------------------------ models
def _log1p_counts(X):
    return np.log1p(np.clip(X, 0, None))


def make_model(kind: str, feats: list, seed: int = 0):
    cat = [f for f in feats if f in CATEGORICAL]
    num = [f for f in feats if f not in CATEGORICAL]
    if kind == "logreg":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(_log1p_counts), StandardScaler()), num)])
        return make_pipeline(pre, LogisticRegression(C=0.5, max_iter=3000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
            categorical_features=[feats.index(c) for c in cat] or None, random_state=seed)
    raise ValueError(kind)


def fit_predict(kind, feats, tr, ev_frames, h):
    y = tr[f"y_{h}h"].to_numpy()
    m = make_model(kind, feats)
    m.fit(tr[feats], y)
    return m, [np.clip(m.predict_proba(e[feats])[:, 1], 1e-4, 1 - 1e-4) for e in ev_frames]


def scores(y, p, p_ref=None) -> dict:
    out = {"brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1]),
           "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
           "ap": average_precision_score(y, p) if y.sum() else np.nan, "base_rate": y.mean()}
    if p_ref is not None:
        out["bss_vs_seasonal"] = 1 - out["brier"] / brier_score_loss(y, p_ref)
    return out


# ------------------------------------------------------------------ operational table
def warning_threshold(p, y, n_loc_weeks, cap) -> float:
    """Lowest threshold whose false warning-hours (p > thr, no onset ahead) stay <= cap per location-week."""
    neg = np.sort(p[y == 0])[::-1]
    allowed = int(np.floor(cap * n_loc_weeks))
    return float(neg[allowed]) if allowed < len(neg) else 0.0


def onset_recall(df: pd.DataFrame, p: np.ndarray, h: int, thr: float, onsets: dict, rng: tuple) -> tuple:
    """Share of onsets in the period with a warning (p > thr) issued by a forecast whose window [t, t+H) contains it."""
    d = df.assign(p=p)
    hit = 0
    total = 0
    for loc, g in d.groupby("location"):
        s = pd.to_datetime(onsets[loc].start)
        s = s[(s >= rng[0] + pd.Timedelta(hours=h)) & (s < rng[1])]
        tt, pp = g.t.to_numpy(), g.p.to_numpy()
        for x in s.to_numpy():
            w = (tt > x - np.timedelta64(h, "h")) & (tt <= x) & (pp > thr)
            total += 1
            hit += bool(w.any())
    return (hit / total if total else np.nan), total


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["volunteer", "official"], required=True)
    ap.add_argument("--train-starts", nargs="+", default=["2025-01-01", "2025-12-01"],
                    help="candidate training-window starts, chosen on validation")
    ap.add_argument("--af", default=str(ROOT / "data" / "tg_kpszsu.csv.gz"), help="parsed Air Force channel")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or ROOT / "reports" / a.source)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    alerts = load_alerts(a.source)
    for n in alerts.notes:
        log(f"[data] {n}")
    tags = tag_messages(pd.read_csv(a.af, parse_dates=["published_at"]))
    last_t = pd.Timestamp(alerts.data_end).floor("h") - DAY            # every horizon's label observed
    splits = make_splits(last_t, min(a.train_starts))
    for k, (s, e) in splits.items():
        log(f"[split] {k:5s} {s} -> {e}")
    grid = hourly_grid(splits["train"][0], last_t)
    df = build_table(alerts, tags, grid)
    log(f"[table] {len(df):,} rows x {df.shape[1]} cols, {time.time() - t0:.0f}s")

    groups = dict(FEATURE_GROUPS)
    if a.source == "volunteer":            # oblast-any series IS the target series there
        groups["spatial"] = [f for f in groups["spatial"] if not f.startswith("obl_")]
    fsets = {name: [f for g in gs for f in groups[g]] for name, gs in SETS.items()}

    va_mask, te_mask = in_split(df, splits["val"]), in_split(df, splits["test"])
    ctx_val = in_split(df, (splits["val"][0] - pd.Timedelta("9D"), splits["val"][1]))
    ctx_test = in_split(df, (splits["test"][0] - pd.Timedelta("9D"), splits["test"][1]))
    metrics, chosen, op_rows, rel_rows, loc_rows = [], {}, [], [], []
    preds_test = df.loc[te_mask, ["location", "t"]].copy()
    n_loc_weeks_val = len(alerts.locations) * 4
    n_loc_weeks_test = len(alerts.locations) * 4

    for h in HORIZONS:
        df[LABEL_HISTORY] = label_history_features(df, h)      # per horizon, from finished labels only
        yv, yt = df.loc[va_mask, f"y_{h}h"].to_numpy(), df.loc[te_mask, f"y_{h}h"].to_numpy()
        base = seasonal_baseline(df, h)
        bv, bt = base[va_mask].to_numpy(), base[te_mask].to_numpy()
        for split, y, b in (("val", yv, bv), ("test", yt, bt)):
            metrics.append({"horizon_h": h, "split": split, "model": "baseline: seasonal 28d",
                            "features": "-", "train_start": "-", **scores(y, b, b)})
        preds_test[f"seasonal_{h}h"] = bt

        # --- 1. model family, training window and online recalibration (on/off), chosen on validation
        #        with the full feature set. Predictions cover 9 days of context before each split so the
        #        online recalibration has a past week of finished labels from the first hour on.
        def evaluate(kind, feats, tr):
            m = make_model(kind, feats).fit(tr[feats], tr[f"y_{h}h"].to_numpy())
            res = {}
            for split, emask, cmask in (("val", va_mask, ctx_val), ("test", te_mask, ctx_test)):
                ctx = df[cmask]
                raw = np.clip(m.predict_proba(ctx[feats])[:, 1], 1e-4, 1 - 1e-4)
                rec = recalibrate_online(ctx.t, raw, ctx[f"y_{h}h"].to_numpy(), h)
                keep = emask[cmask].to_numpy()
                res[split] = {False: raw[keep], True: rec[keep]}
            return res

        cand = []
        for ts in a.train_starts:
            tr = df[in_split(df, (pd.Timestamp(ts), splits["train"][1]))].dropna(subset=[f"y_{h}h"])
            for kind in ("logreg", "hgb"):
                res = evaluate(kind, fsets[FULL], tr)
                for rc in (False, True):
                    cand.append((log_loss(yv, res["val"][rc]), kind, ts, rc))
                    log(f"  [H={h:2d}h] val logloss {cand[-1][0]:.4f}  {kind:6s} train from {ts}  recal={rc}")
        _, kind, ts, rc = min(cand)
        chosen[h] = {"model": kind, "train_start": ts, "online_recalibration": rc}
        tr = df[in_split(df, (pd.Timestamp(ts), splits["train"][1]))].dropna(subset=[f"y_{h}h"])
        label = kind + (" + online recal" if rc else "")

        # --- 2. ablation: same model/window/recal, feature groups added one at a time; scored on val and test.
        #        The feature set used downstream is the one with the best VALIDATION log-loss.
        best = None
        for name, feats in fsets.items():
            res = evaluate(kind, feats, tr)
            for r_ in (False, True):
                for split, y, b in (("val", yv, bv), ("test", yt, bt)):
                    metrics.append({"horizon_h": h, "split": split, "model": kind + (" + online recal" if r_ else ""),
                                    "features": name, "train_start": ts, **scores(y, res[split][r_], b)})
            pv, pt = res["val"][rc], res["test"][rc]
            preds_test[f"{name}_{h}h"] = pt
            vl = log_loss(yv, pv)
            if best is None or vl < best[0]:
                best = (vl, name, pv, pt)
        _, chosen_set, full_pv, full_pt = best
        chosen[h]["features"] = chosen_set
        preds_test[f"selected_{h}h"] = full_pt
        for split, y, p, b in (("val", yv, full_pv, bv), ("test", yt, full_pt, bt)):
            metrics.append({"horizon_h": h, "split": split, "model": "selected on val", "features": f"selected: {chosen_set}",
                            "train_start": ts, **scores(y, p, b)})
        log(f"  [H={h:2d}h] chosen {label} from {ts}, features '{chosen_set}' (val); test BSS vs seasonal: "
            + ", ".join(f"{r['features']}={r['bss_vs_seasonal']:+.3f}" for r in metrics
                        if r["horizon_h"] == h and r["split"] == "test" and r["model"] == label))

        # --- 3. operational table: thresholds from VALIDATION, applied to test
        for cap in CAPS:
            thr = warning_threshold(full_pv, yv, n_loc_weeks_val, cap)
            fw = ((full_pt > thr) & (yt == 0)).sum() / n_loc_weeks_test
            rec, n_on = onset_recall(df[te_mask], full_pt, h, thr, alerts.target, splits["test"])
            rec_b, _ = onset_recall(df[te_mask], bt, h, warning_threshold(bv, yv, n_loc_weeks_val, cap),
                                            alerts.target, splits["test"])
            op_rows.append({"horizon_h": h, "cap_false_per_loc_week": cap, "threshold_from_val": thr,
                            "test_false_per_loc_week": fw, "onsets_warned": rec,
                            "onsets": n_on, "seasonal_onsets_warned": rec_b})

        # --- 4. reliability and per-location skill (full model, test)
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(full_pt, bins) - 1, 0, 9)
        for b_ in range(10):
            in_bin = idx == b_
            if in_bin.sum():
                rel_rows.append({"horizon_h": h, "bin": b_, "mean_p": full_pt[in_bin].mean(),
                                 "observed": yt[in_bin].mean(), "n": int(in_bin.sum())})
        te = df.loc[te_mask, ["location"]].assign(y=yt, p=full_pt, b=bt)
        for loc, g in te.groupby("location"):
            loc_rows.append({"horizon_h": h, "location": loc, "onset_hours": int(g.y.sum()),
                             "brier": brier_score_loss(g.y, g.p),
                             "bss_vs_seasonal": 1 - brier_score_loss(g.y, g.p) / max(brier_score_loss(g.y, g.b), 1e-12)})

    M = pd.DataFrame(metrics)
    M.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out / "operational.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "reliability.csv", index=False)
    pd.DataFrame(loc_rows).to_csv(out / "per_location.csv", index=False)
    preds_test.to_csv(out / "test_predictions.csv.gz", index=False, float_format="%.5f")
    (out / "run_info.json").write_text(json.dumps({
        "source": a.source, "data_end": str(alerts.data_end), "splits": {k: [str(x) for x in v] for k, v in splits.items()},
        "chosen": chosen, "notes": alerts.notes, "rows": len(df), "seconds": round(time.time() - t0)}, indent=2))
    log(f"\n[done] {a.source}: reports in {out} ({time.time() - t0:.0f}s)")
    sel_rows = [(h, c["model"] + (" + online recal" if c["online_recalibration"] else "")) for h, c in chosen.items()]
    keep = M.apply(lambda r: (r.horizon_h, r.model) in sel_rows, axis=1)
    T = M[(M.split == "test") & keep & M.features.isin(list(SETS))].pivot_table(index="features", columns="horizon_h", values="bss_vs_seasonal")
    log("\nTest Brier skill vs seasonal baseline (rows: features, cols: horizon h)\n" + T.round(3).to_string())
    log("\nOperational (test):\n" + pd.DataFrame(op_rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
