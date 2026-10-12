"""Train and evaluate the Kyiv commuter forecaster (two models: alert onset, all-clear).

    python kyiv/train.py --stage tables     # build and cache the 15-min tables (~3 min)
    python kyiv/train.py --stage validate   # model + feature choices on the validation weeks
    python kyiv/train.py --stage test       # run ONCE: frozen choices on the test weeks + scenarios
    python kyiv/train.py --stage final      # fit on all data, save models/ for kyiv/forecast.py

Evaluation = weekly rolling origin: for every week (starting Monday 00:00 Kyiv), models are fitted on all rows
whose label window ended before the week started, then predict that week every 15 minutes.
    validation weeks: 24 Aug – 20 Sep 2026 (4)   -> every choice is made here
    test weeks:       21 Sep – 11 Oct 2026 (3)   -> scored once, after the choices were written to disk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kyiv"))
import features as F  # noqa: E402

TRAIN_START = "2025-01-01"
VAL_WEEKS = pd.date_range("2026-08-23 21:00", periods=4, freq="7D")      # Monday 00:00 Kyiv (UTC+3)
TEST_WEEKS = pd.date_range("2026-09-20 21:00", periods=3, freq="7D")
CACHE = ROOT / "data" / "kyiv" / "cache"
REP = ROOT / "reports" / "kyiv"
MODELS = ROOT / "models"
TASKS = ("onset", "clear")
CONFIGS = [("logreg", None), ("logreg", 45), ("logreg", 14), ("hgb", 45), ("hgb", 14)]   # (model, recency half-life, days)
ABLATION = [("Kyiv history", ["kyiv history"]),
            ("+ regional alerts", ["kyiv history", "regional alerts"]),
            ("+ Air Force", ["kyiv history", "regional alerts", "af"]),
            ("+ alert map", ["kyiv history", "regional alerts", "af", "alertmap"]),
            ("+ Monitoring UA", ["kyiv history", "regional alerts", "af", "alertmap", "monitoring"]),
            ("+ Mykolaiv Vanek", ["kyiv history", "regional alerts", "af", "alertmap", "monitoring", "vanek"])]
SEED = 0


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ tables
def build_tables():
    t0 = time.time()
    inp = F.load_inputs()
    g = F.grid(TRAIN_START, inp.feature_end)
    base = F.base_features(inp, g)
    log(f"[tables] grid {len(g):,} times {g[0]} .. {g[-1]}, features {base.shape[1]} ({time.time() - t0:.0f}s)")
    CACHE.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        tab = F.stacked(base, inp, task)
        tab.to_pickle(CACHE / f"{task}.pkl")
        log(f"[tables] {task}: {len(tab):,} rows, positive share by h {tab.groupby('h').y.mean().round(3).to_dict()} ({time.time() - t0:.0f}s)")
    json.dump({"feature_end": str(inp.feature_end), "label_end": str(inp.label_end)}, open(CACHE / "meta.json", "w"))


def load_table(task):
    tab = pd.read_pickle(CACHE / f"{task}.pkl")
    return tab, json.load(open(CACHE / "meta.json"))


# ------------------------------------------------------------------ models
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(tab: pd.DataFrame, task: str, groups: list) -> tuple[pd.DataFrame, list]:
    cols = [c for gname in groups for c in F.FEATURE_GROUPS[gname]]
    drop = F.ONSET_ONLY if task == "onset" else F.CLEAR_ONLY
    cols = [c for c in dict.fromkeys(cols) if c not in drop and not (task == "clear" and c == "rate7")]
    X = tab[cols].copy().copy()
    X["log_h"] = np.log(tab.h.to_numpy())
    X["base"] = _logit(tab.base.to_numpy())
    if "rate7" in X:
        X["rate7"] = _logit(tab.rate7.to_numpy())
    return X, list(X.columns)


from model_utils import clip5 as _clip5, signed_log as _signed_log  # noqa: E402  (importable for pickles)


def make_model(kind: str, cols: list):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(_signed_log), StandardScaler(),
                             FunctionTransformer(_clip5), LogisticRegression(C=0.3, max_iter=3000))
    mono = [1 if c == "log_h" else 0 for c in cols]                  # longer horizon -> never lower probability
    return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=250, max_leaf_nodes=31, min_samples_leaf=100,
                                          l2_regularization=1.0, monotonic_cst=mono, random_state=SEED)


def weights(t: pd.Series, origin: pd.Timestamp, half_life):
    if not half_life:
        return None
    age = (origin - t).dt.total_seconds() / 86400
    return np.power(0.5, age / half_life).to_numpy()


def train_rows(tab, origin):
    end_ok = tab.t + pd.to_timedelta(tab.h, unit="min") <= origin       # label finished before the week starts
    return tab[end_ok & (tab.t < origin)]


def fit(kind, half_life, tab_tr, task, groups, origin):
    X, cols = design(tab_tr, task, groups)
    m = make_model(kind, cols)
    w = weights(tab_tr.t, origin, half_life)
    if kind == "logreg":
        m.fit(X, tab_tr.y.to_numpy(), logisticregression__sample_weight=w)
    else:
        m.fit(X, tab_tr.y.to_numpy(), sample_weight=w)
    return m, cols


def predict(m, tab, task, groups):
    X, _ = design(tab, task, groups)
    return np.clip(m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)


def rolling(tab, task, weeks, kind, half_life, groups, end):
    """Out-of-sample predictions for each week from a model fitted on everything before it."""
    out = []
    for i, origin in enumerate(weeks):
        stop = weeks[i + 1] if i + 1 < len(weeks) else pd.Timestamp(end)
        wk = tab[(tab.t >= origin) & (tab.t < stop)]
        m, _ = fit(kind, half_life, train_rows(tab, origin), task, groups, origin)
        out.append(wk.assign(p=predict(m, wk, task, groups), week=str(origin.date())).copy())
    return pd.concat(out)


def recalibrate(pr: pd.DataFrame, days: int = 7, min_rows: int = 50) -> np.ndarray:
    """Shift each prediction's log-odds by logit(observed) - logit(predicted) over rows whose label window
    ended before t (t' + h <= t) within the last `days` days. Uses out-of-sample predictions only."""
    out = pr.p.to_numpy().copy()
    for h, g in pr.groupby("h"):
        g = g.sort_values("t")
        tt = g.t.to_numpy("datetime64[ns]").astype(np.int64)
        cy = np.concatenate([[0], np.cumsum(g.y.to_numpy())])
        cp = np.concatenate([[0], np.cumsum(g.p.to_numpy())])
        hi = np.searchsorted(tt, tt - np.int64(h) * 60_000_000_000, side="right")       # t' <= t - h
        lo = np.searchsorted(tt, tt - np.int64(days) * 86_400_000_000_000, side="left")
        n = hi - lo
        oy = (cy[hi] - cy[lo] + 0.5) / (n + 1.0)
        op = (cp[hi] - cp[lo] + 0.5) / (n + 1.0)
        delta = np.where(n >= min_rows, _logit(oy) - _logit(op), 0.0)
        idx = pr.index.get_indexer(g.index)
        out[idx] = 1 / (1 + np.exp(-(_logit(g.p.to_numpy()) + delta)))
    return np.clip(out, 1e-4, 1 - 1e-4)


def with_recal(tab, task, weeks, kind, half_life, groups, end):
    """Rolling predictions for `weeks` plus one warm-up week before them (context for recalibration only)."""
    warm = pd.DatetimeIndex([weeks[0] - pd.Timedelta("7D")]).append(weeks)
    pr = rolling(tab, task, warm, kind, half_life, groups, end).reset_index(drop=True)
    pr["p_raw"] = pr.p
    pr["p_recal"] = recalibrate(pr)
    return pr[pr.t >= weeks[0]].reset_index(drop=True)


def scores(df, col="p"):
    rows = []
    for h, g in df.groupby("h"):
        y, p = g.y.to_numpy(), g[col].to_numpy()
        rows.append({"h": h, "n": len(g), "rate": y.mean(), "logloss": log_loss(y, p, labels=[0, 1]),
                     "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan})
    return pd.DataFrame(rows)


def summary_logloss(df, col="p"):
    return scores(df, col).logloss.mean()


# ------------------------------------------------------------------ stages
def stage_validate(tasks=TASKS):
    REP.mkdir(parents=True, exist_ok=True)
    choice, lines = {}, ["# Validation (24 Aug – 20 Sep 2026, weekly refits)\n"]
    all_groups = ABLATION[-1][1]
    for task in tasks:
        tab, meta = load_table(task)
        end = VAL_WEEKS[-1] + pd.Timedelta("7D")
        base_ll = summary_logloss(tab[(tab.t >= VAL_WEEKS[0]) & (tab.t < end)], "base")
        lines.append(f"\n## {task}\n\nBaseline mean log-loss over horizons: {base_ll:.4f}\n")
        res = []
        for kind, hl in CONFIGS:
            t0 = time.time()
            pr = with_recal(tab, task, VAL_WEEKS, kind, hl, all_groups, end)
            for rc in (False, True):
                ll = summary_logloss(pr, "p_recal" if rc else "p_raw")
                res.append({"model": kind, "half_life_days": hl or "-", "recalibration": rc, "mean_logloss": ll})
                log(f"[validate] {task} {kind:6s} hl={hl} recal={rc}: mean log-loss {ll:.4f} (baseline {base_ll:.4f}) {time.time() - t0:.0f}s")
        R = pd.DataFrame(res)
        best = R.loc[R.mean_logloss.idxmin()]
        kind, hl, rc = best.model, (None if best.half_life_days == "-" else int(best.half_life_days)), bool(best.recalibration)
        lines.append("Model choice (all features):\n\n" + R.round(4).to_markdown(index=False))
        abl = []
        for name, groups in ABLATION:
            pr = with_recal(tab, task, VAL_WEEKS, kind, hl, groups, end)
            ll = summary_logloss(pr, "p_recal" if rc else "p_raw")
            abl.append({"features": name, "mean_logloss": ll, "vs_baseline": ll - base_ll})
            log(f"[validate] {task} {kind} ablation {name:18s}: {ll:.4f}")
        abl.append({"features": "baseline only (no model)", "mean_logloss": base_ll, "vs_baseline": 0.0})
        A = pd.DataFrame(abl)
        sel = A.loc[A.mean_logloss.idxmin(), "features"]
        lines.append(f"\nFeature groups added in turn ({kind}, half-life {hl or 'none'}, recalibration {rc}):\n\n" + A.round(4).to_markdown(index=False))
        lines.append(f"\n**Chosen for {task}: {kind}, half-life {hl or 'none'}, recalibration {rc}, features '{sel}'.**")
        if sel == "baseline only (no model)":
            choice[task] = {"model": "baseline", "half_life": None, "recalibration": False, "features": sel, "groups": []}
        else:
            choice[task] = {"model": kind, "half_life": hl, "recalibration": rc, "features": sel, "groups": dict(ABLATION)[sel]}
    for task in tasks:
        json.dump(choice[task], open(REP / f"choice_{task}.json", "w"), indent=2)
        (REP / f"validation_{task}.md").write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))


def scenario_tables(val, test, task):
    """Decision views for the two commuter scenarios (thresholds set on validation)."""
    out = []
    if task == "onset":                       # Livoberezhna: go now if P(alert within 30 min) is low
        v, t = val[val.h == 30], test[test.h == 30]
        for budget in (0.10, 0.20):
            for col, name in (("p", "model"), ("base", "baseline")):
                order = np.argsort(v[col].to_numpy(), kind="stable")          # lowest risk first
                pv, yv = v[col].to_numpy()[order], v.y.to_numpy()[order]
                caught = np.cumsum(yv) / np.arange(1, len(yv) + 1)            # caught share if 'go' up to here
                ok = np.flatnonzero(caught <= budget)
                thr = float(pv[ok[-1]]) if len(ok) else 0.0
                go_t = t[col] <= thr
                out.append({"scenario": "Livoberezhna: cross now if P(alert within 30 min) ≤ threshold",
                            "risk budget": f"≤{budget:.0%} caught", "forecast": name, "threshold (val)": round(float(thr), 3),
                            "share of times 'go' (test)": f"{go_t.mean():.1%}",
                            "caught by an alert when 'go' (test)": f"{t.y[go_t].mean():.1%}" if go_t.any() else "n/a"})
    else:                                     # Arsenalna: wait for the all-clear if P(all-clear within 30 min) >= 0.5
        t = test[test.h == 30]
        for subset, mask in (("all alerts", np.ones(len(t), bool)),
                             ("red level (two-level system)", (t.level_now == 2).to_numpy())):
            tt = t[mask]
            if len(tt) < 5:
                continue
            for col, name in (("p", "model"), ("base", "baseline")):
                wait = tt[col] >= 0.5
                out.append({"scenario": f"Arsenalna, {subset}: wait if P(all-clear within 30 min) ≥ 0.5",
                            "forecast": name, "rows (15-min)": len(tt), "right decision": f"{(wait == (tt.y == 1)).mean():.1%}",
                            "told to wait, all-clear came": f"{tt.y[wait].mean():.1%}" if wait.any() else "n/a",
                            "told to go, alert lasted >30 min": f"{(1 - tt.y[~wait]).mean():.1%}" if (~wait).any() else "n/a"})
    return pd.DataFrame(out)


def stage_test(tasks=TASKS):
    choice = {t: json.load(open(REP / f"choice_{t}.json")) for t in tasks}
    lines = ["# Test (21 Sep – 11 Oct 2026, weekly refits; choices frozen in choice_<task>.json before this run)\n"]
    for task in tasks:
        tab, meta = load_table(task)
        c = choice[task]
        col = "p_recal" if c["recalibration"] else "p_raw"
        val = with_recal(tab, task, VAL_WEEKS, c["model"], c["half_life"], c["groups"], VAL_WEEKS[-1] + pd.Timedelta("7D"))
        te = with_recal(tab, task, TEST_WEEKS, c["model"], c["half_life"], c["groups"], meta["feature_end"])
        val["p"], te["p"] = val[col], te[col]
        te[["t", "h", "y", "p", "p_raw", "p_recal", "base", "level_now", "week"]].to_csv(
            REP / f"test_pred_{task}.csv.gz", index=False, float_format="%.5f", date_format="%Y-%m-%d %H:%M:%S")
        S, B = scores(te), scores(te, "base")
        T = S.merge(B[["h", "logloss", "brier", "auc"]], on="h", suffixes=("_model", "_baseline"))
        T["brier_skill"] = 1 - T.brier_model / T.brier_baseline
        lines.append(f"\n## {task} ({c['model']}, half-life {c['half_life'] or 'none'}, recalibration {c['recalibration']}, "
                     f"features '{c['features']}')\n")
        lines.append(T.round(3).to_markdown(index=False))
        abl = []
        for name, groups in ABLATION:            # reported for information only; nothing is chosen here
            pr = with_recal(tab, task, TEST_WEEKS, c["model"], c["half_life"], groups, meta["feature_end"])
            pr["p"] = pr[col]
            s = scores(pr)
            abl.append({"features": name, **{f"skill {h}m": 1 - r.brier / bb for h, r, bb in
                                              zip(s.h, s.itertuples(), B.brier)}, "mean_logloss": s.logloss.mean()})
            log(f"[test] {task} ablation {name}: {s.logloss.mean():.4f}")
        lines.append("\nAblation on test (information only; Brier skill vs baseline per horizon):\n")
        lines.append(pd.DataFrame(abl).round(3).to_markdown(index=False))
        sc = scenario_tables(val, te, task)
        lines.append("\nScenario view (test):\n")
        lines.append(sc.to_markdown(index=False))
        # by system level (clear task)
        if task == "clear":
            te30 = te[te.h == 30]
            by = te30.groupby("level_now").apply(lambda g: pd.Series({
                "rows": len(g), "all-clear within 30 min": g.y.mean(), "model brier": brier_score_loss(g.y, g.p),
                "baseline brier": brier_score_loss(g.y, g.base)}), include_groups=False)
            by.index = by.index.map({1: "yellow", 2: "red", 3: "single"})
            lines.append("\nAll-clear within 30 min by alert level (test):\n\n" + by.round(3).to_markdown())
    name = "test.md" if len(tasks) == len(TASKS) else f"test_{tasks[0]}.md"
    (REP / name).write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))


def stage_final():
    import joblib
    choice = {t: json.load(open(REP / f"choice_{t}.json")) for t in TASKS}
    MODELS.mkdir(parents=True, exist_ok=True)
    info = {}
    for task in TASKS:
        tab, meta = load_table(task)
        c = choice[task]
        origin = pd.Timestamp(meta["label_end"])
        tr = train_rows(tab, origin)
        if c["model"] == "baseline":
            joblib.dump({"model": None, "columns": [], "groups": [], "task": task, "choice": c, "deltas": {},
                         "trained_until": str(tr.t.max()), "rows": 0}, MODELS / f"kyiv_{task}.joblib")
            info[task] = {**c, "trained_until": str(tr.t.max()), "rows": 0}
            log(f"[final] {task}: baseline only (validation found no model better than it)")
            continue
        m, cols = fit(c["model"], c["half_life"], tr, task, c["groups"], origin)
        deltas = {}
        if c["recalibration"]:          # offsets for live use: out-of-sample over the last 7 days of data
            ctx_origin = pd.Timestamp(meta["feature_end"]).floor("15min") - pd.Timedelta("7D")
            m_ctx, _ = fit(c["model"], c["half_life"], train_rows(tab, ctx_origin), task, c["groups"], ctx_origin)
            ctx = tab[(tab.t >= ctx_origin) & (tab.t + pd.to_timedelta(tab.h, unit="min") <= origin)].copy()
            ctx["p"] = predict(m_ctx, ctx, task, c["groups"])
            for h, g in ctx.groupby("h"):
                deltas[int(h)] = float(_logit((g.y.sum() + 0.5) / (len(g) + 1)) - _logit((g.p.sum() + 0.5) / (len(g) + 1)))
        joblib.dump({"model": m, "columns": cols, "groups": c["groups"], "task": task, "choice": c, "deltas": deltas,
                     "trained_until": str(tr.t.max()), "rows": len(tr)}, MODELS / f"kyiv_{task}.joblib")
        info[task] = {**c, "trained_until": str(tr.t.max()), "rows": len(tr), "recalibration_offsets": deltas}
        log(f"[final] {task}: {c['model']} on {len(tr):,} rows until {tr.t.max()}, offsets {deltas} -> models/kyiv_{task}.joblib")
    json.dump(info, open(MODELS / "kyiv_models.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["tables", "validate", "test", "final"], required=True)
    ap.add_argument("--task", choices=list(TASKS), default=None, help="validate/test one task only (to run both in parallel)")
    a = ap.parse_args()
    if a.stage == "validate":
        stage_validate([a.task] if a.task else TASKS)
    elif a.stage == "test":
        stage_test([a.task] if a.task else TASKS)
    else:
        {"tables": build_tables, "test": stage_test, "final": stage_final}[a.stage]()
