"""Features and labels for Kyiv commuters, every 15 minutes.

Information rule (tests in kyiv/test_kyiv_leakage.py): a feature at time t uses only
    * KMDA alert posts published before t (an alert whose all-clear is not yet posted is "running");
    * regional alert STARTS before t (volunteer data; end times are not used, ~30 % of them are guessed);
    * Telegram messages published before t.
Labels, for horizon h minutes:
    onset_h  (no Kyiv alert running at t): a new alert starts in [t, t+h)
    clear_h  (an alert running at t):      its all-clear is posted in [t, t+h)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kyiv"))

from clean_alerts import NEW_SYSTEM_UTC, build_episodes, level_changes  # noqa: E402

HORIZONS = (15, 30, 60, 120, 180)                  # minutes
MIN = np.int64(60_000_000_000)
FAR = np.iinfo(np.int64).max // 4                  # "end not known yet"
CHANNELS = ("af", "alertmap", "monitoring", "vanek")
SIGNALS = ("msg", "kyiv", "kyiv_obl", "toward_kyiv", "kyiv_boom", "all_clear",
           "ballistic", "mig31", "strategic", "missile", "jet", "drone", "launch")
TEXT_WINDOWS = (15, 60, 180)
REGIONS = {"kyiv_obl": ["Kyivska oblast"],
           "nbr": ["Chernihivska oblast", "Zhytomyrska oblast", "Cherkaska oblast", "Poltavska oblast",
                   "Sumska oblast", "Vinnytska oblast"]}
LEVEL_CODE = {"none": 0, "yellow": 1, "red": 2, "single": 3}


@dataclass
class Inputs:
    posts: pd.DataFrame        # KMDA alert posts: ts, kind, level
    regional: pd.DataFrame     # volunteer alerts: oblast, started_at
    signals: pd.DataFrame      # ts, channel, signal, n
    feature_end: pd.Timestamp  # last moment every input covers
    label_end: pd.Timestamp    # last KMDA post (labels known up to here)


def load_inputs(root: Path = ROOT) -> Inputs:
    from data import read_raw
    posts = pd.read_csv(root / "data" / "kyiv" / "kmda_alert_posts.csv", parse_dates=["ts"])
    reg = read_raw("volunteer")[["oblast", "started_at"]]
    reg = reg[reg.oblast != "Kyiv City"]
    sig = pd.read_csv(root / "data" / "kyiv" / "signals.csv.gz", parse_dates=["ts"])
    sig = sig[sig.channel.isin(CHANNELS)]
    ends = [sig[sig.channel == c].ts.max() for c in CHANNELS] + [pd.Timestamp(reg.started_at.max())]
    return Inputs(posts, reg, sig, min(ends), posts.ts.max())


def known_at(inp: Inputs, t) -> Inputs:
    """Everything a real-time system would have had at time t (strictly before t)."""
    t = pd.Timestamp(t)
    return Inputs(inp.posts[inp.posts.ts < t], inp.regional[inp.regional.started_at < t],
                  inp.signals[inp.signals.ts < t], t, t)


def grid(start, end) -> np.ndarray:
    return pd.date_range(pd.Timestamp(start).ceil("15min"), pd.Timestamp(end).floor("15min"), freq="15min").to_numpy("datetime64[ns]")


# ------------------------------------------------------------------ episodes as int64 arrays
def episode_arrays(posts: pd.DataFrame):
    ep, _ = build_episodes(posts)
    if len(ep) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    s = ep.start.to_numpy("datetime64[ns]").astype(np.int64)
    e = ep.end.to_numpy("datetime64[ns]").astype(np.int64)
    e = np.where(pd.isna(ep.end).to_numpy(), FAR, e)
    return s, e


def _count(times, t, w_min=None, weights=None):
    hi = np.searchsorted(times, t, side="left")
    lo = np.zeros_like(hi) if w_min is None else np.searchsorted(times, t - np.int64(w_min) * MIN, side="left")
    if weights is None:
        return hi - lo
    c = np.concatenate([[0], np.cumsum(weights)])
    return c[hi] - c[lo]


def _since(times, t, cap_min):
    k = np.searchsorted(times, t, side="left")
    last = np.where(k > 0, times[np.maximum(k - 1, 0)] if len(times) else 0, t - cap_min * MIN)
    return np.minimum((t - last) / MIN, cap_min)


def _occupied(s, e, x):
    """Alert time (ns) before x from episodes that started before x."""
    if len(s) == 0:
        return np.zeros(len(x), np.int64)
    dur = np.concatenate([[0], np.cumsum(np.minimum(e, s + 10**18) - s)])
    k = np.searchsorted(s, x, side="left")
    last = np.maximum(k - 1, 0)
    part = np.where(k > 0, np.minimum(e[last], x) - s[last], 0)
    return np.where(k > 0, dur[last] + part, 0)


# ------------------------------------------------------------------ features (not horizon-specific)
def base_features(inp: Inputs, g: np.ndarray) -> pd.DataFrame:
    t = g.astype(np.int64)
    s, e = episode_arrays(inp.posts)
    k = np.searchsorted(s, t, side="left")               # episodes started before t
    last = np.maximum(k - 1, 0)
    has = k > 0
    active = has & (e[last] >= t) if len(s) else np.zeros(len(t), bool)
    f = {"t": g, "active": active.astype(np.int8)}
    f["elapsed_min"] = np.where(active, (t - s[last]) / MIN, -1.0) if len(s) else -np.ones(len(t))

    lv = level_changes(inp.posts) if len(inp.posts) else pd.DataFrame(columns=["ts", "level"])
    lt = lv.ts.to_numpy("datetime64[ns]").astype(np.int64)
    lcode = lv.level.map(LEVEL_CODE).to_numpy() if len(lv) else np.array([], int)
    j = np.searchsorted(lt, t, side="left")
    f["level_now"] = np.where(active & (j > 0), lcode[np.maximum(j - 1, 0)] if len(lt) else 0, 0)
    f["level_changes"] = np.where(active, j - np.searchsorted(lt, s[last], side="left") - 1, 0) if len(s) else 0
    f["two_level_system"] = (g >= np.datetime64(NEW_SYSTEM_UTC)).astype(np.int8)

    ends_done = np.sort(e[e < FAR])
    f["since_end_min"] = _since(ends_done, t, 7 * 1440)
    f["since_onset_min"] = _since(s, t, 7 * 1440)
    for w in (60, 180, 360, 1440, 7 * 1440):
        f[f"onsets_{w}m"] = _count(s, t, w)
    f["alert_min_24h"] = (_occupied(s, e, t) - _occupied(s, e, t - 1440 * MIN)) / MIN
    # durations of finished episodes (only those whose all-clear was posted before t)
    order = np.argsort(e, kind="stable")
    es, ss = e[order], s[order]
    fin = es < FAR
    es, ss = es[fin], ss[fin]
    q = np.searchsorted(es, t, side="left")
    durs = (es - ss) / MIN
    f["last_dur_min"] = np.where(q > 0, durs[np.maximum(q - 1, 0)] if len(durs) else 0, -1.0)
    q7 = np.searchsorted(es, t - 7 * 1440 * MIN, side="left")
    f["med_dur_7d"] = np.array([np.median(durs[a:b]) if b > a else -1.0 for a, b in zip(q7, q)])

    local = pd.DatetimeIndex(g).tz_localize("UTC").tz_convert("Europe/Kyiv")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    f["hour_local"] = hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = local.dayofweek.to_numpy()

    reg = inp.regional
    for name, obl in REGIONS.items():
        ts = np.sort(reg[reg.oblast.isin(obl)].started_at.to_numpy("datetime64[ns]").astype(np.int64))
        for w in (30, 60, 180):
            f[f"reg_{name}_onsets_{w}m"] = _count(ts, t, w)
        f[f"reg_{name}_since_min"] = _since(ts, t, 1440)
    ts = np.sort(reg.started_at.to_numpy("datetime64[ns]").astype(np.int64))
    for w in (60, 180):
        f[f"reg_nat_onsets_{w}m"] = _count(ts, t, w)

    sig = inp.signals
    for ch in CHANNELS:
        sc = sig[sig.channel == ch]
        for sn in SIGNALS:
            x = sc[sc.signal == sn]
            ts = x.ts.to_numpy("datetime64[ns]").astype(np.int64)
            n = x.n.to_numpy(np.int64)
            for w in TEXT_WINDOWS:
                f[f"{ch}_{sn}_{w}m"] = _count(ts, t, w, n)
            f[f"{ch}_{sn}_since_min"] = _since(ts, t, 1440)
    return pd.DataFrame(f)


FEATURE_GROUPS = {
    "kyiv history": ["elapsed_min", "level_now", "level_changes", "since_end_min", "since_onset_min",
                     "onsets_60m", "onsets_180m", "onsets_360m", "onsets_1440m", "onsets_10080m", "alert_min_24h",
                     "last_dur_min", "med_dur_7d", "hour_local", "hour_sin", "hour_cos", "dow", "base", "rate7"],
    "regional alerts": [f"reg_{n}_{s}" for n in REGIONS for s in ("onsets_30m", "onsets_60m", "onsets_180m", "since_min")]
                       + ["reg_nat_onsets_60m", "reg_nat_onsets_180m"],
}
for _ch in CHANNELS:
    FEATURE_GROUPS[_ch] = [f"{_ch}_{sn}_{w}m" for sn in SIGNALS for w in TEXT_WINDOWS] + [f"{_ch}_{sn}_since_min" for sn in SIGNALS]
ONSET_ONLY = {"elapsed_min", "level_now", "level_changes"}           # meaningless when idle
CLEAR_ONLY = {"since_end_min"}                                        # meaningless during an alert


# ------------------------------------------------------------------ horizon-specific baselines
def onset_baseline(inp: Inputs, g: np.ndarray, h: int, days: int = 28, prior_m: float = 10.0):
    """(seasonal, recent) onset rate for horizon h at each t, from the past only.
    seasonal: same time of day ±15 min on each of the previous `days` days (idle reference times);
    recent:   every hour over the last 7 days ending h minutes before t (labels finished by t)."""
    t = g.astype(np.int64)
    s, e = episode_arrays(inp.posts)

    def rate(refs):
        flat = refs.ravel()
        k = np.searchsorted(s, flat, side="left")
        idle = ~((k > 0) & (e[np.maximum(k - 1, 0)] >= flat)) if len(s) else np.ones(len(flat), bool)
        nxt = s[np.minimum(k, len(s) - 1)] if len(s) else np.full(len(flat), FAR)
        hit = (k < len(s)) & (nxt < flat + h * MIN)
        idle, hit = idle.reshape(refs.shape), (hit & idle).reshape(refs.shape)
        return hit.sum(1), idle.sum(1)

    day = 1440 * MIN
    offs = np.array([-k * day + j * 15 * MIN for k in range(1, days + 1) for j in (-1, 0, 1)])
    hit_s, n_s = rate(t[:, None] + offs[None, :])
    offs7 = -(np.int64(h) * MIN + np.arange(0, 7 * 24) * 60 * MIN)
    hit_r, n_r = rate(t[:, None] + offs7[None, :])
    recent = (hit_r + 1) / (n_r + 2)
    seasonal = (hit_s + prior_m * recent) / (n_s + prior_m)
    return seasonal, recent


def clear_baseline(inp: Inputs, g: np.ndarray, elapsed_min: np.ndarray, h: int):
    """P(all-clear within h | alert has lasted `elapsed`), from alerts finished before t in the last 28 days
    (widened to 90 / 365 days when fewer than 8 of them lasted at least that long). NaN when idle."""
    t = g.astype(np.int64)
    s, e = episode_arrays(inp.posts)
    fin = e < FAR
    order = np.argsort(e[fin], kind="stable")
    es, ss = e[fin][order], s[fin][order]
    dur = (es - ss) / MIN
    out = np.full(len(t), np.nan)
    hi = np.searchsorted(es, t, side="left")
    for i in np.flatnonzero(elapsed_min >= 0):
        el = elapsed_min[i]
        for days in (28, 90, 365):
            lo = np.searchsorted(es, t[i] - days * 1440 * MIN, side="left")
            d = dur[lo:hi[i]]
            at_risk = d >= el
            if at_risk.sum() >= 8 or days == 365:
                n = at_risk.sum()
                out[i] = ((d[at_risk] < el + h).sum() + 0.5) / (n + 1.0)
                break
    return out


# ------------------------------------------------------------------ labels
def labels(inp: Inputs, g: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(onset_h, clear_h) with NaN where the window is not fully observed or the row is the other state."""
    t = g.astype(np.int64)
    s, e = episode_arrays(inp.posts)
    k = np.searchsorted(s, t, side="left")
    last = np.maximum(k - 1, 0)
    active = (k > 0) & (e[last] >= t)
    nxt = s[np.minimum(k, len(s) - 1)]
    onset = ((k < len(s)) & (nxt < t + h * MIN)).astype(float)
    clear = ((e[last] >= t) & (e[last] < t + h * MIN)).astype(float)
    observed = t + h * MIN <= np.int64(pd.Timestamp(inp.label_end).value)
    onset[active | ~observed] = np.nan
    clear[~active | ~observed] = np.nan
    return onset, clear


def stacked(base: pd.DataFrame, inp: Inputs, task: str, horizons=HORIZONS, with_labels=True) -> pd.DataFrame:
    """One row per (time, horizon) for one task ('onset' = idle rows, 'clear' = rows during an alert)."""
    g = base.t.to_numpy("datetime64[ns]")
    rows = []
    for h in horizons:
        if task == "onset":
            m = base.active.to_numpy() == 0
            b, r = onset_baseline(inp, g[m], h)
        else:
            m = base.active.to_numpy() == 1
            b = clear_baseline(inp, g[m], base.elapsed_min.to_numpy()[m], h)
            r = np.full(m.sum(), np.nan)
        part = base[m].copy()
        part["h"] = h
        part["base"], part["rate7"] = b, r
        if with_labels:
            on, cl = labels(inp, g[m], h)
            part["y"] = on if task == "onset" else cl
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["y"]) if with_labels else out
