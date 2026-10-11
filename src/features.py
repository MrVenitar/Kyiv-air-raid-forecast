"""Hourly forecast rows: features known at forecast time t, and labels for each horizon.

Information rule (checked by tests/test_leakage.py):
    a feature at time t may use only alerts that STARTED before t (a running alert's end is unknown)
    and Air Force messages PUBLISHED before t.
Label for horizon H: a new alert episode (onset) at the location starts in [t, t+H).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data import NEIGHBOURS, AlertData, Episodes

HORIZONS = (1, 3, 6, 12, 24)                       # hours
H_NS = np.int64(3_600_000_000_000)
FAR = np.datetime64("2262-01-01")

# Air Force channel: how each location's region is written (any grammatical case)
REGION_PATTERNS = {
    "Cherkaska oblast": r"Черкас",
    "Chernihivska oblast": r"Чернігів|Ніжин|Прилук",
    "Chernivetska oblast": r"Чернівц|Чернівеч|Буковин",
    "Dnipropetrovska oblast": r"Дніпропетров|Дніпр[оау]\b|Дніпром\b|Кривий Ріг|Кривор|Павлоград|Нікопол|Кам.янськ",
    "Donetska oblast": r"Донеч|Донецьк|Краматорськ|Слов.янськ|Покровськ|Костянтинівк|Дружківк",
    "Ivano-Frankivska oblast": r"Івано-Франків|Прикарпат",
    "Kharkivska oblast": r"Харків|Ізюм|Куп.янськ|Чугу[єї]в|Лозов",
    "Khersonska oblast": r"Херсон",
    "Khmelnytska oblast": r"Хмельниц|Хмельнич|Старокостянтин|Шепетів|Кам.янець",
    "Kirovohradska oblast": r"Кіровоград|Кропивниц|Олександрі[їй]",
    "Kyiv City": r"Києв|Київ(?!щ|ськ)|столиц",
    "Kyivska oblast": r"Київщ|Київськ|Біла Церкв|Білоцерк|Бровар|Бориспол|Обухів|Фастів|Вишгород|Буч[аі]",
    "Lvivska oblast": r"Львів|Стрий|Дрогобич|Червоноград|Шептицьк|Золочів|Бродів|Броди",
    "Mykolaivska oblast": r"Миколаїв|Вознесенськ|Первомайськ",
    "Odeska oblast": r"Одес|Ізмаїл|Білгород-Дністров|Подільськ",
    "Poltavska oblast": r"Полтав|Кременчу|Миргород|Лубн",
    "Rivnenska oblast": r"\bРівн|Сарн|Дубн|Вараш",
    "Sumska oblast": r"Сумщ|\bСум[иаіу]\b|Сумськ|Шостк|Конотоп|Охтир|Ромн",
    "Ternopilska oblast": r"Терноп",
    "Vinnytska oblast": r"Вінниц|Вінничч|Калинівк|Жмерин",
    "Volynska oblast": r"Волин|Луцьк|Ковел",
    "Zakarpatska oblast": r"Закарпат|Ужгород|Мукачев",
    "Zaporizka oblast": r"Запоріж",
    "Zhytomyrska oblast": r"Житомир|Коростен|Бердичів|Звягел",
}
THREAT_PATTERNS = {
    "drone": r"БпЛА|[Шш]ахед|дрон",
    "ballistic": r"[Бб]алістик|Іскандер|КН-23",
    "missile": r"[Рр]акет|[Кк]рилат|Калібр|Х-101|Х-59|Х-22",
    "kab": r"\bКАБ",
    "mig31": r"МіГ-31|Кинджал",
    "strategic": r"Ту-95|Ту-160|Ту-22|[Сс]тратегічн",
    "jet_drone": r"🏍|[Рр]еактивн",
    "all_clear": r"[Вв]ідбій",
}
NORM_H = 24 * 28        # text counts are divided by their own trailing-28-day level (posting volume drifts)

FEATURE_GROUPS = {
    "calendar": ["loc_id", "hour_local", "hour_sin", "hour_cos", "dow"],
    "own": ["own_active", "own_since_onset_h", "own_since_end_h", "own_onsets_1h", "own_onsets_3h",
            "own_onsets_6h", "own_onsets_24h", "own_onsets_168h", "own_alert_min_24h", "own_alert_min_168h"],
    "spatial": ["obl_active", "obl_onsets_1h", "obl_onsets_3h", "obl_alert_min_24h",
                "nbr_active_frac", "nbr_onsets_1h", "nbr_onsets_3h", "nbr_onsets_6h",
                "nat_active_frac", "nat_onsets_1h", "nat_onsets_3h", "nat_onsets_6h", "nat_onsets_24h"],
    "recent rates": ["rate_hour_28d", "rate_7d", "rate_28d", "nat_rate_7d"],
    "text": ["af_region_1h_rel", "af_region_3h_rel", "af_region_6h_rel", "af_since_region_h", "af_nbr_3h_rel",
             "af_all_1h_rel", "af_all_24h_rel"]
            + [f"af_{k}_{w}h_rel" for k in THREAT_PATTERNS for w in (1, 6)],
}
LABEL_HISTORY = FEATURE_GROUPS["recent rates"]
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
TABLE_FEATURES = [f for f in ALL_FEATURES if f not in LABEL_HISTORY]   # built by build_table


# ------------------------------------------------------------------ helpers on episode arrays
def _i64(x) -> np.ndarray:
    return np.asarray(x, dtype="datetime64[ns]").astype(np.int64)


def _count_before(times: np.ndarray, t: np.ndarray, w_h: float | None = None) -> np.ndarray:
    """Number of event times in [t - w, t) (or (-inf, t) if w is None)."""
    hi = np.searchsorted(times, t, side="left")
    if w_h is None:
        return hi
    return hi - np.searchsorted(times, t - np.int64(w_h * H_NS), side="left")


def _occupied_before(s: np.ndarray, e: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Total alert time (ns) in (-inf, x), using only episodes that started before x."""
    if len(s) == 0:
        return np.zeros(len(x), np.int64)
    dur_cum = np.concatenate([[0], np.cumsum(e - s)])
    k = np.searchsorted(s, x, side="left")              # episodes started before x
    last = np.maximum(k - 1, 0)
    part = np.where(k > 0, np.minimum(e[last], x) - s[last], 0)
    return np.where(k > 0, dur_cum[last] + part, 0)


def _last(arr: np.ndarray, k: np.ndarray, default: np.ndarray) -> np.ndarray:
    """arr[k-1] where k > 0, else default (safe for empty arr)."""
    if len(arr) == 0:
        return default
    return np.where(k > 0, arr[np.maximum(k - 1, 0)], default)


def _active(s: np.ndarray, e: np.ndarray, t: np.ndarray) -> np.ndarray:
    k = np.searchsorted(s, t, side="left")
    return _last(e, k, np.zeros(len(t), np.int64) + np.iinfo(np.int64).min) > t


def series_features(ep: Episodes, t: np.ndarray, prefix: str, windows=(1, 3, 6, 24, 168),
                    minutes=(24, 168), recency=True) -> dict:
    s, e = _i64(ep.start), _i64(ep.end)
    out = {f"{prefix}_active": _active(s, e, t).astype(np.int8)}
    for w in windows:
        out[f"{prefix}_onsets_{w}h"] = _count_before(s, t, w)
    for w in minutes:
        out[f"{prefix}_alert_min_{w}h"] = (_occupied_before(s, e, t) - _occupied_before(s, e, t - np.int64(w * H_NS))) / 6e10
    if recency:
        k = np.searchsorted(s, t, side="left")
        last_s = _last(s, k, t - 720 * H_NS)
        out[f"{prefix}_since_onset_h"] = np.minimum((t - last_s) / H_NS, 720.0)
        # last END before t: ends are sorted too (episodes don't overlap); an end counts only once it happened
        ends_known = np.searchsorted(e, t, side="left")
        last_e = _last(e, ends_known, t - 720 * H_NS)
        out[f"{prefix}_since_end_h"] = np.minimum((t - last_e) / H_NS, 720.0)
    return out


# ------------------------------------------------------------------ text (Air Force channel)
def tag_messages(af: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per region and threat type; time column `ts` (int ns, UTC)."""
    out = pd.DataFrame({"ts": _i64(af.published_at.dt.tz_convert(None))})
    txt = af.text.fillna("")
    for loc, pat in REGION_PATTERNS.items():
        out[f"reg::{loc}"] = txt.str.contains(pat, regex=True).to_numpy()
    for k, pat in THREAT_PATTERNS.items():
        out[f"thr::{k}"] = txt.str.contains(pat, regex=True).to_numpy()
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def _rel_count(times: np.ndarray, t: np.ndarray, w_h: float) -> np.ndarray:
    """Count in the last w hours relative to its own level over the last 28 days (both from the past only)."""
    c = _count_before(times, t, w_h)
    level = _count_before(times, t, NORM_H) * (w_h / NORM_H)
    return (c + 1.0) / (level + 1.0)


def text_features(tags: pd.DataFrame, loc: str, t: np.ndarray) -> dict:
    ts = tags.ts.to_numpy()
    reg = ts[tags[f"reg::{loc}"].to_numpy()]
    out = {f"af_region_{w}h_rel": _rel_count(reg, t, w) for w in (1, 3, 6)}
    k = np.searchsorted(reg, t, side="left")
    last = _last(reg, k, t - 720 * H_NS)
    out["af_since_region_h"] = np.minimum((t - last) / H_NS, 720.0)
    nb = np.zeros(len(tags), bool)
    for n in NEIGHBOURS[loc]:
        nb |= tags[f"reg::{n}"].to_numpy()
    out["af_nbr_3h_rel"] = _rel_count(ts[nb], t, 3)
    out["af_all_1h_rel"] = _rel_count(ts, t, 1)
    out["af_all_24h_rel"] = _rel_count(ts, t, 24)
    for k_ in THREAT_PATTERNS:
        tt = ts[tags[f"thr::{k_}"].to_numpy()]
        for w in (1, 6):
            out[f"af_{k_}_{w}h_rel"] = _rel_count(tt, t, w)
    return out


# ------------------------------------------------------------------ the full table
def hourly_grid(start, end) -> np.ndarray:
    return pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).floor("h"), freq="h").to_numpy("datetime64[ns]")


def build_table(alerts: AlertData, tags: pd.DataFrame | None, grid: np.ndarray, with_labels: bool = True) -> pd.DataFrame:
    """Rows = (location, hour). Features use only information before each hour."""
    t = _i64(grid)
    locs = alerts.locations
    # national / neighbour building blocks, per location
    act = {l: series_features(alerts.target[l], t, "x", windows=(1, 3, 6, 24), minutes=(), recency=False) for l in locs}
    local = pd.DatetimeIndex(grid).tz_localize("UTC").tz_convert("Europe/Kyiv")
    hour, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    frames = []
    for i, loc in enumerate(locs):
        f = {"location": loc, "t": grid, "loc_id": np.full(len(t), i), "hour_local": hour,
             "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24), "dow": dow}
        f.update(series_features(alerts.target[loc], t, "own"))
        ob = series_features(alerts.oblast_any[loc], t, "obl", windows=(1, 3), minutes=(24,), recency=False)
        f.update(ob)
        nb = NEIGHBOURS[loc]
        f["nbr_active_frac"] = np.mean([act[n]["x_active"] for n in nb], axis=0)
        for w in (1, 3, 6):
            f[f"nbr_onsets_{w}h"] = np.sum([act[n][f"x_onsets_{w}h"] for n in nb], axis=0)
        others = [l for l in locs if l != loc]
        f["nat_active_frac"] = np.mean([act[n]["x_active"] for n in others], axis=0)
        for w in (1, 3, 6, 24):
            f[f"nat_onsets_{w}h"] = np.sum([act[n][f"x_onsets_{w}h"] for n in others], axis=0)
        if tags is not None:
            f.update(text_features(tags, loc, t))
        if with_labels:
            s = _i64(alerts.target[loc].start)
            end_ok = _i64(alerts.data_end)
            for h in HORIZONS:
                n = np.searchsorted(s, t + h * H_NS, side="left") - np.searchsorted(s, t, side="left")
                y = (n > 0).astype(float)
                y[t + h * H_NS > end_ok] = np.nan          # label window not fully observed
                f[f"y_{h}h"] = y
        frames.append(pd.DataFrame(f))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ recent label history (per horizon)


def label_history_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Recent onset frequencies built from PAST labels only.

    The label of row t' covers [t', t'+H). It is fully known at t only if t' + H <= t. With H <= 24 h,
    every row at least 24 h old qualifies, so all series below are shifted by >= 24 hours.
    Rows must be sorted by (location, t) on a contiguous hourly grid (as build_table produces).
    """
    y = df[f"y_{h}h"]
    loc = df["location"]
    g_lh = y.groupby([loc, df["hour_local"]])
    out = pd.DataFrame(index=df.index)
    # same local hour, previous 28 days (shift(1) within the hour group = 24 h ago)
    out["rate_hour_28d"] = g_lh.transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    past = y.groupby(loc).shift(24)                                   # labels at least 24 h old
    gp = past.groupby(loc)
    out["rate_7d"] = gp.transform(lambda s: s.rolling(24 * 7, min_periods=24).mean())
    out["rate_28d"] = gp.transform(lambda s: s.rolling(24 * 28, min_periods=24 * 7).mean())
    nat = past.groupby(df["t"]).transform("mean")                     # all locations, same hour
    out["nat_rate_7d"] = nat.groupby(loc).transform(lambda s: s.rolling(24 * 7, min_periods=24).mean())
    return out


# ------------------------------------------------------------------ online recalibration
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recalibrate_online(t: pd.Series, p: np.ndarray, y: np.ndarray, h: int, days: int = 7) -> np.ndarray:
    """Shift each forecast's log-odds by the recent gap between observed and predicted onset rates.

    For forecast time t, use all locations' rows t' in [t - 24h - days, t - 24h): their labels cover
    [t', t'+H) with H <= 24 h, so they are finished by t. delta(t) = logit(observed) - logit(predicted).
    Rows must cover the hours before the first forecast (context), otherwise delta = 0 there.
    """
    frame = pd.DataFrame({"t": t.to_numpy(), "p": p, "y": y})
    hourly = frame.groupby("t").agg(sp=("p", "sum"), sy=("y", "sum"), n=("y", "count"))
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"), fill_value=0)
    win = hourly.rolling(24 * days, min_periods=24).sum().shift(24)      # rows at least 24 h old
    delta = (_logit(win.sy / win.n) - _logit(win.sp / win.n)).fillna(0.0)
    d = delta.reindex(frame.t).to_numpy()
    return 1 / (1 + np.exp(-(_logit(p) + d)))
