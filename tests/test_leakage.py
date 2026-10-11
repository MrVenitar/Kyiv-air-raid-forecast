"""Streaming-vs-batch leakage test (the tokamak project's self-test, applied to time).

For random forecast times t, rebuild everything from ONLY what a real-time system would have had at t:
  * alert rows that started before t, with the end of any alert still running at t unknown;
  * Air Force messages published before t.
Features computed from that truncated world must equal the batch features at t exactly.
Any feature that peeks at the future (an alert's end, a later message, a later merge) breaks equality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import build_alerts, raw_known_at, read_raw  # noqa: E402
from features import TABLE_FEATURES, LABEL_HISTORY, build_table, hourly_grid, label_history_features, recalibrate_online, tag_messages  # noqa: E402

AF = Path(__file__).resolve().parents[1] / "data" / "tg_kpszsu.csv.gz"
N_TIMES = 12


@lru_cache(maxsize=1)
def _tags():
    return tag_messages(pd.read_csv(AF, parse_dates=["published_at"]))


def test_volunteer_features_use_only_the_past():
    check_source("volunteer")


def test_official_features_use_only_the_past():
    check_source("official")


def check_source(source):
    tags = _tags()
    raw = read_raw(source)
    full = build_alerts(source, raw)
    end = pd.Timestamp(full.data_end) - pd.Timedelta("2D")
    grid = hourly_grid(end - pd.Timedelta("60D"), end)
    batch = build_table(full, tags, grid, with_labels=False).set_index(["location", "t"])

    rng = np.random.default_rng(0)
    # random hours, plus hours when some location had an alert running (the end-time trap)
    picks = list(rng.choice(grid, N_TIMES, replace=False))
    running = batch[batch.own_active == 1].index.get_level_values("t").unique()
    picks += list(rng.choice(running.to_numpy(), N_TIMES, replace=False))

    for t in picks:
        t = np.datetime64(t, "ns")
        known = build_alerts(source, raw_known_at(raw, t), data_end=t)
        tags_known = tags[tags.ts < np.int64(t.astype(np.int64))]
        stream = build_table(known, tags_known, np.array([t]), with_labels=False).set_index(["location", "t"])
        cols = [c for c in TABLE_FEATURES if c in stream.columns]
        a = stream[cols].to_numpy(float)
        b = batch.loc[stream.index, cols].to_numpy(float)
        bad = ~np.isclose(a, b, rtol=0, atol=1e-9)
        if bad.any():
            r, c = np.argwhere(bad)[0]
            raise AssertionError(f"{source} t={t}: {stream.index[r]} feature {cols[c]}: stream={a[r, c]} batch={b[r, c]}")


def test_labels_are_strictly_future():
    """Labels count onsets in [t, t+H): changing everything BEFORE t must not change them."""
    raw = read_raw("volunteer")
    full = build_alerts("volunteer", raw)
    t = np.datetime64(pd.Timestamp(full.data_end).floor("h") - pd.Timedelta("30D"), "ns")
    future_only = build_alerts("volunteer", raw[raw.started_at >= t], data_end=full.data_end)
    ycols = [f"y_{h}h" for h in (1, 3, 6, 12, 24)]
    a = build_table(full, None, np.array([t]))[ycols].to_numpy()
    b = build_table(future_only, None, np.array([t]))[ycols].to_numpy()
    # merged onsets can differ only if an alert running at t swallowed a later start; allow that direction
    assert (b >= a).all()


def test_label_history_uses_only_finished_labels():
    """Scramble every label whose window is not finished by t0; history features at t0 must not move."""
    raw = read_raw("volunteer")
    full = build_alerts("volunteer", raw)
    end = pd.Timestamp(full.data_end).floor("h") - pd.Timedelta("2D")
    df = build_table(full, None, hourly_grid(end - pd.Timedelta("60D"), end))
    rng = np.random.default_rng(1)
    for t0 in rng.choice(df.t.unique()[24 * 35:], 5, replace=False):
        for h in (1, 3, 6, 12, 24):
            col = f"y_{h}h"
            scrambled = df.copy()
            unfinished = scrambled.t + pd.Timedelta(hours=h) > t0
            scrambled.loc[unfinished, col] = rng.integers(0, 2, unfinished.sum())
            a = label_history_features(df, h)[df.t == t0][LABEL_HISTORY].to_numpy(float)
            b = label_history_features(scrambled, h)[df.t == t0][LABEL_HISTORY].to_numpy(float)
            assert np.allclose(a, b, equal_nan=True), f"label history at {t0} H={h} uses unfinished labels"


def test_online_recalibration_uses_only_finished_labels():
    rng = np.random.default_rng(2)
    t = pd.Series(np.repeat(pd.date_range("2026-01-01", periods=24 * 40, freq="h"), 3))
    p = rng.uniform(0.05, 0.95, len(t))
    y = (rng.uniform(size=len(t)) < 0.6).astype(float)
    t0 = t.iloc[len(t) // 2]
    for h in (1, 6, 24):
        y2 = y.copy()
        unfinished = (t + pd.Timedelta(hours=h) > t0).to_numpy()
        y2[unfinished] = 1 - y2[unfinished]
        a = recalibrate_online(t, p, y, h)[(t == t0).to_numpy()]
        b = recalibrate_online(t, p, y2, h)[(t == t0).to_numpy()]
        assert np.allclose(a, b), f"recalibration at {t0} uses unfinished labels (H={h})"


if __name__ == "__main__":          # runs without pytest too
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
