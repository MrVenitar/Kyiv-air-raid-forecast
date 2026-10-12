"""Leakage tests for the Kyiv forecaster (run: python kyiv/test_kyiv_leakage.py).

1. Streaming vs batch: for random 15-min times (half of them during an alert, including times in the
   two-level system), rebuild every input from what existed strictly before t (KMDA posts, regional starts,
   Telegram messages), recompute features and baselines, and require exact equality with the batch run.
2. The same check is run against a deliberately planted leak, which must be caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

N_TIMES = 16


def _compare(inp, picks, base_fn=None, label=""):
    base_fn = base_fn or F.base_features
    g = np.array(sorted(picks), dtype="datetime64[ns]")
    batch = base_fn(inp, g).set_index("t")
    cols = [c for c in batch.columns]
    for t in g:
        known = F.known_at(inp, t)
        row = base_fn(known, np.array([t])).set_index("t")
        a, b = row[cols].to_numpy(float), batch.loc[[t], cols].to_numpy(float)
        bad = ~np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
        if bad.any():
            c = cols[np.argwhere(bad)[0][1]]
            raise AssertionError(f"{label} t={t}: feature {c}: stream={row[c].iloc[0]} batch={batch.loc[t, c]}")
        act = int(row.active.iloc[0])
        for h in F.HORIZONS:
            if act:
                el = row.elapsed_min.to_numpy()
                x1 = F.clear_baseline(known, np.array([t]), el, h)
                x2 = F.clear_baseline(inp, np.array([t]), el, h)
            else:
                x1 = np.r_[F.onset_baseline(known, np.array([t]), h)]
                x2 = np.r_[F.onset_baseline(inp, np.array([t]), h)]
            if not np.allclose(x1, x2, equal_nan=True):
                raise AssertionError(f"{label} t={t} h={h}: baseline stream={x1} batch={x2}")


def _picks(inp, seed=0):
    rng = np.random.default_rng(seed)
    g = F.grid("2026-08-01", inp.feature_end)
    b = F.base_features(inp, g)
    act = b[b.active == 1].t.to_numpy()
    new = g[g >= np.datetime64(F.NEW_SYSTEM_UTC)]
    return list(rng.choice(g, N_TIMES // 4, replace=False)) + list(rng.choice(new, N_TIMES // 4, replace=False)) \
        + list(rng.choice(act, N_TIMES // 2, replace=False))


def test_features_use_only_the_past():
    inp = F.load_inputs()
    _compare(inp, _picks(inp), label="clean")


def test_planted_leak_is_caught():
    """A feature that peeks at the true end of a running alert must fail the check."""
    inp = F.load_inputs()

    def leaky(i, g):
        """Typical bug: 'minutes left in the current alert' from the episode table it is given.
        In the batch table the all-clear of a running alert is already there; in real time it is not."""
        out = F.base_features(i, g)
        s, e = F.episode_arrays(i.posts)
        t = g.astype(np.int64)
        k = np.searchsorted(s, t, side="left")
        e_last = e[np.maximum(k - 1, 0)]
        out["minutes_left"] = np.where(out.active == 1, np.minimum((e_last - t) / F.MIN, 10**6), -1.0)
        return out
    try:
        _compare(inp, _picks(inp, seed=1), base_fn=leaky, label="planted")
    except AssertionError as e:
        print(f"  planted leak caught as expected: {str(e)[:120]}")
        return
    raise AssertionError("planted leak NOT caught")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
