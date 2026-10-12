"""Kyiv commuter forecast at a given moment, every 15 minutes up to 3 hours ahead.

    python kyiv/forecast.py                          # at the end of the available data
    python kyiv/forecast.py --at "2026-10-10 08:15"  # Kyiv local time; uses only data published before it
    python kyiv/live.py                              # right now (fetches the newest Telegram posts first)

If no alert is running: P(a new Kyiv alert starts within 15, 30, 60, 120, 180 min).
If an alert is running: P(the all-clear is posted within 15, 30, 60, 120, 180 min).
Never use this to ignore a real alert: when the siren sounds, follow it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kyiv"))
import features as F  # noqa: E402
from model_utils import predict_logreg  # noqa: E402
from train import _logit, design  # noqa: E402

LEVEL = {0: "none", 1: "yellow (drone danger)", 2: "red (missile / massive attack)", 3: "alert (single-level system)"}


def load_model(task: str, variant: str = "") -> dict:
    """Plain-JSON model (works with any scikit-learn version); falls back to the joblib file."""
    js = ROOT / "models" / f"kyiv_{task}{variant}.json"
    if js.is_file():
        return json.loads(js.read_text())
    import joblib
    art = joblib.load(ROOT / "models" / f"kyiv_{task}{variant}.joblib")
    return {"type": "sklearn", **art}


def forecast(at_utc: pd.Timestamp, inp: F.Inputs | None = None, variant: str = "") -> dict:
    inp = inp or F.load_inputs()
    if at_utc > inp.feature_end:
        raise SystemExit(f"Data ends at {inp.feature_end} UTC: use kyiv/live.py, or export fresh channels.")
    known = F.known_at(inp, at_utc)
    g = np.array([np.datetime64(at_utc, "ns")])
    base = F.base_features(known, g)
    task = "clear" if int(base.active.iloc[0]) else "onset"
    rows = F.stacked(base, known, task, with_labels=False)
    m = load_model(task, variant)
    if m["type"] == "baseline":
        p = rows.base.to_numpy()
    else:
        X, _ = design(rows, task, m["groups"])
        X = X[m["columns"]]
        p = predict_logreg(m, X.to_numpy(float)) if m["type"] == "logreg" else m["model"].predict_proba(X)[:, 1]
        if m.get("deltas"):
            d = rows.h.map(lambda h: m["deltas"].get(str(int(h)), m["deltas"].get(int(h), 0.0))).to_numpy(float)
            p = 1 / (1 + np.exp(-(_logit(p) + d)))
    p = np.maximum.accumulate(np.clip(p, 1e-4, 1 - 1e-4))      # a longer window is never less likely
    return {"task": task, "state": base.iloc[0], "h": rows.h.to_numpy(), "p": p, "baseline": rows.base.to_numpy(),
            "model": m.get("choice", {}), "groups": m.get("groups", [])}


def print_forecast(r: dict, at_utc: pd.Timestamp, note: str = "Data used: published before this moment only.") -> None:
    st = r["state"]
    local = pd.Timestamp(at_utc).tz_localize("UTC").tz_convert("Europe/Kyiv")
    print(f"Kyiv, {local:%Y-%m-%d %H:%M} (Kyiv time). {note}")
    if r["task"] == "clear":
        print(f"ALERT RUNNING for {st.elapsed_min:.0f} min, level: {LEVEL.get(int(st.level_now), '?')}.")
        print("Chance the all-clear comes within ...")
    else:
        print(f"No alert. Last all-clear {st.since_end_min:.0f} min ago; {int(st.onsets_1440m)} alerts in the last 24 h.")
        print("Chance a new alert starts within ...")
    print(f"{'':10s}{'forecast':>10s}{'baseline':>10s}")
    for h, p, b in zip(r["h"], r["p"], r["baseline"]):
        print(f"  {h:>3d} min{p:>10.0%}{b:>10.0%}")
    print(f"(inputs: {', '.join(r['groups']) or 'Kyiv alert history only'}; baseline = recent history of Kyiv alerts only)")
    if r["task"] == "onset":
        print(f"\nLivoberezhna -> across the Dnipro by metro (~15-30 min): chance an alert closes the surface stations "
              f"first: {r['p'][list(r['h']).index(30)]:.0%} within 30 min.")
    else:
        lv = int(st.level_now)
        hint = ("yellow: by the city's rules the green line still crosses the Dnipro (Zvirynetska -> Vydubychi -> Slavutych)"
                if lv == 1 else "red: no metro crossing of the Dnipro until the all-clear")
        print(f"\nArsenalna: wait for the all-clear or reroute? P(all-clear within 30 min) = "
              f"{r['p'][list(r['h']).index(30)]:.0%}. Level {hint}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at", default=None, help="Kyiv local time, e.g. '2026-10-10 08:15' (default: end of data)")
    a = ap.parse_args()
    inp = F.load_inputs()
    if a.at:
        at = pd.Timestamp(a.at).tz_localize("Europe/Kyiv").tz_convert("UTC").tz_localize(None)
    else:
        at = pd.Timestamp(inp.feature_end).floor("15min")
    print_forecast(forecast(at, inp), at)


if __name__ == "__main__":
    main()
