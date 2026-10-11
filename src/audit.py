"""Data audit: the checks that changed the plan (see VERSIONS.md), reproducible.

    python src/audit.py [--official-channel data/tg_air_alert_ua.csv.gz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import build_alerts, merge_intervals, read_raw

ROOT = Path(__file__).resolve().parents[1]
JUN, SEP = np.datetime64("2026-06-01"), np.datetime64("2026-09-01")


def _hours(ep, lo=JUN, hi=SEP) -> float:
    s, e = np.maximum(ep.start, lo), np.minimum(ep.end, hi)
    return float(((e - s)[e > s] / np.timedelta64(1, "h")).sum())


def _near(a: np.ndarray, b: np.ndarray, minutes: float) -> float:
    if not len(a) or not len(b):
        return float("nan")
    i = np.searchsorted(b, a)
    d = np.minimum(np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)]), np.abs(a - b[np.clip(i, 0, len(b) - 1)]))
    return float(np.mean(d <= np.timedelta64(int(minutes * 60), "s")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-channel", default=None, help="parsed @air_alert_ua export (csv.gz), optional")
    ap.add_argument("--af", default=str(ROOT / "data" / "tg_kpszsu.csv.gz"))
    a = ap.parse_args()

    vol_raw, off_raw = read_raw("volunteer"), read_raw("official")
    vol, off = build_alerts("volunteer", vol_raw), build_alerts("official", off_raw)
    print(f"1. Coverage: volunteer ends {vol.data_end}, official ends {off.data_end}")
    if a.official_channel:
        ch = pd.read_csv(a.official_channel, parse_dates=["published_at"])
        last = ch.sort_values("msg_id").iloc[-1]
        print(f"   official channel export: last message id {last.msg_id} at {last.published_at} (Kyiv clock in text: "
              f"{str(last.text)[:40]!r})")

    print("\n2. Kyiv City, Jun-Aug 2026: do the two sources agree?")
    kv, ko = vol.target["Kyiv City"], off.target["Kyiv City"]
    sv = kv.start[(kv.start >= JUN) & (kv.start < SEP)]
    so = ko.start[(ko.start >= JUN) & (ko.start < SEP)]
    print(f"   episodes: volunteer {len(sv)}, official {len(so)}; alert hours: volunteer {_hours(kv):.1f}, official {_hours(ko):.1f}")
    print(f"   volunteer starts within 5 min of an official start: {_near(sv, so, 5):.0%}")
    recent = vol_raw[(vol_raw.oblast == "Kyiv City") & (vol_raw.started_at >= np.datetime64("2026-07-13"))]
    print(f"   volunteer Kyiv City alerts since 13 Jul with a guessed end (start + 30 min): {recent.naive.mean():.0%}")

    print("\n3. Kharkiv oblast, Jun-Aug 2026: share of hours with SOME raion under official alert")
    k = off_raw[off_raw.oblast == "Kharkivska oblast"]
    ep = merge_intervals(k.started_at.to_numpy(), k.finished_at.to_numpy())
    total = (SEP - JUN) / np.timedelta64(1, "h")
    print(f"   {_hours(ep):.0f} of {total:.0f} hours ({_hours(ep) / total:.0%})")

    print("\n4. Air Force channel as arbiter: share of alert starts with the oblast mentioned within 30 min")
    af = pd.read_csv(a.af, parse_dates=["published_at"])
    af = af[af.published_at >= "2026-06-01"]
    ts = af.published_at.dt.tz_convert(None).to_numpy("datetime64[ns]")
    for loc, stem in [("Volynska oblast", "Волин"), ("Lvivska oblast", "Львів"), ("Kharkivska oblast", "Харків"),
                      ("Poltavska oblast", "Полтав")]:
        m = np.sort(ts[af.text.fillna("").str.contains(stem).to_numpy()])
        out = []
        for name, df in (("volunteer", vol_raw), ("official", off_raw)):
            s = np.sort(df[(df.oblast == loc) & (df.started_at >= JUN) & (df.started_at < np.datetime64("2026-09-07"))]
                        .started_at.to_numpy())
            out.append(f"{name} {_near(s, m, 30):.0%} of {len(s)}")
        print(f"   {loc:20s} " + ", ".join(out))

    print("\n5. Air Force channel posting drift (messages per month, share with the 🛵 / 🏍 drone emoji)")
    d = pd.read_csv(a.af, parse_dates=["published_at"])
    d = d[d.published_at >= "2026-03-01"]
    g = d.groupby(d.published_at.dt.strftime("%Y-%m")).text.agg(
        messages="size", moped=lambda s: s.fillna("").str.contains("🛵").mean(),
        motorbike=lambda s: s.fillna("").str.contains("🏍").mean())
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
