"""Kyiv City alert episodes from the KMDA channel («КМДА – офіційний канал»), validated against the official series.

    python kyiv/clean_alerts.py            # writes data/kyiv/kyiv_alerts.csv and reports/kyiv/cleaning.md

Why KMDA: the official alert channel (@air_alert_ua) went silent on 2026-09-07, one day after Kyiv switched to a
two-level system (yellow = drone danger, red = missile / massive attack). KMDA posts every Kyiv alert as a fixed
template, before and after the switch, including the level.

Templates (first line of the post):
    start      «‼️УВАГА! У Києві оголошена (чергова) повітряна тривога!»       old single-level system
    yellow     «🟡 УВАГА! У Києві оголошена (нова загроза –) дронова небезпека!»
    red        «🔴 УВАГА! У Києві оголошена (нова загроза –) ракетна загроза / масована дронова загроза / нова загроза!»
    all-clear  «❕Відбій повітряної тривоги!»
Everything else in the channel (city news, metro notices, reminders "тривога триває") is ignored.

Episode rule (one pass in publication order, so it works identically in real time):
    start while idle   -> new episode (start = publication time)
    start while active -> level change if the level differs, else a repeated post (kept, counted)
    all-clear          -> closes the active episode; while idle it is an orphan (counted, ignored)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEW_SYSTEM_UTC = pd.Timestamp("2026-09-06 03:00")          # 06:00 Kyiv, 6 Sep 2026
LEVELS = {"none": 0, "yellow": 1, "red": 2, "single": 3}    # single = old one-level alert (and its rare later uses)

_START_OLD = re.compile(r"^‼️\s*(?:УВАГА|Увага)!\s*У Києві оголошена (?:чергова )?повітряна тривога", re.I)
_YELLOW = re.compile(r"^🟡\s*УВАГА!\s*У Києві оголошена (?:нова загроза\s*[–—-]\s*)?дронова небезпека")
_RED = re.compile(r"^🔴\s*УВАГА!\s*У Києві оголошена (?:нова загроза|ракетна загроза|масована)")
_END = re.compile(r"^❕\s*Відбій повітряної тривоги")


def classify(text: str) -> tuple[str, str]:
    """(kind, level) of a KMDA post: kind in {start, end, ''}."""
    first = (text or "").strip().split("\n", 1)[0].strip()
    if _END.search(first):
        return "end", "none"
    if _YELLOW.search(first):
        return "start", "yellow"
    if _RED.search(first):
        return "start", "red"
    if _START_OLD.search(first):
        return "start", "single"
    return "", ""


def alert_posts(kmda: pd.DataFrame) -> pd.DataFrame:
    """Alert-related KMDA posts: ts (UTC, naive), msg_id, kind, level."""
    k = kmda[["msg_id", "published_at", "text"]].copy()
    k[["kind", "level"]] = [classify(x) for x in k.text.fillna("")]
    k = k[k.kind != ""].copy()
    k["ts"] = pd.to_datetime(k.published_at, utc=True).dt.tz_convert(None)
    return k.sort_values(["ts", "msg_id"], kind="stable")[["msg_id", "ts", "kind", "level"]].reset_index(drop=True)


def build_episodes(posts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One pass over posts in time order. Returns episodes and counts of anomalies."""
    eps, cur = [], None
    stats = {"repeated_start_posts": 0, "orphan_all_clears": 0, "level_changes": 0}
    for ts, kind, level in zip(posts.ts, posts.kind, posts.level):
        if kind == "start":
            if cur is None:
                cur = {"start": ts, "end": pd.NaT, "level_start": level, "level_max": level,
                       "levels": [(ts, level)], "n_level_changes": 0}
            elif level != cur["levels"][-1][1]:
                cur["levels"].append((ts, level))
                cur["n_level_changes"] += 1
                stats["level_changes"] += 1
                if LEVELS[level] > LEVELS[cur["level_max"]] and level != "single":
                    cur["level_max"] = level
            else:
                stats["repeated_start_posts"] += 1
        elif kind == "end":
            if cur is None:
                stats["orphan_all_clears"] += 1
            else:
                cur["end"] = ts
                eps.append(cur)
                cur = None
    if cur is not None:                       # still running at the end of the data
        eps.append(cur)
    ep = pd.DataFrame(eps)
    if len(ep):
        ep["level_path"] = ep.pop("levels").map(lambda L: ">".join(l for _, l in L))
        ep["first_change_at"] = [pd.NaT] * len(ep)
    return ep, stats


def level_changes(posts: pd.DataFrame) -> pd.DataFrame:
    """(ts, level) of every level the city announced, in order; used for 'level known at t'."""
    out, active, last = [], False, None
    for ts, kind, level in zip(posts.ts, posts.kind, posts.level):
        if kind == "start":
            if not active or level != last:
                out.append((ts, level))
            active, last = True, level
        elif kind == "end":
            if active:
                out.append((ts, "none"))
            active, last = False, None
    return pd.DataFrame(out, columns=["ts", "level"])


# ------------------------------------------------------------------ validation helpers
def _nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each time in a, signed minutes to the nearest time in b (a - b)."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    i = np.searchsorted(b, a)
    lo, hi = b[np.clip(i - 1, 0, len(b) - 1)], b[np.clip(i, 0, len(b) - 1)]
    d_lo, d_hi = (a - lo) / np.timedelta64(1, "m"), (a - hi) / np.timedelta64(1, "m")
    return np.where(np.abs(d_lo) <= np.abs(d_hi), d_lo, d_hi)


def compare(name_a, ep_a, name_b, ep_b, lo, hi, tol=5.0) -> dict:
    a = ep_a[(ep_a.start >= lo) & (ep_a.start < hi)]
    b = ep_b[(ep_b.start >= lo) & (ep_b.start < hi)]
    sa, sb = a.start.to_numpy("datetime64[ns]"), np.sort(b.start.to_numpy("datetime64[ns]"))
    da, db = _nearest(sa, sb), _nearest(sb, np.sort(sa))
    ea = a.end.dropna().to_numpy("datetime64[ns]")
    eb = np.sort(b.end.dropna().to_numpy("datetime64[ns]"))
    de = _nearest(ea, eb)
    dur = lambda e: ((e.end - e.start).dt.total_seconds() / 60).median()  # noqa: E731
    return {"period": f"{lo:%Y-%m-%d} – {hi:%Y-%m-%d}", f"{name_a} episodes": len(a), f"{name_b} episodes": len(b),
            f"{name_a} starts with a {name_b} start ±{tol:g} min": f"{np.mean(np.abs(da) <= tol):.1%}",
            f"{name_b} starts with a {name_a} start ±{tol:g} min": f"{np.mean(np.abs(db) <= tol):.1%}",
            f"median start offset {name_a}−{name_b} (min)": round(float(np.median(da[np.abs(da) <= tol])), 2) if (np.abs(da) <= tol).any() else None,
            f"{name_a} all-clears with a {name_b} end ±{tol:g} min": f"{np.mean(np.abs(de) <= tol):.1%}",
            f"median duration {name_a} / {name_b} (min)": f"{dur(a):.0f} / {dur(b):.0f}"}


def official_kyiv() -> pd.DataFrame:
    from data import merge_intervals, read_raw
    o = read_raw("official")
    o = o[o.oblast == "Kyiv City"]
    ep = merge_intervals(o.started_at.to_numpy(), o.finished_at.to_numpy())
    return pd.DataFrame({"start": ep.start, "end": ep.end}), len(o)


def volunteer_kyiv() -> pd.DataFrame:
    from data import read_raw
    v = read_raw("volunteer")
    v = v[v.oblast == "Kyiv City"].sort_values("started_at")
    return pd.DataFrame({"start": v.started_at.to_numpy(), "end": v.finished_at.to_numpy(), "naive": v.naive.to_numpy()})


def main() -> None:
    out_dir = ROOT / "data" / "kyiv"
    rep_dir = ROOT / "reports" / "kyiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    kmda = pd.read_csv(ROOT / "data" / "tg_kmda.csv.gz", parse_dates=["published_at"])
    posts = alert_posts(kmda)
    ep, stats = build_episodes(posts)
    ep["duration_min"] = (ep.end - ep.start).dt.total_seconds() / 60
    ep["system"] = np.where(ep.start >= NEW_SYSTEM_UTC, "two-level", "single")
    lv = level_changes(posts)

    off, off_rows = official_kyiv()
    vol = volunteer_kyiv()
    lines = ["# Kyiv alert labels: cleaning report\n",
             f"KMDA posts used: {len(posts)} ({posts.kind.value_counts().to_dict()}), first {posts.ts.min()} UTC, last {posts.ts.max()} UTC.",
             f"Episodes: {len(ep)}; anomalies: {stats}.",
             f"Still running at the end of the data: {int(ep.end.isna().sum())}.\n",
             "## Durations (minutes) by year\n",
             ep.assign(year=ep.start.dt.year).groupby("year").duration_min.describe(percentiles=[.25, .5, .75, .95])
               .round(1).to_markdown(), ""]

    lines += ["## KMDA vs official channel (Kyiv City), where both exist\n",
              f"The official dataset stores most Kyiv alerts twice ({off_rows} rows -> {len(off)} merged episodes).\n"]
    cmp = []
    for y in range(2022, 2027):
        lo = max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp("2022-03-15"))
        hi = min(pd.Timestamp(f"{y + 1}-01-01"), NEW_SYSTEM_UTC)
        cmp.append(compare("KMDA", ep, "official", off, lo, hi))
    lines.append(pd.DataFrame(cmp).to_markdown(index=False))

    lines += ["\n## KMDA vs volunteer (eTryvoga) Kyiv City\n"]
    cmp = [compare("KMDA", ep, "volunteer", vol, pd.Timestamp("2026-06-01"), NEW_SYSTEM_UTC),
           compare("KMDA", ep, "volunteer", vol, NEW_SYSTEM_UTC, pd.Timestamp("2026-10-12 05:00"))]
    lines.append(pd.DataFrame(cmp).to_markdown(index=False))
    nv = vol[vol.start >= NEW_SYSTEM_UTC]
    lines.append(f"\nVolunteer Kyiv alerts since the switch: {len(nv)}, of which {nv.naive.mean():.0%} have a guessed end "
                 "(start + 30 min).")

    new = ep[ep.system == "two-level"]
    lines += ["\n## The two-level system (since 06:00 Kyiv, 6 Sep 2026)\n",
              f"Episodes: {len(new)} in {(posts.ts.max() - NEW_SYSTEM_UTC).days} days "
              f"(single-level system in Jan–Aug 2026: {((ep.start >= '2026-01-01') & (ep.system == 'single')).sum()} in "
              f"{(NEW_SYSTEM_UTC - pd.Timestamp('2026-01-01')).days} days).",
              "Level at start / highest level / path:\n",
              new.level_start.value_counts().to_frame("starts at").join(new.level_max.value_counts().to_frame("highest")).to_markdown(),
              "\nMost common level paths:\n", new.level_path.value_counts().head(8).to_frame("episodes").to_markdown(),
              "\nDuration by highest level (minutes):\n",
              new.groupby("level_max").duration_min.describe(percentiles=[.25, .5, .75]).round(1).to_markdown()]

    ep_out = ep[["start", "end", "duration_min", "level_start", "level_max", "level_path", "n_level_changes", "system"]]
    ep_out.to_csv(out_dir / "kyiv_alerts.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    lv.to_csv(out_dir / "kyiv_levels.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    posts.to_csv(out_dir / "kmda_alert_posts.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    (rep_dir / "cleaning.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
