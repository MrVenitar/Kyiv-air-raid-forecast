"""Figures for the Kyiv forecaster from the test-week predictions (reports/kyiv/test_pred_*.pkl).

    python kyiv/report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "kyiv"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
ALERT_FILL = "#f6d7c9"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _kyiv(ts):
    return pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert("Europe/Kyiv").tz_localize(None)


def replay(day_from: str, day_to: str, name: str):
    """One 15-min forecast stream as a commuter would have seen it: P(alert within 30 min) when idle,
    P(all-clear within 30 min) during alerts, with the real alerts shaded."""
    on = pd.read_csv(REP / "test_pred_onset.csv.gz", parse_dates=["t"])
    cl = pd.read_csv(REP / "test_pred_clear.csv.gz", parse_dates=["t"])
    ep = pd.read_csv(ROOT / "data" / "kyiv" / "kyiv_alerts.csv", parse_dates=["start", "end"])
    lo = pd.Timestamp(day_from).tz_localize("Europe/Kyiv").tz_convert("UTC").tz_localize(None)
    hi = pd.Timestamp(day_to).tz_localize("Europe/Kyiv").tz_convert("UTC").tz_localize(None)
    o = on[(on.h == 30) & (on.t >= lo) & (on.t < hi)].sort_values("t")
    c = cl[(cl.h == 30) & (cl.t >= lo) & (cl.t < hi)].sort_values("t")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for s, e, lv in zip(ep.start, ep.end, ep.level_max):
        if e >= lo and s < hi:
            ax.axvspan(_kyiv([max(s, lo)])[0], _kyiv([min(e, hi)])[0], color=ALERT_FILL, lw=0, zorder=0)
    ax.plot(_kyiv(o.t), o.p, ".", ms=5, color=BLUE, label="no alert: P(alert starts within 30 min)")
    ax.plot(_kyiv(c.t), c.p, ".", ms=5, color=AQUA, label="during alert: P(all-clear within 30 min)")
    ax.fill_between([], [], color=ALERT_FILL, label="Kyiv alert (KMDA)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability", color=MUTED, fontsize=9)
    ax.set_title(f"Replay of test days, Kyiv time: {day_from} 00:00 – {day_to} 00:00 (each forecast uses only data published before it)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    _style(ax)
    fig.tight_layout()
    fig.savefig(REP / name, dpi=150)
    plt.close(fig)


def reliability():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, task, title in zip(axes, ("onset", "clear"), ("New alert (no alert now)", "All-clear (alert running)")):
        pr = pd.read_csv(REP / f"test_pred_{task}.csv.gz", parse_dates=["t"])
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, ls="--", label="perfect")
        for col, h in zip((BLUE, ORANGE, AQUA), (15, 30, 60)):
            g = pr[pr.h == h]
            b = np.clip((g.p * 10).astype(int), 0, 9)
            r = g.groupby(b).agg(mp=("p", "mean"), ob=("y", "mean"), n=("y", "size"))
            r = r[r.n >= 20]
            ax.plot(r.mp, r.ob, marker="o", ms=5, lw=2, color=col, label=f"{h} min")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_title(f"{title}\ntest 21 Sep – 11 Oct 2026", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("forecast probability", color=MUTED, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("observed share", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(REP / "reliability.png", dpi=150)
    plt.close(fig)


def skill():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, task, title in zip(axes, ("onset", "clear"), ("New alert within h", "All-clear within h")):
        pr = pd.read_csv(REP / f"test_pred_{task}.csv.gz", parse_dates=["t"])
        hs = sorted(pr.h.unique())
        sk = [1 - brier_score_loss(pr[pr.h == h].y, pr[pr.h == h].p) / brier_score_loss(pr[pr.h == h].y, pr[pr.h == h].base) for h in hs]
        x = np.arange(len(hs))
        ax.bar(x, sk, width=0.6, color=[BLUE if v >= 0 else ORANGE for v in sk])
        for xi, v in zip(x, sk):
            ax.text(xi, v + (0.005 if v >= 0 else -0.005), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5, color=INK)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks(x, [f"{h} min" for h in hs])
        ax.set_title(f"{title}: Brier skill vs baseline (test)", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("skill (0 = baseline)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "skill.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    reliability()
    skill()
    days = sys.argv[1:3] if len(sys.argv) > 2 else ["2026-10-09", "2026-10-11"]
    replay(days[0], days[1], "replay.png")
    print("figures in", REP)
