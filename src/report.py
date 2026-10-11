"""Figures and a markdown summary from reports/<source>/*.csv (written by run.py).

    python src/report.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import json

import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
# reference categorical palette (fixed order), neutral gray for baselines
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY, INK, MUTED, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["calendar", "+recent rates", "+own history", "+spatial", "+Air Force text"]
TITLES = {"volunteer": "Volunteer alerts (oblast level)\ntest 12 Sep – 10 Oct 2026",
          "official": "Official alerts (main city)\ntest 9 Aug – 6 Sep 2026"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def chosen_rows(src: str) -> pd.DataFrame:
    """Test rows of the model/window/recalibration that validation chose, per horizon (+ selected + baseline)."""
    info = json.loads((REP / src / "run_info.json").read_text())
    m = pd.read_csv(REP / src / "metrics.csv")
    lab = {int(h): c["model"] + (" + online recal" if c["online_recalibration"] else "") for h, c in info["chosen"].items()}
    keep = m.apply(lambda r: r.model == lab[r.horizon_h] or r.model in ("selected on val", "baseline: seasonal 28d"), axis=1)
    return m[keep & (m.split == "test")], info


def skill_vs_horizon(sources):
    fig, axes = plt.subplots(1, len(sources), figsize=(5.2 * len(sources), 3.9), sharey=True)
    for ax, src in zip(np.atleast_1d(axes), sources):
        m, _ = chosen_rows(src)
        for i, name in enumerate(ORDER):
            g = m[m.features == name].sort_values("horizon_h")
            ax.plot(g.horizon_h, g.bss_vs_seasonal, marker="o", ms=5, lw=2, color=SERIES[i], label=name)
        ax.axhline(0, color=GRAY, lw=1.2, ls="--")
        ax.text(1, -0.008, "seasonal baseline", color=MUTED, fontsize=8, ha="left", va="top")
        low = m[m.features.isin(ORDER) & (m.bss_vs_seasonal < -0.3)]
        if len(low):
            ax.text(24, -0.295, "off scale: " + ", ".join(f"{r.features} {r.horizon_h} h ({r.bss_vs_seasonal:+.2f})"
                                                      for r in low.itertuples()), color=MUTED, fontsize=7, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 6, 12, 24], ["1 h", "3 h", "6 h", "12 h", "24 h"])
        ax.set_title(TITLES[src], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("forecast horizon", color=MUTED, fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Brier skill vs seasonal baseline", color=MUTED, fontsize=9)
    np.atleast_1d(axes)[0].set_ylim(-0.3, 0.22)
    h_, l_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8.5, loc="lower center", ncol=5, title="features (added cumulatively)", title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(REP / "skill_vs_horizon.png", dpi=150)
    plt.close(fig)


def reliability(sources):
    fig, axes = plt.subplots(1, len(sources), figsize=(4.6 * len(sources), 4.2), sharey=True)
    for ax, src in zip(np.atleast_1d(axes), sources):
        r = pd.read_csv(REP / src / "reliability.csv")
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, ls="--", label="perfect calibration")
        for i, h in enumerate((1, 6, 24)):
            g = r[(r.horizon_h == h) & (r.n >= 30)]
            ax.plot(g.mean_p, g.observed, marker="o", ms=5, lw=2, color=SERIES[i], label=f"{h} h")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_title(TITLES[src], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("forecast probability (test)", color=MUTED, fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("observed share with an onset", color=MUTED, fontsize=9)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5, title="horizon", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(REP / "reliability.png", dpi=150)
    plt.close(fig)


def kyiv_week(src="volunteer", h=3, days=7):
    p = pd.read_csv(REP / src / "test_predictions.csv.gz", parse_dates=["t"])
    p = p[p.location == "Kyiv City"]
    p = p[p.t >= p.t.max() - pd.Timedelta(days=days)]
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from data import load_alerts
    on = pd.to_datetime(load_alerts(src).target["Kyiv City"].start)
    on = on[(on >= p.t.min()) & (on <= p.t.max() + pd.Timedelta(hours=1))]
    loc = lambda s: s.dt.tz_localize("UTC").dt.tz_convert("Europe/Kyiv").dt.tz_localize(None)  # noqa: E731
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(loc(p.t), p[f"seasonal_{h}h"], color=GRAY, lw=1.2, label="seasonal baseline")
    ax.plot(loc(p.t), p[f"selected_{h}h"], color=SERIES[0], lw=2, label="model (selected on validation)")
    for x in loc(pd.Series(on)):
        ax.axvline(x, color=SERIES[1], lw=1, alpha=0.8)
    ax.plot([], [], color=SERIES[1], lw=1, label="alert onset")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"P(new alert within {h} h)", color=MUTED, fontsize=9)
    ax.set_title(f"Kyiv City, last {days} test days (Kyiv time), {src} data", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(REP / f"kyiv_week_{src}.png", dpi=150)
    plt.close(fig)


def summary(sources) -> str:
    lines = []
    for src in sources:
        t, info = chosen_rows(src)
        lines.append(f"\n### {src}\n")
        lines.append("Chosen on validation per horizon: " + "; ".join(
            f"{h} h: {c['model']}{' + online recal' if c['online_recalibration'] else ''}, train from {c['train_start']}, "
            f"features '{c['features']}'" for h, c in info["chosen"].items()) + "\n")
        tab = t[t.features.isin(ORDER)].pivot_table(index="features", columns="horizon_h", values="bss_vs_seasonal").loc[ORDER]
        lines.append("Ablation, Brier skill vs seasonal baseline on test (0 = baseline, higher is better):\n")
        lines.append(tab.round(3).to_markdown())
        sel = t[t.model == "selected on val"].set_index("horizon_h")
        base = t[t.model.str.startswith("baseline")].set_index("horizon_h")
        tab2 = pd.DataFrame({"onset rate": sel.base_rate, "Brier model": sel.brier, "Brier baseline": base.brier,
                             "skill": sel.bss_vs_seasonal, "log-loss model": sel.logloss, "log-loss baseline": base.logloss,
                             "AUC model": sel.auc, "AUC baseline": base.auc})
        lines.append("\nSelected model vs seasonal baseline, test:\n")
        lines.append(tab2.round(3).to_markdown())
        op = pd.read_csv(REP / src / "operational.csv")
        lines.append("\nOperational, test (warning thresholds set on validation):\n")
        lines.append(op.round(3).to_markdown(index=False))
    # per-location skill at 3 h, both sources side by side (n/a when the test period has < 20 onset hours)
    cols = {}
    for src in sources:
        l = pd.read_csv(REP / src / "per_location.csv")
        l = l[l.horizon_h == 3].set_index("location")
        cols[f"{src}: onset hours"] = l.onset_hours
        cols[f"{src}: skill"] = l.bss_vs_seasonal.where(l.onset_hours >= 20)
    lines.append("\n### Per location, 3 h horizon, test (skill vs seasonal baseline; n/a if < 20 onset hours)\n")
    lines.append(pd.DataFrame(cols).sort_index().round(3).to_markdown())
    return "\n".join(lines)


if __name__ == "__main__":
    srcs = [s for s in ("volunteer", "official") if (REP / s / "metrics.csv").is_file()]
    skill_vs_horizon(srcs)
    reliability(srcs)
    for s in srcs:
        kyiv_week(s)
    (REP / "summary.md").write_text(summary(srcs), encoding="utf-8")
    print((REP / "summary.md").read_text())
