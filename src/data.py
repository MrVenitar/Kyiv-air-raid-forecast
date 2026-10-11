"""Alert events for each forecast location, from either alert source.

Two sources (both from https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset, times in UTC):

* ``volunteer``: eTryvoga channel, oblast level, current to the day of download.
* ``official``:  official «Повітряна тривога» channel (@air_alert_ua). Since late 2025 alerts are
  issued per raion. The channel posted its last message on 2026-09-07, so this source ends there.

A *location* is what a forecast is about:
* volunteer: the oblast (as eTryvoga reports it) or Kyiv City;
* official:  the main city of the oblast. It is under alert when there is an oblast-wide alert,
             an alert for the city's raion, or an alert for the city's own hromada.

Luhansk oblast and Crimea are excluded (permanent alerts since 2022: nothing to forecast).

Every location's alerts are merged into *episodes* (overlapping alerts joined). An episode start is
an *onset*. Whether a new alert is an onset only depends on alerts that started before it and are
still running at that moment, so onsets are known in real time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "sirens" / "datasets"

# Oblast (as in the English dataset) -> (main-city raion, main-city hromada prefix or None)
MAIN_CITY = {
    "Cherkaska oblast": ("Cherkaskyi raion", "m. Cherkasy"),
    "Chernihivska oblast": ("Chernihivskyi raion", "m. Chernihiv"),
    "Chernivetska oblast": ("Chernivetskyi raion", "m. Chernivtsi"),
    "Dnipropetrovska oblast": ("Dniprovskyi raion", "m. Dnipro "),
    "Donetska oblast": ("Kramatorskyi raion", "m. Kramatorsk"),      # de facto regional centre
    "Ivano-Frankivska oblast": ("Ivano-Frankivskyi raion", "m. Ivano-Frankivsk"),
    "Kharkivska oblast": ("Kharkivskyi raion", "m. Kharkiv "),
    "Khersonska oblast": ("Khersonskyi raion", "m. Kherson "),
    "Khmelnytska oblast": ("Khmelnytskyi raion", "m. Khmelnytskyi"),
    "Kirovohradska oblast": ("Kropyvnytskyi raion", "m. Kropyvnytskyi"),
    "Kyiv City": (None, None),                                        # its own unit, oblast level only
    "Kyivska oblast": ("Bilotserkivskyi raion", "m. Bila Tserkva"),   # Kyiv itself is a separate unit
    "Lvivska oblast": ("Lvivskyi raion", "m. Lviv "),
    "Mykolaivska oblast": ("Mykolaivskyi raion", "m. Mykolaiv "),
    "Odeska oblast": ("Odeskyi raion", "m. Odesa "),
    "Poltavska oblast": ("Poltavskyi raion", "m. Poltava "),
    "Rivnenska oblast": ("Rivnenskyi raion", "m. Rivne "),
    "Sumska oblast": ("Sumskyi raion", "m. Sumy "),
    "Ternopilska oblast": ("Ternopilskyi raion", "m. Ternopil "),
    "Vinnytska oblast": ("Vinnytskyi raion", "m. Vinnytsia "),
    "Volynska oblast": ("Lutskyi raion", "m. Lutsk "),
    "Zakarpatska oblast": ("Uzhhorodskyi raion", "m. Uzhhorod "),
    "Zaporizka oblast": ("Zaporizkyi raion", "m. Zaporizhzhia "),
    "Zhytomyrska oblast": ("Zhytomyrskyi raion", "m. Zhytomyr "),
}
LOCATIONS = list(MAIN_CITY)           # 24: all oblasts except Luhansk, plus Kyiv City (Crimea not in data)
EXCLUDED = ("Luhanska oblast", "Autonomous Republic of Crimea")

# Land borders between oblasts (Kyiv City sits inside Kyiv oblast). Luhansk is excluded as a location.
NEIGHBOURS = {
    "Volynska oblast": ["Lvivska oblast", "Rivnenska oblast"],
    "Rivnenska oblast": ["Volynska oblast", "Lvivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Zhytomyrska oblast"],
    "Lvivska oblast": ["Volynska oblast", "Rivnenska oblast", "Ternopilska oblast", "Ivano-Frankivska oblast", "Zakarpatska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Ivano-Frankivska oblast": ["Lvivska oblast", "Zakarpatska oblast", "Ternopilska oblast", "Chernivetska oblast"],
    "Ternopilska oblast": ["Lvivska oblast", "Rivnenska oblast", "Khmelnytska oblast", "Chernivetska oblast", "Ivano-Frankivska oblast"],
    "Chernivetska oblast": ["Ivano-Frankivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Vinnytska oblast"],
    "Khmelnytska oblast": ["Rivnenska oblast", "Zhytomyrska oblast", "Vinnytska oblast", "Chernivetska oblast", "Ternopilska oblast"],
    "Zhytomyrska oblast": ["Rivnenska oblast", "Khmelnytska oblast", "Vinnytska oblast", "Kyivska oblast"],
    "Vinnytska oblast": ["Zhytomyrska oblast", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast", "Odeska oblast", "Chernivetska oblast", "Khmelnytska oblast"],
    "Kyivska oblast": ["Zhytomyrska oblast", "Vinnytska oblast", "Cherkaska oblast", "Poltavska oblast", "Chernihivska oblast", "Kyiv City"],
    "Kyiv City": ["Kyivska oblast"],
    "Chernihivska oblast": ["Kyivska oblast", "Poltavska oblast", "Sumska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"],
    "Poltavska oblast": ["Chernihivska oblast", "Sumska oblast", "Kharkivska oblast", "Dnipropetrovska oblast", "Kirovohradska oblast", "Cherkaska oblast", "Kyivska oblast"],
    "Cherkaska oblast": ["Kyivska oblast", "Poltavska oblast", "Kirovohradska oblast", "Vinnytska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Mykolaivska oblast", "Odeska oblast", "Vinnytska oblast"],
    "Odeska oblast": ["Vinnytska oblast", "Kirovohradska oblast", "Mykolaivska oblast"],
    "Mykolaivska oblast": ["Odeska oblast", "Kirovohradska oblast", "Dnipropetrovska oblast", "Khersonska oblast"],
    "Khersonska oblast": ["Mykolaivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"],
    "Dnipropetrovska oblast": ["Poltavska oblast", "Kharkivska oblast", "Donetska oblast", "Zaporizka oblast", "Khersonska oblast", "Mykolaivska oblast", "Kirovohradska oblast"],
    "Zaporizka oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Khersonska oblast"],
    "Donetska oblast": ["Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"],
    "Kharkivska oblast": ["Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Donetska oblast"],
}

_VOLUNTEER_NAMES = {"Kyiv": "Kyiv City"}   # English volunteer file already uses oblast names; safety net


@dataclass
class Episodes:
    """Merged alert episodes of one series: sorted, non-overlapping [start, end)."""
    start: np.ndarray   # datetime64[ns] UTC (naive)
    end: np.ndarray

    def __len__(self) -> int:
        return len(self.start)


def merge_intervals(start: np.ndarray, end: np.ndarray) -> Episodes:
    order = np.argsort(start, kind="stable")
    s, e = start[order], end[order]
    out_s, out_e = [], []
    for a, b in zip(s, e):
        if out_e and a <= out_e[-1]:
            if b > out_e[-1]:
                out_e[-1] = b
        else:
            out_s.append(a)
            out_e.append(b)
    return Episodes(np.array(out_s, dtype="datetime64[ns]"), np.array(out_e, dtype="datetime64[ns]"))


def _naive_utc(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]")


def _read(name: str, dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / name)
    for c in ("started_at", "finished_at"):
        df[c] = _naive_utc(df[c])
    df = df[df.finished_at > df.started_at]
    return df


@dataclass
class AlertData:
    source: str
    locations: list
    target: dict          # location -> Episodes (the forecast target series)
    oblast_any: dict      # location -> Episodes of ANY alert anywhere in its oblast (official only; else = target)
    data_end: np.datetime64   # last moment the source covers (latest start or end in the file)
    notes: list


def read_raw(source: str, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Alert rows of a source with columns oblast, level, raion, hromada, started_at, finished_at (UTC)."""
    if source == "volunteer":
        df = _read("volunteer_data_en.csv", dataset_dir)
        df["oblast"] = df["region"].replace(_VOLUNTEER_NAMES)
        df["level"] = "oblast"
        df["raion"] = df["hromada"] = None
    elif source == "official":
        df = _read("official_data_en.csv", dataset_dir)
    else:
        raise ValueError(f"unknown source {source!r}: use volunteer or official")
    return df


def build_alerts(source: str, df: pd.DataFrame, data_end=None) -> AlertData:
    """Location series from raw rows. Works on any subset of rows (the leakage test feeds truncated ones)."""
    notes = [f"{source}: {len(df)} alerts, levels {df.level.value_counts().to_dict()}"]
    if "naive" in df:
        notes.append(f"guessed (naive) end times: {df.naive.mean():.1%}")
    if data_end is None:
        data_end = max(df.started_at.max(), df.finished_at[df.finished_at < np.datetime64("2262-01-01")].max())
    target, oblast_any = {}, {}
    for loc in LOCATIONS:
        g = df[df.oblast == loc]
        raion, hromada = MAIN_CITY[loc]
        m = g.level.eq("oblast")
        if source == "official":
            if raion:
                m |= g.level.eq("raion") & g.raion.eq(raion)
            if hromada:
                m |= g.level.eq("hromada") & g.hromada.fillna("").str.startswith(hromada.strip())
        target[loc] = merge_intervals(g[m].started_at.to_numpy(), g[m].finished_at.to_numpy())
        oblast_any[loc] = merge_intervals(g.started_at.to_numpy(), g.finished_at.to_numpy())
    return AlertData(source, LOCATIONS, target, oblast_any, np.datetime64(data_end, "ns"), notes)


def load_alerts(source: str, dataset_dir: Path = DATASET_DIR) -> AlertData:
    return build_alerts(source, read_raw(source, dataset_dir))


def raw_known_at(df: pd.DataFrame, t) -> pd.DataFrame:
    """Rows as a real-time system would know them at t: started before t; still running -> end unknown."""
    t = np.datetime64(t, "ns")
    k = df[df.started_at < t].copy()
    k.loc[k.finished_at > t, "finished_at"] = np.datetime64("2262-01-01", "ns")
    return k


def truncate_known(ep: Episodes, t: np.datetime64) -> Episodes:
    """What is known about a series at time t: episodes that started before t; a running one has no end yet."""
    k = np.searchsorted(ep.start, t, side="left")
    s, e = ep.start[:k].copy(), ep.end[:k].copy()
    e[e > t] = np.datetime64("2262-01-01")       # still running at t: end unknown
    return Episodes(s, e)
