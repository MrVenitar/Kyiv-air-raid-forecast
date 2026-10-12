"""Kyiv-relevant signals from Telegram channels, as timestamped events.

    python kyiv/signals.py        # writes data/kyiv/signals.csv.gz and reports/kyiv/signal_samples.md

Each message is split into lines; a line can raise several signals. An event = (publication time, channel,
signal, number of matching lines). Only the publication time is used later, so a forecast at time t sees
messages published before t. (Edits are invisible in Telegram HTML exports: a message counts with its final text.)

Channels (Telegram Desktop exports parsed by src/tg_export.py):
    af          Air Force of the Armed Forces of Ukraine (official, Ukrainian)
    alertmap    «Карта повітряних тривог | Вибухи | Ракети» (aggregator: drone vectors, explosions)
    monitoring  «Моніторинг | Україна» (@monitor_ukr, volunteer monitoring)
    vanek       «Николаевский Ванёк» (volunteer monitoring, mostly Russian)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {"af": "tg_kpszsu.csv.gz", "alertmap": "tg_alertmap.csv.gz",
            "monitoring": "tg_monitoring.csv.gz", "vanek": "tg_vanek.csv.gz"}

# Kyiv city: Ukrainian and Russian case forms; not Kyiv oblast («Київщина», «Київська», «Киевская»),
# not «Києво-Святошинський» (an oblast district); plus city districts.
KYIV = (r"(?<![\w-])(?:Київ(?!щ|ськ|\w)|Києв(?!о-|щ)\w*|Киев(?!щ|ск)\w*|Kyiv\b|столиц\w*|"
        r"Троєщин\w*|Троещин\w*|Оболон\w*|Позняк\w*|Осокорк\w*|Нивк\w*|Виноградар\w*|Борщагівк\w*|Теремк\w*|"
        r"Голосі\w*|Голосее\w*|Соло[мʼ'’]*ʼ?ян\w*|Дарниц\w*|Печерськ\w*|Печерск\w*|Подол\w*|Поділ\w*|Святошин(?!ськ\w* район\w* Київщ)\w*)")
KYIV_OBL = (r"Київщ\w*|Київськ\w* обл\w*|Киевщ\w*|Киевск\w* обл\w*|Бровар\w*|Борисп\w*|Вишгород\w*|Вышгород\w*|"
            r"Ірпін\w*|Ирпен\w*|(?<![\w-])Буч[аіуе]\b|Обухів\w*|Обухов\w*|Васильк(?:ів|ова|ові|ов\b|ове|ову)|"
            r"Біл\w* Церк\w*|Бел\w* Церк\w*|Білоцерк\w*|Белоцерк\w*|Бородянк\w*|Гостомел\w*|Українк\w*|Украинк\w*|"
            r"Переяслав\w*|Яготин\w*|Славутич\w*|Фастів\w*|Фастов\w*|Боярк\w*|Макарів\w*|Макаров\w*|Димер\w*|"
            r"Ржищів\w*|Кагарлик\w*|Березан\w*|Баришів\w*|Згурівк\w*|Іванків\w*|Чорнобил\w*|Чернобыл\w*|Козин\w*|"
            r"Глевах\w*|Вишнев\w*|Коцюбинськ\w*|Тетіїв\w*|Сквир\w*|Богуслав\w*|Кіпті|Козелець")
NEIGHBOUR = r"Чернігів\w*|Чернігівщ\w*|Черниг\w*|Житомир\w*|Черкас\w*|Черкащ\w*|Полтав\w*|Сумщ\w*|Сумск\w*|Брянщ\w*|Брянск\w*|Курськ\w*|Курск\w*|Ніжин\w*|Нежин\w*|Прилук\w*"
BOOM = r"вибух\w*|взрыв\w*|громк\w*|💥|уламк\w*|обломк\w*|ППО|ПВО|прил[её]т\w*|влучан\w*|падіння\w*"
THREATS = {
    "ballistic": r"балістик\w*|баллистик\w*|швидкісн\w*|скоростн\w*|Іскандер\w*|Искандер\w*|КН-23|Циркон\w*|Кинджал\w*|Кінджал\w*",
    "mig31": r"МіГ-31|МиГ-31|Миг-31|МІГ-31|MiG-31",
    "strategic": r"Ту-95|Ту-160|Ту-22|стратег\w*|Енгельс\w*|Энгельс\w*|Олень\w*|Оленя",
    "missile": r"крилат\w*|крылат\w*|Калібр\w*|Калибр\w*|(?<![\w-])КР(?![\w-])|Х-101|Х-555|Х-59|Х-69|ракет\w*",
    "jet": r"реактив\w*|🏍|Герань-3|Гербера",
    "drone": r"БпЛА|БПЛА|БпЛA|БПЛA|[Шш]ахед\w*|мопед\w*|дрон\w*|Shahed|✈️|🛵",
    "launch": r"пуск\w*|зліт\w*|злет\w*|взл[её]т\w*|вил[іе]т\w*|выл[её]т\w*",
}
SIGNALS = ["msg", "kyiv", "kyiv_obl", "toward_kyiv", "kyiv_boom", "all_clear"] + list(THREATS)
# daily/after-the-fact summaries describe the past: they keep 'msg' only
SUMMARY = re.compile(r"У ніч на \d|[Пп]ротивник атакував|протягом (минулої )?доби|[Зз]а попередніми даними|ЗБИТО|збито/подавлено", re.I)
CLEAR = re.compile(r"[Вв]ідбій|[Оо]тбой|🟢", re.I)
# a Kyiv mention right after "from" (з / із / зі / из / с / від / от, up to two words in between) means moving away
AWAY = re.compile(r"(?:^|[\s(/→➡️-])(?:з|із|зі|из|с|від|от)\s+(?:\S+\s+){0,2}$", re.I)

_K, _KO, _N, _B = (re.compile(p, re.I) for p in (KYIV, KYIV_OBL, NEIGHBOUR, BOOM))
_T = {k: re.compile(p, re.I) for k, p in THREATS.items()}
_THREAT_ANY = re.compile("|".join(THREATS.values()) + r"|→|➡️|курс\w*|напрям\w*|летят|летит|летять|летить|в сторону|у бік|на/через|обережно", re.I)


def _toward(line: str) -> bool:
    """True if some Kyiv (city/oblast) mention in the line is not preceded by a 'from' word."""
    for rx in (_K, _KO):
        for m in rx.finditer(line):
            if not AWAY.search(line[:m.start()]):
                return True
    return False


def line_signals(line: str) -> dict:
    """Signals raised by one line of text."""
    if CLEAR.search(line):
        return {"all_clear": 1}
    out = {}
    k, ko = bool(_K.search(line)), bool(_KO.search(line))
    threat = bool(_THREAT_ANY.search(line))
    boom = bool(_B.search(line))
    if k and (threat or boom):
        out["kyiv"] = 1
    if ko and (threat or boom):
        out["kyiv_obl"] = 1
    if (k or ko) and _N.search(line) and threat and _toward(line):
        out["toward_kyiv"] = 1
    if k and boom:
        out["kyiv_boom"] = 1
    for name, rx in _T.items():
        if rx.search(line):
            out[name] = 1
    return out


def message_signals(text: str, context: dict | None = None) -> dict:
    """Sum of line signals in a message, plus 'msg'. Alert-map posts put the region in a header line
    («✈️Київщина:») with the targets below it; such a header is carried to the following lines."""
    out = {"msg": 1}
    if SUMMARY.search(text or ""):
        return out
    header = ""
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            header = ""
            continue
        if re.match(r"^(✈️|🛵|🏍)?\s*[А-ЯІЇЄҐA-Z][\w'ʼ’-]+(щина|ччина|щині|ччині)\s*:\s*$", line):
            header = line
            continue
        for k, v in line_signals(f"{header} {line}" if header else line).items():
            out[k] = out.get(k, 0) + v
    return out


def extract(channel: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ts = pd.to_datetime(df.published_at, utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]")
    for t, mid, txt in zip(ts, df.msg_id.to_numpy(), df.text.fillna("").to_numpy()):
        for sig, n in message_signals(txt).items():
            rows.append((t, channel, sig, n, mid))
    return pd.DataFrame(rows, columns=["ts", "channel", "signal", "n", "msg_id"])


def main() -> None:
    out = ROOT / "data" / "kyiv"
    rep = ROOT / "reports" / "kyiv"
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    parts, samples = [], ["# Text signals: random matched lines (precision spot check)\n"]
    rng = np.random.default_rng(0)
    for ch, f in CHANNELS.items():
        df = pd.read_csv(ROOT / "data" / f, parse_dates=["published_at"])
        df = df[df.published_at >= "2023-06-01"]
        ev = extract(ch, df)
        parts.append(ev)
        recent = df[df.published_at >= "2026-08-01"]
        samples.append(f"\n## {ch}: {len(df)} messages since 2023-06; signal counts since then: "
                       + ", ".join(f"{s}={int(ev[ev.signal == s].n.sum())}" for s in SIGNALS))
        for sig in ("kyiv", "toward_kyiv", "kyiv_boom", "ballistic", "all_clear"):
            hits = []
            for txt in recent.text.fillna("").to_numpy():
                if SUMMARY.search(txt):
                    continue
                for raw in txt.split("\n"):
                    if sig in line_signals(raw.strip()):
                        hits.append(raw.strip())
            if hits:
                pick = rng.choice(len(hits), min(6, len(hits)), replace=False)
                samples.append(f"\n**{sig}** ({len(hits)} lines since Aug 2026), random sample:\n")
                samples += [f"- {hits[i][:160]}" for i in pick]
    ev = pd.concat(parts, ignore_index=True).sort_values(["ts", "channel"], kind="stable")
    ev.to_csv(out / "signals.csv.gz", index=False, date_format="%Y-%m-%d %H:%M:%S")
    (rep / "signal_samples.md").write_text("\n".join(samples), encoding="utf-8")
    print("\n".join(samples))
    print(f"\n{len(ev):,} signal events -> {out / 'signals.csv.gz'}")


if __name__ == "__main__":
    main()
