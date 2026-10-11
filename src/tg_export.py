"""Parse a Telegram Desktop HTML export into a message table.

The export's tooltip shows the message time as the wall clock of the exporting
machine (Kyiv). The offset printed next to it ("UTC+02:00") is not reliable
across DST, so the wall clock is localised as Europe/Kyiv and converted to UTC.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

_MSG = re.compile(r'<div class="message (default|service)[^"]*" id="message-?(\d+)">(.*?)(?=<div class="message (?:default|service)|\Z)', re.S)
_TIME = re.compile(r'title="(\d\d\.\d\d\.\d{4} \d\d:\d\d:\d\d)')
_TEXT = re.compile(r'<div class="text">(.*?)</div>', re.S)


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def parse_export(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    files = sorted(folder.glob("messages*.html"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 1))
    rows = []
    for f in files:
        for kind, mid, body in _MSG.findall(f.read_text(encoding="utf-8")):
            if kind != "default":
                continue
            t = _TIME.search(body)
            x = _TEXT.search(body)
            if not t:
                continue
            rows.append((int(mid), t.group(1), _clean(x.group(1)) if x else ""))
    df = pd.DataFrame(rows, columns=["msg_id", "local", "text"]).drop_duplicates("msg_id")
    local = pd.to_datetime(df.pop("local"), format="%d.%m.%Y %H:%M:%S")
    df["published_at"] = local.dt.tz_localize("Europe/Kyiv", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    # ambiguous hour at the autumn DST switch: fall back to the earlier offset (rare; logged)
    amb = df.published_at.isna()
    if amb.any():
        df.loc[amb, "published_at"] = local[amb].dt.tz_localize("Europe/Kyiv", ambiguous=True).dt.tz_convert("UTC")
    return df.sort_values("msg_id").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    src, out = sys.argv[1], sys.argv[2]
    d = parse_export(src)
    d.to_csv(out, index=False)  # e.g. data/tg_kpszsu.csv.gz
    print(f"{len(d)} messages, {d.published_at.min()} .. {d.published_at.max()} -> {out}")
