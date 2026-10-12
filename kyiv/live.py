"""Forecast for right now: fetch the newest posts from public Telegram channel pages, then forecast.

    python kyiv/live.py
    python kyiv/live.py --kmda KyivCityOfficial --alertmap povitryanatrivogaaa --af kpszsu

How it works
    1. Reads the public web preview of each channel (https://t.me/s/<username>; no login, no token), page by
       page back in time, until it reaches the last message already stored in data/kyiv/ (so there is no gap).
    2. KMDA posts -> alert starts / levels / all-clears (same templates as kyiv/clean_alerts.py);
       Air Force and alert-map posts -> text signals (same rules as kyiv/signals.py).
    3. Regional alert starts (Kyiv oblast, neighbours, country) come from the same Telegram channel the training
       data was built from (eTryvoga, t.me/s/UkraineAlarmSignal), parsed with the dataset's own rules
       (data/sirens/processors/volunteer_etryvoga_processor.py), continuing from its last processed message.
       Then the forecast uses the evaluated models (models/kyiv_onset.json, kyiv_clear.json).
       If that fetch fails, it falls back to models/kyiv_*_live.json: the onset model without regional
       features, and the duration baseline for the all-clear (validation: without regional alerts the
       all-clear model is worse than the baseline).

Nothing is written to disk. Use the forecast as a risk level; when the siren sounds, follow it.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kyiv"))
import features as F  # noqa: E402   (kyiv/features.py also puts src/ on the path, behind kyiv/)
from clean_alerts import classify  # noqa: E402
from forecast import forecast, print_forecast  # noqa: E402
from signals import extract  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
_POST = re.compile(r'data-post="([^"/]+)/(\d+)"')
_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*\bjs-message_text\b[^"]*"[^>]*>(.*?)</div>', re.S)
_TIME = re.compile(r'tgme_widget_message_date"[^>]*>\s*<time[^>]*datetime="([^"]+)"', re.S)


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def parse_page(page: str) -> pd.DataFrame:
    """Messages on one t.me/s/<channel> page: msg_id, published_at (UTC), text."""
    marks = list(_POST.finditer(page))
    rows = []
    for i, m in enumerate(marks):
        chunk = page[m.start(): marks[i + 1].start() if i + 1 < len(marks) else len(page)]
        tm = _TIME.search(chunk)
        if not tm:
            continue
        tx = _TEXT.search(chunk)
        rows.append((int(m.group(2)), pd.Timestamp(tm.group(1)).tz_convert("UTC"), _clean(tx.group(1)) if tx else ""))
    df = pd.DataFrame(rows, columns=["msg_id", "published_at", "text"])
    return df.drop_duplicates("msg_id").sort_values("msg_id").reset_index(drop=True)


def _get(url: str, tries: int = 3) -> str:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise SystemExit(f"Could not read {url}: {e}")
            time.sleep(2 * (k + 1))


def fetch(username: str, stop_id: int | None, max_pages: int = 150, pause: float = 0.6) -> tuple[pd.DataFrame, bool]:
    """Newest messages of a public channel back to stop_id (inclusive). Returns (messages, reached stop_id)."""
    frames, before = [], None
    for _ in range(max_pages):
        url = f"https://t.me/s/{username}" + (f"?before={before}" if before else "")
        df = parse_page(_get(url))
        if df.empty:
            break
        frames.append(df)
        oldest = int(df.msg_id.min())
        if stop_id is not None and oldest <= stop_id:
            return pd.concat(frames).drop_duplicates("msg_id").sort_values("msg_id"), True
        if before is not None and oldest >= before:          # no progress: end of history
            break
        before = oldest
        time.sleep(pause)
    out = pd.concat(frames).drop_duplicates("msg_id").sort_values("msg_id") if frames else pd.DataFrame(
        columns=["msg_id", "published_at", "text"])
    return out, False


SIRENS = ROOT / "data" / "sirens"


def etryvoga_rules():
    """city_keywords and air_raid_keywords exactly as the dataset's processor defines them."""
    import ast
    tree = ast.parse((SIRENS / "processors" / "volunteer_etryvoga_processor.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("city_keywords", "air_raid_keywords"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out["city_keywords"], out["air_raid_keywords"]


def region_starts(msgs: pd.DataFrame) -> pd.DataFrame:
    """Alert starts per region from eTryvoga messages, with the processor's rules: messages with "триває" are
    ignored; the first matching region and keyword decide; a new start within 3 h of an active one replaces it."""
    city, air = etryvoga_rules()
    u = pd.read_csv(SIRENS / "datasets" / "volunteer_data_uk.csv", usecols=["region"])
    e = pd.read_csv(SIRENS / "datasets" / "volunteer_data_en.csv", usecols=["region"])
    uk2en = dict(zip(u.region, e.region))
    active, starts = {}, {}
    for mid, ts, text in zip(msgs.msg_id, pd.to_datetime(msgs.published_at, utc=True).dt.tz_convert(None), msgs.text):
        t = (text or "").lower()
        if "триває" in t:
            continue
        region = next((c for c, kws in city.items() if any(k.lower() in t for k in kws)), None)
        state = next((v for k, v in air.items() if k.lower() in t), None)
        if region is None or state is None:
            continue
        if state:
            if region in active and ts - active[region][0] <= pd.Timedelta("3h"):
                starts.pop(active[region][1], None)                  # rewritten, as the processor does
            active[region] = (ts, mid)
            starts[mid] = (uk2en.get(region, region), ts)
        else:
            active.pop(region, None)
    return pd.DataFrame(list(starts.values()), columns=["oblast", "started_at"])


def regional_live(max_pages: int, say) -> pd.DataFrame:
    from data import read_raw
    stored = read_raw("volunteer")[["oblast", "started_at"]]
    last = int((SIRENS / "processors" / "pkl" / "volunteer_last_processed_id.txt").read_text().strip())
    msgs, ok = fetch("UkraineAlarmSignal", last, max_pages)
    if not ok:
        raise RuntimeError("could not reach the dataset's last processed message (update data/sirens: git pull)")
    new = region_starts(msgs[msgs.msg_id > last])
    newest = pd.to_datetime(msgs.published_at.max(), utc=True).tz_convert("Europe/Kyiv")
    say(f"[live] regional alerts (t.me/s/UkraineAlarmSignal): {len(new)} new starts since the daily dataset, newest post {newest:%d %b %H:%M}")
    reg = pd.concat([stored, new], ignore_index=True)
    reg["started_at"] = pd.to_datetime(reg.started_at)
    return reg[reg.oblast != "Kyiv City"].sort_values("started_at").reset_index(drop=True)


def live_inputs(kmda_user: str, alertmap_user: str, af_user: str, max_pages: int = 150, verbose: bool = True,
                regional: bool = True) -> tuple[F.Inputs, str]:
    """Inputs up to now, and the model variant to use ('' = evaluated models, '_live' = no regional alerts)."""
    root = ROOT / "data" / "kyiv"
    posts = pd.read_csv(root / "kmda_alert_posts.csv", parse_dates=["ts"])
    sig = pd.read_csv(root / "signals.csv.gz", parse_dates=["ts"])
    sig = sig[sig.channel.isin(["af", "alertmap"])]
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    say = print if verbose else (lambda *a, **k: None)

    # KMDA: alert posts (start / level / all-clear); continuity matters most here
    last = int(posts.msg_id.max())
    msgs, ok = fetch(kmda_user, last, max_pages)
    new = msgs[msgs.msg_id > last].copy()
    new[["kind", "level"]] = [classify(x) for x in new.text] if len(new) else np.empty((0, 2))
    new = new[new.kind != ""]
    new["ts"] = pd.to_datetime(new.published_at, utc=True).dt.tz_convert(None)
    posts = pd.concat([posts, new[["msg_id", "ts", "kind", "level"]]], ignore_index=True).sort_values(["ts", "msg_id"])
    say(f"[live] KMDA (t.me/s/{kmda_user}): {len(msgs)} recent posts read, {len(new)} new alert posts"
        + ("" if ok else "  !! could not reach the stored history: alerts in the gap are missing; re-export the channel"))
    if not ok:
        raise SystemExit("Stopping: the Kyiv alert history would have a gap. Export the KMDA channel again "
                         "(Telegram Desktop, HTML), then: python src/tg_export.py <folder> data/tg_kmda.csv.gz && "
                         "python kyiv/clean_alerts.py")

    # Air Force + alert map: text signals of the last hours
    parts = [sig]
    for ch, user in (("af", af_user), ("alertmap", alertmap_user)):
        last = int(sig[sig.channel == ch].msg_id.max())
        msgs, ok = fetch(user, last, max_pages)
        new = msgs[msgs.msg_id > last]
        ev = extract(ch, new) if len(new) else pd.DataFrame(columns=sig.columns)
        parts.append(ev)
        newest = pd.to_datetime(msgs.published_at.max(), utc=True).tz_convert("Europe/Kyiv") if len(msgs) else None
        say(f"[live] {ch} (t.me/s/{user}): {len(new)} new messages, newest {newest:%d %b %H:%M}" if newest is not None
            else f"[live] {ch} (t.me/s/{user}): no messages read")
        if not ok:
            say(f"[live]   note: {ch} history has a gap before the newest {len(msgs)} messages; "
                "features look back at most 24 h, so this only matters if the gap is recent")
    sig = pd.concat(parts, ignore_index=True)
    sig["ts"] = pd.to_datetime(sig.ts)
    reg, variant = pd.DataFrame({"oblast": pd.Series(dtype=str), "started_at": pd.Series(dtype="datetime64[ns]")}), "_live"
    if regional:
        try:
            reg, variant = regional_live(max_pages, say), ""
        except (Exception, SystemExit) as e:  # noqa: BLE001
            say(f"[live] regional alerts unavailable ({str(e)[:120]}); using the models without them")
    return F.Inputs(posts.reset_index(drop=True), reg, sig.sort_values("ts").reset_index(drop=True), now, now), variant


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kmda", default="KyivCityOfficial", help="KMDA channel username")
    ap.add_argument("--alertmap", default="povitryanatrivogaaa", help="«Карта повітряних тривог» channel username")
    ap.add_argument("--af", default="kpszsu", help="Air Force channel username")
    ap.add_argument("--max-pages", type=int, default=150, help="pages of 20 posts to read back per channel")
    ap.add_argument("--no-regional", action="store_true", help="skip the regional alert fetch (use the fallback models)")
    a = ap.parse_args()
    inp, variant = live_inputs(a.kmda, a.alertmap, a.af, a.max_pages, regional=not a.no_regional)
    at = inp.feature_end
    print()
    note = ("Live: Telegram posts up to this minute; evaluated models." if variant == "" else
            "Live: Telegram posts up to this minute; fallback models (no regional alert history).")
    print_forecast(forecast(at, inp, variant=variant), at, note=note)


if __name__ == "__main__":
    main()
