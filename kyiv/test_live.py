"""Offline tests for kyiv/live.py (run: python kyiv/test_live.py).

Telegram pages are simulated with the markup of the public web preview (https://t.me/s/<channel>):
a message block with data-post="<channel>/<id>", the text in div.tgme_widget_message_text.js-message_text
(emoji wrapped in <i class="emoji"><b>..</b></i>, line breaks as <br/>) and the time in
a.tgme_widget_message_date > time[datetime]. The network call is replaced; everything else runs for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live  # noqa: E402


def _msg(channel, mid, ts, text):
    body = text.replace("\n", "<br/>")
    for e in ("🟡", "🔴", "❕", "🏍", "🛵", "💥", "✈️", "‼️"):
        body = body.replace(e, f'<i class="emoji" style="background-image:url(\'//telegram.org/img/emoji/40/X.png\')"><b>{e}</b></i>')
    return (f'<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message '
            f'text_not_supported_wrap js-widget_message" data-post="{channel}/{mid}" data-view="x">'
            f'<div class="tgme_widget_message_bubble"><div class="tgme_widget_message_author accent_color">'
            f'<a class="tgme_widget_message_owner_name" href="https://t.me/{channel}"><span dir="auto">Name</span></a></div>'
            f'<div class="tgme_widget_message_text js-message_text before_footer" dir="auto">{body}</div>'
            f'<div class="tgme_widget_message_footer compact js-message_footer"><div class="tgme_widget_message_info short '
            f'js-message_info"><span class="tgme_widget_message_views">12.3K</span><span class="copyonly"> views</span>'
            f'<span class="tgme_widget_message_meta"><a class="tgme_widget_message_date" href="https://t.me/{channel}/{mid}">'
            f'<time datetime="{ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")}" class="time">{ts:%H:%M}</time></a></span>'
            f'</div></div></div></div></div>')


def _page(channel, msgs):
    return ('<html><head><title>t</title></head><body><section class="tgme_channel_history js-message_history">'
            + "".join(_msg(channel, *m) for m in msgs) + "</section></body></html>")


def test_parse_page():
    t = pd.Timestamp("2026-10-12 16:40:05")
    df = live.parse_page(_page("kpszsu", [(30001, t, "🏍 Реактивний БпЛА курсом на Київ з півночі.\nОбережно!"),
                                          (30002, t + pd.Timedelta("1min"), "💥 Київ - вибухи & ще")]))
    assert list(df.msg_id) == [30001, 30002]
    assert df.text.iloc[0] == "🏍 Реактивний БпЛА курсом на Київ з півночі.\nОбережно!", df.text.iloc[0]
    assert df.text.iloc[1] == "💥 Київ - вибухи & ще"
    assert df.published_at.iloc[0] == pd.Timestamp("2026-10-12 16:40:05", tz="UTC")


def test_end_to_end_offline():
    """Pages newer than the stored data, paged back until the stored history is reached; then a forecast."""
    root = Path(__file__).resolve().parents[1] / "data" / "kyiv"
    posts = pd.read_csv(root / "kmda_alert_posts.csv", parse_dates=["ts"])
    sig = pd.read_csv(root / "signals.csv.gz", parse_dates=["ts"])
    now = pd.Timestamp.now(tz="UTC").tz_convert(None).floor("min")
    k0, a0, m0 = int(posts.msg_id.max()), int(sig[sig.channel == "af"].msg_id.max()), int(sig[sig.channel == "alertmap"].msg_id.max())
    kmda = [(k0 - 1, now - pd.Timedelta("30h"), "Новини міста"), (k0, posts.ts.max(), "❕Відбій повітряної тривоги!"),
            (k0 + 5, now - pd.Timedelta("50min"), "🟡 УВАГА! У Києві оголошена дронова небезпека!\nПросимо всіх..."),
            (k0 + 6, now - pd.Timedelta("40min"), "Гарбузові вікенди в КиївЗоо")]
    af = [(a0, now - pd.Timedelta("20h"), "старе"), (a0 + 1, now - pd.Timedelta("35min"), "🏍 Реактивний БпЛА з Вишгорода у напрямку центру Києва."),
          (a0 + 2, now - pd.Timedelta("10min"), "⚠ Київ\n🏍 Реактивний БпЛА над містом! Перебувайте в укриттях!")]
    am = [(m0, now - pd.Timedelta("20h"), "старе"), (m0 + 1, now - pd.Timedelta("12min"), "✈️ БПЛА→Бровари/Київ")]
    r0 = int((live.SIRENS / "processors" / "pkl" / "volunteer_last_processed_id.txt").read_text().strip())
    reg = [(r0, now - pd.Timedelta("9h"), "старе"),
           (r0 + 1, now - pd.Timedelta("70min"), "🔴 Повітряна тривога в Київській області"),
           (r0 + 2, now - pd.Timedelta("65min"), "🔴 Тривога! Чернігівська область"),
           (r0 + 3, now - pd.Timedelta("30min"), "Повітряна тривога в Київській області триває")]
    pages = {"https://t.me/s/KMDA": _page("KMDA", kmda[2:]), f"https://t.me/s/KMDA?before={k0 + 5}": _page("KMDA", kmda[:2]),
             "https://t.me/s/AF": _page("AF", af), "https://t.me/s/MAP": _page("MAP", am),
             "https://t.me/s/UkraineAlarmSignal": _page("UkraineAlarmSignal", reg)}
    calls = []

    def fake_get(url, tries=3):
        calls.append(url)
        return pages.get(url, _page("x", []))
    live._get = fake_get
    inp, variant = live.live_inputs("KMDA", "MAP", "AF", max_pages=5, verbose=False)
    assert variant == "", "regional fetch should succeed -> evaluated models"
    assert f"https://t.me/s/KMDA?before={k0 + 5}" in calls, calls          # paged back to reach stored history
    recent = inp.regional[inp.regional.started_at >= now - pd.Timedelta("2h")]
    assert sorted(recent.oblast) == ["Chernihivska oblast", "Kyivska oblast"], recent
    new_posts = inp.posts[inp.posts.msg_id > k0]
    assert list(new_posts.kind) == ["start"] and list(new_posts.level) == ["yellow"], new_posts
    new_sig = inp.signals[(inp.signals.channel == "af") & (inp.signals.msg_id > a0)]
    # 1, not 2: «⚠ Київ / 🏍 ... над містом!» names the city on its own line, and the rules work line by line
    # (known limitation, identical in training and live use)
    assert (new_sig.signal == "kyiv").sum() == 1, new_sig
    for v in ("", "_live"):
        r = live.forecast(inp.feature_end, inp, variant=v)
        assert r["task"] == "clear" and int(r["state"].level_now) == 1 and 45 <= r["state"].elapsed_min <= 51
        assert (r["p"] >= 0).all() and (r["p"] <= 1).all() and (pd.Series(r["p"]).diff().dropna() >= 0).all()
        print(f"  forecast on simulated pages, models '{v or 'evaluated'}':",
              {int(h): round(float(p), 3) for h, p in zip(r["h"], r["p"])})
    # regional page unreachable -> fallback models
    pages.pop("https://t.me/s/UkraineAlarmSignal")
    inp2, variant2 = live.live_inputs("KMDA", "MAP", "AF", max_pages=2, verbose=False)
    assert variant2 == "_live" and inp2.regional.empty


def test_gap_stops():
    """If the newest pages never reach the stored KMDA history, live.py must stop instead of forecasting."""
    live._get = lambda url, tries=3: _page("KMDA", [(10**9, pd.Timestamp.now(tz="UTC").tz_convert(None), "текст")]) \
        if "before" not in url else _page("x", [])
    try:
        live.live_inputs("KMDA", "MAP", "AF", max_pages=2, verbose=False)
    except SystemExit as e:
        assert "gap" in str(e)
        return
    raise AssertionError("gap not detected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
