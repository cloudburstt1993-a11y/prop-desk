"""Build tonight's NBA prop board.

    python3 build.py [--date YYYY-MM-DD] [--history history.json] [--out data.json]

1. Downloads (or reads from cache) every regular-season and playoff box score
   for last season and this one (seasons roll over automatically each August).
2. Replays the seasons day by day: project, then learn from that day's results.
   That replay is the backtest.
3. Grades this season's real DraftKings lines once each, then freezes the
   result in history.json (ESPN drops old lines after a while).
4. Projects every DraftKings player prop on the date's slate.
"""
import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import espn
import model as M

MKTS = ["pts", "reb", "ast", "tpm", "pr", "pa", "ra", "pra", "stl", "blk"]
LEAN_EDGE = 0.05               # model vs no-vig market, to label a lean
ROLE_GAP = 0.25                # model this far from the line early on = probably a role change it can't see

# Top 5 per game. Fixed rules, chosen before looking at results; backtested on last season's lines.
TOP_N = 5
TOP_MIN_GAP = 0.03             # model vs no-vig market, either direction
TOP_WORST_PRICE = -135
TOP_MIN_MINUTES = 24


def top_picks(cands):
    """Pick up to TOP_N props for one game.

    A candidate needs: no injury tag, no "Check" flag, 24+ projected minutes, a 3+ point gap
    between the model and the no-vig market, a price no worse than -135, and the player's
    own record agreeing with the model's side over his last 10 games (6+/10 over, or 4-/10)
    and his last 40 (half or more over, or half or fewer). Ranked by gap, one prop per player."""
    ok = []
    for c in cands:
        if c["check"] or c["injured"] or c["min"] < TOP_MIN_MINUTES:
            continue
        gap = c["pModel"] - c["pMarket"]
        if abs(gap) < TOP_MIN_GAP:
            continue
        over = gap > 0
        price = c["over"] if over else c["under"]
        if price < TOP_WORST_PRICE or c["n10"] < 8 or c["nl"] < 20:
            continue
        r10, rl = c["h10"] / c["n10"], c["hl"] / c["nl"]
        if over and not (r10 >= 0.6 and rl >= 0.5):
            continue
        if not over and not (r10 <= 0.4 and rl <= 0.5):
            continue
        ok.append(dict(c, side="over" if over else "under", price=price, gap=round(gap, 3),
                       r10=round(r10, 3), rl=round(rl, 3)))
    ok.sort(key=lambda c: -abs(c["gap"]))
    out, seen = [], set()
    for c in ok:
        if c["pid"] in seen:
            continue
        seen.add(c["pid"])
        out.append(c)
        if len(out) == TOP_N:
            break
    return out


def grade_picks(target, picks):
    """Fold graded Top 5 picks (each with "actual", or None if the player sat) into a record."""
    rec = target.setdefault("record", empty_record())
    nights = {n["day"]: n for n in target.setdefault("nights", [])}
    recent = target.setdefault("recent", [])
    for p in sorted(picks, key=lambda p: p["day"]):
        if p.get("actual") is None:
            continue
        rec["props"] += 1
        night = nights.setdefault(p["day"], {"day": p["day"], "w": 0, "l": 0, "p": 0, "units": 0.0})
        bm = rec["byMkt"].setdefault(p["mkt"], {"w": 0, "l": 0})
        if p["actual"] == p["line"]:
            res = "push"; rec["pushes"] += 1; night["p"] += 1
        elif (p["actual"] > p["line"]) == (p["side"] == "over"):
            won = p["price"] / 100 if p["price"] > 0 else 100 / -p["price"]
            res = "win"; rec["wins"] += 1; rec["units"] += won; bm["w"] += 1; night["w"] += 1; night["units"] += won
        else:
            res = "loss"; rec["losses"] += 1; rec["units"] -= 1; bm["l"] += 1; night["l"] += 1; night["units"] -= 1
        recent.append({k: p.get(k) for k in ("day", "name", "team", "mkt", "line", "side", "price", "mean", "actual")}
                      | {"result": res})
    target["recent"] = recent[-200:]
    target["nights"] = sorted(nights.values(), key=lambda n: n["day"])
    for n in target["nights"]:
        n["units"] = round(n["units"], 3)
    n = rec["wins"] + rec["losses"]
    rec["units"] = round(rec["units"], 3)
    rec["picks"] = n + rec["pushes"]
    rec["roi"] = round(rec["units"] / n, 4) if n else None
    rec["days"] = [x["day"] for x in target["nights"]]
    rec["nights"] = target["nights"]


def decide(pm, pmk, mean, line, n_cur):
    """Lean over / under / none, and whether the gap looks like an unseen role change."""
    check = n_cur < 5 and abs(mean - line) / max(line, 1.0) > ROLE_GAP
    if check:
        return None, True
    side = "over" if pm - pmk >= LEAN_EDGE else "under" if pmk - pm >= LEAN_EDGE else None
    return side, False


def mtn_today():
    # Slates are scheduled in US time; Mountain time is the owner's.
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6)).date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=mtn_today().isoformat())
    ap.add_argument("--history", default="history.json")
    ap.add_argument("--out", default="data.json")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date)
    cur_season = espn.season_of(today)
    prev_season = espn.season_of(today.replace(year=today.year - 1))
    backtest_from = f"{prev_season[:4]}-12-01"   # a month of data before the replay is scored

    history = load_history(a.history)
    pending = history["pending"]                  # eventId -> DraftKings lines not graded yet
    graded_ids = set(history["gradedEvents"])

    # ---------- data ----------
    games = []
    for season in (prev_season, cur_season):
        games += espn.season_games(season, through=today - dt.timedelta(days=1))
    nba = set(espn.team_ids())   # drops All-Star and exhibition teams
    done = [g for g in games if g["completed"] and g["day"] < today.isoformat()
            and g["home"] in nba and g["away"] in nba]
    boxes = espn.boxscores([g["id"] for g in done])
    print(f"{len(done)} completed games", file=sys.stderr)

    # Closing lines for this season's finished games that haven't been graded yet.
    to_fetch = [g for g in done if g["season"] == cur_season and g["id"] not in graded_ids]
    with ThreadPoolExecutor(8) as ex:
        for g, p in zip(to_fetch, ex.map(lambda g: espn.props(g["id"]), to_fetch)):
            if p:
                pending[g["id"]] = p

    slate = [g for g in espn.scoreboard(today, refresh=True)
             if g["seasonType"] in espn.GAME_TYPES and g["home"] in nba and g["away"] in nba]
    with ThreadPoolExecutor(8) as ex:
        slate_props = dict(zip([g["id"] for g in slate], ex.map(lambda g: espn.props(g["id"]), slate)))
    for eid, p in slate_props.items():
        if p:
            pending[eid] = p          # fallback in case ESPN drops the lines after the game

    # One time only: grade whatever DraftKings lines ESPN still has for last season.
    archive_lines = {}
    if "archive" not in history or "top5Archive" not in history:
        prev_done = [g for g in done if g["season"] == prev_season]
        with ThreadPoolExecutor(12) as ex:
            for g, p in zip(prev_done, ex.map(lambda g: espn.props(g["id"]), prev_done)):
                if p:
                    archive_lines[g["id"]] = p
        print(f"archive: lines for {len(archive_lines)} of last season's games", file=sys.stderr)
    inj = espn.injuries()

    # ---------- replay ----------
    st = M.State()
    by_day = defaultdict(list)
    for g in done:
        by_day[g["day"]].append(g)
    bt = defaultdict(list)          # stat -> [(mean, actual, naive_season, naive_l10)]
    to_grade = []                   # real-line results not graded before
    to_archive = []                 # last season's real-line results (first run only)
    for day in sorted(by_day):
        day_games = by_day[day]
        season = day_games[0]["season"]
        st.new_season(season)
        score = day >= backtest_from
        for g in day_games:
            src = archive_lines.get(g["id"]) or ([] if g["id"] in graded_ids else pending.get(g["id"], []))
            ev_lines = {(p["pid"], p["mkt"]): p for p in src}
            sink = to_archive if g["id"] in archive_lines else to_grade
            box = boxes.get(g["id"])
            if not box or not (score or ev_lines):
                continue
            rows = box["rows"]
            out_by_team = defaultdict(list)
            for r in rows:
                if r["dnp"]:
                    out_by_team[r["team"]].append(r["pid"])
            spread = parse_spread(g)
            for r in rows:
                if r["dnp"]:
                    continue
                team = r["team"]
                opp = g["away"] if team == g["home"] else g["home"]
                pr = M.project(st, r["pid"], team, opp, day, season, team == g["home"],
                               out_by_team[team], spread_for(spread, team, g))
                if not pr:
                    continue
                prior = [x for x in st.logs[r["pid"]] if not x["dnp"]]
                cur_prior = [x for x in prior if x["season"] == season]
                for s in MKTS:
                    actual = M.stat_of(r, s)
                    if score and pr["min"] >= 15 and len(prior) >= 10:
                        naive = statistics.fmean(M.stat_of(x, s) for x in (cur_prior or prior))
                        l10 = statistics.fmean(M.stat_of(x, s) for x in prior[-10:])
                        bt[s].append((pr["mean"][s], actual, naive, l10))
                    if (r["pid"], s) in ev_lines:
                        ln = ev_lines[(r["pid"], s)]
                        l10, l40 = prior[-10:], prior[-40:]
                        sink.append({"day": day, "event": g["id"], "pid": r["pid"], "name": r["name"],
                                     "team": team, "mkt": s, "line": ln["line"], "over": ln["over"],
                                     "under": ln["under"], "mean": pr["mean"][s], "actual": actual,
                                     "nCur": pr["nCur"], "min": pr["min"],
                                     "h10": sum(M.stat_of(x, s) > ln["line"] for x in l10), "n10": len(l10),
                                     "hl": sum(M.stat_of(x, s) > ln["line"] for x in l40), "nl": len(l40)})
        for g in day_games:
            st.add_game(g, boxes.get(g["id"]))

    # ---------- fit dispersion and score the backtest ----------
    disp, bt_summary = {}, {}
    for s in MKTS:
        rows = bt[s]
        if len(rows) < 500:
            disp[s] = history.get("fit", {}).get(s) or {"a": 1.3, "b": 0.0, "k": 0.8}
            continue
        disp[s] = fit_spread(rows)
        mae = lambda i: statistics.fmean(abs(r[i] - r[1]) for r in rows)
        bt_summary[s] = {"n": len(rows), "maeModel": mae(0), "maeSeasonAvg": mae(2), "maeLast10": mae(3),
                         "fit": disp[s], "calibration": calibrate(rows, disp[s])}
    history["fit"] = disp
    st.new_season(cur_season)

    if os.environ.get("DUMP_GRADED"):
        with open(os.environ["DUMP_GRADED"], "w") as f:
            json.dump([dict(g, pModel=M.p_over(g["line"], g["mean"], disp[g["mkt"]]),
                            pMarket=M.no_vig(g["over"], g["under"])) for g in to_grade], f)

    # Grade once, then freeze: the record is what the model said at the time.
    grade_into(history, to_grade, disp)
    if "archive" not in history:
        history["archive"] = {"season": prev_season, "record": empty_record(), "recent": [], "nights": []}
        grade_into(history["archive"], to_archive, disp)
    if "top5Archive" not in history:
        history["top5Archive"] = {"season": prev_season}
        grade_picks(history["top5Archive"], backtest_top(to_archive, disp))

    # Tonight's earlier Top 5 picks, locked at tip-off: grade the ones whose games are final.
    t5 = history.setdefault("top5", {})
    t5_pending = history.setdefault("top5Pending", {})
    finished = []
    for eid in [e for e in t5_pending if e in boxes and boxes[e]]:
        played = {r["pid"]: r for r in boxes[eid]["rows"] if not r["dnp"]}
        for pk in t5_pending.pop(eid)["picks"]:
            row = played.get(pk["pid"])
            finished.append(dict(pk, actual=M.stat_of(row, pk["mkt"]) if row else None))
    grade_picks(t5, finished)

    for eid in {r["event"] for r in to_grade} | {g["id"] for g in done if g["id"] in pending}:
        graded_ids.add(eid)
        pending.pop(eid, None)
    history["gradedEvents"] = sorted(graded_ids)

    # ---------- tonight ----------
    out_ids = defaultdict(list)
    questionable = {}
    for pid, i in inj.items():
        info = st.info.get(pid)
        if not info:
            continue
        if (i["status"] or "").lower() in ("out", "doubtful", "suspension"):
            out_ids[info["team"]].append(pid)
        else:
            questionable[pid] = i

    board, players, games_out = [], {}, []
    for g in slate:
        spread = parse_spread(g)
        games_out.append({**g, "spreadNum": spread, "news": team_news([g["home"], g["away"]]),
                          "playoffs": g["seasonType"] == 3,
                          "b2b": {t: st.played_on(t, M.prev_day(today.isoformat())) for t in (g["home"], g["away"])},
                          "pace": {t: round(st.team_pace(t)[0], 1) for t in (g["home"], g["away"])},
                          "out": {t: [inj_row(p, inj, st, cur_season) for p in out_ids[t]] for t in (g["home"], g["away"])}})
        for p in slate_props.get(g["id"], []):
            pid = p["pid"]
            info = st.info.get(pid)
            if not info:
                a_ = espn.athlete(pid)
                if not a_:
                    continue
                info = {"name": a_["name"], "pos": a_["pos"], "team": None}
            team = info.get("team")
            if team not in (g["home"], g["away"]):
                continue
            opp = g["away"] if team == g["home"] else g["home"]
            pr = M.project(st, pid, team, opp, today.isoformat(), cur_season, team == g["home"],
                           out_ids[team], spread_for(spread, team, g))
            if pid not in players:
                players[pid] = player_card(st, pid, team, opp, g, pr, inj.get(pid), questionable.get(pid))
            if not pr:
                continue
            s = p["mkt"]
            mean = pr["mean"][s]
            pm = M.p_over(p["line"], mean, disp[s])
            pmk = M.no_vig(p["over"], p["under"])
            side, check = decide(pm, pmk, mean, p["line"], pr["nCur"])
            hist = [x for x in st.logs[pid] if not x["dnp"]]
            cur = [x for x in hist if x["season"] == cur_season]
            last = [x for x in hist if x["season"] != cur_season]
            vs = [x for x in hist if x["opp"] == opp]
            board.append({
                "pid": pid, "event": g["id"], "mkt": s, "line": p["line"], "openLine": p["openLine"],
                "over": p["over"], "under": p["under"], "overOpen": p["overOpen"], "updated": p["updated"],
                "mean": round(mean, 2), "pModel": round(pm, 3), "pMarket": round(pmk, 3),
                "edge": round(pm - pmk, 3), "side": side, "check": check,
                "ev": round(M.ev_per_100(pm if side != "under" else 1 - pm, p["over"] if side != "under" else p["under"]), 1),
                "hit": {k: hit_rate(v, s, p["line"]) for k, v in
                        (("l5", hist[-5:]), ("l10", hist[-10:]), ("l20", hist[-20:]), ("l40", hist[-40:]),
                         ("season", cur), ("last", last), ("vs", vs))},
                "min": pr["min"], "nCur": pr["nCur"],
                "newTeam": bool(last) and last[-1]["team"] != team,
                "avg": {k: avg(v, s) for k, v in (("l5", hist[-5:]), ("l10", hist[-10:]), ("season", cur), ("last", last))},
                "oppRank": opp_rank(st, opp, pr["bucket"], s),
                "factors": {k: round(v, 3) for k, v in combo_factors(pr, s).items()},
            })

    top5 = {}
    for g in slate:
        cands = []
        for b in board:
            if b["event"] != g["id"] or not b["hit"]["l10"] or not b["hit"]["l40"]:
                continue
            pl = players[b["pid"]]
            cands.append({"pid": b["pid"], "name": pl["name"], "team": pl["team"], "event": g["id"],
                          "day": today.isoformat(), "mkt": b["mkt"], "line": b["line"], "over": b["over"],
                          "under": b["under"], "mean": b["mean"], "pModel": b["pModel"], "pMarket": b["pMarket"],
                          "check": b["check"], "injured": bool(pl.get("injury")), "min": b["min"],
                          "h10": b["hit"]["l10"]["over"], "n10": b["hit"]["l10"]["n"],
                          "hl": b["hit"]["l40"]["over"], "nl": b["hit"]["l40"]["n"],
                          "openLine": b["openLine"], "nCur": b["nCur"], "newTeam": b["newTeam"]})
        picks = top_picks(cands)
        top5[g["id"]] = picks
        if g["status"] == "STATUS_SCHEDULED" and picks:
            t5_pending[g["id"]] = {"day": today.isoformat(), "picks": picks}   # last pre-game version wins

    with ThreadPoolExecutor(8) as ex:
        for pid, n in zip(list(players), ex.map(espn.player_news, list(players))):
            players[pid]["news"] = n

    data = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "date": today.isoformat(),
        "season": cur_season,
        "seasonStart": espn.season_window(cur_season)[0].isoformat(),
        "games": games_out,
        "players": players,
        "props": board,
        "backtest": {"from": backtest_from, "through": max(by_day) if by_day else None, "stats": bt_summary},
        "real": history["record"],
        "archive": {"season": history["archive"]["season"], "record": history["archive"]["record"]},
        "realRecent": history["recent"][-60:],
        "top5": top5,
        "top5Record": {"record": t5.get("record", empty_record()), "recent": t5.get("recent", [])[-40:]},
        "top5Archive": {"season": history["top5Archive"]["season"], "record": history["top5Archive"]["record"]},
        "top5Rules": {"n": TOP_N, "gap": TOP_MIN_GAP, "price": TOP_WORST_PRICE, "min": TOP_MIN_MINUTES},
        "labels": M.LABEL,
        "leanEdge": LEAN_EDGE,
    }
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    with open(a.history, "w") as f:
        json.dump(history, f, separators=(",", ":"))
    print(f"{len(slate)} games, {len(board)} props, {len(players)} players, "
          f"{len(to_grade)} props graded this run -> {a.out}", file=sys.stderr)


def load_history(path):
    """history.json keeps the forward record small: lines waiting to be graded,
    running totals, the last 200 graded leans, and one row per night."""
    h = {}
    if os.path.exists(path):
        with open(path) as f:
            h = json.load(f)
    if "lines" in h:                      # first format: raw lines for every game
        h.setdefault("pending", {}).update(h.pop("lines"))
        h.pop("graded", None)
    h.setdefault("pending", {})
    h.setdefault("gradedEvents", [])
    h.setdefault("recent", [])
    h.setdefault("nights", [])
    h.setdefault("record", empty_record())
    h["record"].setdefault("buckets", {})
    return h


def empty_record():
    return {"props": 0, "picks": 0, "wins": 0, "losses": 0, "pushes": 0, "units": 0.0,
            "roi": None, "days": [], "byMkt": {}, "buckets": {}}


EDGE_BUCKETS = [(0, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.2), (0.2, 1.01)]


# ---------- helpers ----------
def parse_spread(g):
    """Home-team spread from ESPN's 'PHI -2.5' style string (negative = home favored)."""
    d = g.get("spread")
    if not d or d.upper() == "EVEN":
        return 0.0 if d else None
    try:
        team, num = d.rsplit(" ", 1)
        num = float(num)
        return num if team == g["home"] else -num
    except Exception:
        return None


def spread_for(home_spread, team, g):
    if home_spread is None:
        return None
    return home_spread if team == g["home"] else -home_spread


def fit_spread(rows):
    """Least-squares fit of squared error = a*mean + b*mean^2, then pick the
    shrink k that minimizes log-loss on synthetic lines (last-10 average)."""
    sxx = sxy = syy = sx = sy = 0.0
    for m, y, _, _ in rows:
        e2 = (y - m) ** 2
        x1, x2 = m, m * m
        sxx += x1 * x1; sxy += x1 * x2; syy += x2 * x2
        sx += x1 * e2; sy += x2 * e2
    det = sxx * syy - sxy * sxy
    a = (sx * syy - sy * sxy) / det if det else 1.3
    bb = (sy * sxx - sx * sxy) / det if det else 0.0
    a, bb = max(1.0, a), max(0.0, bb)
    sample = rows[::3]
    best = None
    for k in [x / 20 for x in range(10, 21)]:
        fit = {"a": a, "b": bb, "k": k}
        ll = 0.0
        for m, y, _, l10 in sample:
            line = math.floor(l10) + 0.5
            p = min(0.999, max(0.001, M.p_over(line, m, fit)))
            ll -= math.log(p if y > line else 1 - p)
        if best is None or ll < best[0]:
            best = (ll, fit)
    return {k: round(v, 4) for k, v in best[1].items()}


def calibrate(rows, disp):
    """Synthetic-line check: line = last-10 average rounded to x.5.
    Buckets of model P(over) vs how often the over actually hit."""
    buckets = defaultdict(lambda: [0, 0, 0.0])
    for m, y, _, l10 in rows:
        line = math.floor(l10) + 0.5
        p = M.p_over(line, m, disp)
        b = min(9, int(p * 10))
        buckets[b][0] += 1
        buckets[b][1] += y > line
        buckets[b][2] += p
    return [{"lo": b / 10, "hi": (b + 1) / 10, "n": n, "pred": round(ps / n, 3), "actual": round(h / n, 3)}
            for b, (n, h, ps) in sorted(buckets.items()) if n >= 30]


def backtest_top(rows, disp):
    """Apply the Top 5 rules to graded real-line rows, game by game."""
    by_event = defaultdict(list)
    for r in rows:
        pm = M.p_over(r["line"], r["mean"], disp[r["mkt"]])
        pmk = M.no_vig(r["over"], r["under"])
        _, check = decide(pm, pmk, r["mean"], r["line"], r.get("nCur", 99))
        by_event[r["event"]].append(dict(r, pModel=pm, pMarket=pmk, check=check, injured=False))
    picks = []
    for ev in by_event.values():
        picks += top_picks(ev)
    return picks


def grade_into(history, graded, disp):
    """Grade new props against the lines and fold them into the running record.
    A "pick" is siding with the model whenever it is 5+ points away from the market;
    buckets track every prop by the size of the gap, whichever way it points."""
    rec = history["record"]
    for g in graded:
        pm = M.p_over(g["line"], g["mean"], disp[g["mkt"]])
        pmk = M.no_vig(g["over"], g["under"])
        if g["actual"] == g["line"]:
            continue
        gap = abs(pm - pmk)
        lo, hi = next(b for b in EDGE_BUCKETS if b[0] <= gap < b[1])
        b = rec["buckets"].setdefault(f"{lo:g}-{hi:g}", {"lo": lo, "hi": min(hi, 1), "w": 0, "l": 0, "units": 0.0})
        side_over = pm > pmk
        price = g["over"] if side_over else g["under"]
        if (g["actual"] > g["line"]) == side_over:
            b["w"] += 1
            b["units"] = round(b["units"] + (price / 100 if price > 0 else 100 / -price), 3)
        else:
            b["l"] += 1
            b["units"] = round(b["units"] - 1, 3)
    nights = {n["day"]: n for n in history["nights"]}
    for g in sorted(graded, key=lambda g: g["day"]):
        pm = M.p_over(g["line"], g["mean"], disp[g["mkt"]])
        pmk = M.no_vig(g["over"], g["under"])
        side, check = decide(pm, pmk, g["mean"], g["line"], g.get("nCur", 99))
        rec["props"] += 1
        night = nights.setdefault(g["day"], {"day": g["day"], "w": 0, "l": 0, "p": 0, "units": 0.0})
        if not side:
            continue
        price = g["over"] if side == "over" else g["under"]
        bm = rec["byMkt"].setdefault(g["mkt"], {"w": 0, "l": 0})
        if g["actual"] == g["line"]:
            res = "push"
            rec["pushes"] += 1
            night["p"] += 1
        elif (g["actual"] > g["line"]) == (side == "over"):
            res = "win"
            won = price / 100 if price > 0 else 100 / -price
            rec["wins"] += 1; rec["units"] += won; bm["w"] += 1
            night["w"] += 1; night["units"] += won
        else:
            res = "loss"
            rec["losses"] += 1; rec["units"] -= 1; bm["l"] += 1
            night["l"] += 1; night["units"] -= 1
        history["recent"].append({"day": g["day"], "name": g["name"], "mkt": g["mkt"], "line": g["line"],
                                  "side": side, "mean": round(g["mean"], 2), "actual": g["actual"], "result": res,
                                  "pModel": round(pm, 3), "pMarket": round(pmk, 3)})
    history["recent"] = history["recent"][-200:]
    history["nights"] = sorted(nights.values(), key=lambda n: n["day"])
    for n in history["nights"]:
        n["units"] = round(n["units"], 3)
    n = rec["wins"] + rec["losses"]
    rec["units"] = round(rec["units"], 3)
    rec["picks"] = n + rec["pushes"]
    rec["roi"] = round(rec["units"] / n, 4) if n else None
    rec["days"] = [x["day"] for x in history["nights"]]
    rec["nights"] = history["nights"]


def hit_rate(rows, s, line):
    if not rows:
        return None
    return {"over": sum(M.stat_of(r, s) > line for r in rows), "n": len(rows)}


def avg(rows, s):
    return round(statistics.fmean(M.stat_of(r, s) for r in rows), 1) if rows else None


def combo_factors(pr, s):
    parts = M.COMBOS.get(s, (s,))
    tot = sum(pr["mean"][p] for p in parts) or 1
    out = {}
    for k in ("opp", "pace", "usage"):
        out[k] = sum(pr["factors"][p][k] * pr["mean"][p] for p in parts) / tot
    return out


def opp_rank(st, opp, b, s):
    """Rank of the opponent among teams for what it allows at this position (1 = allows the most)."""
    parts = M.COMBOS.get(s, (s,))
    teams = {t for t, _ in st.def_sum}
    score = {t: sum(st.opp_factor(t, b, p) * st.league_rate(b, p) for p in parts) for t in teams}
    order = sorted(score, key=score.get, reverse=True)
    lg = sum(st.league_rate(b, p) for p in parts) or 1
    return {"rank": order.index(opp) + 1 if opp in order else None, "of": len(order),
            "vsLeague": round(score.get(opp, lg) / lg - 1, 3), "bucket": b}


def player_card(st, pid, team, opp, g, pr, injury, q):
    info = st.info.get(pid) or {}
    hist = [x for x in st.logs.get(pid, []) if not x["dnp"]]
    log = [{"day": x["day"], "opp": x["opp"], "home": x["home"], "min": x["min"], **{s: M.stat_of(x, s) for s in MKTS}}
           for x in hist[-20:]]
    return {
        "name": info.get("name"), "team": team, "opp": opp, "pos": info.get("pos"), "event": g["id"],
        "projMin": round(pr["min"], 1) if pr else None, "baseMin": round(pr["baseMin"], 1) if pr else None,
        "mates": pr["mates"] if pr else [], "notes": pr["notes"] if pr else ["Not enough NBA minutes to project."],
        "b2b": pr["b2b"] if pr else False, "games": len(hist), "log": log,
        "injury": injury or q,
        "means": {k: round(v, 2) for k, v in pr["mean"].items()} if pr else None,
    }


def inj_row(pid, inj, st, season):
    i = inj.get(pid, {})
    mi, _ = M.minutes(st, pid, season)
    return {"pid": pid, "name": i.get("name") or st.info[pid]["name"], "status": i.get("status"),
            "type": i.get("type"), "short": i.get("short"), "mpg": round(mi, 1) if mi else None,
            "returns": i.get("returns")}


def team_news(abbrs):
    out = []
    for t in abbrs:
        d = espn.get(f"{espn.SITE}/news?team={t.lower()}&limit=6") or {}
        for art in d.get("articles", [])[:6]:
            if art.get("type") not in ("Story", "HeadlineNews", "Recap", "Preview", "Media"):
                continue
            out.append({"team": t, "headline": art.get("headline"), "desc": art.get("description"),
                        "published": art.get("published"), "type": art.get("type"),
                        "url": ((art.get("links") or {}).get("web") or {}).get("href")})
    return out


if __name__ == "__main__":
    main()
