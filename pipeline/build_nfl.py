"""Build this week's NFL prop board.

    python3 build_nfl.py [--date YYYY-MM-DD] [--history history_nfl.json] [--out data_nfl.json]

Same shape as build.py (NBA): replay last season and this one game by game
(the backtest), grade real DraftKings lines once and lock them, then project
every over/under player prop for this week's games that haven't kicked off.

ESPN publishes the NFL prop lines but not their prices, so every line is
treated as -110 on both sides (a 50% market). Grading uses -110 too.
"""
import argparse
import datetime as dt
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import espn
import model as M
import nfl as N
from build import (calibrate, decide, empty_record, fit_spread, grade_into, grade_picks, mtn_today, parse_spread,
                   spread_for, hit_rate, avg)

ASSUMED_PRICE = -110
TOP_N, TOP_MIN_GAP, LEAN_EDGE = 5, 0.03, 0.05
VOLUME = {"QB": ("pass_att", 20), "RB": ("rush_att", 8), "WR": ("targets", 3), "TE": ("targets", 3)}
MKT_ROLE = {"pass_yds": "QB", "pass_cmp": "QB", "pass_att": "QB", "pass_td": "QB", "pass_int": "QB",
            "rush_yds": "RB", "rush_att": "RB", "rec_yds": "WR", "rec": "WR", "rush_rec_yds": "RB"}


def role_ok(c):
    """The player has a real role for this market: a starting QB for passing props,
    8+ carries a game for rushing, 3+ targets for receiving (last 3 games)."""
    v = c["vol"]
    if c["mkt"].startswith("pass_"):
        return v["pass_att"] >= 20
    if c["mkt"] in ("rush_yds", "rush_att"):
        return v["rush_att"] >= 8
    if c["mkt"] in ("rec_yds", "rec"):
        return v["targets"] >= 3
    return v["rush_att"] >= 8 or v["targets"] >= 3


def top_picks(cands):
    """Up to 5 props per game: no injury tag, no Check flag, a real role, a 3+ point gap between
    the model and a 50% market, and the player's own record agreeing over his last 6 games
    (4+/6 over, or 2-/6) and last 17 (half or more / half or fewer). One prop per player."""
    ok = []
    for c in cands:
        if c["check"] or c["injured"] or not role_ok(c):
            continue
        gap = c["pModel"] - c["pMarket"]
        if abs(gap) < TOP_MIN_GAP or c["n10"] < 4 or c["nl"] < 8:
            continue
        over = gap > 0
        r10, rl = c["h10"] / c["n10"], c["hl"] / c["nl"]
        if over and not (r10 >= 4 / 6 - 1e-9 and rl >= 0.5):
            continue
        if not over and not (r10 <= 2 / 6 + 1e-9 and rl <= 0.5):
            continue
        ok.append(dict(c, side="over" if over else "under", price=ASSUMED_PRICE, gap=round(gap, 3),
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


def priced(p):
    return dict(p, over=ASSUMED_PRICE, under=ASSUMED_PRICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=mtn_today().isoformat())
    ap.add_argument("--history", default="history_nfl.json")
    ap.add_argument("--out", default="data_nfl.json")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date)
    cur = N.season_of(today)
    prev = cur - 1

    h = {}
    if os.path.exists(a.history):
        with open(a.history) as f:
            h = json.load(f)
    for k, v in (("pending", {}), ("gradedEvents", []), ("recent", []), ("nights", []), ("record", empty_record()),
                 ("top5", {}), ("top5Pending", {})):
        h.setdefault(k, v)
    pending, graded_ids = h["pending"], set(h["gradedEvents"])

    pos = N.positions()
    games = N.season_games(prev) + N.season_games(cur)
    teams = set(N.team_ids())
    games = [g for g in games if g["home"] in teams and g["away"] in teams]
    done = [g for g in games if g["completed"]]
    boxes = N.boxscores([g["id"] for g in done])
    print(f"NFL: {len(done)} completed games", file=sys.stderr)

    # This season's finished games: fetch lines once, grade, lock.
    to_fetch = [g for g in done if g["season"] == cur and g["id"] not in graded_ids and g["id"] not in pending]
    with ThreadPoolExecutor(8) as ex:
        for g, p in zip(to_fetch, ex.map(lambda g: N.props(g["id"]), to_fetch)):
            if p:
                pending[g["id"]] = p

    # This week: the current scoreboard week, games not finished.
    sb = espn.get(f"{N.SITE}/scoreboard") or {}
    wk = (sb.get("week") or {}).get("number")
    stype = (sb.get("season") or {}).get("type", 2)
    week = [g for g in N.week_games(cur, stype, wk, refresh=True) if not g["completed"]] if wk else []
    with ThreadPoolExecutor(8) as ex:
        week_props = dict(zip([g["id"] for g in week], ex.map(lambda g: N.props(g["id"]), week)))
    inj = N.injuries()

    # ---------- replay ----------
    st = N.State(pos)
    bt = defaultdict(list)
    to_grade, first_rows = [], []
    for g in sorted(done, key=lambda g: g["date"]):
        st.new_season(g["season"])
        box = boxes.get(g["id"])
        lines = {} if g["id"] in graded_ids else {(p["pid"], p["mkt"]): p for p in pending.get(g["id"], [])}
        score = g["season"] == cur or (g["season"] == prev and g["week"] >= 6) or g["seasonType"] == 3
        if box and (score or lines):
            spread = parse_spread(g)
            for r in box["rows"]:
                team = r["team"]
                opp = g["away"] if team == g["home"] else g["home"]
                pr = N.project(st, r["pid"], team, opp, g["season"], spread_for(spread, team, g), g.get("total"))
                if not pr:
                    continue
                prior = st.logs[r["pid"]]
                for s in N.MKTS:
                    actual = N.stat_of(r, s)
                    role = MKT_ROLE[s]
                    if score and len(prior) >= 4 and role_ok({"mkt": s, "vol": pr["vol"]}):
                        cur_prior = [x for x in prior if x["season"] == g["season"]] or prior
                        bt[s].append((pr["mean"][s], actual, statistics.fmean(N.stat_of(x, s) for x in cur_prior),
                                      statistics.fmean(N.stat_of(x, s) for x in prior[-5:])))
                    if (r["pid"], s) in lines:
                        ln = lines[(r["pid"], s)]
                        l6, l17 = prior[-6:], prior[-17:]
                        to_grade.append({"day": g["date"][:10], "event": g["id"], "pid": r["pid"], "name": r["name"],
                                         "team": team, "mkt": s, "line": ln["line"], "over": ASSUMED_PRICE,
                                         "under": ASSUMED_PRICE, "mean": pr["mean"][s], "actual": actual,
                                         "nCur": pr["nCur"], "vol": pr["vol"],
                                         "h10": sum(N.stat_of(x, s) > ln["line"] for x in l6), "n10": len(l6),
                                         "hl": sum(N.stat_of(x, s) > ln["line"] for x in l17), "nl": len(l17)})
        st.add_game(g, box)

    disp, bt_summary = {}, {}
    for s in N.MKTS:
        rows = bt[s]
        if len(rows) < 200:
            disp[s] = h.get("fit", {}).get(s) or {"a": 3.0, "b": 0.05, "k": 0.8}
            continue
        disp[s] = fit_spread(rows)
        mae = lambda i: statistics.fmean(abs(r[i] - r[1]) for r in rows)
        bt_summary[s] = {"n": len(rows), "maeModel": mae(0), "maeSeasonAvg": mae(2), "maeLast10": mae(3),
                         "fit": disp[s], "calibration": calibrate(rows, disp[s])}
    h["fit"] = disp
    st.new_season(cur)

    # Weeks graded on the first run become the starting record; Top 5 rules are run on them too.
    first_run = not h["gradedEvents"]
    grade_into(h, to_grade, disp)
    if first_run:
        by_ev = defaultdict(list)
        for r in to_grade:
            pm = M.p_over(r["line"], r["mean"], disp[r["mkt"]])
            _, check = decide(pm, 0.5, r["mean"], r["line"], r["nCur"])
            by_ev[r["event"]].append(dict(r, pModel=pm, pMarket=0.5, check=check, injured=False))
        picks = [p for ev in by_ev.values() for p in top_picks(ev)]
        h["top5Backtest"] = {}
        grade_picks(h["top5Backtest"], picks)
    t5, t5p = h["top5"], h["top5Pending"]
    finished = []
    for eid in [e for e in t5p if boxes.get(e)]:
        rows = {r["pid"]: r for r in boxes[eid]["rows"]}
        for pk in t5p.pop(eid)["picks"]:
            row = rows.get(pk["pid"])
            finished.append(dict(pk, actual=N.stat_of(row, pk["mkt"]) if row else None))
    grade_picks(t5, finished)
    for eid in {r["event"] for r in to_grade} | {g["id"] for g in done if g["id"] in pending}:
        graded_ids.add(eid)
        pending.pop(eid, None)
    h["gradedEvents"] = sorted(graded_ids)

    # ---------- this week ----------
    out_ids = defaultdict(set)
    for pid, i in inj.items():
        info = st.info.get(pid)
        if info and (i["status"] or "").lower() in ("out", "doubtful", "injured reserve", "ir"):
            out_ids[info["team"]].add(pid)

    def starting_qb(team):
        qbs = [(sum(r.get("pass_att", 0) for r in st.logs[p][-3:]), p) for p, inf in st.info.items()
               if inf["team"] == team and inf["bucket"] == "QB" and st.logs[p] and st.logs[p][-1]["season"] == cur]
        return max(qbs)[1] if qbs else None

    board, players, games_out, top5 = [], {}, [], {}
    for g in week:
        spread = parse_spread(g)
        qb_out = {t: (starting_qb(t) in out_ids[t]) for t in (g["home"], g["away"])}
        games_out.append({**g, "spreadNum": spread, "news": N.team_news([g["home"], g["away"]]),
                          "qbOut": qb_out, "b2b": {}, "pace": {},
                          "out": {t: [{"pid": p, "name": inj[p]["name"], "status": inj[p]["status"],
                                       "type": inj[p]["type"], "short": inj[p]["short"]}
                                      for p in out_ids[t] if p in inj] for t in (g["home"], g["away"])}})
        cands = []
        for p in week_props.get(g["id"], []):
            pid = p["pid"]
            info = st.info.get(pid)
            if not info or info["team"] not in (g["home"], g["away"]):
                continue
            team = info["team"]
            opp = g["away"] if team == g["home"] else g["home"]
            pr = N.project(st, pid, team, opp, cur, spread_for(spread, team, g), g.get("total"))
            if not pr:
                continue
            s = p["mkt"]
            mean = pr["mean"][s]
            pm = M.p_over(p["line"], mean, disp[s])
            side, check = decide(pm, 0.5, mean, p["line"], pr["nCur"])
            notes = []
            if qb_out[team] and (s.startswith("pass_") or s in ("rec", "rec_yds", "rush_rec_yds")):
                check, side = True, None
                notes.append("Starting QB is out; passing-game numbers are unreliable this week.")
            hist = st.logs[pid]
            curh = [x for x in hist if x["season"] == cur]
            last = [x for x in hist if x["season"] != cur]
            vs = [x for x in hist if x["opp"] == opp]
            f = pr["factors"][s]
            injury = inj.get(pid)
            if pid not in players:
                players[pid] = {"name": info["name"], "team": team, "opp": opp, "pos": info["bucket"], "event": g["id"],
                                "games": len(hist), "injury": injury, "notes": [],
                                "log": [{"day": x["day"], "opp": x["opp"], "home": x["home"],
                                         **{m: N.stat_of(x, m) for m in N.MKTS}} for x in hist[-17:]]}
            why = [["Weighted per-game average", f"{pr['base'].get(s, 0) if s not in N.COMBOS else sum(pr['base'][c] for c in N.COMBOS[s]):.1f}"],
                   [f"{opp} defense vs {info['bucket']}s", f"×{f['opp']:.3f}"],
                   ["Game environment" + (f" (team implied {pr['implied']:.1f} pts)" if pr["implied"] else ""), f"×{f['env']:.3f}"],
                   ["Recent volume (last 3 games)", (f"{pr['vol']['pass_att']} pass attempts" if s.startswith("pass_")
                                                     else f"{pr['vol']['rush_att']} carries · {pr['vol']['targets']} targets" if s == "rush_rec_yds"
                                                     else f"{pr['vol']['rush_att']} carries" if s.startswith("rush_")
                                                     else f"{pr['vol']['targets']} targets")]]
            entry = {
                "pid": pid, "event": g["id"], "mkt": s, "line": p["line"], "openLine": p["openLine"],
                "over": ASSUMED_PRICE, "under": ASSUMED_PRICE, "assumedPrice": True, "updated": p["updated"],
                "mean": round(mean, 2), "pModel": round(pm, 3), "pMarket": 0.5, "edge": round(pm - 0.5, 3),
                "side": side, "check": check, "ev": 0,
                "hit": {k: hit_rate(v, s, p["line"]) for k, v in
                        (("l3", hist[-3:]), ("l6", hist[-6:]), ("l17", hist[-17:]), ("season", curh), ("last", last), ("vs", vs))},
                "avg": {k: avg(v, s) for k, v in (("l3", hist[-3:]), ("l6", hist[-6:]), ("season", curh), ("last", last))},
                "why": why, "notes": notes, "nCur": pr["nCur"], "newTeam": bool(last) and last[-1]["team"] != team,
                "min": 0, "vol": pr["vol"],
            }
            board.append(entry)
            if entry["hit"]["l6"] and entry["hit"]["l17"]:
                cands.append({"pid": pid, "name": info["name"], "team": team, "event": g["id"], "day": g["date"][:10],
                              "mkt": s, "line": p["line"], "over": ASSUMED_PRICE, "under": ASSUMED_PRICE,
                              "mean": entry["mean"], "pModel": entry["pModel"], "pMarket": 0.5, "check": check,
                              "injured": bool(injury), "vol": pr["vol"], "min": 0,
                              "h10": entry["hit"]["l6"]["over"], "n10": entry["hit"]["l6"]["n"],
                              "hl": entry["hit"]["l17"]["over"], "nl": entry["hit"]["l17"]["n"],
                              "openLine": p["openLine"], "nCur": pr["nCur"], "newTeam": entry["newTeam"]})
        top5[g["id"]] = top_picks(cands)
        if g["status"] == "STATUS_SCHEDULED" and top5[g["id"]]:
            t5p[g["id"]] = {"day": g["date"][:10], "picks": top5[g["id"]]}

    with ThreadPoolExecutor(8) as ex:
        for pid, n in zip(list(players), ex.map(N.player_news, list(players))):
            players[pid]["news"] = n

    data = {
        "sport": "nfl",
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "date": today.isoformat(), "season": str(cur), "seasonStart": f"{cur}-08-01",
        "week": wk, "seasonType": stype,
        "games": games_out, "players": players, "props": board,
        "backtest": {"from": f"{prev} week 6", "through": max((g["date"][:10] for g in done), default=None), "stats": bt_summary},
        "real": h["record"], "archive": None, "realRecent": h["recent"][-60:],
        "top5": top5,
        "top5Record": {"record": t5.get("record", empty_record()), "recent": t5.get("recent", [])[-40:]},
        "top5Archive": {"season": f"{cur} weeks before this app", "record": h.get("top5Backtest", {}).get("record", empty_record())},
        "top5Rules": {"n": TOP_N, "gap": TOP_MIN_GAP, "price": ASSUMED_PRICE, "min": 0},
        "labels": N.LABEL, "markets": N.MKTS, "leanEdge": LEAN_EDGE, "assumedPrice": ASSUMED_PRICE,
    }
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    with open(a.history, "w") as f:
        json.dump(h, f, separators=(",", ":"))
    print(f"NFL week {wk}: {len(week)} games, {len(board)} props, {len(to_grade)} graded this run -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
