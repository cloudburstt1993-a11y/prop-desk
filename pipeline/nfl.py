"""NFL data (ESPN public JSON, no key) and projection model.

Projection for a stat = the player's recency-weighted per-game average
  x opponent factor (what this defense allows per game to the position, vs league)
  x game environment (the team's implied points from the spread and total).
Outcomes use the same negative binomial as the NBA side, with spread fitted on
walk-forward residuals.
"""
import datetime as dt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import espn

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

BASE = ["pass_yds", "pass_cmp", "pass_att", "pass_td", "pass_int", "rush_yds", "rush_att", "rec", "rec_yds", "targets"]
COMBOS = {"rush_rec_yds": ("rush_yds", "rec_yds")}
MKTS = ["pass_yds", "pass_cmp", "pass_att", "pass_td", "pass_int", "rush_yds", "rush_att", "rec_yds", "rec", "rush_rec_yds"]
LABEL = {"pass_yds": "Pass yards", "pass_cmp": "Completions", "pass_att": "Pass attempts", "pass_td": "Pass TDs",
         "pass_int": "Interceptions", "rush_yds": "Rush yards", "rush_att": "Carries", "rec_yds": "Receiving yards",
         "rec": "Receptions", "rush_rec_yds": "Rush + rec yards"}
PROP_MARKETS = {
    "Total Passing Yards (incl. overtime)": "pass_yds",
    "Total Pass Completions (incl. overtime)": "pass_cmp",
    "Total Passing Attempts (incl. overtime)": "pass_att",
    "Total Passing Touchdowns (incl. overtime)": "pass_td",
    "Total Passing Interceptions (incl. overtime)": "pass_int",
    "Total Rushing Yards (incl. overtime)": "rush_yds",
    "Total Carries (incl. overtime)": "rush_att",
    "Total Receiving Yards (incl. overtime)": "rec_yds",
    "Total Receptions (incl. overtime)": "rec",
    "Total Rushing Plus Receiving Yards (incl. overtime)": "rush_rec_yds",
}
GAME_TYPES = (2, 3)
REG_WEEKS, POST_WEEKS = 18, 5

GAME_DECAY = 0.85        # per played game
LAST_SEASON_W = 0.6
DEF_DECAY = 0.9          # per team game
DEF_PRIOR_G = 5          # team-games of league-average prior
SEASON_CARRY = 0.4
LEAGUE_POINTS = 22.5


def season_of(day):
    """NFL season year: Sep 2026 through Feb 2027 is the 2026 season."""
    return day.year if day.month >= 3 else day.year - 1


def stat_of(row, s):
    if s in COMBOS:
        return sum(row.get(c, 0) for c in COMBOS[s])
    return row.get(s, 0)


# ---------- fetching ----------
def week_games(year, stype, week, refresh=False):
    def fetch():
        d = espn.get(f"{SITE}/scoreboard?dates={year}&seasontype={stype}&week={week}")
        out = []
        for e in (d or {}).get("events", []):
            c = e["competitions"][0]
            teams = {t["homeAway"]: t for t in c.get("competitors", [])}
            if set(teams) != {"home", "away"}:
                continue
            odds = (c.get("odds") or [{}])[0]
            out.append({
                "id": e["id"], "date": e["date"], "seasonType": stype, "week": week, "season": year,
                "status": c["status"]["type"]["name"], "completed": c["status"]["type"]["completed"],
                "home": teams["home"]["team"].get("abbreviation", "TBD"),
                "away": teams["away"]["team"].get("abbreviation", "TBD"),
                "homeName": teams["home"]["team"].get("displayName", "TBD"),
                "awayName": teams["away"]["team"].get("displayName", "TBD"),
                "homeScore": teams["home"].get("score"), "awayScore": teams["away"].get("score"),
                "spread": odds.get("details"), "total": odds.get("overUnder"),
                "neutral": c.get("neutralSite", False),
            })
        return out
    name = f"nfl/weeks/{year}-{stype}-{week}.json"
    cached = espn._cached(name, fetch, refresh=refresh)
    if cached and not refresh and not all(g["completed"] for g in cached):
        cached = espn._cached(name, fetch, refresh=True)
    return cached or []


def season_games(year):
    specs = [(year, 2, w) for w in range(1, REG_WEEKS + 1)] + [(year, 3, w) for w in range(1, POST_WEEKS + 1)]
    with ThreadPoolExecutor(12) as ex:
        weeks = list(ex.map(lambda s: week_games(*s), specs))
    return [g for w in weeks for g in w]


def boxscore(event_id):
    def fetch():
        d = espn.get(f"{SITE}/summary?event={event_id}")
        if not d or "boxscore" not in d:
            return None
        players = {}
        for team in d["boxscore"].get("players", []):
            abbr = team["team"]["abbreviation"]
            for cat in team.get("statistics", []):
                keys = cat.get("keys") or []
                for a in cat.get("athletes", []):
                    ath = a.get("athlete") or {}
                    if not ath.get("id"):
                        continue
                    row = players.setdefault(ath["id"], {"pid": ath["id"], "name": ath.get("displayName"), "team": abbr})
                    st = dict(zip(keys, a.get("stats") or []))
                    num = lambda k: _num(st.get(k))
                    if cat["name"] == "passing":
                        cmp_att = (st.get("completions/passingAttempts") or "0/0").split("/")
                        row.update(pass_cmp=_num(cmp_att[0]), pass_att=_num(cmp_att[-1]), pass_yds=num("passingYards"),
                                   pass_td=num("passingTouchdowns"), pass_int=num("interceptions"))
                    elif cat["name"] == "rushing":
                        row.update(rush_att=num("rushingAttempts"), rush_yds=num("rushingYards"),
                                   rush_td=num("rushingTouchdowns"))
                    elif cat["name"] == "receiving":
                        row.update(rec=num("receptions"), rec_yds=num("receivingYards"),
                                   rec_td=num("receivingTouchdowns"), targets=num("receivingTargets"))
        return {"id": event_id, "rows": list(players.values())}
    return espn._cached(f"nfl/box/{event_id}.json", fetch)


def _num(v):
    try:
        return int(float(str(v).replace(",", "")))
    except Exception:
        return 0


def boxscores(ids):
    with ThreadPoolExecutor(12) as ex:
        return dict(zip(ids, ex.map(boxscore, ids)))


def props(event_id):
    items, page = [], 1
    while True:
        d = espn.get(f"{CORE}/events/{event_id}/competitions/{event_id}/odds/{espn.DRAFTKINGS}/propBets?limit=1000&page={page}")
        if not d:
            break
        items += d.get("items", [])
        if page >= d.get("pageCount", 1):
            break
        page += 1
    pairs = {}
    for it in items:
        mkt = PROP_MARKETS.get(it.get("type", {}).get("name"))
        if not mkt or "athlete" not in it:
            continue
        pid = it["athlete"]["$ref"].split("/athletes/")[1].split("?")[0]
        pairs.setdefault((pid, mkt), []).append(it)
    out = []
    for (pid, mkt), rows in pairs.items():
        if len(rows) != 2:
            continue
        over, under = rows
        try:
            line = over["current"]["target"]["value"]
        except (KeyError, TypeError):
            continue
        # ESPN's NFL feed carries the lines without prices; keep prices when present.
        price = lambda r: espn._int(((r.get("odds") or {}).get("american") or {}).get("value"))
        out.append({"pid": pid, "mkt": mkt, "line": line,
                    "openLine": over.get("open", {}).get("target", {}).get("value", line),
                    "over": price(over), "under": price(under), "overOpen": None,
                    "updated": over.get("lastUpdated")})
    return out


def team_ids():
    d = espn.get(f"{SITE}/teams") or {}
    return {t["team"]["abbreviation"]: t["team"]["id"] for t in d["sports"][0]["leagues"][0]["teams"]}


def positions():
    """pid -> position abbreviation from current rosters."""
    teams = team_ids()

    def roster(tid):
        d = espn.get(f"{SITE}/teams/{tid}/roster") or {}
        out = {}
        for grp in d.get("athletes", []):
            for a in grp.get("items", []):
                out[a["id"]] = (a.get("position") or {}).get("abbreviation", "")
        return out
    pos = {}
    with ThreadPoolExecutor(12) as ex:
        for r in ex.map(roster, teams.values()):
            pos.update(r)
    return pos


def injuries():
    d = espn.get(f"{SITE}/injuries") or {}
    out = {}
    for team in d.get("injuries", []):
        for inj in team.get("injuries", []):
            a = inj.get("athlete", {})
            href = (a.get("links") or [{}])[0].get("href", "")
            pid = str(a.get("id") or href.split("/id/")[-1].split("/")[0])
            det = inj.get("details") or {}
            if (inj.get("status") or "").lower() == "active":
                continue          # practice-report entries for healthy players
            out[pid] = {"name": a.get("displayName"), "team": team.get("displayName"), "status": inj.get("status"),
                        "type": " ".join(x for x in (det.get("side"), det.get("type")) if x and x != "Not Specified"),
                        "short": inj.get("shortComment"), "returns": det.get("returnDate")}
    return out


def player_news(pid):
    d = espn.get(f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{pid}/overview")
    if not d:
        return None
    rw = d.get("rotowire") or {}
    news = [{"headline": n.get("headline"), "published": n.get("lastModified") or n.get("published"),
             "url": ((n.get("links") or {}).get("web") or {}).get("href")} for n in (d.get("news") or [])[:4]]
    return {"note": {"headline": rw.get("headline"), "story": rw.get("story"), "published": rw.get("published")}
            if rw.get("headline") else None, "news": news}


def team_news(abbrs):
    out = []
    for t in abbrs:
        d = espn.get(f"{SITE}/news?team={t.lower()}&limit=5") or {}
        for art in d.get("articles", [])[:5]:
            out.append({"team": t, "headline": art.get("headline"), "desc": art.get("description"),
                        "published": art.get("published"), "type": art.get("type"),
                        "url": ((art.get("links") or {}).get("web") or {}).get("href")})
    return out


# ---------- model ----------
def bucket(pos, row=None):
    pos = (pos or "").upper()
    if pos in ("QB", "RB", "WR", "TE"):
        return pos
    if pos == "FB":
        return "RB"
    if row:   # unknown position: infer from usage
        if row.get("pass_att", 0) >= 5:
            return "QB"
        if row.get("rush_att", 0) > row.get("targets", 0):
            return "RB"
        return "WR"
    return None


class State:
    def __init__(self, pos):
        self.pos = pos
        self.logs = defaultdict(list)
        self.info = {}
        self.team_games = defaultdict(list)
        self.def_sum = defaultdict(lambda: defaultdict(float))     # (team, bucket) -> stat sums allowed, + "g"
        self.league = defaultdict(lambda: defaultdict(float))      # bucket -> stat sums, + "g" (team-games)
        self.season = None

    def new_season(self, season):
        if self.season is not None and season != self.season:
            for d in self.def_sum.values():
                for k in d:
                    d[k] *= SEASON_CARRY
        self.season = season

    def add_game(self, g, box):
        if not box:
            return
        teams = {g["home"]: g["away"], g["away"]: g["home"]}
        allowed = defaultdict(lambda: defaultdict(float))
        for r in box["rows"]:
            if r["team"] not in teams:
                continue
            b = bucket(self.pos.get(r["pid"]), r)
            r = dict(r, day=g["date"][:10], season=g["season"], opp=teams[r["team"]], home=r["team"] == g["home"],
                     event=g["id"], bucket=b)
            self.logs[r["pid"]].append(r)
            self.info[r["pid"]] = {"name": r["name"], "team": r["team"], "bucket": b}
            for s in BASE:
                allowed[(teams[r["team"]], b)][s] += r.get(s, 0)
        for team, opp in teams.items():
            self.team_games[team].append((g["date"][:10], opp, g["id"]))
            for b in ("QB", "RB", "WR", "TE"):
                d = self.def_sum[(opp, b)]
                for k in list(d):
                    d[k] *= DEF_DECAY
                d["g"] += 1
                self.league[b]["g"] += 1
                for s in BASE:
                    v = allowed[(opp, b)][s]
                    d[s] += v
                    self.league[b][s] += v

    def opp_factor(self, opp, b, s):
        lg = self.league[b]
        if not lg["g"] or not lg[s]:
            return 1.0
        lr = lg[s] / lg["g"]
        d = self.def_sum[(opp, b)]
        return ((d[s] + lr * DEF_PRIOR_G) / (d["g"] + DEF_PRIOR_G)) / lr


def project(st, pid, team, opp, season, spread, total):
    logs = st.logs.get(pid, [])[-20:]
    if len(logs) < 2:
        return None
    b = st.info[pid]["bucket"]
    base, wsum = defaultdict(float), 0.0
    for i, r in enumerate(reversed(logs)):
        w = GAME_DECAY ** i * (LAST_SEASON_W if r["season"] != season else 1.0)
        wsum += w
        for s in BASE:
            base[s] += w * r.get(s, 0)
    for s in BASE:
        base[s] /= wsum
    implied = None
    env = 1.0
    if spread is not None and total:
        implied = total / 2 - spread / 2
        env = max(0.6, min(1.5, implied / LEAGUE_POINTS))
    means, factors = {}, {}
    for s in BASE:
        of = st.opp_factor(opp, b, s)
        ef = env if s in ("pass_td",) else env ** 0.5
        if s == "pass_int":
            of, ef = of, 1.0
        means[s] = base[s] * of * ef
        factors[s] = {"opp": of, "env": ef}
    for c, parts in COMBOS.items():
        means[c] = sum(means[p] for p in parts)
        tot = means[c] or 1
        factors[c] = {k: sum(factors[p][k] * means[p] for p in parts) / tot for k in ("opp", "env")}
    n_cur = sum(1 for r in st.logs[pid] if r["season"] == season)
    recent = st.logs[pid][-3:]
    vol = {k: round(sum(r.get(k, 0) for r in recent) / len(recent), 1) for k in ("pass_att", "rush_att", "targets")}
    return {"mean": means, "base": dict(base), "factors": factors, "bucket": b, "implied": implied,
            "nCur": n_cur, "vol": vol}
