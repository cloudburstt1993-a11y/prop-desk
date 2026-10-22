"""Fetch NBA data from ESPN's public JSON endpoints (no key, no account).

Everything is cached on disk under CACHE_DIR. Completed games never change,
so their box scores are fetched once; today's slate, props and injuries are
always re-fetched.
"""
import datetime as dt
import gzip
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = os.environ.get("NBA_CACHE", os.path.join(os.path.dirname(__file__), "cache"))
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
UA = {"User-Agent": "Mozilla/5.0 (nba-props research; personal project)", "Accept-Encoding": "gzip"}

DRAFTKINGS = 100
GAME_TYPES = (2, 3)   # regular season and playoffs; preseason (1) and All-Star are skipped


def season_of(day):
    """'2026-27' for any date from Aug 2026 through Jul 2027."""
    y = day.year if day.month >= 8 else day.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def season_window(season):
    y = int(season[:4])
    return dt.date(y, 10, 1), dt.date(y + 1, 6, 30)


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(1.5 * (i + 1))


def _cached(name, fetch, refresh=False):
    path = os.path.join(CACHE_DIR, name)
    if not refresh and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = fetch()
    if data is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    return data


def scoreboard(day, refresh=False):
    """Slim list of games on a date."""
    def fetch():
        d = get(f"{SITE}/scoreboard?dates={day:%Y%m%d}")
        out = []
        for e in (d or {}).get("events", []):
            c = e["competitions"][0]
            teams = {t["homeAway"]: t for t in c.get("competitors", [])}
            if set(teams) != {"home", "away"}:
                continue
            odds = (c.get("odds") or [{}])[0]
            out.append({
                "id": e["id"],
                "date": e["date"],
                "seasonType": e["season"]["type"],
                "status": c["status"]["type"]["name"],
                "completed": c["status"]["type"]["completed"],
                "home": teams["home"]["team"].get("abbreviation", "TBD"),
                "away": teams["away"]["team"].get("abbreviation", "TBD"),
                "homeId": teams["home"]["team"].get("id"),
                "awayId": teams["away"]["team"].get("id"),
                "homeName": teams["home"]["team"].get("displayName", "TBD"),
                "awayName": teams["away"]["team"].get("displayName", "TBD"),
                "homeScore": teams["home"].get("score"),
                "awayScore": teams["away"].get("score"),
                "spread": odds.get("details"),
                "total": odds.get("overUnder"),
            })
        return out
    # Past days are frozen once every game is final.
    today = dt.date.today()
    name = f"scoreboard/{day:%Y%m%d}.json"
    path = os.path.join(CACHE_DIR, name)
    if not refresh and day < today - dt.timedelta(days=1) and os.path.exists(path):
        return _cached(name, fetch)
    return _cached(name, fetch, refresh=True)


def _split(made_att):
    try:
        m, a = made_att.split("-")
        return int(m), int(a)
    except Exception:
        return 0, 0


def boxscore(event_id):
    """Per-player box score rows for a completed game (cached forever)."""
    def fetch():
        d = get(f"{SITE}/summary?event={event_id}")
        if not d or "boxscore" not in d:
            return None
        rows = []
        for team in d["boxscore"].get("players", []):
            abbr = team["team"]["abbreviation"]
            for block in team.get("statistics", []):
                labels = block.get("labels") or []
                for a in block.get("athletes", []):
                    ath = a.get("athlete") or {}
                    if not ath.get("id"):
                        continue
                    row = {
                        "pid": ath["id"],
                        "name": ath.get("displayName"),
                        "pos": (ath.get("position") or {}).get("abbreviation", ""),
                        "team": abbr,
                        "starter": a.get("starter", False),
                        "dnp": a.get("didNotPlay", False) or not a.get("stats"),
                        "reason": a.get("reason", ""),
                    }
                    s = dict(zip(labels, a.get("stats") or []))
                    if not row["dnp"]:
                        fgm, fga = _split(s.get("FG", "0-0"))
                        tpm, tpa = _split(s.get("3PT", "0-0"))
                        ftm, fta = _split(s.get("FT", "0-0"))
                        num = lambda k: int(float(s.get(k) or 0)) if str(s.get(k, "0")).lstrip("-+").replace(".", "").isdigit() else 0
                        row.update(min=num("MIN"), pts=num("PTS"), reb=num("REB"), ast=num("AST"),
                                   stl=num("STL"), blk=num("BLK"), tov=num("TO"), oreb=num("OREB"),
                                   fgm=fgm, fga=fga, tpm=tpm, tpa=tpa, ftm=ftm, fta=fta)
                    rows.append(row)
        return {"id": event_id, "rows": rows}
    return _cached(f"box/{event_id}.json", fetch)


def season_games(season, through=None):
    start, end = season_window(season)
    end = min(end, through or dt.date.today())
    if end < start:
        return []
    days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
    with ThreadPoolExecutor(12) as ex:
        boards = list(ex.map(scoreboard, days))
    games = []
    for day, board in zip(days, boards):
        for g in board or []:
            if g["seasonType"] in GAME_TYPES:
                g["day"] = day.isoformat()
                g["season"] = season
                games.append(g)
    return games


def boxscores(event_ids):
    with ThreadPoolExecutor(12) as ex:
        return dict(zip(event_ids, ex.map(boxscore, event_ids)))


PROP_MARKETS = {
    "Total Points": "pts",
    "Total Rebounds": "reb",
    "Total Assists": "ast",
    "Total 3-Point Field Goals": "tpm",
    "Total Points and Rebounds": "pr",
    "Total Points and Assists": "pa",
    "Total Assists and Rebounds": "ra",
    "Total Points, Rebounds, and Assists": "pra",
    "Total Steals": "stl",
    "Total Blocks": "blk",
}


def props(event_id):
    """DraftKings over/under player props for one game, as served by ESPN.

    ESPN lists each over/under as two rows for the same athlete and market;
    the first row is the Over (verified against the matching milestone price).
    """
    items, page = [], 1
    while True:
        d = get(f"{CORE}/events/{event_id}/competitions/{event_id}/odds/{DRAFTKINGS}/propBets?limit=1000&page={page}")
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
        line = over["current"]["target"]["value"]
        out.append({
            "pid": pid, "mkt": mkt, "line": line,
            "openLine": over.get("open", {}).get("target", {}).get("value", line),
            "over": int(over["odds"]["american"]["value"].replace("+", "")),
            "under": int(under["odds"]["american"]["value"].replace("+", "")),
            "overOpen": _int(over["odds"]["american"].get("open")),
            "updated": over.get("lastUpdated"),
        })
    return out


def _int(v):
    try:
        return int(str(v).replace("+", ""))
    except Exception:
        return None


def athlete(pid):
    def fetch():
        d = get(f"{CORE}/athletes/{pid}")
        if not d:
            return None
        return {"id": pid, "name": d.get("displayName"), "pos": (d.get("position") or {}).get("abbreviation", ""),
                "headshot": (d.get("headshot") or {}).get("href"),
                "team": (d.get("team") or {}).get("$ref", "").split("/teams/")[-1].split("?")[0]}
    return _cached(f"athletes/{pid}.json", fetch)


def player_news(pid):
    """Latest RotoWire note (role, usage, off-court context) and ESPN headlines for a player."""
    d = get(f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/overview")
    if not d:
        return None
    rw = d.get("rotowire") or {}
    news = []
    for n in (d.get("news") or [])[:4]:
        news.append({"headline": n.get("headline"), "published": n.get("lastModified") or n.get("published"),
                     "url": ((n.get("links") or {}).get("web") or {}).get("href")})
    return {"note": {"headline": rw.get("headline"), "story": rw.get("story"), "published": rw.get("published")}
            if rw.get("headline") else None, "news": news}


def injuries():
    d = get(f"{SITE}/injuries") or {}
    out = {}
    for team in d.get("injuries", []):
        for inj in team.get("injuries", []):
            a = inj.get("athlete", {})
            pid = (a.get("links") or [{}])[0].get("href", "")
            pid = a.get("id") or pid.split("/id/")[-1].split("/")[0]
            out[str(pid)] = {
                "name": a.get("displayName"),
                "team": team.get("displayName"),
                "status": inj.get("status"),
                "type": " ".join(x for x in ((inj.get("details") or {}).get(k) for k in ("side", "type", "detail")) if x),
                "returns": (inj.get("details") or {}).get("returnDate"),
                "short": inj.get("shortComment"),
                "long": inj.get("longComment"),
                "date": inj.get("date"),
            }
    return out


def team_ids():
    d = get(f"{SITE}/teams") or {}
    out = {}
    for t in d["sports"][0]["leagues"][0]["teams"]:
        t = t["team"]
        out[t["abbreviation"]] = {"id": t["id"], "name": t["displayName"], "color": t.get("color"),
                                  "logo": (t.get("logos") or [{}])[0].get("href")}
    return out
