"""Player-prop projection model.

The season is replayed day by day. On each day every player is projected using
only games that finished before that day, then that day's results are folded
into the state. The same code projects tonight's slate (the backtest and the
live picks can't drift apart).

Projection for a stat = projected minutes x per-minute rate x adjustments:
  - per-minute rate: recency-weighted, shrunk toward the league rate for the
    player's position, last season's games count half
  - minutes: recent played minutes, blended with last season early on
  - opponent: how much this opponent allows per minute to the player's
    position, relative to league (shrunk)
  - pace: expected possessions of the two teams vs league
  - teammates out: the player's own with/without splits (shrunk hard)
  - back-to-back and big-spread blowout risk trim minutes
The spread of outcomes is a negative binomial whose dispersion is fitted on
the walk-forward residuals.
"""
import math
from collections import defaultdict

BASE = ["pts", "reb", "ast", "tpm", "stl", "blk"]
COMBOS = {"pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast"), "pra": ("pts", "reb", "ast")}
LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "tpm": "3-Pointers", "stl": "Steals",
         "blk": "Blocks", "pr": "Pts + Reb", "pa": "Pts + Ast", "ra": "Reb + Ast", "pra": "Pts + Reb + Ast"}

RATE_DECAY = 0.94       # per played game
LAST_SEASON_W = 0.5     # extra weight on last season's games
RATE_PRIOR_MIN = 150    # minutes of league-average prior in the long-run rate
RECENT_PRIOR_MIN = 250  # minutes of long-run prior in the recent rate
MIN_DECAY = 0.75        # per played game, for minutes
DEF_DECAY = 0.97        # per team game, opponent-allowed sums
DEF_PRIOR_MIN = 900     # minutes of league-average prior per position
SEASON_CARRY = 0.5      # how much of last season's team state carries over


def bucket(pos):
    pos = (pos or "").upper()
    if pos in ("C", "FC", "CF"):
        return "C"
    if pos in ("PG", "SG", "G", "GF"):
        return "G"
    return "F"


def stat_of(row, s):
    if s in COMBOS:
        return sum(row.get(c, 0) for c in COMBOS[s])
    return row.get(s, 0)


class State:
    def __init__(self):
        self.logs = defaultdict(list)           # pid -> [row] played and DNP, chronological
        self.info = {}                           # pid -> latest name/pos/team
        self.team_games = defaultdict(list)      # team -> [(day, opp, eventId)]
        self.def_sum = defaultdict(lambda: defaultdict(float))   # (team,bucket) -> stat sums allowed (+'min')
        self.pace = defaultdict(lambda: [0.0, 0.0])              # team -> [decayed poss, decayed weight]
        self.league = defaultdict(lambda: defaultdict(float))    # bucket -> stat sums (+'min')
        self.league_pace = [0.0, 0.0]
        self.season = None

    # ---------- updating ----------
    def new_season(self, season):
        if self.season and season != self.season:
            for d in self.def_sum.values():
                for k in d:
                    d[k] *= SEASON_CARRY
            for p in self.pace.values():
                p[0] *= SEASON_CARRY
                p[1] *= SEASON_CARRY
        self.season = season

    def add_game(self, game, box):
        if not box:
            return
        rows = box["rows"]
        teams = {game["home"]: game["away"], game["away"]: game["home"]}
        team_tot = defaultdict(lambda: defaultdict(float))
        for r in rows:
            if r["team"] not in teams:
                continue
            r = dict(r, day=game["day"], season=game["season"], opp=teams[r["team"]],
                     home=r["team"] == game["home"], event=game["id"])
            self.logs[r["pid"]].append(r)
            self.info[r["pid"]] = {"name": r["name"], "pos": r["pos"] or self.info.get(r["pid"], {}).get("pos", ""),
                                   "team": r["team"]}
            if r["dnp"]:
                continue
            for k in ("min", "fga", "fta", "oreb", "tov"):
                team_tot[r["team"]][k] += r.get(k, 0)
        for team, opp in teams.items():
            self.team_games[team].append((game["day"], opp, game["id"]))
            d_key = [(opp, b) for b in "GFC"]
            for k in d_key:
                for s in self.def_sum[k]:
                    self.def_sum[k][s] *= DEF_DECAY
        for r in rows:
            if r.get("dnp") or r["team"] not in teams or not r.get("min"):
                continue
            b = bucket(self.info[r["pid"]]["pos"])
            dk = self.def_sum[(teams[r["team"]], b)]
            lg = self.league[b]
            dk["min"] += r["min"]
            lg["min"] += r["min"]
            for s in BASE:
                dk[s] += r.get(s, 0)
                lg[s] += r.get(s, 0)
        for team, t in team_tot.items():
            if t["min"] < 200:
                continue
            poss = (t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]) * 240.0 / t["min"]
            p = self.pace[team]
            p[0] = p[0] * DEF_DECAY + poss
            p[1] = p[1] * DEF_DECAY + 1
            self.league_pace[0] += poss
            self.league_pace[1] += 1

    # ---------- reading ----------
    def league_rate(self, b, s):
        lg = self.league[b]
        return lg[s] / lg["min"] if lg["min"] else 0.0

    def team_pace(self, team):
        p = self.pace[team]
        lp = self.league_pace[0] / self.league_pace[1] if self.league_pace[1] else 100.0
        return (p[0] + lp * 5) / (p[1] + 5), lp

    def opp_factor(self, opp, b, s):
        d = self.def_sum[(opp, b)]
        lr = self.league_rate(b, s)
        if not lr:
            return 1.0
        allowed = (d[s] + lr * DEF_PRIOR_MIN) / (d["min"] + DEF_PRIOR_MIN)
        return allowed / lr

    def played_on(self, team, day):
        return any(d == day for d, _, _ in self.team_games.get(team, []))


def rates(st, pid, season):
    """Per-minute rates: a recency-weighted rate shrunk toward the player's own
    long-run rate, which is itself shrunk toward the league rate at his position."""
    logs = [r for r in st.logs.get(pid, []) if not r["dnp"] and r.get("min", 0) > 0][-120:]
    b = bucket(st.info.get(pid, {}).get("pos"))
    rec, lng = defaultdict(float), defaultdict(float)
    rec_min = lng_min = 0.0
    for i, r in enumerate(reversed(logs)):
        ls = LAST_SEASON_W if r["season"] != season else 1.0
        w = RATE_DECAY ** i * ls
        rec_min += w * r["min"]
        lng_min += ls * r["min"]
        for s in BASE:
            rec[s] += w * r.get(s, 0)
            lng[s] += ls * r.get(s, 0)
    out = {}
    for s in BASE:
        lr = st.league_rate(b, s)
        long_run = (lng[s] + lr * RATE_PRIOR_MIN) / (lng_min + RATE_PRIOR_MIN)
        out[s] = (rec[s] + long_run * RECENT_PRIOR_MIN) / (rec_min + RECENT_PRIOR_MIN)
    return out, lng_min


def minutes(st, pid, season):
    logs = [r for r in st.logs.get(pid, []) if not r["dnp"] and r.get("min", 0) > 0]
    cur = [r["min"] for r in logs if r["season"] == season]
    prev = [r["min"] for r in logs if r["season"] != season][-30:]

    def wavg(xs, decay):
        if not xs:
            return None
        num = den = 0.0
        for i, x in enumerate(reversed(xs[-15:])):
            w = decay ** i
            num += w * x
            den += w
        return num / den

    c, p = wavg(cur, MIN_DECAY), wavg(prev, 0.9)
    if c is None and p is None:
        return None, 0
    if c is None:
        return p, 0
    if p is None:
        return c, len(cur)
    k = min(len(cur), 6) / 6.0
    return k * c + (1 - k) * p, len(cur)


def with_without(st, pid, mate, season):
    """Player's per-minute pts/reb/ast/tpm and minutes with and without a teammate."""
    mate_out = {r["event"] for r in st.logs.get(mate, []) if r["dnp"]}
    mate_in = {r["event"] for r in st.logs.get(mate, []) if not r["dnp"]}
    w, wo = defaultdict(float), defaultdict(float)
    for r in st.logs.get(pid, [])[-120:]:
        if r["dnp"] or not r.get("min"):
            continue
        tgt = wo if r["event"] in mate_out else w if r["event"] in mate_in else None
        if tgt is None:
            continue
        tgt["n"] += 1
        tgt["min"] += r["min"]
        for s in BASE:
            tgt[s] += r.get(s, 0)
    return w, wo


def project(st, pid, team, opp, day, season, home, out_ids, spread=None):
    """Mean projection for every stat for one player in one game."""
    info = st.info.get(pid)
    if not info:
        return None
    mins, n_cur = minutes(st, pid, season)
    if not mins:
        return None
    r, m_hist = rates(st, pid, season)
    b = bucket(info["pos"])
    notes = []

    # Teammates out tonight who normally play real minutes.
    usage = defaultdict(lambda: 1.0)
    min_adj = 1.0
    mates = []
    for mate in out_ids:
        if mate == pid:
            continue
        mi, _ = minutes(st, mate, season)
        if not mi or mi < 20:
            continue
        w, wo = with_without(st, pid, mate, season)
        if wo["n"] < 2 or w["n"] < 5 or not w["min"] or not wo["min"]:
            mates.append({"pid": mate, "name": st.info[mate]["name"], "mpg": round(mi, 1), "n": int(wo["n"])})
            continue
        shrink = wo["n"] / (wo["n"] + 10.0)
        entry = {"pid": mate, "name": st.info[mate]["name"], "mpg": round(mi, 1), "n": int(wo["n"]),
                 "with": {}, "without": {}}
        for s in ("pts", "reb", "ast", "tpm"):
            rw, rwo = w[s] / w["min"], wo[s] / wo["min"]
            if rw > 0:
                usage[s] *= 1 + shrink * max(-0.25, min(0.25, rwo / rw - 1))
            entry["with"][s] = round(w[s] / w["n"], 1)
            entry["without"][s] = round(wo[s] / wo["n"], 1)
        mw, mwo = w["min"] / w["n"], wo["min"] / wo["n"]
        min_adj *= 1 + shrink * max(-0.2, min(0.2, mwo / mw - 1))
        entry["with"]["min"], entry["without"]["min"] = round(mw, 1), round(mwo, 1)
        mates.append(entry)

    b2b = st.played_on(team, prev_day(day))
    if b2b:
        min_adj *= 0.97
        notes.append("Second night of a back-to-back (minutes trimmed 3%).")
    if spread is not None and abs(spread) > 10 and mins >= 28:
        cut = min(0.08, 0.006 * (abs(spread) - 10))
        min_adj *= 1 - cut
        notes.append(f"Spread of {abs(spread):g} raises blowout risk (starter minutes trimmed {cut*100:.0f}%).")
    proj_min = min(42.0, mins * min_adj)

    tp, lp = st.team_pace(team)
    op, _ = st.team_pace(opp)
    pace = ((tp + op) / 2) / lp if lp else 1.0

    means, factors = {}, {}
    for s in BASE:
        of = st.opp_factor(opp, b, s)
        hf = 1.01 if home else 0.99
        means[s] = proj_min * r[s] * of * pace * hf * usage[s]
        factors[s] = {"opp": of, "pace": pace, "usage": usage[s]}
    for c, parts in COMBOS.items():
        means[c] = sum(means[p] for p in parts)
    return {"mean": means, "min": proj_min, "baseMin": mins, "rates": r, "factors": factors, "b2b": b2b,
            "bucket": b, "mates": mates, "notes": notes, "nCur": n_cur, "histMin": m_hist}


def prev_day(day):
    import datetime as dt
    return (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()


# ---------- distributions ----------
def nb_cdf(k, mean, disp):
    """P(X <= k) for a negative binomial with Var = mean * disp (disp > 1), else Poisson."""
    if k < 0:
        return 0.0
    mean = max(mean, 1e-6)
    if disp <= 1.0001:
        p, tot = math.exp(-mean), 0.0
        for i in range(int(k) + 1):
            if i > 0:
                p *= mean / i
            tot += p
        return min(1.0, tot)
    var = mean * disp
    rr = mean * mean / (var - mean)
    pp = rr / (rr + mean)
    logp = rr * math.log(pp)
    p, tot = math.exp(logp), 0.0
    for i in range(int(k) + 1):
        if i > 0:
            p *= (rr + i - 1) / i * (1 - pp)
        tot += p
    return min(1.0, tot)


def p_over(line, mean, fit):
    """fit = {"a", "b", "k"}: Var = a*mean + b*mean^2, then probabilities are
    pulled toward 50% by k (fitted so stated probabilities match hit rates)."""
    disp = fit["a"] + fit["b"] * mean
    p = 1.0 - nb_cdf(math.floor(line), mean, disp)
    return 0.5 + fit["k"] * (p - 0.5)


def implied(american):
    return 100.0 / (american + 100) if american > 0 else -american / (-american + 100.0)


def no_vig(over, under):
    po, pu = implied(over), implied(under)
    return po / (po + pu)


def ev_per_100(p, american):
    win = american if american > 0 else 100 * 100.0 / -american
    return p * win - (1 - p) * 100
