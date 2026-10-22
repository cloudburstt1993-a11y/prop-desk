# NBA & NFL Prop Desk

Player-prop research for NBA and NFL games: every DraftKings over/under on the slate, a projection, recent form, matchups, injuries and news, a Top 5 per game, and an honest track record graded on real lines.

Live at **https://raw.githack.com/cloudburstt1993-a11y/prop-desk/main/index.html** (served straight from this repository). If GitHub Pages is turned on in Settings → Pages (branch main, root), it is also at https://cloudburstt1993-a11y.github.io/prop-desk/. Rebuilt automatically three times a day (10:38 AM, 2:38 PM, 4:38 PM Mountain).

21+. Research, not picks. The NBA model has not beaten DraftKings on real lines; the NFL results are a small sample with assumed −110 prices. If gambling stops being fun, call 1-800-GAMBLER.

## How it's built

`pipeline/` holds the code (Python standard library only; data from ESPN's public feeds, no keys):

```
python3 build.py --history history.json --out data.json            # NBA
python3 build_nfl.py --history history_nfl.json --out data_nfl.json  # NFL
python3 render.py data.json index.html data_nfl.json                # page body
python3 render_site.py index.html ..                                 # site files at repo root
```
