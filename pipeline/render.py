"""Fill template.html with the NBA and NFL data -> index.html.

    python3 render.py [data.json] [index.html] [data_nfl.json]
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data.json")
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "index.html")
nfl_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(data_path)), "data_nfl.json")


def blob(path):
    if not os.path.exists(path):
        return "null"
    with open(path) as f:
        return json.dumps(json.load(f), separators=(",", ":")).replace("</", "<\\/")


with open(os.path.join(here, "template.html")) as f:
    tpl = f.read()
html = tpl.replace("__DATA_NFL__", blob(nfl_path)).replace("__DATA__", blob(data_path))
with open(out_path, "w") as f:
    f.write(html)
print(f"wrote {out_path} ({os.path.getsize(out_path) // 1024} KB)")
