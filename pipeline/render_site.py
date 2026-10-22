"""Wrap index.html (artifact page content) into a standalone site for GitHub Pages.

    python3 render_site.py [index.html] [site_dir] [domain]
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "index.html")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "site")
domain = sys.argv[3] if len(sys.argv) > 3 else "props.bookcliffweb.com"

head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<meta name="description" content="NBA and NFL player-prop research with an honest real-line track record.">'
        '<style>:root{color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}'
        'img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
os.makedirs(out, exist_ok=True)
with open(src) as f:
    body = f.read()
with open(os.path.join(out, "index.html"), "w") as f:
    f.write(head + body + "</body></html>")
with open(os.path.join(out, "CNAME"), "w") as f:
    f.write(domain + "\n")
open(os.path.join(out, ".nojekyll"), "w").close()
print(f"site ready in {out} for {domain}")
