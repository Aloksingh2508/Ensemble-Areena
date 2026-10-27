"""Build a static website (runs fully in the browser via stlite / WebAssembly).

    python scripts/build_web.py          ->  creates ./web/index.html
Then host the `web/` folder on any static host (GitHub Pages, Netlify, Cloudflare Pages...).
Locally:  python -m http.server -d web 8000   and open http://localhost:8000
(Opening index.html by double-click does NOT work: browsers block module workers on file://)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web"
STLITE = "1.12.0"  # bundles Streamlit 1.65 + Pyodide 314 (Python 3.14)
REQUIREMENTS = ["numpy", "pandas", "scikit-learn", "matplotlib", "plotly"]

INCLUDE = [ROOT / "app", ROOT / "src"]
EXTRA = [ROOT / ".streamlit" / "config.toml"]

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ensemble Learning - ML Battle Arena</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.css">
<style>
 body{margin:0;background:#0B1026;color:#E8ECFF;font-family:system-ui,sans-serif}
 #loading{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:24px}
 #loading h1{margin:0 0 8px}#loading p{max-width:560px;color:#9FB0FF}
</style>
</head>
<body>
<div id="loading"><h1>🎮 ML Battle Arena</h1>
<p>Loading Python, scikit-learn and Streamlit <b>inside your browser</b>. The first visit downloads roughly 60-100 MB
and can take a minute; later visits are much faster. Nothing is sent to a server.</p></div>
<div id="root"></div>
<script type="module">
import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__V__/build/stlite.js";
const config = __CONFIG__;
try {
  mount(config, document.getElementById("root"));
  new MutationObserver((_, obs) => {
    if (document.querySelector("#root [data-testid], #root .stApp")) { document.getElementById("loading").remove(); obs.disconnect(); }
  }).observe(document.getElementById("root"), {childList: true, subtree: true});
} catch (e) {
  document.getElementById("loading").innerHTML = "<h1>Could not start</h1><p>" + String(e) + "</p>";
}
</script>
</body>
</html>
"""


def collect() -> dict:
    files = {}
    paths = [p for base in INCLUDE for p in base.rglob("*") if p.is_file()] + EXTRA
    for p in sorted(paths):
        if "__pycache__" in p.parts or p.suffix not in {".py", ".toml"}:
            continue
        files[p.relative_to(ROOT).as_posix()] = p.read_text(encoding="utf-8")
    return files


def main() -> None:
    files = collect()
    config = {"entrypoint": "app/main.py", "requirements": REQUIREMENTS, "files": files}
    blob = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")  # keep </script> safe inside HTML
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(TEMPLATE.replace("__V__", STLITE).replace("__CONFIG__", blob), encoding="utf-8")
    (OUT / ".nojekyll").write_text("")
    print(f"Wrote {OUT / 'index.html'} with {len(files)} files ({(OUT / 'index.html').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
