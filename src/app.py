"""App init, MantineProvider, cache, page routing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import dash_mantine_components as dmc
from dash import Dash, html, page_container
from flask import jsonify, request, send_from_directory

from components.shared_state import nav_rail, shared_state
from components.theme_toggle import THEMES, css_variables
from utils import status
from utils.cache import cache
from utils.settings import PRECOMPUTE_DIR, URL_BASE_PATHNAME

# No background callback manager, deliberately. The regional page's cold read
# takes ~10-14 s, which would normally argue for `background=True` — but Dash's
# DiskcacheManager runs the callback in a *forked* child, and the gridded stack
# (icechunk's Rust/tokio runtime and its TLS connection pool) is not
# fork-safe. On macOS the child dies outright, taking the callback with it and
# raising an OS crash dialog, with nothing in the server log. This is the same
# hazard the deployment notes flag for gunicorn `--preload`.
#
# The regional page therefore runs a plain callback and drives its own loading
# overlay. Only the first read per box is slow; `@cache.memoize` serves
# repeats in ~1 ms.
app = Dash(
    __name__,
    use_pages=True,
    url_base_pathname=URL_BASE_PATHNAME,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        (
            "https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700"
            "&family=Barlow+Semi+Condensed:wght@600&display=swap"
        )
    ],
)
server = app.server

# Theme tokens as CSS custom properties, generated from the one table in
# `components/theme_toggle.py`, so stylesheets never carry a colour of their own.
app.index_string = app.index_string.replace(
    "{%css%}",
    "{%css%}\n        <style>"
    + css_variables()
    + "</style>\n        <script>window.MX_THEMES = "
    + json.dumps(THEMES)
    + ";</script>",
)

cache.init_app(
    server,
    config={
        "CACHE_TYPE": os.environ.get("CACHE_TYPE", "FileSystemCache"),
        "CACHE_DIR": os.environ.get("CACHE_DIR", str(Path(".cache"))),
        "CACHE_DEFAULT_TIMEOUT": 3600,
    },
)



# Precomputed preset frames (scripts/precompute.py), served from disk. Once
# hosted they come from the bucket instead (`PRECOMPUTE_URL`), same paths.
@server.route(f"{URL_BASE_PATHNAME}precomputed/<path:path>")
def precomputed_file(path: str):
    return send_from_directory(Path(PRECOMPUTE_DIR).resolve(), path, max_age=3600)


@server.route(f"{URL_BASE_PATHNAME}status")
def request_status():
    """What this browser session's requests are doing (polled by assets/status.js)."""
    return jsonify(status.snapshot(request.cookies.get(status.COOKIE)))


# Cold-front blue as the primary colour (light: shade 7 #1F5FAE; dark: shade 4,
# lifted for contrast on the night ground). Body type is Atkinson Hyperlegible,
# drawn to keep look-alike glyphs (0/O, 1/l/I) apart in coordinates and run
# times; titles are Barlow Semi Condensed, narrow like synoptic-chart lettering.
# Mantine's `dark` palette is rebuilt on the night tokens so its inputs, menus
# and popovers sit on the same blue ground as the shell.
_THEME = {
    "primaryColor": "front",
    "primaryShade": {"light": 7, "dark": 4},
    "colors": {
        "front": [
            "#eaf1fb", "#d5e3f6", "#a9c6ec", "#7ba7e1", "#5B9BEA",
            "#3d7dd1", "#2f72c6", "#1f5fae", "#174c8d", "#0f3a6e",
        ],
        "dark": [
            "#E4ECF4", "#C3CFDC", "#9FB0C3", "#8FA3BA", "#3A5274",
            "#294160", "#1D3352", "#15263D", "#0E1B2C", "#09131F",
        ],
    },
    "fontFamily": "'Atkinson Hyperlegible', system-ui, sans-serif",
    "headings": {
        "fontFamily": "'Barlow Semi Condensed', 'Atkinson Hyperlegible', sans-serif",
        "fontWeight": "600",
    },
    "defaultRadius": "sm",
}

app.layout = dmc.MantineProvider(
    id="mantine-provider",
    theme=_THEME,
    defaultColorScheme="dark",
    children=html.Div(
        [
            nav_rail(),
            html.Div([*shared_state(), page_container], className="app-main"),
            # Filled by assets/status.js while a request runs (utils/status.py).
            html.Div(id="status-bar", className="status-bar", role="status", **{"aria-live": "polite"}),
        ],
        className="app",
    ),
)

if __name__ == "__main__":
    app.run(
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_PORT", "8050")),
        debug=True,
    )
