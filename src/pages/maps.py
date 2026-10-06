"""Climate · Maps: a gallery of ERA5 monthly maps drawn offline.

The full workbench retrieves any CDS/ADS field on demand; a public site cannot
— every request would queue under one Copernicus account. Here the maps are
retrieved and drawn by the scheduled job (`utils/map_gallery.py`, run from
`scripts/precompute.py --maps`) with the same request builder and renderer, and
this page only shows the images. Nothing on it calls Copernicus.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback, html

from components.page_shell import page_shell
from components.shared_state import url_sync
from utils.map_gallery import gallery_entries
from utils.settings import MAP_GALLERY, MAP_GALLERY_REFERENCE, PRECOMPUTE_URL

dash.register_page(__name__, path="/maps", title="Climate maps · preddyfast")

_HIDDEN = {"display": "none"}
_IMAGE = {"width": "100%", "maxWidth": "980px", "display": "block"}


def layout(**query_params):
    entries = gallery_entries()
    keys = [e["key"] for e in entries]
    chosen = query_params.get("map")
    value = chosen if chosen in keys else (keys[0] if keys else None)
    return page_shell(
        title="Climate maps",
        description=(
            "ERA5 monthly means for the latest month Copernicus serves, retrieved "
            "and drawn offline by a scheduled job. Anomalies are against "
            f"{MAP_GALLERY_REFERENCE.replace('-', '–')}."
        ),
        sidebar=[
            html.Div(id="mp-current-url", style=_HIDDEN),
            dmc.RadioGroup(
                dmc.Stack(
                    [dmc.Radio(label=e["label"], value=e["key"]) for e in entries],
                    gap="xs",
                ),
                id="mp-gallery-select",
                label="Map",
                value=value,
            ),
            dmc.Text(
                "The full workbench retrieves any ERA5 or CAMS field, area and time "
                "on demand; this public demo shows a fixed set.",
                size="xs",
                c="dimmed",
            ),
        ],
        main=[
            html.Div(
                dmc.Alert(
                    "The maps gallery has not been published yet.",
                    color="yellow",
                    withCloseButton=False,
                )
                if not entries
                else None
            ),
            dmc.Stack(
                [
                    dmc.Text(id="mp-gallery-caption", size="sm", c="dimmed"),
                    html.Img(id="mp-gallery-image", alt="Climate map", style=_HIDDEN),
                    html.A(
                        dmc.Button("Download map PNG", variant="light", size="xs"),
                        id="mp-gallery-download",
                        target="_blank",
                        style=_HIDDEN,
                    ),
                ],
                gap="xs",
            ),
        ],
        graph_id="mp-gallery-image",
        share_target="mp-current-url",
    )


@callback(
    Output("mp-gallery-image", "src"),
    Output("mp-gallery-image", "alt"),
    Output("mp-gallery-image", "style"),
    Output("mp-gallery-caption", "children"),
    Output("mp-gallery-download", "href"),
    Output("mp-gallery-download", "style"),
    Input("mp-gallery-select", "value"),
)
def show_map(key):
    """Point the image at the chosen item's published PNG."""
    entry = next((e for e in gallery_entries() if e["key"] == key), None)
    if entry is None or key not in MAP_GALLERY:
        return "", "", _HIDDEN, "", "", _HIDDEN
    src = f"{PRECOMPUTE_URL}{entry['path']}"
    caption = f"{entry['label']} · {entry['month']} · {entry['caption']}"
    return src, entry["label"], _IMAGE, caption, src, {"display": "inline-block"}


url_sync("mp-current-url", {"mp-gallery-select": "map"})
