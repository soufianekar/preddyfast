"""App-level state shared by every page: focus stores, nav, and URL syncing.

The stores live in `app.layout`, not in a page layout, so a location, box or
variable picked on one page is still there when the user moves to another.
They use `session` storage so a reload keeps them too; a query string, when
present, wins over what the session held.
"""

from __future__ import annotations

import json
import urllib.parse

import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from components.theme_toggle import theme_toggle
from utils.settings import URL_BASE_PATHNAME
from utils.url_state import parse_location
from utils.variable_family import publish, resolve

_BASE = URL_BASE_PATHNAME.rstrip("/")

# Two sections, each with its own views. A view is
# (link id, label, path, variable namespace the page reads); the section's
# header link opens its first view.
_SECTIONS = (
    (
        "climate",
        "Climate",
        (
            ("nav-climate", "At a location", "/", "daily"),
            ("nav-maps", "Maps & indices", "/maps", "era5"),
        ),
    ),
    (
        "weather",
        "Weather",
        (
            ("nav-meteogram", "At a location", "/meteogram", "daily"),
            ("nav-regional", "Area average", "/regional", "grid"),
            ("nav-evaluation", "Evaluation", "/evaluation", "grid"),
        ),
    ),
)
_VIEWS = tuple(view for _, _, views in _SECTIONS for view in views)


def shared_state() -> list:
    """Components that must exist on every page; mount once in `app.layout`."""
    return [
        dcc.Location(id="url"),
        dcc.Store(id="location-store", storage_type="session"),
        dcc.Store(id="bbox-store", storage_type="session"),
        dcc.Store(id="variable-store", storage_type="session"),
        # Theme survives reloads and new tabs; see components/theme_toggle.py.
        dcc.Store(id="theme-store", storage_type="local"),
    ]


def nav_rail() -> html.Nav:
    """Left rail: the two sections with their views, and the theme switch.

    Each section is marked by its front symbol, as drawn on a synoptic
    chart — a warm front's semicircle for Climate, a cold front's triangle
    for Weather — the app's one decorative element. The active view carries
    the occluded-front bar. On narrow screens the rail becomes a top bar
    (`assets/shell.css`). `nav_links` keeps every link carrying the current
    variable and marks what is active.
    """
    sections = [
        html.Div(
            [
                dcc.Link(
                    [html.Span(className="front-mark"), html.Span(label)],
                    id=f"sec-{key}",
                    href=f"{_BASE}{views[0][2]}",
                    className="rail-section-link",
                ),
                html.Div(
                    [
                        dcc.Link(view_label, id=link_id, href=f"{_BASE}{path}", className="rail-view")
                        for link_id, view_label, path, _ in views
                    ],
                    className="rail-views",
                ),
            ],
            id=f"rail-{key}",
            className=f"rail-section section-{key}",
        )
        for key, label, views in _SECTIONS
    ]
    return html.Nav(
        [
            dcc.Link("asifast", href=f"{_BASE}/", className="rail-brand"),
            html.Div(sections, className="rail-sections"),
            html.Div(theme_toggle(), className="rail-foot"),
        ],
        className="rail",
        **{"aria-label": "Sections and views"},
    )


@callback(
    *(Output(link_id, "href") for link_id, *_ in _VIEWS),
    *(Output(link_id, "className") for link_id, *_ in _VIEWS),
    *(Output(f"sec-{key}", "href") for key, *_ in _SECTIONS),
    *(Output(f"rail-{key}", "className") for key, *_ in _SECTIONS),
    Input("variable-store", "data"),
    Input("url", "pathname"),
)
def nav_links(store, pathname):
    """Point each link at its view with that view's member of the current family.

    The page's `layout(**query_params)` applies it, so arriving needs no
    callback and cannot race the page's own controls.
    """
    current = (pathname or "").rstrip("/")
    hrefs, classes, active_section = {}, {}, _SECTIONS[0][0]
    for key, _, views in _SECTIONS:
        for link_id, _, path, namespace in views:
            variable = resolve(store, namespace)
            query = (
                f"?{urllib.parse.urlencode({'variable': variable})}" if variable else ""
            )
            hrefs[link_id] = f"{_BASE}{path}{query}"
            active = current == f"{_BASE}{path}".rstrip("/")
            classes[link_id] = "rail-view active" if active else "rail-view"
            if active:
                active_section = key
    view_ids = [link_id for link_id, *_ in _VIEWS]
    return (
        *(hrefs[i] for i in view_ids),
        *(classes[i] for i in view_ids),
        *(hrefs[views[0][0]] for _, _, views in _SECTIONS),
        *(
            f"rail-section section-{key}" + (" active" if key == active_section else "")
            for key, *_ in _SECTIONS
        ),
    )


def publish_variable(select_id: str, namespace: str, default: str) -> None:
    """Record explicit variable choices on a page in `variable-store`.

    `dmc.Select` reports its value once on mount with the prop marked as
    changed, so `prevent_initial_call` does not filter it. The mount value is
    the one the page opens with — `resolve(store) or default` — and a user
    cannot *change to* the value already shown, so that value is skipped.
    Republishing it would let a page with no equivalent (regional, after
    snowfall) overwrite the choice with its default.
    """

    @callback(
        Output("variable-store", "data", allow_duplicate=True),
        Input(select_id, "value"),
        State("variable-store", "data"),
        prevent_initial_call=True,
    )
    def _publish(key, store):
        if not key or key == (resolve(store, namespace) or default):
            return no_update
        return publish(store, namespace, key)


@callback(
    Output("location-store", "data", allow_duplicate=True),
    Input("url", "search"),
    prevent_initial_call="initial_duplicate",
)
def restore_location(search):
    """Seed the location store from a shared link.

    Page controls are restored by each page's `layout(**query_params)`, which
    runs before the controls exist and so needs no callback. The box is
    restored the same way (`bbox_select(bounds=...)`); it cannot be done here,
    because `sync_bbox` publishes on mount and would race this write.
    """
    if not search:
        return no_update

    params = dict(urllib.parse.parse_qsl(search.lstrip("?")))
    return parse_location(params) or no_update


def url_sync(sink_id: str, controls: dict[str, str | None]) -> None:
    """Keep the query string in step with a page's controls, client-side.

    Parameters
    ----------
    sink_id
        Id of a hidden `html.Div` on the page; a clientside callback needs an
        output, and this one carries no information.
    controls
        Component id → query key, in the order to write them. A `None` key
        marks a store whose dict is flattened into the query under its own
        keys (`location-store` → lat/lon/name). While such a store is empty
        the URL is left alone. Lists are comma-joined; empty values are
        omitted.
    """
    keys = json.dumps(list(controls.values()))
    dash.clientside_callback(
        f"""
        function() {{
            const keys = {keys};
            const p = new URLSearchParams();
            for (let i = 0; i < keys.length; i++) {{
                const v = arguments[i];
                if (keys[i] === null) {{
                    if (!v) {{ return window.location.href; }}
                    for (const k in v) {{ p.set(k, v[k]); }}
                }} else if (Array.isArray(v)) {{
                    if (v.length) {{ p.set(keys[i], v.join(',')); }}
                }} else if (v !== null && v !== undefined && v !== '') {{
                    p.set(keys[i], v);
                }}
            }}
            window.history.replaceState(null, '', '?' + p.toString());
            return window.location.href;
        }}
        """,
        Output(sink_id, "children"),
        *(
            Input(component_id, "data" if key is None else "value")
            for component_id, key in controls.items()
        ),
    )
