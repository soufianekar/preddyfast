"""Shared page frame: collapsible control sidebar, canvas, and a tool column.

Layout follows the Copernicus Interactive Climate Atlas: controls on the
left, the figure filling the rest, view actions stacked on its right edge.
Only one page is mounted at a time, so the ids here are fixed; the page tells
the tools which graph to act on through `shell-graph`. Responsive rules live
in `assets/shell.css`, fullscreen resizing in `assets/shell.js`.
"""

from __future__ import annotations

import urllib.parse

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html


def icon(path: str) -> str:
    """Data URI for a 24px stroke icon; no icon package is a dependency."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        'viewBox="0 0 24 24" fill="none" stroke="#1f5fae" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
    )
    return f"data:image/svg+xml;utf8,{urllib.parse.quote(svg)}"


_ICON_COLLAPSE = icon('<path d="M11 7l-5 5 5 5M17 7l-5 5 5 5"/>')
_ICON_EXPAND = icon('<path d="M7 7l5 5-5 5M13 7l5 5-5 5"/>')
_ICON_FULLSCREEN = icon('<path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4"/>')
_ICON_DOWNLOAD = icon('<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>')


def _glyph(icon: str) -> html.Span:
    """An icon drawn in the theme's accent: the SVG is a mask, the colour CSS."""
    return html.Span(className="tool-icon", style={"--mx-icon": f'url("{icon}")'})


def _tool(icon: str, tool_id: str, label: str) -> dmc.Tooltip:
    return dmc.Tooltip(
        dmc.ActionIcon(
            _glyph(icon),
            id=tool_id,
            variant="default",
            size="lg",
            **{"aria-label": label},
        ),
        label=label,
        position="left",
        withArrow=True,
    )


def _grouped(sidebar: list) -> list:
    """Sidebar items; a string is a group heading, as is a `(heading, [controls])` tuple.

    Pages group their controls as Where / What / When / Models. A plain
    heading, no rule or eyebrow: the grouping is the information.
    """
    out = []
    for item in sidebar:
        if isinstance(item, str):
            out.append(dmc.Text(item, className="panel-heading"))
        elif isinstance(item, tuple):
            heading, controls = item
            out.append(dmc.Text(heading, className="panel-heading"))
            out.extend(controls)
        else:
            out.append(item)
    return out


def page_shell(
    title: str,
    description: str,
    sidebar: list,
    main: list,
    graph_id: str,
    share_target: str,
) -> html.Div:
    """Frame a page's controls and figure.

    Parameters
    ----------
    title, description
        Shown at the top of the sidebar.
    sidebar
        Controls, top to bottom.
    main
        Alerts, figure and anything else in the canvas.
    graph_id
        `dcc.Graph` the fullscreen and download tools act on.
    share_target
        Id of the page's `url_sync` sink; its text is the current URL, which
        the share tool copies.
    """
    return html.Div(
        [
            html.Aside(
                dmc.Stack(
                    [
                        dmc.Group(
                            [
                                dmc.Title(title, order=4, className="view-title"),
                                dmc.ActionIcon(
                                    _glyph(_ICON_COLLAPSE),
                                    id="shell-collapse",
                                    variant="subtle",
                                    **{"aria-label": "Hide controls"},
                                ),
                            ],
                            justify="space-between",
                            wrap="nowrap",
                        ),
                        dmc.Text(description, size="sm", className="view-description"),
                        *_grouped(sidebar),
                    ],
                    gap="sm",
                    p="md",
                ),
                className="shell-sidebar",
            ),
            html.Main(
                [
                    dmc.Tooltip(
                        dmc.ActionIcon(
                            _glyph(_ICON_EXPAND),
                            id="shell-expand",
                            variant="default",
                            size="lg",
                            className="shell-expand",
                            **{"aria-label": "Show controls"},
                        ),
                        label="Show controls",
                        position="right",
                    ),
                    html.Div(
                        [
                            html.Div(main, className="shell-content"),
                            html.Div(
                                [
                                    _tool(
                                        _ICON_FULLSCREEN,
                                        "shell-fullscreen",
                                        "Fullscreen",
                                    ),
                                    _tool(_ICON_DOWNLOAD, "shell-download", "Save PNG"),
                                    dmc.Tooltip(
                                        dcc.Clipboard(
                                            target_id=share_target,
                                            className="shell-tool-clipboard",
                                            title="Copy link to this view",
                                        ),
                                        label="Copy link to this view",
                                        position="left",
                                        withArrow=True,
                                    ),
                                ],
                                className="shell-toolbar",
                            ),
                        ],
                        id="shell-canvas",
                        className="shell-canvas",
                    ),
                    dcc.Store(id="shell-graph", data=graph_id),
                    html.Div(id="shell-sink", style={"display": "none"}),
                ],
                className="shell-main",
            ),
        ],
        id="shell",
        className="shell",
    )


dash.clientside_callback(
    """
    function(collapse, expand) {
        const hide = dash_clientside.callback_context.triggered_id === 'shell-collapse';
        // Plotly sizes to its container only on a window resize.
        setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
        return hide ? 'shell collapsed' : 'shell';
    }
    """,
    Output("shell", "className"),
    Input("shell-collapse", "n_clicks"),
    Input("shell-expand", "n_clicks"),
    prevent_initial_call=True,
)

dash.clientside_callback(
    """
    function(fullscreen, download, graphId) {
        const t = dash_clientside.callback_context.triggered_id;
        if (t === 'shell-fullscreen') {
            const canvas = document.getElementById('shell-canvas');
            if (document.fullscreenElement) { document.exitFullscreen(); }
            else if (canvas) { canvas.requestFullscreen(); }
        } else if (t === 'shell-download') {
            const gd = document.querySelector('#' + graphId + ' .js-plotly-plot');
            if (gd && window.Plotly) {
                // Paper is transparent on screen; a saved PNG gets the
                // theme's panel colour so its text stays legible anywhere.
                const name = document.documentElement.getAttribute('data-mx-theme') || 'dark';
                const panel = ((window.MX_THEMES || {})[name] || {}).panel || '#ffffff';
                // plot_theme.js skips the figure while it is exported, or its
                // afterplot handler would restore transparency mid-export.
                gd._mxExporting = true;
                const restore = () => {
                    gd._mxExporting = false;
                    return window.Plotly.relayout(gd, {paper_bgcolor: 'rgba(0,0,0,0)'});
                };
                window.Plotly.relayout(gd, {paper_bgcolor: panel}).then(() =>
                    window.Plotly.downloadImage(gd, {
                        // Not document.title: Dash sets it to "Updating..." mid-callback.
                        format: 'png', scale: 2,
                        filename: graphId + '-' + new Date().toISOString().slice(0, 10),
                        width: gd._fullLayout.width, height: gd._fullLayout.height,
                    })
                ).then(restore, restore);
            }
        }
        return dash_clientside.no_update;
    }
    """,
    Output("shell-sink", "children"),
    Input("shell-fullscreen", "n_clicks"),
    Input("shell-download", "n_clicks"),
    State("shell-graph", "data"),
    prevent_initial_call=True,
)
