"""Dark (default) and light themes: one token table for CSS, Plotly and maps.

`THEMES` (in `utils/settings.py`) is the only place theme colours are written down. `css_variables`
turns it into the `--mx-*` custom properties the shell stylesheet uses, so
`assets/shell.css` never carries a hex value of its own; the Plotly theme
(`assets/plot_theme.js`) and the static map renderer read the same table.

The choice lives in `theme-store` (browser `localStorage`, so it survives a
reload) and is applied client-side: `data-mx-theme` on `<html>` drives the
CSS, and Mantine's colour scheme follows. No server round trip.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, html

from utils.settings import THEMES

DEFAULT_THEME = "dark"


def css_variables() -> str:
    """`--mx-*` custom properties for both themes, keyed on `data-mx-theme`."""
    blocks = []
    for name, tokens in THEMES.items():
        props = "".join(f"--mx-{key}:{value};" for key, value in tokens.items())
        selector = f':root[data-mx-theme="{name}"]'
        if name == DEFAULT_THEME:
            selector = f":root,{selector}"
        blocks.append(f"{selector}{{{props}color-scheme:{name};}}")
    return "\n".join(blocks)


def theme_toggle() -> dmc.ActionIcon:
    """Switch between dark and light; the label names the theme it switches to."""
    return dmc.ActionIcon(
        html.Span(className="theme-icon"),
        id="theme-toggle",
        variant="subtle",
        size="lg",
        className="theme-toggle",
        **{"aria-label": "Switch to light theme"},
    )


dash.clientside_callback(
    """
    function(clicks, stored) {
        const ctx = dash_clientside.callback_context;
        let theme = stored || 'dark';
        if (ctx.triggered_id === 'theme-toggle' && clicks) {
            theme = theme === 'dark' ? 'light' : 'dark';
        }
        document.documentElement.setAttribute('data-mx-theme', theme);
        const next = theme === 'dark' ? 'light' : 'dark';
        return [theme, theme, 'Switch to ' + next + ' theme'];
    }
    """,
    Output("theme-store", "data"),
    Output("mantine-provider", "forceColorScheme"),
    Output("theme-toggle", "aria-label"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
)
