import json
import shutil
import subprocess
from pathlib import Path

import pytest

from components.theme_toggle import THEMES

JS = Path(__file__).resolve().parents[1] / "src" / "assets" / "plot_theme.js"


def _update(layout: dict, theme: str) -> dict:
    prog = f"""
    const window = {{MX_THEMES: {json.dumps(THEMES)}}};
    const document = {{addEventListener() {{}}, documentElement: {{getAttribute() {{ return null; }}}}}};
    const MutationObserver = function() {{ return {{observe() {{}}}}; }};
    {JS.read_text()}
    console.log(JSON.stringify(window.mxPlotUpdate({json.dumps(layout)}, {json.dumps(theme)})));
    """
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def test_backgrounds_transparent_and_text_from_tokens():
    u = _update({"xaxis": {}, "yaxis": {}}, "dark")
    assert u["paper_bgcolor"] == "rgba(0,0,0,0)" and u["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert u["font.color"] == THEMES["dark"]["ink"]


def test_every_axis_including_numbered_ones():
    u = _update({"xaxis": {}, "yaxis": {}, "xaxis2": {}, "yaxis3": {}}, "light")
    for ax in ("xaxis", "yaxis", "xaxis2", "yaxis3"):
        assert u[f"{ax}.gridcolor"] == THEMES["light"]["grid"]
        assert u[f"{ax}.zerolinecolor"] == THEMES["light"]["zero"]
    assert "xaxis4.gridcolor" not in u


def test_data_colours_untouched():
    u = _update({"xaxis": {}, "colorway": ["#123456"]}, "dark")
    assert not any(k.startswith("colorway") or "marker" in k or "line.color" == k for k in u)


def _events(script: str) -> list:
    prog = f"""
    const window = {{MX_THEMES: {json.dumps(THEMES)}}};
    const handlers = {{}};
    let attr = 'dark', relayouts = [];
    const gd = {{layout: {{xaxis: {{}}, paper_bgcolor: 'rgba(0,0,0,0)'}},
                on(ev, fn) {{ handlers[ev] = fn; }}}};
    window.Plotly = {{relayout(g, u) {{ relayouts.push(u['font.color']); return Promise.resolve(); }}}};
    const document = {{
      addEventListener(ev, fn) {{ if (ev === 'DOMContentLoaded') setTimeout(fn, 0); }},
      documentElement: {{getAttribute() {{ return attr; }}}},
      body: {{}},
      querySelectorAll() {{ return [gd]; }},
    }};
    let observers = [];
    const MutationObserver = function(cb) {{ observers.push(cb); return {{observe() {{}}}}; }};
    {JS.read_text()}
    setTimeout(() => {{ {script}; console.log(JSON.stringify(relayouts)); }}, 5);
    """
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_a_new_figure_from_a_callback_is_themed_again():
    # dcc.Graph swaps in a new layout object (Plotly.react), then afterplot fires.
    calls = _events("gd.layout = {xaxis: {}, paper_bgcolor: 'rgba(0,0,0,0)'}; handlers['plotly_afterplot']()")
    assert len(calls) == 2


def test_the_restyle_itself_does_not_loop():
    calls = _events("handlers['plotly_afterplot'](); handlers['plotly_afterplot']()")
    assert len(calls) == 1


def test_export_is_left_alone():
    calls = _events("gd._mxExporting = true; gd.layout = {xaxis: {}}; handlers['plotly_afterplot']()")
    assert len(calls) == 1


def test_subtitle_and_provenance_footer_are_muted_text():
    layout = {"xaxis": {}, "title": {"text": "t", "subtitle": {"text": "s"}},
              "annotations": [{"text": "trend"}, {"name": "provenance", "text": "p"}]}
    u = _update(layout, "light")
    assert u["title.subtitle.font.color"] == THEMES["light"]["dim"]
    assert u["annotations[1].font.color"] == THEMES["light"]["dim"]
    assert "annotations[0].font.color" not in u  # other annotations keep their own colour


def _fit(full_layout: dict, layout: dict):
    prog = f"""
    const window = {{MX_THEMES: {json.dumps(THEMES)}}};
    const document = {{addEventListener() {{}}, documentElement: {{getAttribute() {{ return null; }}}}}};
    const MutationObserver = function() {{ return {{observe() {{}}}}; }};
    {JS.read_text()}
    console.log(JSON.stringify(window.mxFrameFit({json.dumps(full_layout)}, {json.dumps(layout)})));
    """
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


FRAME = {"axis_px": 64, "gap_px": 8, "line_px": 15, "lines": 2, "pad_px": 10}
LAYOUT = {"meta": {"mx_frame": FRAME}, "annotations": [{"text": "x"}, {"name": "provenance", "text": "a<br>b"}]}


def test_footer_refits_to_the_measured_legend_height():
    # A narrow screen wraps the legend to 6 rows (132 px): footer and margin follow it.
    u = _fit({"height": 620, "margin": {"t": 74, "b": 160}, "legend": {"_height": 132}}, LAYOUT)
    footer_top = 64 + 132 + 8
    assert u["annotations[1].yshift"] == -footer_top
    assert u["margin.b"] == footer_top + 2 * 15 + 10
    assert u["legend.y"] == pytest.approx(-64 / (620 - 74 - u["margin.b"]))


def test_no_refit_once_converged_or_without_a_frame():
    b = 64 + 44 + 8 + 30 + 10
    assert _fit({"height": 620, "margin": {"t": 74, "b": b}, "legend": {"_height": 44}}, LAYOUT) is None
    assert _fit({"height": 620, "margin": {"t": 74, "b": 99}, "legend": {"_height": 44}}, {"annotations": []}) is None
