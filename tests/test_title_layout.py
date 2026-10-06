"""Time-series figures: bold title, one muted context line, provenance footer under the legend."""

import pandas as pd
import pytest
from test_eval_figures import CONTEXT, SCORES
from test_figures_multi import _fig as regional_fig

from utils.figures import PROVENANCE, figure_text, make_meteogram_fig, make_scores_fig
from utils.settings import EVAL_VARIABLES, GRID_STORES


def _meteogram():
    times = pd.date_range("2026-10-01 06:00", periods=4, freq="6h")
    df = pd.DataFrame([{"date": t, "member": m, "value": float(m + i)} for i, t in enumerate(times) for m in range(3)])
    return make_meteogram_fig(
        {"ifs_ens": df}, {"ifs_ens": "ECMWF IFS ENS"}, {"label": "Temperature", "unit": "°C"}, "Paris",
        subtitle="Paris · dynamical.org archive",
        provenance=["ECMWF IFS ENS run 2026-10-01 00 UTC · 51 members · 0.25° cell 48.75°N 2.25°E"],
    )


FIGURES = {
    "regional": regional_fig,
    "meteogram": _meteogram,
    "scores": lambda: make_scores_fig(SCORES, "crps", EVAL_VARIABLES["temperature_2m"], CONTEXT),
}


def _footer(fig):
    return next(a for a in fig.layout.annotations if a.name == PROVENANCE)


@pytest.mark.parametrize("name", FIGURES)
def test_title_is_one_bold_line_with_a_context_subtitle(name):
    title = FIGURES[name]().layout.title
    assert title.text.startswith("<b>") and "<br>" not in title.text
    assert title.subtitle.text and "<br>" not in title.subtitle.text
    assert title.font.size == 16 and title.subtitle.font.size == 12


@pytest.mark.parametrize("name", FIGURES)
def test_provenance_footer_sits_under_the_legend(name):
    fig = FIGURES[name]()
    footer = _footer(fig)
    assert footer.font.size == 11 and footer.yref == "paper" and footer.yanchor == "top"
    plot_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    legend_px = -fig.layout.legend.y * plot_px
    assert fig.layout.legend.yanchor == "top" and legend_px > 0
    # Footer starts below where the legend starts, and the bottom margin holds it.
    assert -footer.yshift > legend_px
    lines = footer.text.count("<br>") + 1
    assert fig.layout.margin.b >= -footer.yshift + 11 * lines


def test_regional_footer_has_one_line_per_model():
    lines = _footer(regional_fig()).text.split("<br>")
    assert len(lines) == 2
    assert GRID_STORES["ifs_ens"]["model_label"] in lines[0] and GRID_STORES["aifs_ens"]["model_label"] in lines[1]


def test_figure_text_gathers_title_subtitle_and_footer():
    text = figure_text(_meteogram())
    assert "Temperature" in text and "dynamical.org archive" in text and "51 members" in text
