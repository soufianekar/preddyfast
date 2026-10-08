"""write_aggregates / load_aggregates and the Scorecard page, on synthetic rows."""

import importlib
import os

import dash
import numpy as np
import pandas as pd
import pytest

from utils import scorecard, scores_db, station_network
from utils.figures import figure_text as figures_text
from utils.settings import SCORECARD_DAYS, SCORECARD_PERIODS

TODAY = pd.Timestamp("2026-10-06")
NEWEST = pd.Timestamp("2026-10-05")


def _synthetic_rows(variables=("temperature_2m",), stores=("ifs_ens", "gefs"), n_days=10):
    frames = []
    inits = pd.date_range(NEWEST - pd.Timedelta(days=n_days - 1), NEWEST)
    for variable in variables:
        for k, store in enumerate(stores):
            idx = pd.MultiIndex.from_product(
                [inits, ["A", "B"], [24.0, 72.0, 120.0, 240.0]],
                names=["init", "station", "lead_hours"],
            ).to_frame(index=False)
            err = 1.0 + 0.5 * k + 0.1 * (idx["lead_hours"] / 24)
            frames.append(idx.assign(
                store=store, variable=variable,
                valid_time=idx["init"] + pd.to_timedelta(idx["lead_hours"], unit="h"),
                obs=0.0, fc_mean=0.0, err=err, ae=err.abs(), se=err**2, crps=err / 2,
                spread2=1.0, wet_fc=np.nan, wet_obs=np.nan, brier=np.nan,
            ))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(scores_db, "SCORECARD_DIR", str(tmp_path / "scorecard"))
    monkeypatch.setattr(scorecard, "PRECOMPUTE_DIR", str(tmp_path / "precomputed"))
    rows = _synthetic_rows()
    monkeypatch.setattr(
        scores_db, "load_rows",
        lambda stores, variables, start, end: rows[
            rows["store"].isin(stores) & rows["variable"].isin(variables)
        ].reset_index(drop=True),
    )
    net = pd.DataFrame({
        "id": ["A", "B"], "name": ["Alpha", "Bravo"], "lat": [50.0, 52.0], "lon": [8.0, 13.0],
    })
    monkeypatch.setattr(
        station_network, "load_network", lambda: (net, TODAY - pd.Timedelta(days=SCORECARD_DAYS), TODAY)
    )
    return tmp_path


def _agg():
    scorecard.write_aggregates(TODAY)
    return scorecard.load_aggregates()


def test_aggregates_hold_every_kind_period_and_lead_day(env):
    agg, meta = _agg()
    assert set(agg["kind"]) == {"lead", "lead_relative", "station"}
    for kind in ("lead", "lead_relative", "station"):
        assert set(agg.loc[agg["kind"] == kind, "period_days"]) == set(SCORECARD_PERIODS)
    station = agg[agg["kind"] == "station"]
    assert set(station["lead_day"]) == {1, 3, 5, 10}
    assert meta["n_stations"] == 2
    assert meta["stores"] == ["ifs_ens", "gefs"]
    assert meta["end"] == "2026-10-05"
    assert meta["generated"].endswith("Z")


def test_relative_baseline_rows_are_zero(env):
    agg, _ = _agg()
    rel = agg[(agg["kind"] == "lead_relative") & (agg["store"] == "ifs_ens")]
    assert len(rel) and (rel["rmse"].abs() < 1e-9).all()
    other = agg[(agg["kind"] == "lead_relative") & (agg["store"] == "gefs")]
    assert (other["rmse"] < 0).all()  # gefs has the larger error


def test_station_rows_carry_coordinates_and_names(env):
    agg, _ = _agg()
    station = agg[agg["kind"] == "station"]
    a = station[station["station"] == "A"]
    assert (a["lat"] == 50.0).all() and (a["lon"] == 8.0).all() and (a["name"] == "Alpha").all()


def test_load_rereads_after_mtime_change(env):
    first, _ = _agg()
    assert scorecard.load_aggregates()[0] is first  # memoized
    path = scorecard.aggregates_path()
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert scorecard.load_aggregates()[0] is not first


def test_load_returns_none_without_file(env):
    assert scorecard.load_aggregates() is None


def test_stale_check_follows_newest_init_file(env):
    assert scorecard._aggregates_stale()
    scorecard.write_aggregates(TODAY)
    assert not scorecard._aggregates_stale()
    init_file = scores_db.scorecard_path("ifs_ens", "temperature_2m", NEWEST)
    init_file.parent.mkdir(parents=True, exist_ok=True)
    init_file.write_bytes(b"x")
    assert scorecard._aggregates_stale()


# --- the page ----------------------------------------------------------------------


@pytest.fixture
def page(env):
    dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    return importlib.import_module("pages.evaluation")


def _call(page, **kw):
    args = {"variable": "temperature_2m", "score": "rmse", "period": str(SCORECARD_DAYS),
            "display": "relative", "models": ["ifs_ens", "gefs"], "lead": "3",
            "map_model": "gefs"}
    args.update(kw)
    return page.update_scorecard(**args)


def test_layout_builds_without_aggregates_and_with_query(page):
    assert page.layout() is not None
    assert page.layout(variable="wind_speed_10m", score="nope", period="7", models="gefs") is not None


def test_missing_aggregates_show_the_alert(page):
    heat, lines, station_map, alert, _ = _call(page)
    assert alert.children == page._MISSING and alert.color == "red"
    for fig in (heat, lines, station_map):
        assert fig.layout.annotations  # "No scorecard published yet"


def _alerts(component) -> list:
    """Every dmc.Alert in a layout tree."""
    found = [component] if type(component).__name__ == "Alert" else []
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and hasattr(child, "to_plotly_json"):
            found += _alerts(child)
    return found


def test_layout_shows_a_red_alert_when_nothing_is_published(page):
    alerts = _alerts(page.layout())
    assert [a.color for a in alerts] == ["red"] and alerts[0].children == page._MISSING


def test_layout_shows_no_alert_once_published(page):
    scorecard.write_aggregates(TODAY)
    assert _alerts(page.layout()) == []


def test_callback_returns_three_figures_and_no_alert(page):
    scorecard.write_aggregates(TODAY)
    heat, lines, station_map, alert, provenance = _call(page)
    assert alert == []
    assert any(t.type == "heatmap" for t in heat.data)
    assert any(t.type == "scatter" for t in lines.data)
    assert any(t.type == "scatter" and t.name == "stations" for t in station_map.data)
    assert "2 stations" in provenance and "2026-10-05" in provenance


def test_absolute_display_and_baseline_not_selected(page):
    scorecard.write_aggregates(TODAY)
    _, _, _, alert, _ = _call(page, display="absolute", models=["gefs"])
    assert alert == []
    heat = _call(page, display="relative", models=["gefs"])[0]
    assert "ECMWF IFS ENS" in str(heat.data[0].y)  # baseline kept so the percent is defined


def test_invalid_score_for_variable_is_ignored_by_the_figure_callback(page):
    scorecard.write_aggregates(TODAY)
    assert _call(page, score="ets") == (dash.no_update,) * 5


def test_changing_variable_resets_an_invalid_score(page):
    data, value = page.options_for_variable("precipitation_6h", "rmse")
    assert value == "mae" and "rmse" not in [o["value"] for o in data]
    assert page.options_for_variable("wind_speed_10m", "crps")[1] == "crps"


def test_score_labels_carry_their_direction(page):
    labels = {o["value"]: o["label"] for o in page._score_options("precipitation_6h")}
    assert labels["mae"] == "MAE (lower is better)"
    assert labels["brier"] == "fair Brier ≥ 0.6 mm/6 h (lower is better)"
    assert labels["frequency_bias"].endswith("(ideal 1)")


def test_station_bias_correction_uses_the_all_lead_offset(env):
    """The station map's rmse_bc at one lead matches the By-lead table's definition."""
    agg, _ = _agg()
    rows = _synthetic_rows()
    err = rows[(rows["store"] == "ifs_ens") & (rows["station"] == "A")]
    offset = err["err"].mean()
    day1 = err[err["lead_hours"] == 24.0]
    expected = np.sqrt(((day1["err"] - offset) ** 2).mean())
    got = agg[(agg["kind"] == "station") & (agg["store"] == "ifs_ens")
              & (agg["station"] == "A") & (agg["lead_day"] == 1)
              & (agg["period_days"] == max(SCORECARD_PERIODS))]["rmse_bc"]
    assert len(got) == 1 and got.iloc[0] == pytest.approx(expected)
    assert expected > 0  # a per-lead offset would give 0 here


def test_stored_relative_rows_equal_relative_to_of_stored_lead_rows(env):
    agg, _ = _agg()
    keys = ["period_days", "variable", "store", "lead_day"]
    lead = agg[agg["kind"] == "lead"]
    stored = agg[agg["kind"] == "lead_relative"].sort_values(keys).reset_index(drop=True)
    parts = [scorecard.relative_to(g.drop(columns=["kind"])) for _, g in lead.groupby("period_days")]
    again = pd.concat(parts).sort_values(keys).reset_index(drop=True)
    for col in ("rmse", "mae", "crps", "bias"):
        np.testing.assert_allclose(stored[col], again[col], equal_nan=True)


def test_relative_heatmap_labels_the_baseline_row(page):
    scorecard.write_aggregates(TODAY)
    heat = _call(page, display="relative", models=["gefs"])[0]
    assert any("(baseline)" in y for y in heat.data[0].y)


def test_map_model_follows_selected_models(page):
    data, value = page.options_for_map_model(["gefs", "ifs_ens"], "icon_eu")
    assert value == "ifs_ens" and {o["value"] for o in data} == {"gefs", "ifs_ens"}
    assert page.options_for_map_model(["gefs", "ifs_ens"], "gefs")[1] == "gefs"
    scorecard.write_aggregates(TODAY)
    station_map = _call(page, map_model="ifs_ens")[2]
    assert "ECMWF IFS ENS" in figures_text(station_map)


def test_corrupt_aggregates_give_a_red_alert_not_an_exception(page):
    scorecard.write_aggregates(TODAY)
    scorecard.aggregates_path().write_bytes(b"not parquet")
    heat, lines, station_map, alert, _ = _call(page)
    assert alert.color == "red"
    assert page.layout() is not None


def test_none_period_or_lead_is_ignored(page):
    scorecard.write_aggregates(TODAY)
    assert _call(page, period=None) == (dash.no_update,) * 5
    assert _call(page, lead=None) == (dash.no_update,) * 5


def test_description_names_the_configured_network_not_germany(page):
    from utils.settings import SCORECARD_STATIONS

    text = str(page.layout().children if hasattr(page.layout(), "children") else page.layout())
    assert "Germany" not in text
    assert f"{len(SCORECARD_STATIONS)} major European city airports" in text


def test_station_map_offers_only_lead_days_with_rows(page, monkeypatch):
    """Day 10 is dropped when no selected model has station rows there."""
    rows = _synthetic_rows()
    rows = rows[rows["lead_hours"] < 240.0].reset_index(drop=True)
    monkeypatch.setattr(
        scores_db, "load_rows",
        lambda stores, variables, start, end: rows[
            rows["store"].isin(stores) & rows["variable"].isin(variables)
        ].reset_index(drop=True),
    )
    scorecard.write_aggregates(TODAY)
    data, value = page.options_for_lead("temperature_2m", "7", ["gefs"], "10")
    assert [o["value"] for o in data] == ["1", "3", "5"] and value == "3"
    assert page.options_for_lead("temperature_2m", "7", ["gefs"], "5")[1] == "5"


def test_station_map_of_spread_skill_on_a_deterministic_model_is_a_note(page, monkeypatch):
    rows = _synthetic_rows(stores=("ifs_ens", "gfs"))
    monkeypatch.setattr(
        scores_db, "load_rows",
        lambda stores, variables, start, end: rows[
            rows["store"].isin(stores) & rows["variable"].isin(variables)
        ].reset_index(drop=True),
    )
    scorecard.write_aggregates(TODAY)
    station_map = _call(page, score="spread_skill", models=["ifs_ens", "gfs"],
                        map_model="gfs")[2]
    assert not [t for t in station_map.data if t.name == "stations"]
    note = " ".join(a.text for a in station_map.layout.annotations)
    assert "GFS" in note and "deterministic" in note
