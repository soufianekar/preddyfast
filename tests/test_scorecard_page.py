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
    # No ranks or thresholds in these rows: no rank, reliability or roc kinds.
    assert set(agg["kind"]) == {"lead", "lead_relative", "station", "diurnal"}
    for kind in ("lead", "lead_relative", "station", "diurnal"):
        assert set(agg.loc[agg["kind"] == kind, "period_days"]) == set(SCORECARD_PERIODS)
    assert set(agg["mode"]) == {"raw"}  # no hc / perturbed columns in these rows
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
            "map_model": "gefs", "obs_mode": "raw"}
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
    data, value = page.options_for_variable("precipitation_24h", "rmse")
    flat = [o["value"] for g in data for o in g["items"]]
    assert value == "mae" and "rmse" not in flat
    assert page.options_for_variable("wind_speed_10m", "crps")[1] == "crps"


def test_score_labels_carry_their_direction(page):
    labels = {o["value"]: o["label"] for o in page._flat_score_options("precipitation_24h")}
    assert labels["mae"] == "MAE (lower is better)"
    assert labels["brier"] == "fair Brier ≥ 1 mm/24 h (lower is better)"
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
    parts = [scorecard.relative_to(g.drop(columns=["kind"]))
             for _, g in lead.groupby(["mode", "period_days"])]
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


# --- Task 5: modes, new kinds, significance, metadata -----------------------------


def _full_rows(n_days=10):
    """Synthetic rows with every Task 4 column, for two ensembles and GFS."""
    rng = np.random.default_rng(7)
    frames = []
    inits = pd.date_range(NEWEST - pd.Timedelta(days=n_days - 1), NEWEST)
    for variable in ("temperature_2m", "precipitation_24h"):
        for k, store in enumerate(("ifs_ens", "gefs", "gfs")):
            leads = [30.0, 54.0, 126.0] if variable == "precipitation_24h" else [
                6.0, 12.0, 24.0, 72.0, 120.0, 240.0]
            idx = pd.MultiIndex.from_product(
                [inits, ["A", "B"], leads], names=["init", "station", "lead_hours"]
            ).to_frame(index=False)
            n = len(idx)
            m = 1.0 if store == "gfs" else 50.0
            err = rng.normal(0.5 * k, 1.0, n)
            obs = rng.gamma(1.0, 3.0, n)
            p = rng.integers(0, 51, n) / 50
            cols = dict(
                store=store, variable=variable,
                valid_time=idx["init"] + pd.to_timedelta(idx["lead_hours"], unit="h"),
                obs=obs, fc_mean=obs + err, err=err, ae=np.abs(err), se=err**2,
                crps=np.abs(err) / 2, spread2=np.nan if m == 1 else 1.0,
                wet_fc=(obs + err >= 1).astype(float), wet_obs=(obs >= 1).astype(float),
                brier=np.nan if m == 1 else 0.1,
                dz=np.where(idx["station"] == "A", 50.0, np.nan),
                err_hc=np.where(idx["station"] == "A", err + 0.3, np.nan),
                crps_hc=np.where(idx["station"] == "A", np.abs(err) / 2, np.nan),
                crps_pert=np.abs(err) / 3, spread2_pert=1.2, err_pert=err,
                rank=np.nan if m == 1 else rng.integers(0, 51, n).astype(float),
                rank_pert=rng.integers(0, 51, n).astype(float),
                n_members=m, n_members_pert=50.0, obs_persist=obs + rng.normal(0, 2, n),
            )
            if variable == "temperature_2m":
                cols.update(wet_fc=np.nan, wet_obs=np.nan, brier=np.nan)
            for t in (1, 5, 10):
                on = variable == "precipitation_24h"
                cols.update({
                    f"p_{t}": p if on and m > 1 else (obs + err >= t) * 1.0 if on else np.nan,
                    f"p_pert_{t}": p if on else np.nan,
                    f"wet_fc_{t}": (obs + err >= t) * 1.0 if on else np.nan,
                    f"wet_obs_{t}": (obs >= t) * 1.0 if on else np.nan,
                    f"brier_{t}": 0.1 if on and m > 1 else np.nan,
                    f"brier_pert_{t}": 0.12 if on else np.nan,
                })
            frames.append(idx.assign(**cols))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def full_env(env, monkeypatch):
    rows = _full_rows()
    monkeypatch.setattr(
        scores_db, "load_rows",
        lambda stores, variables, start, end: rows[
            rows["store"].isin(stores) & rows["variable"].isin(variables)
        ].reset_index(drop=True),
    )
    return rows


def test_full_aggregates_hold_every_kind_and_mode(full_env):
    from utils.settings import SCORECARD_MODES

    agg, meta = _agg()
    assert set(agg["kind"]) == {
        "lead", "lead_relative", "station", "diurnal", "rank", "reliability", "roc"
    }
    assert set(agg["mode"]) == set(SCORECARD_MODES)
    assert meta["modes"] == list(SCORECARD_MODES)
    assert meta["height_corrected_excluded"] == 1  # station B has no dz
    assert meta["orography_source"].startswith("ERA5")
    assert meta["mode_raw_scores"] == {
        k: list(v) for k, v in scorecard.MODE_RAW_SCORES.items()
    }
    assert "ets_1" in meta["mode_raw_scores"]["representativeness"]
    hc = agg[(agg["kind"] == "lead") & (agg["mode"] == "height_corrected")]
    t2m = hc[hc["variable"] == "temperature_2m"]
    assert t2m["mode_applied"].all() and (t2m["n_stations"] == 1).all()
    assert not hc[hc["variable"] == "precipitation_24h"]["mode_applied"].any()
    rel = agg[(agg["kind"] == "reliability")]
    assert set(rel["threshold"]) == {1.0, 5.0, 10.0}
    raw_gfs = rel[(rel["mode"] == "raw") & (rel["store"] == "gfs")]
    pert_gfs = rel[(rel["mode"] == "representativeness") & (rel["store"] == "gfs")]
    assert raw_gfs.empty and not pert_gfs.empty


def test_relative_rows_carry_significance_columns(full_env):
    agg, _ = _agg()
    rel = agg[(agg["kind"] == "lead_relative") & (agg["mode"] == "raw")
              & (agg["variable"] == "temperature_2m")]
    assert {"diff_rmse", "lo_rmse", "hi_rmse", "sig_rmse", "sig_crps"} <= set(rel.columns)
    base = rel[rel["store"] == "ifs_ens"]
    assert (base["diff_rmse"] == 0).all() and base["sig_rmse"].isna().all()
    gefs = rel[rel["store"] == "gefs"]
    assert gefs["sig_rmse"].notna().all()  # 10 runs >= SCORECARD_MIN_INITS


def test_page_reads_raw_rows_only_and_old_files_default_to_raw(full_env, page):
    import pyarrow as pa
    import pyarrow.parquet as pq

    agg, _ = _agg()
    heat, _, _, alert, _ = _call(page, display="absolute")
    assert alert is None or getattr(alert, "color", None) != "red"
    assert heat.data
    sel = page._select(agg, "lead", "temperature_2m", 7, ["ifs_ens"])
    assert set(sel["mode"]) == {"raw"} and sel["lead_day"].is_unique
    old = agg[(agg["mode"] == "raw")].drop(columns=["mode", "mode_applied"])
    table = pa.Table.from_pandas(old, preserve_index=False)
    pq.write_table(table, scorecard.aggregates_path())
    again, _ = scorecard.load_aggregates()
    assert set(again["mode"]) == {"raw"} and again["mode_applied"].all()


# --- Task 7: observations mode, Diagnostics tab, significance, method note -------


def _find(component, component_id):
    """The component with `component_id` anywhere under `component`, or None."""
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found = _find(child, component_id)
        if found is not None:
            return found
    return None


def test_obs_modes_offer_only_modes_that_apply():
    from utils.settings import SCORECARD_MODES

    assert scorecard.obs_modes("temperature_2m") == SCORECARD_MODES
    for variable in ("wind_speed_10m", "precipitation_24h"):
        assert scorecard.obs_modes(variable) == ("raw", "representativeness")
    assert scorecard.obs_modes("z500") == ("raw",)  # no TM865 model for Z500


def test_z500_disables_representativeness_and_falls_back(page):
    data, value = page.options_for_obs_mode("z500", "representativeness")
    assert value == "raw"
    assert next(d for d in data if d["value"] == "representativeness")["disabled"]


def test_layout_obs_mode_defaults_to_raw_and_reads_the_query(page):
    assert _find(page.layout(), "sc-obs-mode").value == "raw"
    assert _find(page.layout(obs="nope"), "sc-obs-mode").value == "raw"
    hc = _find(page.layout(obs="height_corrected"), "sc-obs-mode")
    assert hc.value == "height_corrected"
    wind = page.layout(obs="height_corrected", variable="wind_speed_10m")
    assert _find(wind, "sc-obs-mode").value == "raw"
    disabled = {d["value"]: d.get("disabled", False) for d in _find(wind, "sc-obs-mode").data}
    assert disabled == {"raw": False, "height_corrected": True, "representativeness": False}
    assert _find(page.layout(obs="representativeness"), "sc-obs-mode").value == (
        "representativeness"
    )


def test_height_corrected_is_disabled_and_falls_back_for_wind_and_precip(page):
    for variable in ("wind_speed_10m", "precipitation_24h"):
        data, value = page.options_for_obs_mode(variable, "height_corrected")
        assert value == "raw"
        assert next(d for d in data if d["value"] == "height_corrected")["disabled"]
        assert page.options_for_obs_mode(variable, "representativeness")[1] == (
            "representativeness"
        )
    data, value = page.options_for_obs_mode("temperature_2m", "height_corrected")
    assert value == "height_corrected" and not any(d.get("disabled") for d in data)
    assert page.options_for_obs_mode(None, "raw") == (dash.no_update, dash.no_update)


def test_obs_mode_is_in_the_url(page):
    assert page.URL_KEYS["sc-obs-mode"] == "obs"


def test_switching_mode_changes_the_footer(full_env, page):
    scorecard.write_aggregates(TODAY)
    raw = _call(page, obs_mode="raw")
    hc = _call(page, obs_mode="height_corrected")
    for fig in raw[:3]:
        assert "Mode: raw" in figures_text(fig)
    for fig in hc[:3]:
        text = figures_text(fig)
        assert "height-corrected with ERA5 orography" in text
        assert "1 station without elevation or orography excluded" in text
    pert = _call(page, obs_mode="representativeness")[0]
    assert "TM865" in figures_text(pert)
    assert _call(page, obs_mode="bogus") == (dash.no_update,) * 5


def test_height_corrected_heatmap_shows_hc_values(full_env, page):
    agg, _ = _agg()
    heat_raw = _call(page, display="absolute", score="bias", obs_mode="raw")[0]
    heat_hc = _call(page, display="absolute", score="bias", obs_mode="height_corrected")[0]
    assert not np.allclose(
        np.asarray(heat_raw.data[0].z, dtype=float),
        np.asarray(heat_hc.data[0].z, dtype=float),
        equal_nan=True,
    )


def test_score_select_offers_new_scores_grouped_by_threshold(page):
    t2m = [o["value"] for o in page._flat_score_options("temperature_2m")]
    assert {"rmse", "error_sd", "spread_error", "rmse_ss_persist", "crpss_persist"} <= set(t2m)
    groups = page._score_options("precipitation_24h")
    names = [g["group"] for g in groups]
    assert names == ["Amounts · headline events ≥ 1 mm", "≥ 1 mm", "≥ 5 mm", "≥ 10 mm"]
    five = {o["value"]: o["label"] for o in groups[2]["items"]}
    assert set(five) == {"ets_5", "fbi_5", "pod_5", "far_5", "brier_5", "roc_area_5"}
    assert five["brier_5"] == "Fair Brier score (≥ 5 mm) (lower is better)"
    assert "crpss_persist" in [o["value"] for o in groups[0]["items"]]
    # The headline threshold's duplicates of the general scores are dropped;
    # its ROC area (no general twin) stays.
    from utils.settings import SCORECARD_VARIABLES

    head = SCORECARD_VARIABLES["precipitation_24h"]["threshold"]
    one = [o["value"] for o in groups[1]["items"]]
    assert names[1] == f"≥ {head:g} mm" and one == [f"roc_area_{head:g}"]
    flat = [o["value"] for o in page._flat_score_options("precipitation_24h")]
    assert len(flat) == len(set(flat))
    for dup in ("ets", "fbi", "pod", "far", "brier"):
        assert f"{dup}_{head:g}" not in flat
    data, value = page.options_for_variable("temperature_2m", "ets_5")
    assert value == "rmse"
    assert page.options_for_variable("precipitation_24h", "roc_area_10")[1] == "roc_area_10"


def test_new_scores_draw_on_every_view(full_env, page):
    scorecard.write_aggregates(TODAY)
    for variable, score in (("temperature_2m", "rmse_ss_persist"),
                            ("temperature_2m", "error_sd"),
                            ("precipitation_24h", "ets_5"),
                            ("precipitation_24h", "crpss_persist")):
        heat, lines, smap, alert, _ = _call(page, variable=variable, score=score,
                                            display="absolute")
        assert getattr(alert, "color", None) != "red", (variable, score)
        assert any(t.type == "heatmap" for t in heat.data), (variable, score)


def test_relative_heatmap_marks_significance_from_stored_rows(full_env, page):
    scorecard.write_aggregates(TODAY)
    heat = _call(page, display="relative", score="rmse", models=["gefs"])[0]
    hover = " ".join(str(c) for t in heat.data for c in np.ravel(t.customdata if t.customdata is not None else []))
    assert "significant at 95 %" in hover
    assert "pairs both models share" in hover


def _diag(page, **kw):
    args = {"variable": "temperature_2m", "period": str(SCORECARD_DAYS),
            "obs_mode": "raw", "models": ["ifs_ens", "gefs", "gfs"],
            "diurnal_score": "bias", "rank_lead": "3", "threshold": "1",
            "lead_group": "1-3"}
    args.update(kw)
    return page.update_diagnostics(**args)


def test_diagnostics_return_figures_on_the_real_schema(full_env, page):
    import plotly.graph_objects as go

    scorecard.write_aggregates(TODAY)
    for variable in ("temperature_2m", "precipitation_24h"):
        for mode in scorecard.obs_modes(variable):
            for score in ("bias", "error_sd"):
                out = _diag(page, variable=variable, obs_mode=mode, diurnal_score=score)
                assert len(out) == 6
                assert all(isinstance(f, go.Figure) for f in out[:5])
    spread, diurnal, rank, rel, roc, note = _diag(page)
    assert any(t.type == "scatter" for t in spread.data)
    assert any(t.type == "scatter" for t in diurnal.data)
    assert any(t.type == "bar" for t in rank.data)
    assert "Mode: raw" in figures_text(rank)
    # Reliability / ROC are for precipitation: a note on temperature.
    assert "precipitation" in " ".join(a.text for a in rel.layout.annotations)
    assert "precipitation" in " ".join(a.text for a in roc.layout.annotations)
    # GFS is deterministic: absent from rank/reliability in raw mode, said so.
    assert "NOAA GFS" in str(note) and "deterministic" in str(note)
    pert = _diag(page, obs_mode="representativeness")[5]
    assert "NOAA GFS" not in str(pert)
    p_spread, p_diurnal, p_rank, p_rel, p_roc, _ = _diag(
        page, variable="precipitation_24h", threshold="5", lead_group="4-7"
    )
    assert any(t.type == "scatter" for t in p_rel.data)
    assert any(t.type == "scatter" for t in p_roc.data)
    assert "diurnal" in " ".join(a.text for a in p_diurnal.layout.annotations).lower()


def test_diagnostics_never_raise(page):
    out = _diag(page)  # no aggregates file
    assert getattr(out[5], "color", None) == "yellow"
    scorecard.write_aggregates(TODAY)
    scorecard.aggregates_path().write_bytes(b"not parquet")
    assert getattr(_diag(page)[5], "color", None) == "red"
    for bad in ({"variable": None}, {"period": None}, {"obs_mode": "x"},
                {"rank_lead": "7"}, {"threshold": None}, {"lead_group": "2-9"},
                {"diurnal_score": "crps"}):
        assert _diag(page, **bad) == (dash.no_update,) * 6
    out = _diag(page, models=[])
    assert len(out) == 6


def test_diagnostic_notes_are_pure_and_per_variable():
    notes = scorecard.diagnostic_notes("temperature_2m", ["ifs_ens", "gfs"], "raw")
    assert notes["spread_error"] is None and notes["diurnal"] is None
    assert "precipitation" in notes["reliability"]
    assert "NOAA GFS" in notes["deterministic"]
    assert scorecard.diagnostic_notes("temperature_2m", ["gfs"], "representativeness")[
        "deterministic"] is None
    precip = scorecard.diagnostic_notes("precipitation_24h", ["ifs_ens"], "raw")
    assert precip["reliability"] is None and precip["deterministic"] is None
    assert "06 UTC" in precip["diurnal"] and precip["spread_error"]


def test_method_note_lists_every_method(page):
    text = str(_find(page.layout(), "sc-method"))
    for words in ("nearest grid cell", "00/06/12/18 UTC", "06 UTC", "never hourly sums",
                  "fair CRPS", "0.0065 K/m", "ERA5 orography", "TM865",
                  "Ben Bouallègue", "ECMWF IFS ENS 9 km", "DWD ICON-EU 6.5 km",
                  "moving-block bootstrap", "3 days", "1000", "95 %",
                  "24·⌈lead/24⌉", "50 pairs", "5 runs"):
        assert words in text, words
    assert _find(page.layout(), "sc-method").value is None  # collapsed
    stations = next(line for line in page._method_lines() if "00/06/12/18" in line)
    assert "station" in stations and "500 hPa" not in stations
    z500 = next(line for line in page._method_lines() if "500 hPa" in line)
    for words in ("00 UTC valid times only", "IFS control T+0", "favours IFS and AIFS"):
        assert words in z500, words


def test_event_control_takes_unit_and_label_from_the_variable(page):
    from utils.settings import SCORECARD_VARIABLES

    spec = SCORECARD_VARIABLES["precipitation_24h"]
    layout = page.layout()
    data = _find(layout, "sc-threshold").data
    assert [d["label"] for d in data] == [f"≥ {t:g} {spec['unit']}" for t in spec["thresholds"]]
    assert f"Event ({spec['label']})" in str(layout)


def test_deterministic_note_stays_when_representativeness_falls_back_to_raw():
    """Z500 has no TM865 model: its deterministic stores still have no ranks."""
    notes = scorecard.diagnostic_notes("z500", ["ifs_ens", "gfs"], "representativeness")
    assert "NOAA GFS" in notes["deterministic"]
