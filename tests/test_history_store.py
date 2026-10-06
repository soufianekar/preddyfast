"""Open-Meteo history kept on disk: one full fetch per place, then only new days."""

import numpy as np
import pandas as pd
import pytest

from utils import openmeteo_api as om

TODAY = pd.Timestamp("2026-10-04")


def _series(start, end, nan_tail=0, value=1.0):
    dates = pd.date_range(start, end, freq="D")
    values = np.full(len(dates), value)
    if nan_tail:
        values[-nan_tail:] = np.nan
    return pd.DataFrame({"date": dates, "value": values})


@pytest.fixture
def fetched(monkeypatch, tmp_path):
    calls = []

    def fake(lat, lon, variable, model, start, end):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        # The reanalysis lags: the last 5 requested days come back empty.
        return _series(start, end, nan_tail=min(5, (pd.Timestamp(end) - pd.Timestamp(start)).days + 1), value=2.0)

    monkeypatch.setattr(om, "HISTORY_DIR", str(tmp_path))
    monkeypatch.setattr(om, "_fetch_daily_range", fake)
    monkeypatch.setattr(om, "_today", lambda: TODAY)
    return calls


def test_first_view_fetches_the_whole_history_once(fetched):
    df = om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    assert fetched == [(pd.Timestamp(f"{om.HISTORICAL_START_YEAR}-01-01"), TODAY - pd.Timedelta("1D"))]
    assert df["date"].iloc[-1] == TODAY - pd.Timedelta("1D") and {"date", "value", "doy", "variable"} <= set(df.columns)


def test_next_view_fetches_only_the_tail(fetched, monkeypatch):
    om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    monkeypatch.setattr(om, "_today", lambda: TODAY + pd.Timedelta("2D"))
    om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    start, end = fetched[-1]
    # From the first empty trailing day (5 NaN days before yesterday), at most a week back.
    assert end == TODAY + pd.Timedelta("1D")
    assert start == min(TODAY - pd.Timedelta("5D"), end - pd.Timedelta("6D"))
    assert (end - start).days < 14  # one weighted Open-Meteo call


def test_revised_days_replace_stored_ones(fetched, monkeypatch, tmp_path):
    om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    monkeypatch.setattr(om, "_fetch_daily_range", lambda lat, lon, v, m, s, e: _series(s, e, value=7.0))
    monkeypatch.setattr(om, "_today", lambda: TODAY + pd.Timedelta("1D"))
    df = om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    assert df["date"].is_unique and df["value"].iloc[-1] == 7.0 and df["value"].iloc[0] == 2.0


def test_same_day_repeat_makes_no_call(fetched):
    om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    n = len(fetched)
    om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    # Still a NaN tail, but the store was refreshed today: nothing new to ask for.
    assert len(fetched) == n


def test_places_models_and_variables_have_their_own_files(fetched, tmp_path):
    om.history_series(48.85, 2.35, "temperature_2m_mean", "era5_seamless")
    om.history_series(48.86, 2.35, "temperature_2m_mean", "era5_seamless")
    om.history_series(48.85, 2.35, "precipitation_sum", "era5")
    assert len(list(tmp_path.glob("*.parquet"))) == 3


def test_weighted_call_count_follows_open_meteo_rules():
    assert om.weighted_calls(days=14, n_vars=1) == pytest.approx(1.0)
    assert om.weighted_calls(days=7, n_vars=1) == pytest.approx(1.0)
    assert om.weighted_calls(days=28, n_vars=1) == pytest.approx(2.0)
    assert om.weighted_calls(days=14, n_vars=15) == pytest.approx(1.5)
