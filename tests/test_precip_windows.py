"""Precipitation verification only where a station can supply the window, aligned to it."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import observations as obs
from utils.verification import common_leads, imerg_window_totals

DATA = Path(__file__).parent / "data"


@pytest.fixture
def koeln(monkeypatch):
    monkeypatch.setattr(obs, "_cached_download", lambda url, path, max_age_h: DATA / "ghcnh_dwd_sample.psv")
    return obs.fetch_station_year("GMI0000EDDK", 2026)


def test_reported_24h_totals_are_read_with_their_end_time(koeln):
    totals = obs.reported_totals(koeln, 24)
    assert list(totals["valid_time"].dt.strftime("%m-%d %H")) == ["09-10 06", "09-11 06", "09-12 06"]
    assert list(totals["value"]) == [0.3, 0.0, 2.0]


def _hourly(start, hours, every=1, value=0.5):
    t = pd.date_range(start, periods=hours // every, freq=f"{every}h", tz="UTC")
    return pd.DataFrame({"valid_time": t, "precipitation_1h": value,
                         "precip_total_6h": np.nan, "precip_total_12h": np.nan, "precip_total_24h": np.nan})


def test_inventory_counts_complete_hourly_windows_by_end_hour():
    frame = _hourly("2026-09-01 01:00", 24 * 5)  # every hour, 5 days
    inv = obs.precip_inventory(frame, 24)
    best = inv[0]
    assert best["source"] == "hourly" and best["count"] >= 4


def test_inventory_prefers_reported_totals_when_hours_have_gaps(koeln):
    inv = obs.precip_inventory(koeln, 24)
    assert inv[0] == {"source": "reported", "hours": 24, "end_hour": 6, "count": 3}


def test_inventory_is_empty_when_nothing_can_be_verified():
    frame = _hourly("2026-09-01 00:00", 24 * 3, every=6)  # 4 reports a day, no totals
    assert obs.precip_inventory(frame, 24) == []


def test_window_end_alignment_picks_leads_ending_at_the_stations_hour():
    a = pd.DataFrame({"lead_hours": [24.0, 30.0, 48.0, 54.0, 78.0]})
    # 00 UTC runs against 06–06 UTC totals: windows end at T+30, T+54, T+78.
    assert common_leads({"a": a}, 24, offset=6) == [30.0, 54.0, 78.0]
    assert common_leads({"a": a}, 24) == [24.0, 48.0]


def test_imerg_half_hours_are_start_labelled_and_windows_need_every_one():
    t = pd.date_range("2026-09-01 00:00", periods=96, freq="30min")
    halfhours = pd.DataFrame({"time": t, "mm": 0.25})
    ends = [pd.Timestamp("2026-09-02 00:00"), pd.Timestamp("2026-09-02 12:00"), pd.Timestamp("2026-09-03 00:30")]
    totals = imerg_window_totals(halfhours, ends, 24)
    # [00:00, 24:00) holds 48 half-hours starting 00:00..23:30 -> 12 mm; the
    # 12:00 window needs half-hours from 12:00 the day before (present) -> 12 mm;
    # the last needs one past the data -> missing, never short.
    assert totals.iloc[0] == pytest.approx(12.0) and totals.iloc[1] == pytest.approx(12.0)
    assert np.isnan(totals.iloc[2])
