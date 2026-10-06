from pathlib import Path

import pandas as pd
import pytest

from utils import observations as obs

DATA = Path(__file__).parent / "data"


@pytest.fixture
def stations(monkeypatch):
    monkeypatch.setattr(obs, "_cached_download", lambda url, path, max_age_h: DATA / "ghcnh_stations_sample.csv")
    return obs.station_list()


def test_station_list_columns(stations):
    row = stations.set_index("id").loc["FRM00007156"]
    assert row["name"] == "PARIS-MONTSOURIS" and row["elev_m"] == 77.0
    assert row["lat"] == pytest.approx(48.8167)


def test_nearest_stations_ordered_by_distance(stations):
    near = obs.nearest_stations(48.85, 2.35, n=3, max_km=50)
    assert near["id"].iloc[0] == "FRM00007156"
    assert near["dist_km"].is_monotonic_increasing
    assert (near["dist_km"] < 50).all()


def test_nearest_stations_respects_max_distance(stations):
    assert obs.nearest_stations(0.0, 0.0, n=3, max_km=50).empty


@pytest.fixture
def paris(monkeypatch):
    monkeypatch.setattr(obs, "_cached_download", lambda url, path, max_age_h: DATA / "ghcnh_sample.psv")
    return obs.fetch_station_year("FRM00007156", 2026)


def test_units_and_timestamps(paris):
    assert str(paris["valid_time"].dt.tz) == "UTC"
    t = paris.dropna(subset=["temperature_2m"])
    assert t["temperature_2m"].between(-60, 60).all()
    assert paris["pressure_reduced_to_mean_sea_level"].dropna().between(900, 1100).all()


def test_quality_flagged_value_is_dropped(paris):
    flagged = pd.Timestamp("2026-06-16T01:00:00", tz="UTC")
    assert pd.isna(paris.set_index("valid_time").loc[flagged, "temperature_2m"])


def test_trace_precipitation_is_zero(paris):
    p = paris["precipitation_1h"].dropna()
    assert (p >= 0).all() and (p == 0).any()


def test_missing_station_raises_observation_error(monkeypatch):
    def gone(url, path, max_age_h):
        raise obs.ObservationError("GHCNh has no 2026 file for station XXX.")

    monkeypatch.setattr(obs, "_cached_download", gone)
    with pytest.raises(obs.ObservationError):
        obs.fetch_station_year("XXX", 2026)


def test_available_stations_skips_those_without_the_year(stations, monkeypatch):
    monkeypatch.setattr(obs, "_year_exists", lambda station_id, year: station_id.startswith("FRM"))
    got = obs.available_stations(48.85, 2.35, 2026, n=3, max_km=50)
    assert got["id"].tolist() == ["FRM00007156"]


def _psv(tmp_path, rows):
    cols = ["DATE", "temperature", "temperature_Quality_Code", "temperature_Measurement_Code", "temperature_Source_Code"]
    lines = ["|".join(cols)] + ["|".join(r) for r in rows]
    path = tmp_path / "s.psv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_qc_keeps_passing_legacy_codes_and_drops_failing_ones(tmp_path, monkeypatch):
    rows = [
        ["2026-06-01T00:00:00", "10.0", "1", "", "313"],   # passed all QC -> keep
        ["2026-06-01T01:00:00", "11.0", "5", "", "313"],   # passed, NCEI source -> keep
        ["2026-06-01T02:00:00", "12.0", "2", "", "313"],   # suspect -> drop
        ["2026-06-01T03:00:00", "13.0", "7", "", "313"],   # erroneous -> drop
        ["2026-06-01T04:00:00", "14.0", "5", "", "220"],   # "removed" in this source -> drop
        ["2026-06-01T05:00:00", "15.0", "s", "", "313"],   # GHCNh spike check -> drop
        ["2026-06-01T06:00:00", "16.0", "", "", "313"],    # unflagged -> keep
    ]
    path = _psv(tmp_path, rows)
    monkeypatch.setattr(obs, "_cached_download", lambda url, p, max_age_h: path)
    got = obs.fetch_station_year("X", 2026)["temperature_2m"].tolist()
    assert got[:2] == [10.0, 11.0] and got[6] == 16.0
    assert all(pd.isna(v) for v in got[2:6])


def test_last_obs_time_is_the_newest_report_of_that_variable(monkeypatch):
    t = pd.date_range("2026-09-28", periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame({"valid_time": t, "temperature_2m": [1, 2, 3, 4, None, None],
                          "precipitation_1h": [0.0, None, None, None, None, None]})
    monkeypatch.setattr(obs, "fetch_station_year", lambda sid, year: frame)
    assert obs.last_obs_time("X", "temperature_2m") == pd.Timestamp("2026-09-28 03:00")
    assert obs.last_obs_time("X", "precipitation_1h") == pd.Timestamp("2026-09-28 00:00")


def test_last_obs_time_falls_back_to_last_year_early_in_january(monkeypatch):
    empty = pd.DataFrame({"valid_time": pd.to_datetime([], utc=True), "temperature_2m": []})
    last = pd.DataFrame({"valid_time": [pd.Timestamp("2025-12-31 23:00", tz="UTC")], "temperature_2m": [2.0]})
    now = pd.Timestamp.now(tz="UTC").year

    def fetch(sid, year):
        if year == now:
            raise obs.ObservationError("no file yet")
        return last if year == now - 1 else empty

    monkeypatch.setattr(obs, "fetch_station_year", fetch)
    assert obs.last_obs_time("X", "temperature_2m") == pd.Timestamp("2025-12-31 23:00")


def test_last_obs_time_none_when_never_reported(monkeypatch):
    frame = pd.DataFrame({"valid_time": pd.to_datetime(["2026-09-28"], utc=True), "temperature_2m": [None]})
    monkeypatch.setattr(obs, "fetch_station_year", lambda sid, year: frame)
    assert obs.last_obs_time("X", "temperature_2m") is None
