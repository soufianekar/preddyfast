"""Station network selection: synthetic observations, no network."""

import numpy as np
import pandas as pd
import pytest

from utils import observations as obs
from utils import station_network as sn
from utils.settings import SCORECARD_COUNTRY, SCORECARD_VARIABLES

START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-01-10 18:00", tz="UTC")  # 10 days: 40 synoptic hours


@pytest.fixture(autouse=True)
def general_mode(monkeypatch):
    """Existing tests exercise the country + thinning path; fixed-list tests opt in."""
    monkeypatch.setattr(sn, "SCORECARD_STATIONS", ())


def _frame(times, **cols):
    n = len(times)
    out = pd.DataFrame({"valid_time": pd.DatetimeIndex(times, tz="UTC")})
    for name in ("temperature_2m", "wind_speed_10m", "precipitation_1h", "precip_total_6h"):
        out[name] = cols.get(name, [np.nan] * n)
        out[f"{name}_report_type"] = cols.get(f"{name}_report_type", ["FM-15"] * n)
    return out


def _synoptic(start=START, end=END):
    t = pd.date_range(start, end, freq="h")
    return t[t.hour % 6 == 0]


def test_completeness_counts_only_hours_within_15_min():
    hrs = _synoptic()
    # first 20 exactly on the hour, next 10 at +10 min, last 10 at +20 min (outside)
    times = list(hrs[:20]) + list(hrs[20:30] + pd.Timedelta("10min")) + list(
        hrs[30:] + pd.Timedelta("20min")
    )
    f = _frame(times, temperature_2m=[10.0] * 40)
    assert sn.completeness(f, "temperature_2m", START, END) == pytest.approx(30 / 40)


def test_completeness_ignores_qc_masked_values():
    hrs = _synoptic()
    t = [10.0] * 36 + [np.nan] * 4
    f = _frame(hrs, temperature_2m=t)
    assert sn.completeness(f, "temperature_2m", START, END) == pytest.approx(36 / 40)


def test_missing_15_percent_fails_temperature_threshold():
    hrs = _synoptic()
    t = [10.0] * 34 + [np.nan] * 6  # 85 %
    f = _frame(hrs, temperature_2m=t)
    c = sn.completeness(f, "temperature_2m", START, END)
    assert c == pytest.approx(0.85)
    assert c < SCORECARD_VARIABLES["temperature_2m"]["completeness"]


def test_duplicate_synop_and_metar_same_hour_counted_once_with_synop_value():
    t0 = pd.Timestamp("2026-01-05 12:00", tz="UTC")
    f = _frame(
        [t0 - pd.Timedelta("5min"), t0],
        temperature_2m=[1.0, 2.0],
        temperature_2m_report_type=["FM-15", "FM-12"],
    )
    got = sn.dedupe_hourly(f)
    row = got[got["valid_time"] == t0]
    assert len(row) == 1 and row["temperature_2m"].iloc[0] == 2.0
    # a nearer METAR does not beat an eligible SYNOP
    f = _frame(
        [t0, t0 + pd.Timedelta("10min")],
        temperature_2m=[1.0, 2.0],
        temperature_2m_report_type=["FM-15", "FM-12"],
    )
    got = sn.dedupe_hourly(f)
    assert got.loc[got["valid_time"] == t0, "temperature_2m"].iloc[0] == 2.0
    assert got["temperature_2m"].notna().sum() == 1


def test_report_at_1220_not_counted_for_12utc():
    f = _frame([pd.Timestamp("2026-01-05 12:20", tz="UTC")], temperature_2m=[3.0])
    assert sn.dedupe_hourly(f)["temperature_2m"].notna().sum() == 0


def test_utc_report_not_shifted_by_dst():
    # 2026-07-01 12:00 UTC is 14:00 in Berlin; it must stay the 12 UTC hour.
    t = pd.Timestamp("2026-07-01 12:00", tz="UTC")
    got = sn.dedupe_hourly(_frame([t], temperature_2m=[20.0]))
    assert got["valid_time"].iloc[0] == t
    assert str(got["valid_time"].dt.tz) == "UTC"
    c = sn.completeness(
        _frame([t], temperature_2m=[20.0]),
        "temperature_2m",
        t,
        t,
    )
    assert c == 1.0


def test_precip_window_from_reported_total_or_six_hourly_values():
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-01-02 18:00", tz="UTC")  # 8 windows ending 00..18
    ends = list(_synoptic(start, end))
    # windows 1-2 from reported totals
    rep = _frame(ends[:2], precip_total_6h=[0.5, 0.0])
    # window 3 (ending ends[2]): six complete hourly values
    h3 = pd.date_range(ends[2] - pd.Timedelta("5h"), ends[2], freq="h")
    hourly = _frame(h3, precipitation_1h=[0.1] * 6)
    # window 4: only five hourly values -> incomplete
    h4 = pd.date_range(ends[3] - pd.Timedelta("4h"), ends[3], freq="h")
    part = _frame(h4, precipitation_1h=[0.1] * 5)
    f = pd.concat([rep, hourly, part], ignore_index=True)
    assert sn.completeness(f, "precipitation_6h", start, end) == pytest.approx(3 / 8)


def test_thin_by_distance_exact_n_keeps_farthest_pair_and_spreads():
    rng = np.random.default_rng(0)
    lat = np.r_[rng.uniform(50.0, 50.5, 60), 47.5, 55.0]  # dense cluster + extremes
    lon = np.r_[rng.uniform(7.0, 7.5, 60), 9.0, 9.0]
    st = pd.DataFrame({"id": [f"S{i}" for i in range(62)], "lat": lat, "lon": lon})
    out = sn.thin_by_distance(st, 10)
    assert len(out) == 10
    assert {"S60", "S61"} <= set(out["id"])  # farthest pair

    def min_pair(df):
        d = sn.haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy())
        return d[np.triu_indices(len(df), 1)].min()

    base = st.iloc[rng.choice(62, 10, replace=False)]
    assert min_pair(out) >= min_pair(base)


def test_thin_by_distance_small_input_returned_whole():
    st = pd.DataFrame({"id": ["a", "b"], "lat": [50.0, 51.0], "lon": [8.0, 9.0]})
    assert len(sn.thin_by_distance(st, 5)) == 2


def _stations(n, country=SCORECARD_COUNTRY):
    return pd.DataFrame(
        {
            "id": [f"G{i:03d}" for i in range(n)],
            "name": [f"S{i}" for i in range(n)],
            "lat": 48.0 + np.arange(n) * 0.1,
            "lon": 8.0 + np.arange(n) * 0.1,
            "elev_m": 100.0,
            "country": country,
        }
    )


def _full(start, end):
    t = pd.date_range(start.floor("h"), end, freq="h")
    t = t[t.hour % 6 == 0]
    return _frame(
        t,
        temperature_2m=[5.0] * len(t),
        wind_speed_10m=[3.0] * len(t),
        precip_total_6h=[0.0] * len(t),
    )


def test_year_boundary_window_reads_both_years(monkeypatch):
    calls = []

    def fake(sid, year):
        calls.append(year)
        t = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="6h")
        return _frame(t, temperature_2m=[1.0] * len(t))

    monkeypatch.setattr(obs, "fetch_station_year", fake)
    s = pd.Timestamp("2025-12-20", tz="UTC")
    e = pd.Timestamp("2026-01-10", tz="UTC")
    out = sn.network_obs(["A"], s, e)["A"]
    assert sorted(set(calls)) == [2025, 2026]
    assert out["valid_time"].min() >= s and out["valid_time"].max() <= e
    years = set(out["valid_time"].dt.year)
    assert years == {2025, 2026}
    assert {"valid_time", "temperature_2m", "wind_speed_10m", "precipitation_1h",
            "precip_total_6h"} <= set(out.columns)


def test_select_network_caps_downloads_and_thins(monkeypatch):
    monkeypatch.setattr(sn, "SCORECARD_MAX_STATIONS", 5)
    st = pd.concat([_stations(40), _stations(5, "FR").assign(id=lambda d: "F" + d["id"])])
    monkeypatch.setattr(obs, "station_list", lambda: st)
    monkeypatch.setattr(obs, "_year_exists", lambda sid, y: sid != "G001")
    fetched = []

    def fake(sid, year):
        fetched.append(sid)
        if sid == fetched[0]:  # the first screened candidate is sparse
            return _frame([pd.Timestamp("2026-01-02", tz="UTC")], temperature_2m=[1.0])
        return _full(START, END)

    monkeypatch.setattr(obs, "fetch_station_year", fake)
    net = sn.select_network(START, END)
    assert len(set(fetched)) == 10
    assert not any(f.startswith("F") for f in fetched) and "G001" not in fetched
    assert fetched[0] not in set(net["id"])
    assert len(net) == 5
    assert list(net.columns) == [
        "id", "name", "lat", "lon", "elev_m", "ok_temperature", "ok_precip", "ok_wind",
    ]
    assert net["ok_temperature"].all() and net["ok_wind"].all() and net["ok_precip"].all()


def test_select_network_marks_per_variable_ok(monkeypatch):
    st = _stations(3)
    monkeypatch.setattr(obs, "station_list", lambda: st)
    monkeypatch.setattr(obs, "_year_exists", lambda sid, y: True)

    def fake(sid, year):
        f = _full(START, END)
        if sid == "G001":
            f["wind_speed_10m"] = np.nan
        if sid == "G002":
            f["precip_total_6h"] = np.nan
        return f

    monkeypatch.setattr(obs, "fetch_station_year", fake)
    net = sn.select_network(START, END).set_index("id")
    assert net.loc["G000", ["ok_temperature", "ok_precip", "ok_wind"]].all()
    assert not net.loc["G001", "ok_wind"] and net.loc["G001", "ok_precip"]
    assert not net.loc["G002", "ok_precip"] and net.loc["G002", "ok_wind"]


def test_save_and_load_network_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(sn, "SCORECARD_DIR", str(tmp_path))
    assert sn.load_network() is None
    net = _stations(2).drop(columns="country").assign(
        ok_temperature=True, ok_precip=False, ok_wind=True
    )
    sn.save_network(net, START, END)
    got, s, e = sn.load_network()
    assert s == START and e == END
    pd.testing.assert_frame_equal(got.reset_index(drop=True), net.reset_index(drop=True))


def test_empty_network_keeps_its_window(monkeypatch, tmp_path):
    monkeypatch.setattr(sn, "SCORECARD_DIR", str(tmp_path))
    empty = _stations(2).drop(columns="country").assign(
        ok_temperature=True, ok_precip=False, ok_wind=True
    ).iloc[0:0]
    sn.save_network(empty, START, END)
    got, s, e = sn.load_network()
    assert got.empty and s == START and e == END


def _precip_hourly(report_times):
    return _frame(report_times, precipitation_1h=[0.2] * len(report_times))


def test_hourly_precip_report_within_15_min_closes_the_nearest_hour():
    start = pd.Timestamp("2026-01-01 06:00", tz="UTC")
    end = pd.Timestamp("2026-01-01 06:00", tz="UTC")
    # hours 01..06 reported at :50 (->next hour), :00, :10 - all within 15 min
    off = ["00:50", "02:00", "02:50", "04:10", "05:00", "05:50"]
    t = [pd.Timestamp(f"2026-01-01 {x}", tz="UTC") for x in off]
    assert sn.completeness(_precip_hourly(t), "precipitation_6h", start, end) == 1.0


def test_hourly_precip_report_30_min_off_does_not_count():
    start = pd.Timestamp("2026-01-01 06:00", tz="UTC")
    end = pd.Timestamp("2026-01-01 06:00", tz="UTC")
    off = ["00:50", "02:00", "02:50", "03:30", "05:00", "05:50"]
    t = [pd.Timestamp(f"2026-01-01 {x}", tz="UTC") for x in off]
    assert sn.completeness(_precip_hourly(t), "precipitation_6h", start, end) == 0.0


def test_missing_station_year_skipped_but_unreachable_source_raises(monkeypatch):
    def fake(sid, year):
        if year == 2025:
            raise obs.ObservationNotFound("no file")
        return _full(START, END)

    monkeypatch.setattr(obs, "fetch_station_year", fake)
    s = pd.Timestamp("2025-12-20", tz="UTC")
    assert not sn.network_obs(["A"], s, END)["A"].empty

    def down(sid, year):
        raise obs.ObservationError("GHCNh is unreachable")

    monkeypatch.setattr(obs, "fetch_station_year", down)
    with pytest.raises(obs.ObservationError):
        sn.network_obs(["A"], s, END)
    monkeypatch.setattr(obs, "station_list", lambda: _stations(3))
    monkeypatch.setattr(obs, "_year_exists", lambda sid, y: True)
    with pytest.raises(obs.ObservationError):
        sn.select_network(START, END)


def test_early_january_accepts_previous_year_file(monkeypatch):
    s = pd.Timestamp("2025-10-10", tz="UTC")
    e = pd.Timestamp("2026-01-08", tz="UTC")
    monkeypatch.setattr(obs, "station_list", lambda: _stations(2))
    monkeypatch.setattr(obs, "_year_exists", lambda sid, y: y == 2025)
    monkeypatch.setattr(obs, "fetch_station_year", lambda sid, y: _full(s, e))
    assert len(sn.select_network(s, e)) == 2


# --- fixed station list ---------------------------------------------------------

FIXED = ("EGLL", "LFPG", "EDDB")


def _fixed_world(monkeypatch, frames):
    """station_list holds the fixed ids plus a German decoy; `frames[sid]` is the file."""
    st = pd.concat([
        pd.DataFrame({
            "id": list(FIXED), "name": ["London", "Paris", "Berlin"],
            "lat": [51.5, 49.0, 52.4], "lon": [-0.5, 2.5, 13.5], "elev_m": 25.0,
            "country": ["GB", "FR", "DE"],
        }),
        _stations(3),
    ])
    monkeypatch.setattr(sn, "SCORECARD_STATIONS", FIXED)
    monkeypatch.setattr(obs, "station_list", lambda: st)
    monkeypatch.setattr(obs, "_year_exists", lambda sid, y: pytest.fail("no year probe"))
    fetched = []

    def fake(sid, year):
        fetched.append(sid)
        return frames[sid]

    monkeypatch.setattr(obs, "fetch_station_year", fake)

    def no_thin(*a, **k):
        raise AssertionError("fixed list must not be thinned")

    monkeypatch.setattr(sn, "thin_by_distance", no_thin)
    return fetched


def test_fixed_list_is_exactly_those_stations_in_order(monkeypatch):
    full = _full(START, END)
    fetched = _fixed_world(monkeypatch, {s: full for s in FIXED})
    net = sn.select_network(START, END)
    assert list(net["id"]) == list(FIXED)
    assert list(net["name"]) == ["London", "Paris", "Berlin"]  # not filtered by country
    assert set(fetched) == set(FIXED)  # the German decoys were never read


def test_fixed_station_failing_temperature_is_logged_and_excluded(monkeypatch, caplog):
    full = _full(START, END)
    bad = full.copy()
    bad["temperature_2m"] = np.nan
    _fixed_world(monkeypatch, {"EGLL": full, "LFPG": bad, "EDDB": full})
    with caplog.at_level("WARNING", logger="utils.station_network"):
        net = sn.select_network(START, END)
    assert list(net["id"]) == ["EGLL", "EDDB"]
    assert any("LFPG" in r.message and "temperature" in r.message for r in caplog.records)


def test_precip_threshold_is_looser_for_the_fixed_list_only(monkeypatch):
    hrs = _synoptic()
    keep = 22  # 22 of 40 windows = 0.55
    f = _full(START, END)
    f["precip_total_6h"] = [0.0] * keep + [np.nan] * (len(f) - keep)
    assert sn.completeness(f, "precipitation_6h", START, END) == pytest.approx(0.55)
    assert len(hrs) == 40
    _fixed_world(monkeypatch, {s: f for s in FIXED})
    assert sn.select_network(START, END)["ok_precip"].all()

    monkeypatch.setattr(sn, "SCORECARD_STATIONS", ())
    monkeypatch.setattr(sn, "thin_by_distance", lambda df, n: df.reset_index(drop=True))
    monkeypatch.setattr(obs, "station_list", lambda: _stations(2))
    monkeypatch.setattr(obs, "_year_exists", lambda sid, y: True)
    monkeypatch.setattr(obs, "fetch_station_year", lambda sid, y: f)
    net = sn.select_network(START, END)
    assert len(net) == 2 and not net["ok_precip"].any()


def test_save_network_records_the_configured_list(monkeypatch, tmp_path):
    monkeypatch.setattr(sn, "SCORECARD_DIR", str(tmp_path))
    assert sn.load_configured() is None
    net = _stations(1).drop(columns="country").assign(
        ok_temperature=True, ok_precip=True, ok_wind=True
    )
    sn.save_network(net, START, END, ("B", "A"))
    assert sn.load_configured() == ("A", "B")
    sn.save_network(net, START, END)
    assert sn.load_configured() == ()
