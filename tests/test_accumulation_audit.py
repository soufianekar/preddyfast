"""Pins for the accumulation audit (docs/accumulation-audit.md)."""

import pandas as pd

from utils import earthmover_api as em


def test_past_run_meteogram_windows_are_disjoint_from_init(monkeypatch):
    init = pd.Timestamp("2026-10-01 12:00")
    rows = [{"valid_time": init + pd.Timedelta(hours=h), "lead_hours": float(h), "member": 0, "value": 1.0}
            for h in (24, 30, 36, 42, 48, 54, 72)]

    monkeypatch.setattr(em, "fetch_point_ensemble", lambda *a: (pd.DataFrame(rows), {"init_time": init}))
    monkeypatch.setattr(em, "store_steps", lambda store: "6 h")
    df, meta_var, _ = em.archive_meteogram(48.85, 2.35, "precipitation_sum", "2026-10-01T12:00", 5, 24, store="ifs_ens")
    # Boxes must not share hours: 24 h totals ending T+24, T+48, T+72 — 12–12 UTC for a 12 UTC run.
    assert list(df["date"].dt.strftime("%d %H")) == ["02 12", "03 12", "04 12"]
    assert "24 h total" in meta_var["label"]


import numpy as np
import pytest
import xarray as xr

from utils.climatology import accumulate, calendar_day, compute_acc_pct


def test_ytd_total_stops_at_the_first_gap():
    out = accumulate(pd.Series([1.0, 2.0, np.nan, 4.0]))
    assert out.iloc[1] == 3.0 and out.iloc[2:].isna().all()


def test_a_year_with_a_gap_drops_out_of_the_envelope_past_it():
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "value": 1.0})
    df.loc[df["date"] == "2022-04-10", "value"] = np.nan
    df["doy"] = calendar_day(df["date"])
    pct = compute_acc_pct(df, 2021, 2023)
    # Day 200: only the complete years count, each 200 mm; the gap year is not 199.
    assert pct.loc[200, ["p10", "p50", "p90"]].tolist() == [200.0, 200.0, 200.0]


def test_forecast_extension_starts_from_the_last_complete_day():
    from utils.figures import make_accumulation_fig

    hist = pd.DataFrame({"date": pd.date_range("2021-01-01", "2023-12-31", freq="D"), "value": 1.0})
    hist["doy"] = calendar_day(hist["date"])
    pct = compute_acc_pct(hist, 2021, 2023)
    days = pd.date_range("2026-01-01", "2026-01-10", freq="D")
    current = pd.DataFrame({"date": days, "value": [1.0] * 7 + [np.nan] * 3})
    current["doy"] = calendar_day(current["date"])
    current["ytd"] = accumulate(current["value"])
    forecast = pd.DataFrame({"date": pd.date_range("2026-01-09", periods=3, freq="D"), "value": 2.0})
    forecast["doy"] = calendar_day(forecast["date"])
    fig = make_accumulation_fig(pct, current, forecast, {"label": "Precipitation", "unit": "mm"}, "ERA5", "1981–2020", 2026)
    ext = next(t for t in fig.data if t.name and "orecast" in t.name)
    y = [v for v in ext.y if v is not None and not np.isnan(v)]
    # Anchored at 7 mm (7 Jan, the last complete day), never NaN, then +2 a day.
    assert y[0] == pytest.approx(7.0) and y[-1] == pytest.approx(13.0)


def test_era5_day_is_0100_through_0000_of_the_next_day():
    from utils.cds_api import _sum_window

    t = pd.date_range("2026-07-01 00:00", "2026-07-02 01:00", freq="h")
    da = xr.DataArray(np.arange(len(t), dtype=float), dims="time", coords={"time": t})
    total = float(_sum_window(da, "time", pd.Timestamp("2026-07-01"), 24))
    assert total == pytest.approx(sum(range(1, 25)))  # hours 01..24 (00:00 next day), not 00..23


def test_era5_window_missing_an_hour_is_refused():
    from utils.cds_api import CDSDataError, _sum_window

    t = pd.date_range("2026-07-01 00:00", "2026-07-01 23:00", freq="h")
    da = xr.DataArray(np.ones(len(t)), dims="time", coords={"time": t})
    with pytest.raises(CDSDataError):
        _sum_window(da, "time", pd.Timestamp("2026-07-01"), 24)


def test_monthly_product_with_the_wrong_step_type_is_refused():
    from utils.cds_api import CDSDataError, _check_step_type

    da = xr.DataArray([1.0], attrs={"GRIB_stepType": "accum"})
    with pytest.raises(CDSDataError, match="avgad"):
        _check_step_type(da, {"expected_step_type": "avgad"}, "tp")


def test_month_total_uses_that_months_calendar_length():
    from utils.cds_api import monthly_rate_factor

    assert monthly_rate_factor("per_month", pd.Timestamp("2024-02-01"))[0] == 29.0
    assert monthly_rate_factor("per_month", pd.Timestamp("2026-02-01"))[0] == 28.0
    assert monthly_rate_factor("per_day", pd.Timestamp("2026-02-01"))[0] == 1.0


def test_window_totals_matches_frame_cube():
    lead_h = np.arange(0.0, 73.0, 6.0)
    rate = np.random.default_rng(0).random((2, lead_h.size, 3, 4)).astype(np.float32)
    totals, keep = em.window_totals(rate, lead_h, 24, lead_axis=1)
    # Hand-built: 6 h steps at a mean rate, 24 h total ending at lead L.
    assert lead_h[keep][0] == 24.0 and totals.shape[1] == len(keep)
    i = int(np.flatnonzero(lead_h == 48)[0])
    expect = rate[:, i - 3 : i + 1].sum(axis=1) * 6.0
    j = list(lead_h[keep]).index(48.0)
    np.testing.assert_allclose(totals[:, j], expect, rtol=1e-5)
    with pytest.raises(em.GridDataError):
        em.window_totals(rate[:, :3], lead_h[:3], 24, lead_axis=1)
