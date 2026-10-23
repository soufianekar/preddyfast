"""Precomputed preset animations: written as files plus a manifest, served by URL."""

import base64
import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils import precomputed as pc

INIT = pd.Timestamp("2026-10-04 00:00")


def _png_uri(color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _animation(leads=(24, 48, 72)):
    return {
        "panels": [{"store": "ifs_ens", "label": "ECMWF IFS ENS"}],
        "frames": [{"lead_hours": h, "valid": f"T{h}", "srcs": [_png_uri()]} for h in leads],
        "errors": [],
        "caption": "3 frames",
        "filename": "temperature_2m-mean-2026100400",
    }


@pytest.fixture(autouse=True)
def _dir(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "PRECOMPUTE_DIR", str(tmp_path))
    # The job saves each preset's fields from the cube it just rendered.
    cube = {"lat": np.array([45.0, 45.25]), "lon": np.array([5.0, 5.25]), "lead_hours": [24, 48, 72],
            "valid_time": [INIT + pd.Timedelta(hours=h) for h in (24, 48, 72)], "unit": "°C"}
    monkeypatch.setattr(pc, "frame_cube", lambda *a, **k: cube)
    monkeypatch.setattr(pc, "frame_statistic", lambda c, *a: (np.zeros((3, 2, 2), np.float32), "ensemble mean"))
    return tmp_path


def test_frames_written_as_webp_with_relative_paths(_dir):
    entry = pc.write_animation("ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", _animation())
    paths = [f["path"] for f in entry["frames"]]
    assert paths[0] == "frames/ifs_ens/2026100400/ecmwf_europe/temperature_2m/mean/T024.webp"
    with Image.open(_dir / paths[0]) as img:
        assert img.format == "WEBP"
    assert entry["caption"] == "3 frames" and entry["frames"][1]["lead_hours"] == 48


def test_manifest_round_trip_and_current_init(_dir):
    manifest = pc.load_manifest()
    assert manifest == {"stores": {}}
    entry = pc.write_animation("ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", _animation())
    pc.record(manifest, "ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", entry)
    pc.save_manifest(manifest)
    again = pc.load_manifest()
    assert pc.is_current(again, "ifs_ens", INIT) and not pc.is_current(again, "ifs_ens", INIT - pd.Timedelta("12h"))
    assert json.loads((_dir / "latest.json").read_text())["stores"]["ifs_ens"]["init"] == "2026-10-04T00:00"


def test_a_new_run_replaces_the_stores_old_entries(_dir):
    manifest = {"stores": {}}
    old = pc.write_animation("ifs_ens", INIT - pd.Timedelta("1D"), "ecmwf_europe", "temperature_2m", "mean", _animation())
    pc.record(manifest, "ifs_ens", INIT - pd.Timedelta("1D"), "ecmwf_europe", "temperature_2m", "mean", old)
    new = pc.write_animation("ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", _animation())
    pc.record(manifest, "ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", new)
    assert manifest["stores"]["ifs_ens"]["init"] == "2026-10-04T00:00"
    assert "2026100300" not in json.dumps(manifest)


def _served(monkeypatch, init=INIT):
    manifest = {"stores": {}}
    entry = pc.write_animation("ifs_ens", init, "ecmwf_europe", "temperature_2m", "mean", _animation((24, 48, 72, 96)))
    pc.record(manifest, "ifs_ens", init, "ecmwf_europe", "temperature_2m", "mean", entry)
    pc.save_manifest(manifest)
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT)


ARGS = dict(area_key="ecmwf_europe", variable="temperature_2m", run=None, step_hours=24, horizon_hours=72,
            accum_hours=None, statistic="mean", projection="auto", extent_key="data", palette="default", stores=("ifs_ens",))


def test_preset_served_by_url_and_cut_at_the_horizon(monkeypatch):
    _served(monkeypatch)
    out = pc.preset_animation(**ARGS, base_url="/climate/precomputed/")
    assert [f["lead_hours"] for f in out["frames"]] == [24, 48, 72]
    assert out["frames"][0]["srcs"] == ["/climate/precomputed/frames/ifs_ens/2026100400/ecmwf_europe/temperature_2m/mean/T024.webp"]
    assert out["panels"][0]["store"] == "ifs_ens" and out["precomputed"] is True


@pytest.mark.parametrize("change", [
    {"area_key": "box:45,50,5,10"}, {"palette": "viridis"}, {"step_hours": 12}, {"projection": "lambert"},
    {"statistic": "percentile"}, {"stores": ("ifs_ens", "gefs")}, {"run": "2026-10-03T00:00"},
])
def test_anything_else_is_rendered_live(monkeypatch, change):
    _served(monkeypatch)
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT if run in (None, "latest") else pd.Timestamp(run))
    assert pc.preset_animation(**{**ARGS, **change}, base_url="/") is None


def test_stale_preset_is_not_served(monkeypatch):
    _served(monkeypatch, init=INIT - pd.Timedelta("1D"))
    assert pc.preset_animation(**ARGS, base_url="/") is None


def test_refresh_renders_each_preset_once_per_new_run(monkeypatch, _dir):
    calls = []

    def fake_animation(area_key, variable, init_time, step, horizon, accum, statistic, pct, pct_range,
                       projection, extent, palette, stores, theme, budget_mb=None):
        calls.append((area_key, variable, statistic, accum, stores, theme, palette, step, horizon))
        return _animation()

    monkeypatch.setattr(pc, "forecast_animation", fake_animation)
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT)
    monkeypatch.setattr(pc, "store_variables", lambda store: ["temperature_2m", "precipitation_surface"])
    monkeypatch.setattr(pc, "PRECOMPUTE_REGIONS", ("ecmwf_europe",))
    monkeypatch.setattr(pc, "PRECOMPUTE_VARIABLES", {"temperature_2m": None, "precipitation_surface": 24, "wind_gust_10m": None})
    monkeypatch.setattr(pc, "PRECOMPUTE_STATISTICS", {"precipitation_surface": ("mean", "range")})
    manifest = pc.load_manifest()
    done = pc.refresh_store("ifs_ens", manifest)
    # T2m mean, precip mean + range; gust is not in the store and is skipped.
    assert sorted((c[1], c[2]) for c in calls) == [("precipitation_surface", "mean"), ("precipitation_surface", "range"),
                                                   ("temperature_2m", "mean")]
    assert all(c[5] == "dark" and c[6] == "default" and c[7] == 24 for c in calls)
    assert done == 3 and pc.load_manifest()["stores"]["ifs_ens"]["products"]["ecmwf_europe/precipitation_surface/mean"]["accum_hours"] == 24
    assert pc.refresh_store("ifs_ens", pc.load_manifest()) == 0  # same run: nothing to do


def test_deterministic_store_draws_its_field_only(monkeypatch, _dir):
    stats = []
    monkeypatch.setattr(pc, "forecast_animation", lambda *a, **k: stats.append(a[6]) or _animation())
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT)
    monkeypatch.setattr(pc, "store_variables", lambda store: ["precipitation_surface"])
    monkeypatch.setattr(pc, "PRECOMPUTE_REGIONS", ("ecmwf_europe",))
    monkeypatch.setattr(pc, "PRECOMPUTE_VARIABLES", {"precipitation_surface": 24})
    monkeypatch.setattr(pc, "PRECOMPUTE_STATISTICS", {"precipitation_surface": ("mean", "range")})
    pc.refresh_store("aifs_single", pc.load_manifest())
    assert stats == ["mean"]


def test_upload_sends_frames_then_the_manifest_last(_dir):
    entry = pc.write_animation("ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", _animation((24,)))
    manifest = {"stores": {}}
    pc.record(manifest, "ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", entry)
    pc.save_manifest(manifest)
    sent = []

    class Client:
        def upload_file(self, filename, bucket, key, ExtraArgs=None):
            sent.append((key, ExtraArgs["ContentType"]))

    pc.upload(Client(), "bucket")
    assert sent[-1] == ("latest.json", "application/json")
    assert ("frames/ifs_ens/2026100400/ecmwf_europe/temperature_2m/mean/T024.webp", "image/webp") in sent


def test_control_renders_right_after_its_source_so_the_cube_is_shared(monkeypatch, _dir):
    order = []
    monkeypatch.setattr(pc, "forecast_animation", lambda *a, **k: order.append((a[0], a[1], a[12][0])) or _animation())
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT)
    monkeypatch.setattr(pc, "store_variables", lambda store: ["temperature_2m", "pressure_reduced_to_mean_sea_level"])
    monkeypatch.setattr(pc, "PRECOMPUTE_REGIONS", ("ecmwf_europe", "ecmwf_n_atlantic"))
    monkeypatch.setattr(pc, "PRECOMPUTE_VARIABLES", {"temperature_2m": None, "pressure_reduced_to_mean_sea_level": None})
    pc.refresh(("ifs_ens", "gefs", "ifs_control"), pc.load_manifest())
    for i, (region, variable, store) in enumerate(order):
        if store == "ifs_control":
            # Only other stores' renders between the source's and the control's for the same field.
            src = max(j for j, o in enumerate(order[:i]) if o == (region, variable, "ifs_ens"))
            assert i - src <= 2


def test_estimate_counts_a_view_as_free_when_its_source_is_read(monkeypatch):
    monkeypatch.setattr(pc, "PRECOMPUTE_REGIONS", ("ecmwf_europe",))
    monkeypatch.setattr(pc, "area_bounds", lambda region: (35.0, 75.0, -12.5, 42.5))
    monkeypatch.setattr(pc, "read_estimate", lambda stores, bounds, horizon_hours=None: (10, 800.0))
    rows = dict(((s, r), mb) for s, r, mb in pc.estimate(("ifs_ens", "ifs_control")))
    assert rows[("ifs_ens", "ecmwf_europe")] == 800.0 and rows[("ifs_control", "ecmwf_europe")] == 0.0
    assert dict(((s, r), mb) for s, r, mb in pc.estimate(("ifs_control",)))[("ifs_control", "ecmwf_europe")] == 800.0


def _fields(lat, lon, leads):
    values = np.arange(len(leads) * lat.size * lon.size, dtype=np.float32).reshape(len(leads), lat.size, lon.size)
    return values


def test_preset_fields_round_trip_and_cover_a_sub_area(_dir, monkeypatch):
    lat = np.arange(35.0, 72.25, 0.25)
    lon = np.arange(-25.0, 45.25, 0.25)
    leads = [24, 48, 72]
    values = _fields(lat, lon, leads)
    manifest = {"stores": {}}
    entry = pc.write_animation("ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", _animation(), None)
    entry["fields"] = pc.write_fields("ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", values, lat, lon,
                                      leads, [INIT + pd.Timedelta(hours=h) for h in leads], "°C", "ensemble mean of 51 members")
    pc.record(manifest, "ifs_ens", INIT, "ecmwf_europe", "temperature_2m", "mean", entry)
    pc.save_manifest(manifest)
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT)
    cube = pc.fields_covering((45.0, 50.0, 5.0, 10.0), "temperature_2m", None, 24, 72, None, "mean", "ifs_ens")
    assert cube["cube"].shape == (3, 1, 21, 21) and cube["description"] == "ensemble mean of 51 members"
    assert float(cube["lat"].min()) == 45.0 and float(cube["lon"].max()) == 10.0 and cube["lead_hours"] == leads
    # Values are the preset's, cut out, not re-read.
    i, j = int((45.0 - 35.0) / 0.25), int((5.0 - -25.0) / 0.25)
    assert cube["cube"][0, 0, 0, 0] == pytest.approx(float(values[0, i, j]), rel=1e-3)


@pytest.mark.parametrize("change", [
    dict(bounds=(30.0, 50.0, 5.0, 10.0)),     # leaves the preset region
    dict(statistic="percentile"),              # a statistic not precomputed
    dict(step=12), dict(accum=24),
])
def test_anything_the_preset_does_not_hold_reads_the_archive(_dir, monkeypatch, change):
    test_preset_fields_round_trip_and_cover_a_sub_area(_dir, monkeypatch)
    args = dict(bounds=(45.0, 50.0, 5.0, 10.0), variable="temperature_2m", init=None, step=24, horizon=72,
                accum=None, statistic="mean", store="ifs_ens")
    args.update(change)
    assert pc.fields_covering(args["bounds"], args["variable"], args["init"], args["step"], args["horizon"],
                              args["accum"], args["statistic"], args["store"]) is None


def _m(**stores):
    return {
        "stores": {
            s: {"init": i, "products": {f"r/v/{s}": {"frames": []}}}
            for s, i in stores.items()
        }
    }


def test_merge_part_replaces_its_stores():
    out = pc.merge_manifest(
        _m(ifs_ens="2026-10-21T00:00"), _m(ifs_ens="2026-10-22T00:00")
    )
    assert out["stores"]["ifs_ens"]["init"] == "2026-10-22T00:00"


def test_merge_keeps_stores_missing_from_part():
    out = pc.merge_manifest(
        _m(ifs_ens="2026-10-21T00:00", gefs="2026-10-21T00:00"),
        _m(gefs="2026-10-22T00:00"),
    )
    assert out["stores"]["ifs_ens"]["init"] == "2026-10-21T00:00"


def test_merge_ignores_older_init():
    out = pc.merge_manifest(_m(gefs="2026-10-22T00:00"), _m(gefs="2026-10-21T00:00"))
    assert out["stores"]["gefs"]["init"] == "2026-10-22T00:00"


def test_merge_newer_init_wins():
    base = pc.merge_manifest({"stores": {}}, _m(aifs_ens="2026-10-21T00:00"))
    out = pc.merge_manifest(base, _m(aifs_ens="2026-10-21T12:00"))
    assert out["stores"]["aifs_ens"]["init"] == "2026-10-21T12:00"


def test_merge_does_not_mutate_inputs():
    base, part = _m(gefs="2026-10-21T00:00"), _m(gefs="2026-10-22T00:00")
    pc.merge_manifest(base, part)
    assert base["stores"]["gefs"]["init"] == "2026-10-21T00:00"


def test_refresh_reads_under_the_batch_budget_not_the_interactive_one(monkeypatch, _dir):
    budgets = []
    monkeypatch.setattr(pc, "forecast_animation", lambda *a, **k: budgets.append(k.get("budget_mb")) or _animation())
    monkeypatch.setattr(pc, "current_init", lambda stores, run: INIT)
    monkeypatch.setattr(pc, "store_variables", lambda store: ["temperature_2m"])
    monkeypatch.setattr(pc, "PRECOMPUTE_REGIONS", ("ecmwf_n_atlantic",))
    monkeypatch.setattr(pc, "PRECOMPUTE_VARIABLES", {"temperature_2m": None})
    pc.refresh_store("ifs_ens", pc.load_manifest())
    # IFS ENS over the N Atlantic to T+360 is ~960 MB; the interactive limit is 400.
    assert budgets == [pc.PRECOMPUTE_READ_BUDGET_MB] and pc.PRECOMPUTE_READ_BUDGET_MB >= 1000
