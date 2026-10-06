import json

import pandas as pd
import pytest

from utils import map_gallery as mg
from utils.settings import CDS_DATASETS, MAP_AREA_PRESETS, MAP_GALLERY


@pytest.fixture
def _dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "PRECOMPUTE_DIR", str(tmp_path))
    return tmp_path


def test_month_is_the_one_the_dataset_lag_allows():
    lag = CDS_DATASETS["era5_monthly"]["lag_days"]
    now = pd.Timestamp("2026-10-06")
    month = mg.gallery_month(now)
    assert month == (now - pd.Timedelta(days=lag)).to_period("M").to_timestamp()
    assert month.day == 1


def test_anomaly_job_queues_the_month_and_its_reference_period():
    job, requests = mg.gallery_job("t2m_anomaly_europe", pd.Timestamp("2026-08-01"))
    assert set(requests) == {"field", "reference"}
    field, reference = requests["field"], requests["reference"]
    assert field["year"] == ["2026"] and field["month"] == ["08"]
    ref_years = {int(y) for y in reference["year"]}
    assert ref_years == set(range(1991, 2021))
    assert job["anomaly"] and job["reference"] == [1991, 2020]
    assert job["request_meta"]["month_label"] == "August 2026 monthly mean"
    assert job["request_meta"]["region"] == "ecmwf_europe"


def test_absolute_item_queues_one_request():
    job, requests = mg.gallery_job("t2m_europe", pd.Timestamp("2026-08-01"))
    assert set(requests) == {"field"} and not job["anomaly"] and job["reference"] is None


def test_global_items_use_the_global_preset():
    job, requests = mg.gallery_job("t2m_anomaly_global", pd.Timestamp("2026-08-01"))
    assert job["request_meta"]["region"] == "global"
    assert tuple(MAP_AREA_PRESETS["global"]) == (-90.0, 90.0, -180.0, 180.0)
    # No `area` key (MARS would shave the wrap column) and CDS regrids it.
    assert "area" not in requests["field"] and job["grid"] is not None


def test_precipitation_is_expressed_per_day():
    job, _ = mg.gallery_job("tp_anomaly_europe", pd.Timestamp("2026-08-01"))
    assert job["monthly_mode"] == "per_day"


def test_manifest_round_trip_and_currency(_dir):
    assert mg.load_gallery() == {"items": {}}
    manifest = {"items": {}}
    mg.record(manifest, "t2m_europe", pd.Timestamp("2026-08-01"), "maps/t2m_europe-202608.png", "cap")
    mg.save_gallery(manifest)
    loaded = mg.load_gallery()
    entry = loaded["items"]["t2m_europe"]
    assert entry["month"] == "2026-08" and entry["label"] == MAP_GALLERY["t2m_europe"]["label"]
    assert mg.is_current(loaded, "t2m_europe", pd.Timestamp("2026-08-01"))
    assert not mg.is_current(loaded, "t2m_europe", pd.Timestamp("2026-09-01"))
    assert "generated" in json.loads((_dir / "maps" / "gallery.json").read_text())


def test_gallery_entries_keep_the_settings_order(_dir):
    manifest = {"items": {}}
    for key in reversed(list(MAP_GALLERY)):
        mg.record(manifest, key, pd.Timestamp("2026-08-01"), f"maps/{key}.png", "")
    mg.save_gallery(manifest)
    assert [e["key"] for e in mg.gallery_entries()] == list(MAP_GALLERY)


def test_data_uri_is_decoded_to_png_bytes():
    assert mg._png_bytes("data:image/png;base64,iVBORw0KGgo=") == b"\x89PNG\r\n\x1a\n"
