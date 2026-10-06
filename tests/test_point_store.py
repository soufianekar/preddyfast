import pandas as pd

from utils import earthmover_api as em


def test_point_extract_read_once_then_from_disk(tmp_path, monkeypatch):
    calls = []

    def reader(lat, lon, variable, init_time, accum_hours, store):
        calls.append(init_time)
        return pd.DataFrame({"valid_time": [pd.Timestamp("2026-10-01 03:00")], "lead_hours": [3.0],
                             "member": [0], "value": [1.5]}), {"store": store, "n_members": 1}

    monkeypatch.setattr(em, "POINT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(em, "_read_point_run", reader)
    a = em._point_run(48.85, 2.35, "temperature_2m", "2026-10-01T00:00", None, "ifs_ens")
    b = em._point_run(48.85, 2.35, "temperature_2m", "2026-10-01T00:00", None, "ifs_ens")
    assert calls == ["2026-10-01T00:00"]
    pd.testing.assert_frame_equal(a[0], b[0])
    assert b[1]["n_members"] == 1


def test_point_cache_key_separates_store_window_and_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "POINT_CACHE_DIR", str(tmp_path))
    keys = {
        em._point_cache_path(48.85, 2.35, "precipitation_surface", "2026-10-01T00:00", 24, "ifs_ens"),
        em._point_cache_path(48.85, 2.35, "precipitation_surface", "2026-10-01T00:00", 24, "aifs_ens"),
        em._point_cache_path(48.85, 2.35, "precipitation_surface", "2026-10-01T00:00", 6, "ifs_ens"),
        em._point_cache_path(48.20, 2.35, "precipitation_surface", "2026-10-01T00:00", 24, "ifs_ens"),
    }
    assert len(keys) == 4


def test_points_in_the_same_cell_share_an_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "POINT_CACHE_DIR", str(tmp_path))
    assert em._point_cache_path(48.85, 2.35, "t", "2026-10-01T00:00", None, "ifs_ens") == em._point_cache_path(48.80, 2.30, "t", "2026-10-01T00:00", None, "ifs_ens")
