"""IFS Control: member 0 of the IFS ENS store, drawn and scored as a deterministic model."""

import numpy as np
import pandas as pd

from utils import earthmover_api as em
from utils.settings import GRID_STORES


def test_control_is_a_deterministic_view_of_ifs_ens_member_0():
    spec = GRID_STORES["ifs_control"]
    assert spec["kind"] == "deterministic" and spec["source_store"] == "ifs_ens" and spec["member"] == 0
    assert spec["res_deg"] == GRID_STORES["ifs_ens"]["res_deg"] and spec["members"] == 1


def _members_frame():
    t = pd.date_range("2026-10-01 06:00", periods=3, freq="6h")
    return pd.DataFrame([{"valid_time": v, "lead_hours": 6.0 * (i + 1), "member": m, "value": 10.0 * m + i}
                         for m in range(3) for i, v in enumerate(t)])


def test_member_subset_keeps_member_0_and_relabels_meta():
    meta = {"n_members": 3, "store": "ifs_ens", "model_label": GRID_STORES["ifs_ens"]["model_label"], "init_time": "x"}
    df, out = em._member_subset(_members_frame(), meta, "ifs_control")
    assert set(df["member"]) == {0} and list(df["value"]) == [0.0, 1.0, 2.0]
    assert out["n_members"] == 1 and out["store"] == "ifs_control"
    assert out["model_label"] == GRID_STORES["ifs_control"]["model_label"] and out["init_time"] == "x"
    assert meta["n_members"] == 3  # the source's cached meta is not mutated


def test_point_run_reads_the_ens_extract_once_for_both(monkeypatch, tmp_path):
    monkeypatch.setattr(em, "POINT_CACHE_DIR", str(tmp_path))
    reads = []

    def fake(lat, lon, variable, init_time, accum_hours, store):
        reads.append(store)
        return _members_frame(), {"n_members": 3, "store": store, "model_label": "m"}

    monkeypatch.setattr(em, "_read_point_run", fake)
    args = (48.85, 2.35, "temperature_2m", "2026-10-01T00:00", None)
    ens, _ = em._point_run(*args, "ifs_ens")
    assert em.point_run_cached(*args, "ifs_control")  # control is free once the ENS run is on disk
    ctrl, meta = em._point_run(*args, "ifs_control")
    assert reads == ["ifs_ens"] and set(ctrl["member"]) == {0} and meta["n_members"] == 1


def test_cube_member_slices_one_member():
    cube = {"cube": np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2), "members": [0, 1, 2], "store": "ifs_ens"}
    out = em._cube_member(cube, 0, "ifs_control")
    assert out["cube"].shape == (2, 1, 2, 2) and out["members"] == [0] and out["store"] == "ifs_control"
    assert np.array_equal(out["cube"][:, 0], cube["cube"][:, 0])


def test_chunk_dims_accepts_a_kept_member_axis_on_a_deterministic_view():
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    chunks = (1, 85, 51, 32, 32)
    assert em._chunk_dims(dims, chunks, deterministic=True)["ensemble_member"] == 51
    # A true deterministic store: encoding has no member axis, the open adds none.
    assert em._chunk_dims(dims[:2] + dims[3:], (1, 85, 32, 32), deterministic=True) == {
        "init_time": 1, "lead_time": 85, "latitude": 32, "longitude": 32}


def test_physical_stores_count_a_shared_source_once():
    assert em.physical_stores(("ifs_ens", "ifs_control", "gefs")) == ("ifs_ens", "gefs")
    assert em.physical_stores(("ifs_control",)) == ("ifs_ens",)


def test_view_uses_its_sources_calendar_and_variables(monkeypatch):
    calls = []
    monkeypatch.setattr(em, "_open_dataset", lambda store: calls.append(store) or (_ for _ in ()).throw(AssertionError(store)))
    monkeypatch.setattr(em, "_store_calendar", lambda store: {"store": store})
    monkeypatch.setattr(em, "_store_variables", lambda store: [store])
    monkeypatch.setattr(em, "_store_inits", lambda store: [store])
    assert em.run_calendar("ifs_control") == {"store": "ifs_ens"}
    assert em.store_variables("ifs_control") == ["ifs_ens"]
    assert em.available_inits("ifs_control") == ["ifs_ens"]
    assert calls == []
