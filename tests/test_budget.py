import numpy as np
import pytest

from utils import earthmover_api as em

CHUNK = 85 * 51 * 32 * 32 * 4  # IFS ENS float32 chunk, bytes


def test_one_tile_box():
    assert em._tiles((0, 31), (0, 31)) == 1


def test_box_straddling_a_tile_edge():
    assert em._tiles((30, 33), (0, 31)) == 2


def test_large_region_two_models_refused_before_any_read(monkeypatch):
    # ~36 000 cells: 120 x 300 at 0.25 deg -> 4-5 x 10 tiles per store.
    monkeypatch.setattr(em, "_store_chunks", lambda store, bounds, horizon=None: (em._tiles((0, 119), (0, 299)), CHUNK))
    reads = []
    monkeypatch.setattr(em, "_open_dataset", lambda *a, **k: reads.append(a))
    with pytest.raises(em.GridDataError, match="MB across 2 models"):
        em.check_read_budget(("ifs_ens", "aifs_ens"), (0, 30, 0, 75))
    assert reads == []


def test_small_box_two_models_allowed(monkeypatch):
    monkeypatch.setattr(em, "_store_chunks", lambda store, bounds, horizon=None: (1, CHUNK))
    em.check_read_budget(("ifs_ens", "aifs_ens"), (45, 50, 5, 10))


def test_inits_multiply_the_cost(monkeypatch):
    monkeypatch.setattr(em, "_store_chunks", lambda store, bounds, horizon=None: (1, CHUNK))
    with pytest.raises(em.GridDataError):
        em.check_read_budget(("ifs_ens", "aifs_ens"), (45, 50, 5, 10), n_inits=40)


def test_read_estimate_counts_chunks_and_mb(monkeypatch):
    monkeypatch.setattr(em, "_store_chunks", lambda store, bounds, horizon=None: (1, CHUNK))
    chunks, mb = em.read_estimate(("ifs_ens", "aifs_ens"), (45, 45, 5, 5), n_inits=10)
    assert chunks == 20 and mb == pytest.approx(20 * CHUNK / 1e6)


GEFS = {"chunks": {"init_time": 1, "ensemble_member": 31, "lead_time": 64, "latitude": 17, "longitude": 16},
        "lead_hours": np.r_[np.arange(0, 241, 3), np.arange(246, 841, 6)].astype(float), "n_members": 31}
IFS46 = {"chunks": {"init_time": 1, "lead_time": 93, "ensemble_member": 51, "latitude": 32, "longitude": 30},
         "lead_hours": np.arange(0, 1105, 6.0), "n_members": 101}
GFS = {"chunks": {"init_time": 1, "lead_time": 105, "latitude": 121, "longitude": 121},
       "lead_hours": np.r_[np.arange(0, 121, 1), np.arange(123, 385, 3)].astype(float), "n_members": 1}


def test_gefs_counts_lead_blocks():
    assert em._chunk_count(GEFS, (0, 40), (0, 40), None) == 3 * 3 * 3
    assert em._chunk_count(GEFS, (0, 40), (0, 40), 72) == 3 * 3 * 1


def test_ifs46_counts_member_and_lead_blocks():
    assert em._chunk_count(IFS46, (5, 5), (7, 7), None) == 2 * 2
    assert em._chunk_count(IFS46, (5, 5), (7, 7), 72) == 2


def test_deterministic_has_no_member_axis():
    assert em._chunk_count(GFS, (0, 0), (0, 0), None) == 2  # 209 leads / 105
    assert em._chunk_count(GFS, (0, 0), (0, 0), 72) == 1


def test_many_small_chunks_fit_when_their_bytes_do(monkeypatch):
    # GEFS-like: 80 chunks of ~2 MB is 160 MB — well inside the budget.
    monkeypatch.setattr(em, "_store_chunks", lambda store, bounds, horizon=None: (80, 2_158_592))
    em.check_read_budget(("gefs",), (45, 55, 5, 15))
