import numpy as np
import pandas as pd

from utils.climatology import compute_member_pct
from utils.figures import figure_text, make_regional_multi_fig
from utils.settings import GRID_STORES, GRID_VARIABLES


def _series(offset: float, n_members: int, init: str) -> tuple:
    times = pd.date_range("2026-10-01 06:00", periods=4, freq="6h")
    df = pd.DataFrame(
        [{"valid_time": t, "member": m, "value": offset + m + i} for i, t in enumerate(times) for m in range(n_members)]
    )
    meta = {
        "init_time": pd.Timestamp(init), "n_members": n_members, "n_lat": 21, "n_lon": 21,
        "bbox": (45.0, 50.0, 5.0, 10.0), "resolution_deg": 0.25, "unit": "°C",
        "accum_hours": None, "step_window": None,
    }
    return df, compute_member_pct(df), meta


def _fig():
    results = {"ifs_ens": _series(0.0, 3, "2026-10-01 00:00"), "aifs_ens": _series(100.0, 4, "2026-10-01 06:00")}
    return make_regional_multi_fig(results, GRID_VARIABLES["temperature_2m"])


def test_every_trace_names_one_system():
    labels = [GRID_STORES[k]["label"] for k in ("ifs_ens", "aifs_ens")]
    for trace in _fig().data:
        assert sum(label in trace.name for label in labels) == 1, trace.name


def test_members_hidden_when_several_systems():
    members = [t for t in _fig().data if t.legendgroup.endswith("-members")]
    assert len(members) == 2 and all(t.visible == "legendonly" for t in members)


def test_no_trace_mixes_systems():
    for trace in _fig().data:
        y = np.asarray([v for v in trace.y if v is not None], dtype=float)
        assert (y < 50).all() or (y >= 100).all(), trace.name


def test_subtitle_states_each_init_and_model():
    title = figure_text(_fig())
    assert "2026-10-01 00 UTC" in title and "2026-10-01 06 UTC" in title
    assert GRID_STORES["aifs_ens"]["model_label"] in title


def test_single_system_shows_members():
    fig = make_regional_multi_fig({"ifs_ens": _series(0.0, 3, "2026-10-01 00:00")}, GRID_VARIABLES["temperature_2m"])
    members = [t for t in fig.data if t.legendgroup.endswith("-members")]
    assert members and members[0].visible in (None, True)


def test_extent_uses_hemisphere_letters_never_signs():
    from utils.figures import _bbox_label
    label = _bbox_label({"bbox": (29.5, 32.0, -23.5, -20.0), "resolution_deg": 0.0625, "n_lat": 41, "n_lon": 57})
    assert "-" not in label.replace("–", "") and "W" in label
