import pytest

from utils import earthmover_api as em

ICON = {"global": False, "lat_range": (29.5, 70.5), "lon_range": (-23.5, 62.5), "res_deg": 0.0625}
GLOBAL = {"global": True, "lat_range": (-90.0, 90.0), "lon_range": (-180.0, 179.75), "res_deg": 0.25}


@pytest.fixture
def geometry(monkeypatch):
    monkeypatch.setattr(em, "store_geometry", lambda store: ICON if store == "icon_eu" else GLOBAL)


def test_inside_passes_unchanged(geometry):
    assert em.clip_to_store("icon_eu", (45.0, 50.0, 5.0, 10.0)) == ((45.0, 50.0, 5.0, 10.0), None)


def test_straddling_box_is_clipped_and_says_so(geometry):
    bounds, note = em.clip_to_store("icon_eu", (25.0, 40.0, -30.0, 0.0))
    assert bounds == (29.5, 40.0, -23.5, 0.0)
    assert "clipped to" in note and "ICON-EU" in note


def test_entirely_outside_is_refused(geometry):
    with pytest.raises(em.GridDataError, match="outside"):
        em.clip_to_store("icon_eu", (-40.0, -30.0, 140.0, 150.0))


def test_global_store_never_clips(geometry):
    assert em.clip_to_store("ifs_ens", (-40.0, -30.0, 140.0, 150.0)) == ((-40.0, -30.0, 140.0, 150.0), None)
