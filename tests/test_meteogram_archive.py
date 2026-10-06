import pytest

from utils.earthmover_api import archive_systems


def test_keeps_only_archive_backed_systems():
    assert archive_systems(["icon_seamless", "ecmwf_aifs025", "gfs_seamless"]) == ["ecmwf_aifs025"]


def test_keeps_selection_order():
    assert archive_systems(["ecmwf_aifs025", "ecmwf_ifs025"]) == ["ecmwf_aifs025", "ecmwf_ifs025"]


@pytest.mark.parametrize("selected", [[], None, ["icon_seamless"]])
def test_falls_back_to_ifs(selected):
    assert archive_systems(selected) == ["ecmwf_ifs025"]
