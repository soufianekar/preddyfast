from utils.settings import ENSEMBLE_MODELS, GRID_STORES


def test_archive_stores_are_known_grid_stores():
    for key, model in ENSEMBLE_MODELS.items():
        if "archive_store" in model:
            assert model["archive_store"] in GRID_STORES, key


def test_systems_with_an_archive():
    mapped = {k: m["archive_store"] for k, m in ENSEMBLE_MODELS.items() if "archive_store" in m}
    assert mapped == {"ecmwf_ifs025": "ifs_ens", "ecmwf_aifs025": "aifs_ens", "gfs025": "gefs"}
