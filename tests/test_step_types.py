import numpy as np
import pytest
import xarray as xr

from utils import earthmover_api as em


def _ds(var, step_type):
    da = xr.DataArray(np.zeros(3), dims=["lead_time"], attrs={"step_type": step_type})
    return xr.Dataset({var: da})


def test_cloud_cover_accepts_instant_and_avg():
    assert em._check_step_type(_ds("total_cloud_cover_atmosphere", "avg"), "total_cloud_cover_atmosphere") == "avg"
    assert em._check_step_type(_ds("total_cloud_cover_atmosphere", "instant"), "total_cloud_cover_atmosphere") == "instant"


def test_precipitation_accumulation_refused():
    with pytest.raises(em.GridDataError, match="accum"):
        em._check_step_type(_ds("precipitation_surface", "accum"), "precipitation_surface")


def test_window_words_follow_the_store_step_type():
    assert em._window_phrase("avg") == "mean over each preceding step"
    assert em._window_phrase("max") == "maximum over each preceding step"
    assert em._window_phrase("instant") is None

