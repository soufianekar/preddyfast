"""TM865 representativeness models and the perturbed-ensemble draw."""

import numpy as np
import pytest

from utils import obs_uncertainty as ou
from utils.settings import GRID_STORES


def test_native_km_declared():
    got = {k: v["native_km"] for k, v in GRID_STORES.items()}
    assert got == {
        "ifs_ens": 9, "ifs_control": 9, "aifs_ens": 31, "aifs_single": 31,
        "gefs": 25, "gfs": 13, "icon_eu": 6.5,
    }


def test_t2m_params_flat_and_height():
    shift, sigma = ou.t2m_params(np.array([0.0]), 18)
    assert shift[0] == 0 and sigma[0] == pytest.approx(0.36)
    shift, sigma = ou.t2m_params(np.array([100.0]), 18)
    assert shift[0] == pytest.approx(0.65)
    assert sigma[0] == pytest.approx(0.36 + 0.314 * 100**0.25)


def test_t2m_params_large_scale_branch():
    _, sigma = ou.t2m_params(np.array([16.0]), 100)
    assert sigma[0] == pytest.approx(2 + 0.15 * 2)


def test_wind_params_18km():
    a0, a1, b1 = ou.wind_params(18)
    assert (a0, a1) == (pytest.approx(-0.36), pytest.approx(1.036))
    assert b1 == pytest.approx(-0.72 + 0.17 * 18**0.75)
    assert b1 == pytest.approx(0.76, abs=0.01)


def test_precip_params_eq12():
    p = ou.precip_params(18)
    assert p["alpha0"] == pytest.approx(0.005 * 18**0.5, abs=1e-12)
    assert p["alpha1"] == 1
    assert p["beta0"] == pytest.approx(0.0005 * 18, abs=1e-12)
    assert p["beta1"] == pytest.approx(-0.02 * 18 + 0.55 * 18**0.5, abs=1e-12)
    assert p["delta"] == pytest.approx(0.005 * 18**0.5, abs=1e-12)


def test_perturb_temperature_moments():
    rng = np.random.default_rng(1)
    x = np.full((1, 10_000), 12.0)
    out = ou.perturb(x, "temperature_2m", 18, rng, dz_m=np.array([0.0]))
    assert out.shape == x.shape
    assert abs(out.mean() - 12.0) < 0.02
    assert out.std() == pytest.approx(0.36, rel=0.02)


def test_perturb_temperature_applies_height_shift():
    rng = np.random.default_rng(1)
    x = np.full((1, 1, 20_000), 10.0)
    out = ou.perturb(x, "temperature_2m", 18, rng, dz_m=np.array([100.0]))
    assert out.mean() == pytest.approx(10.65, abs=0.05)


def test_perturb_wind_non_negative():
    rng = np.random.default_rng(2)
    out = ou.perturb(np.full((5, 2000), 0.2), "wind_speed_10m", 18, rng)
    assert (out >= 0).all() and np.isfinite(out).all()


def test_perturb_precip_zero_keeps_mass_at_zero():
    rng = np.random.default_rng(3)
    out = ou.perturb(np.zeros((1, 10_000)), "precipitation_24h", 18, rng)
    assert (out >= 0).all()
    assert (out == 0).mean() >= 0.5


def test_perturb_precip_wet_member_centred():
    rng = np.random.default_rng(3)
    out = ou.perturb(np.full((1, 20_000), 10.0), "precipitation_24h", 18, rng)
    assert out.mean() == pytest.approx(10.0, rel=0.05)


def test_same_seed_identical():
    x = np.random.default_rng(0).gamma(2, 2, (3, 4, 10))
    dz = np.array([0.0, 50.0, -30.0])
    a = ou.perturb(x, "temperature_2m", 9, np.random.default_rng(5), dz_m=dz)
    b = ou.perturb(x, "temperature_2m", 9, np.random.default_rng(5), dz_m=dz)
    np.testing.assert_array_equal(a, b)
    assert a.shape == x.shape


def test_draws_expand_member_axis():
    x = np.ones((3, 1, 7))
    out = ou.perturb(x, "wind_speed_10m", 9, np.random.default_rng(0), draws=50)
    assert out.shape == (3, 1, 350)
    x1 = np.ones((3, 1, 1))
    out = ou.perturb(x1, "precipitation_24h", 9, np.random.default_rng(0), draws=50)
    assert out.shape == (3, 1, 50)


def test_nan_dz_gives_nan_rows():
    x = np.full((2, 3, 4), 5.0)
    out = ou.perturb(
        x, "temperature_2m", 9, np.random.default_rng(0), dz_m=np.array([np.nan, 1.0])
    )
    assert np.isnan(out[0]).all() and np.isfinite(out[1]).all()


def test_temperature_requires_dz_and_unknown_variable_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ou.perturb(np.ones((1, 3)), "temperature_2m", 9, rng)
    with pytest.raises(ValueError):
        ou.perturb(np.ones((1, 3)), "humidity", 9, rng)
