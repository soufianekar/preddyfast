"""Observation representativeness uncertainty (ECMWF Tech Memo 865).

Ben Bouallègue (2020), "Accounting for representativeness in the verification of
ensemble forecasts", ECMWF Technical Memorandum 865. Pure NumPy/SciPy; the
perturbed-ensemble draw follows Ferro (2017): each member is replaced by one
sample from the observation-error distribution centred on it.
"""

import numpy as np
from scipy import stats

VARIABLES: tuple[str, ...] = ("temperature_2m", "wind_speed_10m", "precipitation_24h")
LAPSE_K_PER_M: float = 0.0065


def t2m_params(dz_m: np.ndarray, delta_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (shift_K, sigma_K) for 2 m temperature (TM865 Eq. 2-4).

    Parameters
    ----------
    dz_m : array
        Model minus station elevation (m); any shape, NaN propagates.
    delta_km : float
        Averaging scale dA, the model's native grid spacing (km).
    """
    dz = np.asarray(dz_m, dtype=float)
    if delta_km < 100:
        b0, b1 = 0.02 * delta_km, 0.35 - 0.002 * delta_km
    else:
        b0, b1 = 2.0, 0.15
    return LAPSE_K_PER_M * dz, b0 + b1 * np.abs(dz) ** 0.25


def wind_params(delta_km: float) -> tuple[float, float, float]:
    """Return (alpha0, alpha1, beta1) for 10 m wind speed (TM865 Eq. 7).

    The truncated normal (Eq. 5-6) has location ``alpha0 + alpha1 * x`` and
    scale ``beta1 * sqrt(x) + 0.01``, truncated at 0.
    """
    d = float(delta_km)
    return -0.02 * d, 1 + 0.002 * d, -0.04 * d + 0.17 * d**0.75


def precip_params(delta_km: float) -> dict[str, float]:
    """Return the censored shifted gamma parameters for 24 h precipitation.

    TM865 Eq. 12 gives alpha0, alpha1, beta0, beta1 and delta as functions of
    dA; Eq. 8-11 use them as ``mu = alpha0 + alpha1 * x``,
    ``sigma = beta0 + beta1 * sqrt(x)``, ``k = mu**2 / sigma**2``,
    ``theta = sigma**2 / mu``, shifted by ``-delta`` and censored at 0.
    """
    d = float(delta_km)
    return {
        "alpha0": 0.005 * d**0.5,
        "alpha1": 1.0,
        "beta0": 0.0005 * d,
        "beta1": -0.02 * d + 0.55 * d**0.5,
        "delta": 0.005 * d**0.5,
    }


def perturb(
    members: np.ndarray,
    variable: str,
    delta_km: float,
    rng: np.random.Generator,
    dz_m: np.ndarray | None = None,
    draws: int | None = None,
) -> np.ndarray:
    """Perturbed ensemble: one representativeness draw per member (Ferro 2017).

    Parameters
    ----------
    members : array, shape (..., member)
        Forecast values; the member axis is last. Precipitation in mm per 24 h,
        wind in m/s, temperature in the unit of the station (K offsets are
        added as-is, so use degC or K consistently).
    variable : str
        ``temperature_2m`` (Eq. 2-4; the draw is centred on ``x + shift``),
        ``wind_speed_10m`` (Eq. 5-7) or ``precipitation_24h`` (Eq. 8-12).
    delta_km : float
        Model native grid spacing dA (km).
    rng : numpy.random.Generator
        Seeded by the caller (``SCORECARD_RNG_SEED``) for reproducibility.
    dz_m : array, optional
        Required for temperature: model minus station elevation (m). Shape must
        broadcast against ``members`` from the left, e.g. ``(station,)`` for
        ``(station, lead, member)``. NaN gives NaN output.
    draws : int, optional
        Deterministic stores: expand each member to ``draws`` draws, giving a
        member axis of ``n_member * draws`` (member-major).
    """
    if variable not in VARIABLES:
        raise ValueError(f"Unsupported variable for TM865 perturbation: {variable}")
    x = np.asarray(members, dtype=float)
    if draws:
        x = np.repeat(x, int(draws), axis=-1)
    if variable == "temperature_2m":
        if dz_m is None:
            raise ValueError("temperature_2m needs dz_m (model minus station, m)")
        dz = np.asarray(dz_m, dtype=float)
        dz = dz.reshape(dz.shape + (1,) * (x.ndim - dz.ndim))
        shift, sigma = t2m_params(dz, delta_km)
        return x + shift + sigma * rng.standard_normal(x.shape)
    if variable == "wind_speed_10m":
        a0, a1, b1 = wind_params(delta_km)
        mu = a0 + a1 * x
        sigma = b1 * np.sqrt(np.maximum(x, 0.0)) + 0.01
        return stats.truncnorm.rvs(
            (0.0 - mu) / sigma, np.inf, loc=mu, scale=sigma, random_state=rng
        )
    p = precip_params(delta_km)
    root = np.sqrt(np.maximum(x, 0.0))
    mu = p["alpha0"] + p["alpha1"] * x
    sigma = p["beta0"] + p["beta1"] * root
    g = stats.gamma.rvs(mu**2 / sigma**2, scale=sigma**2 / mu, random_state=rng)
    return np.maximum(g - p["delta"], 0.0)
