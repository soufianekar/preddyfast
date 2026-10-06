import pytest

from components.theme_toggle import THEMES
from utils.settings import SYSTEM_STYLES


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("theme", list(THEMES))
@pytest.mark.parametrize("system", sorted(SYSTEM_STYLES))
def test_system_lines_readable_on_both_panels(system, theme):
    assert contrast(SYSTEM_STYLES[system]["color"], THEMES[theme]["panel"]) >= 3.0


import math

from utils.settings import SYSTEM_PALETTE


def _lab(h):
    rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116  # noqa: E731
    return (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))


def test_store_keys_never_repaint_an_open_meteo_system():
    # icon_eu is both an Open-Meteo ensemble key and a store key.
    assert SYSTEM_STYLES["icon_eu"]["color"] in SYSTEM_PALETTE


def test_ecmwf_ai_ens_is_well_apart_from_every_other_hue():
    own = SYSTEM_STYLES["ecmwf_aifs025"]["color"]
    others = {s["color"] for k, s in SYSTEM_STYLES.items() if s["color"] != own}
    assert min(math.dist(_lab(own), _lab(o)) for o in others) >= 25


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_observed_line_readable_on_both_panels(theme):
    from utils.settings import OBSERVED_COLOR

    assert contrast(OBSERVED_COLOR, THEMES[theme]["panel"]) >= 3.0
