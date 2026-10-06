from components.theme_toggle import THEMES, theme_toggle


def test_two_themes_dark_first():
    assert list(THEMES) == ["dark", "light"]


def test_every_theme_defines_every_token():
    keys = set(THEMES["dark"])
    assert keys == set(THEMES["light"]) and {"night", "panel", "ink", "dim", "rule", "warm", "cold", "occl"} <= keys


def test_toggle_is_labelled_for_screen_readers():
    toggle = theme_toggle()
    assert toggle.id == "theme-toggle" and "aria-label" in toggle.to_plotly_json()["props"]
