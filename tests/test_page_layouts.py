import importlib

import dash
import pytest


@pytest.fixture(scope="module", autouse=True)
def _app():
    app = dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    yield app


@pytest.mark.parametrize("page", ["maps", "meteogram", "regional", "evaluation"])
def test_layout_builds_with_and_without_query(page):
    module = importlib.import_module(f"pages.{page}")
    assert module.layout() is not None
    assert module.layout(variable="temperature_2m", models="gefs,gfs", run="latest") is not None


@pytest.mark.parametrize("page", ["maps", "meteogram", "regional", "evaluation"])
def test_every_callback_takes_its_inputs_and_states(page):
    """A decorator stacked on the wrong function still registers; catch it here."""
    import inspect

    from dash._callback import GLOBAL_CALLBACK_MAP

    module = importlib.import_module(f"pages.{page}")
    for key, entry in GLOBAL_CALLBACK_MAP.items():
        if "callback" not in entry:  # clientside
            continue
        fn = inspect.unwrap(entry["callback"])
        if fn.__module__ != module.__name__:
            continue
        params = inspect.signature(fn).parameters.values()
        if any(p.kind is p.VAR_POSITIONAL for p in params):
            continue
        n_args = len(entry["inputs"]) + len(entry["state"])
        assert len(params) == n_args, f"{fn.__name__} for {key}"
