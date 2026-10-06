import sys
from pathlib import Path

import flask
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.cache import cache  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _app_context():
    """Bind the memoize cache to a throwaway app; `app.py` normally does this."""
    app = flask.Flask("tests")
    cache.init_app(app, config={"CACHE_TYPE": "NullCache"})
    with app.app_context():
        yield


@pytest.fixture(autouse=True)
def _no_real_precomputed(monkeypatch, tmp_path):
    """Never read the developer's .precomputed tree (it would ask the live store for runs)."""
    from utils import precomputed

    monkeypatch.setattr(precomputed, "PRECOMPUTE_DIR", str(tmp_path / "precomputed"))
