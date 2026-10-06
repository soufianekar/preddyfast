"""Shared flask-caching instance.

Defined standalone (no `server` bound yet) so `utils/openmeteo_api.py` can
import and decorate with it without a circular import on `app.py`. `app.py`
calls `cache.init_app(server, config=...)` once at startup.
"""

from __future__ import annotations

from flask_caching import Cache

cache = Cache()
