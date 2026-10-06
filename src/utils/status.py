"""Request status for the status bar: what each long request is doing right now.

Long paths (archive reads, frame rendering, station scoring, CDS queues) call
`report(stage, done, total)` inside an `activity(label)`. Entries are keyed by
the browser's session cookie, which Flask can read inside any callback, so no
callback signature changes. They live in the shared cache, so a poll that lands
on another gunicorn worker still sees them. `assets/status.js` polls
`/status` only while a Dash request is in flight.

A plain callback cannot push progress while it runs and background callbacks
fork (unsafe for the gridded stack, see app.py); this is the alternative.
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from contextlib import contextmanager

import flask

from utils.cache import cache

COOKIE = "preddyfast_sid"
DONE_SHOWN_S = 4.0  # a finished entry stays visible this long
_STALE_S = 300.0  # a running entry not updated for this long is dropped (worker died)
_TTL_S = 600

_current: contextvars.ContextVar = contextvars.ContextVar("status_activity", default=None)
_lock = threading.Lock()


def _session() -> str | None:
    try:
        return flask.request.cookies.get(COOKIE)
    except RuntimeError:  # outside a request (scripts, tests, the precompute job)
        return None


def _key(sid: str) -> str:
    return f"status:{sid}"


def _write(sid: str, activity_id: str, **fields) -> None:
    with _lock:
        entries = cache.get(_key(sid)) or {}
        entry = {**entries.get(activity_id, {}), **fields, "updated": time.time()}
        entries[activity_id] = entry
        cache.set(_key(sid), entries, timeout=_TTL_S)


@contextmanager
def activity(label: str):
    """Report a long request under `label`; marks it done or failed when it ends."""
    sid = _session()
    if not sid:
        yield
        return
    activity_id = uuid.uuid4().hex[:8]
    token = _current.set((sid, activity_id))
    _write(sid, activity_id, label=label, stage="Starting", done=None, total=None,
           detail="", state="running", started=time.time())
    try:
        yield
    except BaseException:
        _write(sid, activity_id, state="failed", stage="Failed")
        raise
    else:
        _write(sid, activity_id, state="done", stage="Done")
    finally:
        _current.reset(token)


def report(stage: str, done: int | None = None, total: int | None = None, detail: str = "") -> None:
    """Update the innermost running activity; a no-op when nothing is listening."""
    ctx = _current.get()
    if ctx is None:
        return
    sid, activity_id = ctx
    _write(sid, activity_id, stage=stage, done=done, total=total, detail=detail)


def track(
    key: str,
    label: str,
    stage: str,
    done: int | None = None,
    total: int | None = None,
    state: str = "running",
) -> None:
    """An entry that outlives one request (a CDS job across its polls), keyed by `key`."""
    sid = _session()
    if not sid:
        return
    _write(sid, key, label=label, stage=stage, done=done, total=total, detail="",
           state=state, started=time.time() if state == "running" else 0)


def snapshot(sid: str | None) -> list[dict]:
    """The session's entries worth showing, running first, newest first."""
    if not sid:
        return []
    now = time.time()
    entries = cache.get(_key(sid)) or {}
    keep = [
        e
        for e in entries.values()
        if (e["state"] == "running" and now - e["updated"] < _STALE_S)
        or (e["state"] != "running" and now - e["updated"] < DONE_SHOWN_S)
    ]
    return sorted(keep, key=lambda e: (e["state"] != "running", -e.get("started", 0)))
