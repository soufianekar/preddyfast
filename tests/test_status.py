"""Request status for the status bar: stages reported by long paths, read by session."""

import flask
import pytest
from flask_caching import Cache

from utils import status


@pytest.fixture
def app(monkeypatch):
    app = flask.Flask("status-test")
    c = Cache(app, config={"CACHE_TYPE": "SimpleCache"})
    monkeypatch.setattr(status, "cache", c)
    return app


def test_activity_reports_stages_under_the_session_cookie(app):
    with app.test_request_context(headers={"Cookie": f"{status.COOKIE}=abc"}):
        with status.activity("Area average"):
            status.report("Reading archive", done=3, total=10, detail="ECMWF IFS ENS")
            snap = status.snapshot("abc")
            assert snap[0]["label"] == "Area average" and snap[0]["stage"] == "Reading archive"
            assert (snap[0]["done"], snap[0]["total"], snap[0]["state"]) == (3, 10, "running")
        assert status.snapshot("abc")[0]["state"] == "done"


def test_failure_is_shown_then_the_error_propagates(app):
    with app.test_request_context(headers={"Cookie": f"{status.COOKIE}=abc"}):
        with pytest.raises(ValueError), status.activity("Evaluation"):
            raise ValueError("boom")
        assert status.snapshot("abc")[0]["state"] == "failed"


def test_report_outside_an_activity_or_request_is_a_no_op(app):
    status.report("nothing listening")  # no request context at all
    with app.test_request_context():  # no cookie
        with status.activity("x"):
            status.report("still fine")
        assert status.snapshot(None) == []


def test_sessions_are_kept_apart_and_nested_activities_report_to_the_innermost(app):
    with app.test_request_context(headers={"Cookie": f"{status.COOKIE}=a"}):
        with status.activity("outer"):
            with status.activity("inner"):
                status.report("deep")
                snap = {e["label"]: e for e in status.snapshot("a")}
                assert snap["inner"]["stage"] == "deep" and snap["outer"]["stage"] == "Starting"
            status.report("back out")
            snap = {e["label"]: e for e in status.snapshot("a")}
            assert snap["inner"]["state"] == "done" and snap["outer"]["stage"] == "back out"
        assert status.snapshot("b") == []


def test_finished_entries_expire(app, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(status.time, "time", lambda: clock[0])
    with app.test_request_context(headers={"Cookie": f"{status.COOKIE}=abc"}):
        with status.activity("quick"):
            pass
        clock[0] += status.DONE_SHOWN_S + 1
        assert status.snapshot("abc") == []
