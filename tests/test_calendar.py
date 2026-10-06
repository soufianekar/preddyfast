from components.run_select import pick_hour
from utils.earthmover_api import _merge_calendars

IFS = {
    "first": "2024-04-01",
    "last": "2026-10-02",
    "hours": [0],
    "days": {"2026-10-01": [0], "2026-10-02": [0]},
    "missing": [],
    "unfinished": [],
}
AIFS = {
    "first": "2025-07-02",
    "last": "2026-10-02",
    "hours": [0, 6, 12, 18],
    "days": {"2026-10-01": [0, 6, 12, 18], "2026-10-02": [0, 6]},
    "missing": [],
    "unfinished": ["2026-10-02T06:00"],
}


def test_intersection_keeps_shared_hours_only():
    cal = _merge_calendars([IFS, AIFS])
    assert cal["hours"] == [0]
    assert cal["days"] == {"2026-10-01": [0], "2026-10-02": [0]}
    assert cal["first"] == "2026-10-01"
    assert cal["unfinished"] == ["2026-10-02T06:00"]


def test_single_calendar_unchanged():
    assert _merge_calendars([AIFS])["hours"] == [0, 6, 12, 18]


def test_day_missing_in_one_store_is_missing_in_common():
    other = {**IFS, "days": {"2026-10-02": [0]}}
    cal = _merge_calendars([other, AIFS])
    assert cal["missing"] == []  # span starts at the first common day
    assert list(cal["days"]) == ["2026-10-02"]


def test_snap_moves_an_hour_the_day_lacks():
    assert pick_hour([0], "06", snap=True) == "00"


def test_no_snap_keeps_the_url_hour():
    assert pick_hour([0], "06", snap=False) == "06"


def test_keeps_an_available_hour():
    assert pick_hour([0, 6], "06", snap=True) == "06"


def test_day_without_runs_has_no_hour():
    assert pick_hour([], "06", snap=True) is None
