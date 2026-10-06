import pytest

from utils import openmeteo_api as om


class _Cache:
    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, value, timeout=None):
        self.d[key] = value


class _Block:
    def __init__(self, n):
        self.n = n

    def VariablesLength(self):
        return self.n


class _Response:
    def __init__(self, n, hourly=True):
        self.block = _Block(n)
        self.hourly = hourly

    def Hourly(self):
        return self.block if self.hourly else None

    def Daily(self):
        return None if self.hourly else self.block


@pytest.fixture
def shared(monkeypatch):
    c = _Cache()
    monkeypatch.setattr(om, "cache", c)
    monkeypatch.setattr(om, "OPENMETEO_DAILY_BUDGET", 20.0)
    return c


def test_open_meteo_example_weighs_four_calls():
    # Their docs: one variable, ICON Seamless (40 members), 7 days = 4.0.
    assert om.weighted_calls(7, 40) == 4.0
    assert om.weighted_calls(35, 1) == 2.5


def test_members_counted_from_the_response_per_model(shared, monkeypatch):
    monkeypatch.setattr(om._client, "weather_api", lambda url, params: [_Response(51)])
    om._get(om.ENSEMBLE_URL, {}, 7)
    assert om.calls_today() == pytest.approx(5.1)
    monkeypatch.setattr(om._client, "weather_api", lambda url, params: [_Response(1, False)] * 3)
    om._get(om.FORECAST_URL, {}, 7)
    assert om.calls_today() == pytest.approx(8.1)  # one call per deterministic model


def test_refused_once_the_budget_is_spent_without_asking(shared, monkeypatch):
    asked = []
    monkeypatch.setattr(om._client, "weather_api", lambda url, params: asked.append(1) or [_Response(200)])
    om._get(om.ENSEMBLE_URL, {}, 7)  # 20 calls: reaches the budget
    with pytest.raises(om.OpenMeteoQuotaError, match="today's share"):
        om._get(om.ENSEMBLE_URL, {}, 7)
    assert len(asked) == 1


def test_the_count_is_per_utc_day(shared, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(om._client, "weather_api", lambda url, params: [_Response(200)])
    monkeypatch.setattr(om, "_today", lambda: pd.Timestamp("2026-10-06", tz="UTC"))
    om._get(om.ENSEMBLE_URL, {}, 7)
    monkeypatch.setattr(om, "_today", lambda: pd.Timestamp("2026-10-07", tz="UTC"))
    assert om.calls_today() == 0.0
    om._get(om.ENSEMBLE_URL, {}, 7)  # a new day is not refused
