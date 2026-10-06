import os

import pytest


def test_client_is_taken_hops_from_the_right():
    from app import client_address

    # A client forging "1.2.3.4" cannot pick its bucket: the proxy's entry wins.
    assert client_address("1.2.3.4, 203.0.113.9", "10.0.0.1", hops=1) == "203.0.113.9"
    assert client_address("203.0.113.9", "10.0.0.1", hops=1) == "203.0.113.9"
    assert client_address(None, "10.0.0.1", hops=1) == "10.0.0.1"
    assert client_address("1.2.3.4", "10.0.0.1", hops=0) == "10.0.0.1"


@pytest.mark.skipif(os.environ.get("CI_SKIP_APP") == "1", reason="app import is slow")
def test_callbacks_are_rate_limited_per_client():
    import app as a

    a.limiter.reset()
    client = a.server.test_client()
    url = next(r.rule for r in a.server.url_map.iter_rules() if r.rule.endswith("_dash-update-component"))
    limit = int(a.RATE_LIMIT.split()[0])
    codes = [
        client.post(url, json={}, headers={"X-Forwarded-For": "198.51.100.7"}).status_code
        for _ in range(limit + 1)
    ]
    assert 429 not in codes[:limit] and codes[-1] == 429
    other = client.post(url, json={}, headers={"X-Forwarded-For": "198.51.100.8"}).status_code
    assert other != 429
