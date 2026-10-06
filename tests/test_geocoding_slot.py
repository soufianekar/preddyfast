import pytest

from utils import geocoding as gc


class _Cache:
    def __init__(self):
        self.taken = False

    def add(self, key, value, timeout=None):
        if self.taken:
            return False
        self.taken = True
        return True


def test_second_request_within_the_second_is_refused(monkeypatch):
    monkeypatch.setattr(gc, "cache", _Cache())
    gc._nominatim_slot()
    with pytest.raises(gc.GeocodingError, match="busy"):
        gc._nominatim_slot(wait_s=0.3)


def test_reverse_geocode_falls_back_to_coordinates_when_busy(monkeypatch):
    def busy(wait_s=3.0):
        raise gc.GeocodingError("busy")

    monkeypatch.setattr(gc, "_nominatim_slot", busy)
    assert gc.reverse_geocode.uncached(48.85, 2.35) == "48.85, 2.35"
