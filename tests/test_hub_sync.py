import os
import sys
import time
import types

import pytest

from utils import hub_sync


@pytest.fixture
def _hub(tmp_path, monkeypatch):
    """A configured repo, a temp tree, and a fake snapshot_download that counts calls."""
    calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda **kw: calls.append(kw)
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.setattr(hub_sync, "PRECOMPUTE_HF_REPO", "someone/preddyfast-data")
    monkeypatch.setattr(hub_sync, "PRECOMPUTE_DIR", str(tmp_path))
    monkeypatch.setattr(hub_sync, "PRECOMPUTE_SYNC_S", 900.0)
    monkeypatch.setattr(hub_sync, "_frozen", False)
    return calls, tmp_path


def test_nothing_happens_without_a_repo(monkeypatch):
    monkeypatch.setattr(hub_sync, "PRECOMPUTE_HF_REPO", "")
    assert hub_sync.sync() is False


def test_syncs_metadata_only_then_waits_a_period(_hub):
    calls, root = _hub
    assert hub_sync.sync() is True
    assert hub_sync.sync() is False  # within the period: another worker's stamp counts too
    assert len(calls) == 1
    kw = calls[0]
    assert kw["repo_type"] == "dataset" and kw["local_dir"] == str(root)
    assert not any(p.endswith((".webp", ".png")) for p in kw["allow_patterns"])


def test_syncs_again_once_the_stamp_is_old(_hub):
    calls, root = _hub
    hub_sync.sync()
    old = time.time() - 901
    os.utime(root / ".synced", (old, old))
    assert hub_sync.sync() is True and len(calls) == 2


def test_frozen_process_never_syncs(_hub):
    calls, _ = _hub
    hub_sync.freeze()
    assert hub_sync.sync(force=True) is False and calls == []


def _manifest(stamp: str) -> dict:
    return {
        "stores": {
            "ifs_ens": {
                "init": "2026-10-06T00:00",
                "products": {
                    "ecmwf_europe/temperature_2m/mean": {
                        "frames": [{"path": f"frames/ifs_ens/{stamp}/ecmwf_europe/temperature_2m/mean/T024.webp"}],
                        "fields": f"frames/ifs_ens/{stamp}/ecmwf_europe/temperature_2m/mean/fields.npz",
                    }
                },
            }
        }
    }


def test_prunes_superseded_runs_and_gallery_months_only():
    files = [
        "latest.json",
        "maps/gallery.json",
        "maps/t2m_europe-202608.png",
        "maps/t2m_europe-202607.png",
        "frames/ifs_ens/2026100600/ecmwf_europe/temperature_2m/mean/T024.webp",
        "frames/ifs_ens/2026100600/ecmwf_europe/temperature_2m/mean/fields.npz",
        "frames/ifs_ens/2026100512/ecmwf_europe/temperature_2m/mean/T024.webp",
        "scores/stations/ifs_ens/temperature_2m/X.parquet",
        "README.md",
    ]
    gallery = {"items": {"t2m_europe": {"path": "maps/t2m_europe-202608.png"}}}
    assert hub_sync.stale_paths(files, _manifest("2026100600"), gallery) == [
        "frames/ifs_ens/2026100512/ecmwf_europe/temperature_2m/mean/T024.webp",
        "maps/t2m_europe-202607.png",
    ]


def test_an_empty_manifest_never_wipes_the_frames():
    files = ["frames/ifs_ens/2026100600/x/T024.webp", "maps/a.png"]
    assert hub_sync.stale_paths(files, {"stores": {}}, {"items": {}}) == []
