import json
from pathlib import Path

import publish


def _part(root: Path, store: str, init: str) -> Path:
    frame = root / "frames" / store / "x" / "T024.webp"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"w")
    entry = {"init": init, "products": {"r/v/mean": {"frames": [{"path": frame.name}]}}}
    (root / "latest.json").write_text(json.dumps({"stores": {store: entry}}))
    return root


def test_publish_merges_parts_and_copies_files(tmp_path, monkeypatch):
    out = tmp_path / "pre"
    out.mkdir()
    base = {"stores": {"gefs": {"init": "2026-10-21T00:00", "products": {}}}}
    (out / "latest.json").write_text(json.dumps(base))
    monkeypatch.setattr(publish.precomputed, "_root", lambda: out)
    parts = [
        _part(tmp_path / "a", "ifs_ens", "2026-10-22T00:00"),
        _part(tmp_path / "b", "aifs_ens", "2026-10-22T00:00"),
    ]
    copied = publish.publish(parts, out, upload=None)
    manifest = json.loads((out / "latest.json").read_text())
    assert set(manifest["stores"]) == {"gefs", "ifs_ens", "aifs_ens"}
    assert (out / "frames/ifs_ens/x/T024.webp").exists() and copied == 2


def test_publish_copies_scorecard_and_maps_parts(tmp_path, monkeypatch):
    out = tmp_path / "pre"
    out.mkdir()
    monkeypatch.setattr(publish.precomputed, "_root", lambda: out)
    part = tmp_path / "sc"
    (part / "eval/scorecard").mkdir(parents=True)
    (part / "eval/scorecard/aggregates.parquet").write_bytes(b"p")
    publish.publish([part], out, upload=None)
    assert (out / "eval/scorecard/aggregates.parquet").read_bytes() == b"p"


def test_publish_does_not_let_a_mirrored_copy_overwrite_new_output(
    tmp_path, monkeypatch
):
    out = tmp_path / "pre"
    (out / "maps").mkdir(parents=True)
    (out / "maps/gallery.json").write_bytes(b"old")
    monkeypatch.setattr(publish.precomputed, "_root", lambda: out)
    maps_part, scorecard_part = tmp_path / "maps", tmp_path / "sc"
    for part, data in ((maps_part, b"new"), (scorecard_part, b"old")):
        (part / "maps").mkdir(parents=True)
        (part / "maps/gallery.json").write_bytes(data)
    copied = publish.publish([maps_part, scorecard_part], out, upload=None)
    assert (out / "maps/gallery.json").read_bytes() == b"new"
    assert copied == 1


def test_main_refuses_without_its_merge_base(tmp_path, monkeypatch):
    from utils import hub_sync

    part = _part(tmp_path / "a", "ifs_ens", "2026-10-22T00:00")
    called = []
    monkeypatch.setattr(hub_sync, "PRECOMPUTE_HF_REPO", "me/data")
    monkeypatch.setattr(hub_sync, "sync", lambda force=False: False)
    monkeypatch.setattr(publish, "publish", lambda *a, **k: called.append(a) or 0)
    assert publish.main([str(part)]) != 0
    assert called == []


def test_publish_takes_the_globe_part_manifest_over_stale_mirrors(
    tmp_path, monkeypatch
):
    out = tmp_path / "pre"
    (out / "globe").mkdir(parents=True)
    (out / "globe/manifest.json").write_text(json.dumps({"init_time": "OLD"}))
    monkeypatch.setattr(publish.precomputed, "_root", lambda: out)
    stale, fresh = tmp_path / "a", tmp_path / "globe"
    for part, init in ((stale, "OLD"), (fresh, "NEW")):
        (part / "globe").mkdir(parents=True)
        (part / "globe/manifest.json").write_text(json.dumps({"init_time": init}))
    publish.publish([stale, fresh], out, upload=None)
    assert json.loads((out / "globe/manifest.json").read_text())["init_time"] == "NEW"
