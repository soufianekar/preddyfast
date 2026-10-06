"""Precomputed products on a public Hugging Face Dataset: upload, prune, mirror.

The scheduled job (`scripts/precompute.py --upload-hf`) uploads everything it
wrote under `PRECOMPUTE_DIR` — animation frames, gallery images, preset fields,
station scores, manifests — to the Dataset `PRECOMPUTE_HF_REPO`, then deletes
images no manifest points at any more, so the Dataset holds the newest run only.

The public app never writes there. Images load straight from the Dataset by
URL (`PRECOMPUTE_URL`); what the server itself reads — manifests, `fields.npz`,
score Parquet — is mirrored into `PRECOMPUTE_DIR` by `sync`, at most once per
`PRECOMPUTE_SYNC_S` across all workers. Without `PRECOMPUTE_HF_REPO` (a local
run) every function here does nothing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from utils.settings import PRECOMPUTE_DIR, PRECOMPUTE_HF_REPO, PRECOMPUTE_SYNC_S

_log = logging.getLogger(__name__)

# What the server reads from disk; images are fetched by the browser instead.
SYNC_PATTERNS = ["latest.json", "maps/gallery.json", "frames/**/fields.npz", "scores/**/*.parquet"]
_STAMP = ".synced"
_NEVER_UPLOAD = [_STAMP, "**/*.part", ".cache/**"]
_frozen = False


def freeze() -> None:
    """Stop all further syncs in this process.

    For the precompute job: it mirrors once at the start, then writes new
    manifests locally; a later sync must not put the Dataset's older ones back.
    """
    global _frozen
    _frozen = True


def sync(force: bool = False) -> bool:
    """Mirror the Dataset's metadata into `PRECOMPUTE_DIR`; True if it ran.

    The stamp is touched *before* downloading, so the other workers skip
    rather than all fetching at once; a failed fetch is retried next period.
    """
    if not PRECOMPUTE_HF_REPO or _frozen:
        return False
    stamp = Path(PRECOMPUTE_DIR) / _STAMP
    if not force and stamp.exists() and time.time() - stamp.stat().st_mtime < PRECOMPUTE_SYNC_S:
        return False
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        snapshot_download(
            repo_id=PRECOMPUTE_HF_REPO,
            repo_type="dataset",
            local_dir=PRECOMPUTE_DIR,
            allow_patterns=SYNC_PATTERNS,
        )
    except (HfHubHTTPError, OSError) as exc:
        _log.warning("precomputed sync from %s failed: %s", PRECOMPUTE_HF_REPO, exc)
        return False
    return True


def referenced_paths(manifest: dict, gallery: dict) -> set[str]:
    """Every file path the two manifests point at."""
    paths = {item["path"] for item in gallery.get("items", {}).values()}
    for store in manifest.get("stores", {}).values():
        for product in store.get("products", {}).values():
            paths.update(frame["path"] for frame in product.get("frames", []))
            if product.get("fields"):
                paths.add(product["fields"])
    return paths


def stale_paths(files: list[str], manifest: dict, gallery: dict) -> list[str]:
    """Images and preset fields on the Dataset that no manifest references.

    Only `frames/` and `maps/` content is ever pruned — manifests and scores
    are kept whatever they contain. With an empty animation manifest nothing
    under `frames/` is touched: that is a failed or partial job, not a
    decision to delete every run.
    """
    keep = referenced_paths(manifest, gallery)
    prune_frames = bool(manifest.get("stores"))
    prune_maps = bool(gallery.get("items"))
    stale = []
    for path in files:
        if path in keep or path.endswith(".json"):
            continue
        if (prune_frames and path.startswith("frames/")) or (
            prune_maps and path.startswith("maps/") and path.endswith(".png")
        ):
            stale.append(path)
    return sorted(stale)


def _read_json(path: Path, empty: dict) -> dict:
    return json.loads(path.read_text()) if path.exists() else empty


def upload(token: str, squash: bool = True) -> int:
    """Upload `PRECOMPUTE_DIR` to the Dataset, prune what no manifest references.

    Returns the number of files pruned. `squash` collapses the Dataset's git
    history afterwards: frames are replaced several times a day and every old
    version would otherwise stay in the repository's storage.
    """
    from huggingface_hub import CommitOperationDelete, HfApi

    if not PRECOMPUTE_HF_REPO:
        raise ValueError("Set PRECOMPUTE_HF_REPO to upload.")
    api = HfApi(token=token)
    root = Path(PRECOMPUTE_DIR)
    api.create_repo(PRECOMPUTE_HF_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=PRECOMPUTE_HF_REPO,
        repo_type="dataset",
        folder_path=str(root),
        ignore_patterns=_NEVER_UPLOAD,
        commit_message="Precomputed products",
    )
    files = api.list_repo_files(PRECOMPUTE_HF_REPO, repo_type="dataset")
    stale = stale_paths(
        files,
        _read_json(root / "latest.json", {"stores": {}}),
        _read_json(root / "maps" / "gallery.json", {"items": {}}),
    )
    if stale:
        api.create_commit(
            repo_id=PRECOMPUTE_HF_REPO,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=p) for p in stale],
            commit_message=f"Prune {len(stale)} superseded files",
        )
    if squash:
        api.super_squash_history(repo_id=PRECOMPUTE_HF_REPO, repo_type="dataset")
    return len(stale)
