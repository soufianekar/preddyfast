"""Merge the producer jobs' outputs into the published tree and upload it.

    python scripts/publish.py parts/* --upload-hf

Each part is one job's PRECOMPUTE_DIR (an animations model group, the
scorecard, the maps gallery). The Dataset's current manifest is mirrored
first, so a store no job refreshed keeps its published entry and frames.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import precomputed  # noqa: E402
from utils.settings import PRECOMPUTE_DIR  # noqa: E402

_log = logging.getLogger("publish")


def publish(parts: list[Path], root: Path, upload: Callable[[], int] | None) -> int:
    """Copy each part's new or changed files into `root`, merge manifests.

    Returns the number of files copied. A file identical to what `root` held
    before this call (a mirrored copy) is skipped, and `latest.json` is merged,
    not copied.
    """
    manifest = precomputed.load_manifest()
    copied = 0
    held: dict[Path, bytes | None] = {}  # what root held before any part wrote
    for part in parts:
        for path in part.rglob("*"):
            rel = path.relative_to(part)
            if not path.is_file() or rel.name in ("latest.json", ".synced"):
                continue
            target = root / rel
            # Every part carries a mirrored copy of the Dataset's files; a
            # stale one must not overwrite another part's new output.
            if target not in held:
                held[target] = target.read_bytes() if target.exists() else None
            if held[target] == path.read_bytes():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
        part_manifest = part / "latest.json"
        if part_manifest.exists():
            manifest = precomputed.merge_manifest(
                manifest, json.loads(part_manifest.read_text())
            )
    precomputed.save_manifest(manifest)
    if upload is not None:
        _log.info("%d superseded files pruned", upload())
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--upload-hf", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from utils import hub_sync

    hub_sync.sync(force=True)  # the published manifest is the merge base
    hub_sync.freeze()
    upload = (
        (lambda: hub_sync.upload(os.environ["HF_TOKEN"])) if args.upload_hf else None
    )
    parts = [p for p in args.parts if p.is_dir()]
    copied = publish(parts, Path(PRECOMPUTE_DIR), upload)
    _log.info("published %d files from %d parts", copied, len(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
