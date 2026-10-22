"""Mirror the scheduled job's working state to remote storage and back.

The station scorecard keeps per-init rows, the frozen network and its window
on disk. CI runners start empty, so `pull` restores that directory before the
job and `push` sends back what changed. A JSON manifest of SHA-256 hashes and
a generation counter make the mirror checkable: push refuses unless this
directory was pulled successfully and nobody else pushed since. A failed pull
followed by a push would otherwise delete every remote row the empty directory
"lost".

A push must never leave the remote unpullable: the HF backend commits files
and manifest in one atomic commit, so an interrupted push leaves the previous
state whole.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

MARKER = ".state-pulled.json"
# Temporaries the writers rename into place (`scores_db` writes `*.<pid>.part`).
_SKIP_SUFFIXES = (".part", ".tmp")


class StateError(RuntimeError):
    """The mirror cannot proceed safely; the remote state is left untouched."""


class Backend(Protocol):
    def read_manifest(self) -> dict | None: ...

    def download_all(self, wanted: dict[str, str], root: Path) -> None: ...

    def apply(
        self, uploads: dict[str, Path], deletes: list[str], manifest: dict
    ) -> None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_state(path: Path, root: Path) -> bool:
    """A file the mirror owns: not a temporary, not a dotfile (marker, .DS_Store)."""
    parts = path.relative_to(root).parts
    return (
        path.is_file()
        and not any(part.startswith(".") for part in parts)
        and not path.name.endswith(_SKIP_SUFFIXES)
    )


def local_files(root: Path) -> dict[str, str]:
    """Relative POSIX path -> sha256 of every state file under `root`."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if _is_state(path, root)
    }


def pull(root: Path, backend: Backend) -> int:
    """Restore `root` from the remote; returns the number of files in the state.

    No remote manifest is a first run (generation 0): local files are kept so
    they can seed the remote. Otherwise `root` is made to match the manifest,
    including removing state files the remote no longer has. The marker is
    written only after every file has arrived and matched its hash.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).unlink(missing_ok=True)
    remote = backend.read_manifest()
    manifest = remote or {"generation": 0, "files": {}}
    have = local_files(root)
    wanted = {
        rel: sha for rel, sha in manifest["files"].items() if have.get(rel) != sha
    }
    if wanted:
        backend.download_all(wanted, root)
    for rel, sha in wanted.items():
        if not (root / rel).exists() or _sha256(root / rel) != sha:
            raise StateError(f"Hash mismatch on {rel} after download.")
    if remote is not None:
        for rel in set(have) - set(manifest["files"]):
            (root / rel).unlink()
    (root / MARKER).write_text(json.dumps({"generation": manifest["generation"]}))
    return len(manifest["files"])


def push(root: Path, backend: Backend) -> tuple[int, int]:
    """Send what changed since the pull; returns (uploaded, deleted)."""
    marker = root / MARKER
    if not marker.exists():
        raise StateError("No successful pull in this directory; refusing to push.")
    pulled = json.loads(marker.read_text())["generation"]
    remote = backend.read_manifest() or {"generation": 0, "files": {}}
    if remote["generation"] != pulled:
        raise StateError(
            f"Remote generation {remote['generation']} differs from pulled "
            f"{pulled}; another job pushed since. Pull again."
        )
    local = local_files(root)
    uploads = {
        rel: root / rel for rel, sha in local.items() if remote["files"].get(rel) != sha
    }
    deletes = sorted(set(remote["files"]) - set(local))
    if not uploads and not deletes:
        return 0, 0
    manifest = {"generation": pulled + 1, "files": local}
    backend.apply(uploads, deletes, manifest)
    marker.write_text(json.dumps({"generation": pulled + 1}))
    return len(uploads), len(deletes)


class HfBackend:
    """A private Hugging Face Dataset; files and manifest land in one commit."""

    def __init__(self, api: Any, repo_id: str) -> None:
        self.api, self.repo_id = api, repo_id

    def read_manifest(self) -> dict | None:
        import tempfile

        from huggingface_hub.errors import EntryNotFoundError

        # A missing Dataset is not caught: it is a configuration error, and a
        # first run must not be mistaken for it.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename="manifest.json",
                    repo_type="dataset",
                    local_dir=tmp,
                )
            except EntryNotFoundError:
                return None
            return json.loads(Path(path).read_text())

    def download_all(self, wanted: dict[str, str], root: Path) -> None:
        """One snapshot for every wanted file: per-file requests hit Hub rate limits."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.api.snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=tmp,
                allow_patterns=list(wanted),
            )
            for rel in wanted:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(Path(tmp) / rel, root / rel)

    def apply(
        self, uploads: dict[str, Path], deletes: list[str], manifest: dict
    ) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        # No create_repo: the Dataset exists already and the fine-grained token
        # (read + write on the two Datasets only) cannot create repositories.
        ops = [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path))
            for rel, path in uploads.items()
        ]
        ops += [CommitOperationDelete(path_in_repo=rel) for rel in deletes]
        ops.append(
            CommitOperationAdd(
                path_in_repo="manifest.json",
                path_or_fileobj=json.dumps(manifest).encode(),
            )
        )
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Scorecard state generation {manifest['generation']}",
        )
        # Rows are rewritten daily; old versions would otherwise pile up in storage.
        self.api.super_squash_history(repo_id=self.repo_id, repo_type="dataset")
