import hashlib
import json
from pathlib import Path

import pytest

from utils import state_store as ss


class FakeBackend:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.manifest: dict | None = None

    def read_manifest(self):
        return None if self.manifest is None else json.loads(json.dumps(self.manifest))

    def download(self, rel, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[rel])

    def apply(self, uploads, deletes, manifest):
        for rel, path in uploads.items():
            self.files[rel] = path.read_bytes()
        for rel in deletes:
            self.files.pop(rel, None)
        self.manifest = manifest


def _write(root: Path, rel: str, data: bytes) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)


def test_first_run_round_trip(tmp_path):
    remote, a, b = FakeBackend(), tmp_path / "a", tmp_path / "b"
    assert ss.pull(a, remote) == 0
    _write(a, "ifs_ens/temperature_2m/20261020.parquet", b"rows")
    _write(a, "stations.parquet", b"net")
    _write(a, "x.parquet.123.part", b"junk")
    assert ss.push(a, remote) == (2, 0)
    assert remote.manifest["generation"] == 1
    assert ss.pull(b, remote) == 2
    assert (b / "stations.parquet").read_bytes() == b"net"
    assert not (b / "x.parquet.123.part").exists()


def test_push_refuses_without_marker(tmp_path):
    remote = FakeBackend()
    remote.manifest = {"generation": 3, "files": {"keep.parquet": "h"}}
    with pytest.raises(ss.StateError, match="pull"):
        ss.push(tmp_path, remote)
    assert remote.manifest["generation"] == 3


def test_push_refuses_on_generation_change(tmp_path):
    remote = FakeBackend()
    ss.pull(tmp_path, remote)
    remote.manifest = {"generation": 5, "files": {}}  # someone else pushed
    _write(tmp_path, "a.parquet", b"1")
    with pytest.raises(ss.StateError, match="generation"):
        ss.push(tmp_path, remote)


def test_pull_hash_mismatch_leaves_no_marker(tmp_path):
    remote = FakeBackend()
    remote.files["a.parquet"] = b"actual"
    remote.manifest = {
        "generation": 1,
        "files": {"a.parquet": hashlib.sha256(b"other").hexdigest()},
    }
    with pytest.raises(ss.StateError, match="a.parquet"):
        ss.pull(tmp_path, remote)
    assert not (tmp_path / ss.MARKER).exists()


def test_push_deletes_pruned_only(tmp_path):
    remote, root = FakeBackend(), tmp_path / "s"
    ss.pull(root, remote)
    for name in ("old.parquet", "new.parquet"):
        _write(root, name, name.encode())
    ss.push(root, remote)
    ss.pull(root, remote)
    (root / "old.parquet").unlink()
    _write(root, "newer.parquet", b"n")
    assert ss.push(root, remote) == (1, 1)
    assert set(remote.files) == {"new.parquet", "newer.parquet"}


def test_unchanged_files_not_reuploaded(tmp_path):
    remote = FakeBackend()
    ss.pull(tmp_path, remote)
    _write(tmp_path, "a.parquet", b"1")
    ss.push(tmp_path, remote)
    ss.pull(tmp_path, remote)
    assert ss.push(tmp_path, remote) == (0, 0)


class FakeHfApi:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.commits = 0
        self.squashed = 0

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        from huggingface_hub import CommitOperationAdd

        self.commits += 1
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                src = op.path_or_fileobj
                self.files[op.path_in_repo] = (
                    Path(src).read_bytes() if isinstance(src, str) else src
                )
            else:
                self.files.pop(op.path_in_repo, None)

    def hf_hub_download(self, repo_id, filename, repo_type, local_dir):
        from huggingface_hub.errors import EntryNotFoundError

        if filename not in self.files:
            raise EntryNotFoundError(filename)
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[filename])
        return str(out)

    def super_squash_history(self, repo_id, repo_type):
        self.squashed += 1


def test_hf_backend_one_commit_per_push(tmp_path):
    api = FakeHfApi()
    backend = ss.HfBackend(api, "me/preddyfast-state")
    ss.pull(tmp_path / "a", backend)
    _write(tmp_path / "a", "ifs_ens/t2m/20261020.parquet", b"r")
    assert ss.push(tmp_path / "a", backend) == (1, 0)
    assert api.commits == 1 and api.squashed == 1
    assert set(api.files) == {"ifs_ens/t2m/20261020.parquet", "manifest.json"}
    assert ss.pull(tmp_path / "b", backend) == 1
    assert (tmp_path / "b" / "ifs_ens/t2m/20261020.parquet").read_bytes() == b"r"
    (tmp_path / "a" / "ifs_ens/t2m/20261020.parquet").unlink()
    assert ss.push(tmp_path / "a", backend) == (0, 1)
    assert set(api.files) == {"manifest.json"}
