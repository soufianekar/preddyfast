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

    def download_all(self, wanted, root):
        for rel in wanted:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(self.files[rel])

    def apply(self, uploads, deletes, manifest):
        for rel, path in uploads.items():
            self.files[rel] = path.read_bytes()
        for rel in deletes:
            self.files.pop(rel, None)
        self.manifest = manifest


def _write(root: Path, rel: str, data: bytes) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    remote.manifest = {"generation": 1, "files": {"a.parquet": _sha(b"other")}}
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


def test_pull_removes_local_files_the_remote_dropped(tmp_path):
    remote = FakeBackend()
    remote.files["keep.parquet"] = b"k"
    remote.manifest = {"generation": 2, "files": {"keep.parquet": _sha(b"k")}}
    _write(tmp_path, "pruned/20260901.parquet", b"stale")
    _write(tmp_path, "x.parquet.1.part", b"tmp")
    ss.pull(tmp_path, remote)
    assert not (tmp_path / "pruned/20260901.parquet").exists()
    assert (tmp_path / "keep.parquet").read_bytes() == b"k"
    assert (tmp_path / "x.parquet.1.part").exists()  # not state: left alone


def test_first_pull_keeps_local_files_to_seed_the_remote(tmp_path):
    remote = FakeBackend()
    _write(tmp_path, "seed.parquet", b"s")
    ss.pull(tmp_path, remote)
    assert ss.push(tmp_path, remote) == (1, 0)


def test_dotfiles_are_not_state(tmp_path):
    remote = FakeBackend()
    ss.pull(tmp_path, remote)
    _write(tmp_path, ".DS_Store", b"mac")
    _write(tmp_path, "gefs/.DS_Store", b"mac")
    _write(tmp_path, "a.parquet", b"1")
    assert ss.push(tmp_path, remote) == (1, 0)
    assert set(remote.files) == {"a.parquet"}


class FakeHfApi:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.commits = 0
        self.squashed = 0
        self.snapshots = 0
        self.file_downloads = 0

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

        self.file_downloads += 1
        if filename not in self.files:
            raise EntryNotFoundError(filename)
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[filename])
        return str(out)

    def snapshot_download(self, repo_id, repo_type, local_dir, allow_patterns):
        self.snapshots += 1
        for name in allow_patterns:
            out = Path(local_dir) / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self.files[name])
        return str(local_dir)

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


def test_hf_pull_fetches_all_files_in_one_snapshot(tmp_path):
    api = FakeHfApi()
    backend = ss.HfBackend(api, "me/preddyfast-state")
    a = tmp_path / "a"
    ss.pull(a, backend)
    for i in range(20):
        _write(a, f"gefs/z500/202610{i:02d}.parquet", str(i).encode())
    ss.push(a, backend)
    api.file_downloads = 0
    assert ss.pull(tmp_path / "b", backend) == 20
    assert api.snapshots == 1
    assert api.file_downloads == 1  # the manifest only
