"""Pull or push the station scorecard's working state (private HF Dataset).

    python scripts/state.py pull    # before the job
    python scripts/state.py push    # after it, even if it stopped part-way

Needs HF_TOKEN (read + write on the Dataset) and PRECOMPUTE_HF_STATE_REPO.
Exit codes: 0 done, 1 failed (remote untouched by a refused push), 2 not
configured or bad usage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import scores_db, state_store
from utils.settings import PRECOMPUTE_HF_STATE_REPO

_NEEDED = ("HF_TOKEN", "PRECOMPUTE_HF_STATE_REPO")


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("pull", "push"):
        print(__doc__)
        return 2
    missing = [name for name in _NEEDED if not os.environ.get(name)]
    if missing:
        print(f"Remote state not configured: set {', '.join(missing)}.")
        return 2
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    backend = state_store.HfBackend(
        HfApi(token=os.environ["HF_TOKEN"]), PRECOMPUTE_HF_STATE_REPO
    )
    root = scores_db.scorecard_root()
    try:
        if argv[0] == "pull":
            print(f"pulled state: {state_store.pull(root, backend)} files")
        else:
            uploaded, deleted = state_store.push(root, backend)
            print(f"pushed state: {uploaded} uploaded, {deleted} deleted")
    except (state_store.StateError, HfHubHTTPError, OSError) as exc:
        print(f"state {argv[0]} failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
