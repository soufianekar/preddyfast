"""Precompute preset animations and station scores; optionally upload them to R2.

Runs outside the app (cron, or the GitHub Actions workflow once hosting is
decided), so it has no gunicorn timeout and never forks the app's workers.
Each store is refreshed only when the manifest does not already hold its
newest finished run, so running it hourly costs a few metadata reads.

    python scripts/precompute.py --estimate          # MB per store/region, reads nothing
    python scripts/precompute.py                     # all preset stores, animations only
    python scripts/precompute.py --stores ifs_ens --scores
    python scripts/precompute.py --upload            # then copy to R2 (env below)

Output goes to PRECOMPUTE_DIR (default .precomputed) and SCORES_DIR. Upload
reads R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import flask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.cache import cache
from utils.settings import (
    PRECOMPUTE_STATIONS,
    PRECOMPUTE_STORES,
    SCORES_DIR,
)

_log = logging.getLogger("precompute")


def _r2_client():
    import boto3  # only the upload needs it

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stores", nargs="+", default=list(PRECOMPUTE_STORES))
    parser.add_argument("--regions", nargs="+", help="limit to these preset regions")
    parser.add_argument("--variables", nargs="+", help="limit to these preset variables")
    parser.add_argument("--estimate", action="store_true", help="print read volumes and exit")
    parser.add_argument("--force", action="store_true", help="re-render runs already done")
    parser.add_argument("--no-animations", action="store_true")
    parser.add_argument("--scores", action="store_true", help="also score PRECOMPUTE_STATIONS")
    parser.add_argument("--upload", action="store_true", help="copy outputs to R2 afterwards")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    app = flask.Flask("precompute")
    cache.init_app(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})
    with app.app_context():
        from utils import precomputed
        from utils.evaluation import refresh_scores

        stores = tuple(args.stores)
        # Narrow the preset matrix for a trial run (the manifest records what was done).
        if args.regions:
            precomputed.PRECOMPUTE_REGIONS = tuple(args.regions)
        if args.variables:
            precomputed.PRECOMPUTE_VARIABLES = {
                v: a for v, a in precomputed.PRECOMPUTE_VARIABLES.items() if v in args.variables
            }
        if args.estimate:
            for store, region, mb in precomputed.estimate(stores):
                note = " (shares its source's reads)" if mb == 0 else ""
                print(f"{store:12s} {region:18s} ~{mb:7,.0f} MB per variable{note}")
            return 0

        started = time.time()
        if not args.no_animations:
            manifest = precomputed.load_manifest()
            written = precomputed.refresh(stores, manifest, force=args.force)
            _log.info("%d animations written", written)
        if args.scores:
            done = refresh_scores(PRECOMPUTE_STATIONS, stores)
            _log.info("scores: %d evaluations persisted", done)
        if args.upload:
            sent = precomputed.upload(_r2_client(), os.environ["R2_BUCKET"], roots=(Path(SCORES_DIR),))
            _log.info("uploaded %d files", sent)
        _log.info("done in %.0f s", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
