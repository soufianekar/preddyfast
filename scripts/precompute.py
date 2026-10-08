"""Precompute preset animations and the station scorecard; publish them.

Runs outside the app (cron, or the GitHub Actions workflow once hosting is
decided), so it has no gunicorn timeout and never forks the app's workers.
Each store is refreshed only when the manifest does not already hold its
newest finished run, so running it hourly costs a few metadata reads.

    python scripts/precompute.py --estimate          # MB per store/region, reads nothing
    python scripts/precompute.py                     # all preset stores, animations only
    python scripts/precompute.py --no-animations --scorecard   # newest inits first
    python scripts/precompute.py --no-animations --scorecard --backfill \
        --max-minutes 300
    python scripts/precompute.py --no-animations --maps   # ERA5 maps gallery (needs CDS)
    python scripts/precompute.py --upload-hf         # then publish to the HF Dataset

Output goes to PRECOMPUTE_DIR (default .precomputed; the scorecard's published
aggregates under eval/scorecard/, its working state in SCORECARD_DIR).
With PRECOMPUTE_HF_REPO set, the Dataset's manifests are mirrored first so
products already current are skipped; --upload-hf publishes the tree there and
prunes superseded runs (needs HF_TOKEN with write access). --upload copies to
R2 instead (R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET).
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
from utils.settings import PRECOMPUTE_STORES

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
    parser.add_argument(
        "--scorecard",
        action="store_true",
        help="refresh the station scorecard (rolling window) and its aggregates",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="scorecard: walk the window oldest init first",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="scorecard: stop cleanly between inits after this long; rerun to resume",
    )
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="scorecard: select a new station network (re-reads every init)",
    )
    parser.add_argument(
        "--aggregates-only",
        action="store_true",
        help="rebuild only the scorecard aggregates file from stored rows",
    )
    parser.add_argument(
        "--maps",
        nargs="*",
        metavar="ITEM",
        help="draw the ERA5 maps gallery (all, or these items)",
    )
    parser.add_argument(
        "--upload", action="store_true", help="copy outputs to R2 afterwards"
    )
    parser.add_argument(
        "--upload-hf", action="store_true", help="publish outputs to PRECOMPUTE_HF_REPO"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    app = flask.Flask("precompute")
    cache.init_app(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})
    with app.app_context():
        from utils import hub_sync, precomputed

        # One mirror of what is already published, then none: this run's new
        # manifests must not be overwritten by the Dataset's older ones.
        hub_sync.sync(force=True)
        hub_sync.freeze()

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
        if not (args.no_animations or args.aggregates_only):
            manifest = precomputed.load_manifest()
            written = precomputed.refresh(stores, manifest, force=args.force)
            _log.info("%d animations written", written)
        if args.aggregates_only:
            import pandas as pd

            from utils import scorecard

            path = scorecard.write_aggregates(
                scorecard.window_anchor(pd.Timestamp.now("UTC").tz_localize(None))
            )
            _log.info("scorecard aggregates: %s", path)
        if args.scorecard:
            import pandas as pd

            from utils import scorecard
            from utils.settings import SCORECARD_STORES, SCORECARD_VARIABLES

            # Fixed models, not --stores: the scorecard compares these.
            try:
                wrote = scorecard.refresh(
                    SCORECARD_STORES,
                    tuple(SCORECARD_VARIABLES),
                    pd.Timestamp.now("UTC").tz_localize(None).normalize(),
                    backfill=args.backfill,
                    max_minutes=args.max_minutes,
                    reselect=args.reselect,
                )
                _log.info("scorecard: %d init files written", wrote)
            except scorecard.ObservationError as exc:
                # GHCNh unreachable: keep the published scorecard, still publish
                # the rest.
                _log.error("scorecard not refreshed: %s", exc)
        if args.maps is not None:
            from utils import map_gallery

            drawn = map_gallery.refresh(force=args.force, keys=args.maps or None)
            _log.info("maps gallery: %d images written", drawn)
        if args.upload:
            # The scorecard aggregates live under PRECOMPUTE_DIR, so the tree
            # carries them.
            sent = precomputed.upload(_r2_client(), os.environ["R2_BUCKET"])
            _log.info("uploaded %d files", sent)
        if args.upload_hf:
            pruned = hub_sync.upload(os.environ["HF_TOKEN"])
            _log.info("published to %s, %d superseded files pruned", hub_sync.PRECOMPUTE_HF_REPO, pruned)
        _log.info("done in %.0f s", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
