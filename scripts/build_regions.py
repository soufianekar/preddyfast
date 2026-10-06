"""Rebuild src/data/ar6_regions.json from the IPCC WGI Atlas reference regions.

Source: IPCC-WG1/Atlas, reference-regions/IPCC-WGI-reference-regions-v4_coordinates.csv
(Iturbide et al. 2020, the CMIP6 / AR6 reference regions). Each row holds a
region's continent, surface type, name, acronym, and polygon vertices as
"lon|lat" pairs.

    python scripts/build_regions.py [path-or-url-to-csv]
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/IPCC-WG1/Atlas/main/reference-regions/"
    "IPCC-WGI-reference-regions-v4_coordinates.csv"
)
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "data" / "ar6_regions.json"


def _read(source: str) -> str:
    if source.startswith("http"):
        with urllib.request.urlopen(source, timeout=60) as response:  # noqa: S310
            return response.read().decode()
    return Path(source).read_text()


def main(source: str = SOURCE_URL) -> None:
    rows = list(csv.reader(io.StringIO(_read(source))))[1:]
    regions = []
    for continent, surface, name, acronym, *vertices in rows:
        points = [
            [float(lon), float(lat)]
            for lon, lat in (v.split("|") for v in vertices if v.strip())
        ]
        regions.append(
            {
                "acronym": acronym.strip(),
                # "N.W.North-America" -> "N.W. North America"
                "name": re.sub(r"\.(?=[A-Z][a-z])", ". ", name.strip().replace("-", " ")),
                "continent": continent.strip().title(),
                "surface": surface.strip(),
                "vertices": points,
            }
        )
    OUTPUT.write_text(json.dumps({"source": SOURCE_URL, "regions": regions}, indent=1))
    print(f"wrote {len(regions)} regions to {OUTPUT}")


if __name__ == "__main__":
    main(*sys.argv[1:])
