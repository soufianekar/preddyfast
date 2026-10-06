"""Model choices for the gridded views, grouped by producing centre."""

from __future__ import annotations

from utils.settings import GRID_STORES


def model_label(key: str) -> str:
    """'NOAA GEFS · 31 members · 0.25°', or '· deterministic ·' for one run."""
    spec = GRID_STORES[key]
    size = "deterministic" if spec["kind"] == "deterministic" else f"{spec['members']} members"
    return f"{spec['label']} · {size} · {spec['res_deg']:g}°"


def model_options(disabled: frozenset[str] = frozenset()) -> list[dict]:
    """`dmc.MultiSelect` data: one group per family, in settings order.

    `disabled` greys out stores that cannot serve the current choice (e.g.
    a variable they do not archive) while keeping them visible.
    """
    groups: dict[str, list[dict]] = {}
    for key, spec in GRID_STORES.items():
        item = {"value": key, "label": model_label(key)}
        if key in disabled:
            item["disabled"] = True
        groups.setdefault(spec["family"], []).append(item)
    return [{"group": family, "items": items} for family, items in groups.items()]
