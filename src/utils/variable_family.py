"""Carry a variable choice between pages whose catalogues differ (no Dash).

The shared `variable-store` holds `{"family": str | None, "keys": {ns: key}}`:
the family of the last explicit choice, plus the last key chosen in each
namespace while that family held. Keeping per-namespace keys means Max
Temperature on the climate page survives a round-trip through the regional
page, which only has instantaneous 2 m temperature.
"""

from __future__ import annotations

from utils.settings import VARIABLE_FAMILIES


def family_of(namespace: str, key: str) -> str | None:
    """Family a variable belongs to, or None if it is page-specific."""
    for family, members in VARIABLE_FAMILIES.items():
        if key in members.get(namespace, ()):
            return family
    return None


def publish(store: dict | None, namespace: str, key: str) -> dict:
    """New store contents after the user picks `key` on a `namespace` page."""
    family = family_of(namespace, key)
    keys = {namespace: key}
    if store and family and store.get("family") == family:
        keys = {**store.get("keys", {}), **keys}
    return {"family": family, "keys": keys}


def resolve(store: dict | None, namespace: str) -> str | None:
    """Variable a `namespace` page should open with, or None for its default."""
    if not store:
        return None
    if key := store.get("keys", {}).get(namespace):
        return key
    members = VARIABLE_FAMILIES.get(store.get("family") or "", {}).get(namespace)
    return members[0] if members else None
