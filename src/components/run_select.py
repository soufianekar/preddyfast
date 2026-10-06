"""Forecast-run picker for the Weather views: latest, or any archived run.

A past run is chosen as a date on a calendar plus an init hour, instead of
one long list. Both come from the store itself (`earthmover_api.run_calendar`):
dates with no servable run are disabled on the calendar, and only the hours
the store holds for the chosen day are enabled — IFS ENS has 00 UTC only,
AIFS ENS four a day. A run still being written is shown but disabled.

The availability is filled by a callback when the page mounts rather than in
the layout, because it opens the Icechunk store (~5 s cold) and a layout
function must return immediately.

The composed value — `"latest"` or an ISO init `YYYY-MM-DDTHH:MM` — lives in
a hidden input with the picker's id, so pages and `url_sync` read
`Input(select_id, "value")` as before.
"""

from __future__ import annotations

import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, no_update

from utils.earthmover_api import (
    GridDataError,
    common_run_calendar,
    run_calendar,
    run_in_calendar,
)
from utils.settings import GRID_STORE_DEFAULT, GRID_STORES

LATEST = "latest"
_PAST = "past"


def _parse_run(value: str | None) -> pd.Timestamp | None:
    """The init time a run value names, or None for latest or junk."""
    if not value or value == LATEST:
        return None
    try:
        return pd.Timestamp(value)
    except ValueError:
        return None


def pick_hour(day_hours: list[int], current: str | None, snap: bool) -> str | None:
    """The init hour to show after a date or model change.

    Without `snap` the current hour stands, even if that day lacks it, so a
    run from the URL is reported by the fetch rather than swapped. With it, an
    hour the day has is kept, else the day's newest; a day with no run has
    none.
    """
    if not snap:
        return current
    if current is not None and int(current) in day_hours:
        return current
    return f"{max(day_hours):02d}" if day_hours else None


def run_select(
    select_id: str, latest_label: str, description: str, value: str
) -> dmc.Stack:
    """Latest/past switch, init-date calendar and init-hour select.

    Parameters
    ----------
    select_id
        Id of the hidden input carrying the composed run value. The visible
        controls are `{select_id}-mode`, `-date` and `-hour`.
    latest_label
        Short label for the latest-run segment.
    description
        Page-specific note shown under the switch.
    value
        Initial run value, from the URL.
    """
    init = _parse_run(value)
    past = init is not None
    return dmc.Stack(
        [
            dcc.Input(id=select_id, type="hidden", value=value),
            dcc.Store(id=f"{select_id}-calendar"),
            dmc.Text("Forecast run", size="sm", fw=500),
            dmc.SegmentedControl(
                id=f"{select_id}-mode",
                data=[
                    {"value": LATEST, "label": latest_label},
                    {"value": _PAST, "label": "Past run"},
                ],
                value=_PAST if past else LATEST,
                fullWidth=True,
                size="xs",
            ),
            dmc.Text(description, size="xs", c="dimmed"),
            dmc.Group(
                [
                    dmc.DatePickerInput(
                        id=f"{select_id}-date",
                        label="Init date",
                        value=f"{init:%Y-%m-%d}" if past else None,
                        valueFormat="ddd D MMM YYYY",
                        clearable=False,
                        size="xs",
                        style={"flex": 1},
                    ),
                    dmc.Select(
                        id=f"{select_id}-hour",
                        label="Init (UTC)",
                        data=[f"{init:%H}"] if past else [],
                        value=f"{init:%H}" if past else None,
                        allowDeselect=False,
                        size="xs",
                        w=110,
                    ),
                ],
                id=f"{select_id}-past",
                gap="xs",
                align="flex-end",
                wrap="nowrap",
                display="flex" if past else "none",
            ),
            dmc.Text(id=f"{select_id}-avail", size="xs", c="dimmed"),
        ],
        gap=6,
    )


def register_run_options(
    select_id: str,
    store: str = GRID_STORE_DEFAULT,
    store_input: str | tuple[str, str] | None = None,
) -> None:
    """Wire a run picker to a store's available runs.

    Parameters
    ----------
    select_id
        As passed to `run_select`.
    store
        `GRID_STORES` key, when the page reads one fixed store.
    store_input
        Id of a control whose `value` is a `GRID_STORES` key or a list of
        them, or an `(id, property)` pair, when the page lets the user pick
        the models. The calendar then follows it — for several models, only
        the inits they all hold — and a date or hour the new choice lacks
        moves to its newest run.
    """
    inputs = [Input(f"{select_id}-mode", "id")]
    input_id = None
    if store_input:
        input_id, prop = (
            store_input if isinstance(store_input, tuple) else (store_input, "value")
        )
        inputs.append(Input(input_id, prop))

    @callback(
        Output(f"{select_id}-calendar", "data"),
        Output(f"{select_id}-date", "minDate"),
        Output(f"{select_id}-date", "maxDate"),
        Output(f"{select_id}-date", "disabledDates"),
        Output(f"{select_id}-avail", "children"),
        *inputs,
        State(select_id, "value"),
    )
    def _fill(_, *args):
        *rest, current = args
        chosen = rest[0] if rest else None
        keys = chosen if isinstance(chosen, list) else [chosen]
        keys = sorted(k for k in keys if k in GRID_STORES) or [store]
        label = " + ".join(GRID_STORES[k]["label"] for k in keys)
        try:
            if len(keys) == 1:
                cal = run_calendar(keys[0])
            else:
                cal = common_run_calendar(tuple(keys))
        except GridDataError as exc:
            return no_update, no_update, no_update, no_update, str(exc)
        # Only a model switch by the user re-snaps the date and hour; on
        # mount the URL's run is kept so the fetch can report it.
        # A model switch by the user re-snaps the date and hour, and so does a
        # run (e.g. from the URL) the models cannot serve together: the data
        # callbacks wait for a run in the calendar rather than fetch a wrong one.
        switched = input_id is not None and ctx.triggered_id == input_id
        cal = {**cal, "snap": switched or not run_in_calendar(cal, current)}
        hours = "/".join(f"{h:02d}" for h in cal["hours"]) or "no"
        first, last = pd.Timestamp(cal["first"]), pd.Timestamp(cal["last"])
        common = " in common" if len(keys) > 1 else ""
        avail = (
            f"{label} archive: {hours} UTC runs{common}, "
            f"{first:%-d %b %Y} – {last:%-d %b %Y}."
        )
        if cal["unfinished"]:
            avail += f" {len(cal['unfinished'])} newest still arriving."
        return cal, cal["first"], cal["last"], cal["missing"], avail

    @callback(
        Output(f"{select_id}-past", "display"),
        Output(f"{select_id}-date", "value"),
        Input(f"{select_id}-mode", "value"),
        Input(f"{select_id}-calendar", "data"),
        State(f"{select_id}-date", "value"),
    )
    def _mode(mode, cal, date):
        """Show the past-run controls; default the date to the newest servable day."""
        display = "flex" if mode == _PAST else "none"
        if mode != _PAST or not cal or not cal["days"]:
            return display, no_update
        if not date or (cal.get("snap") and date[:10] not in cal["days"]):
            return display, max(cal["days"])
        return display, no_update

    @callback(
        Output(f"{select_id}-hour", "data"),
        Output(f"{select_id}-hour", "value"),
        Input(f"{select_id}-date", "value"),
        Input(f"{select_id}-calendar", "data"),
        State(f"{select_id}-hour", "value"),
    )
    def _hours(date, cal, hour):
        """Every cycle hour of the store, enabled only where that day has a run.

        A date the user picks keeps the hour if that day has it, else takes
        the day's newest. On mount the URL's hour is left alone even if it is
        unavailable, so the fetch reports it rather than it being swapped.
        """
        if not date or not cal:
            return no_update, no_update
        day = date[:10]
        have = set(cal["days"].get(day, []))
        unfinished = {t[11:13] for t in cal["unfinished"] if t[:10] == day}
        data = [
            {
                "value": f"{h:02d}",
                "label": f"{h:02d}:00"
                + (" · arriving" if f"{h:02d}" in unfinished else ""),
                "disabled": h not in have,
            }
            for h in cal["hours"]
        ]
        snap = ctx.triggered_id == f"{select_id}-date" or bool(cal.get("snap"))
        picked = pick_hour(sorted(have), hour, snap)
        return data, picked if snap else no_update

    @callback(
        Output(select_id, "value"),
        Input(f"{select_id}-mode", "value"),
        Input(f"{select_id}-date", "value"),
        Input(f"{select_id}-hour", "value"),
        State(select_id, "value"),
    )
    def _compose(mode, date, hour, current):
        """The single run value pages read: `latest` or `YYYY-MM-DDTHH:00`."""
        if mode != _PAST:
            value = LATEST
        elif date and hour is not None:
            value = f"{date[:10]}T{hour}:00"
        else:
            return no_update
        return no_update if value == current else value
