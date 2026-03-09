from __future__ import annotations


def modelling_context(
    calendars: dict | None = None,
    variables: dict | None = None,
) -> dict:
    """Create a modelling context with calendars and external regressors.

    Args:
        calendars: Dict mapping names to calendar objects (NationalCalendar, etc.).
        variables: Dict mapping group names to dicts of named time series.

    Returns:
        Dict with ``"calendars"`` and ``"variables"`` keys.
    """
    if calendars is None:
        calendars = {}
    if variables is None:
        variables = {}
    return {"calendars": calendars, "variables": variables}
