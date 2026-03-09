from __future__ import annotations

import copy


def add_outlier(
    spec: dict,
    type: str,
    date: str,
    name: str | None = None,
    coef: float = 0.0,
) -> dict:
    """Add an outlier to a RegARIMA specification.

    Args:
        spec (dict): The specification dict.
        type (str): Outlier type (``"AO"``, ``"LS"``, ``"TC"``, ``"SO"``).
        date (str): Date in ``"YYYY-MM-DD"`` format.
        name (str | None): Name for the outlier.
        coef (float): Initial coefficient value.

    Returns:
        dict: Modified specification dict.
    """
    spec = copy.deepcopy(spec)
    if "outliers" not in spec:
        spec["outliers"] = []
    if name is None:
        name = f"{type}.{date}"
    spec["outliers"].append(
        {
            "name": name,
            "type": type,
            "date": date,
            "coef": coef,
        }
    )
    return spec


def remove_outlier(
    spec: dict,
    type: str | None = None,
    date: str | None = None,
    name: str | None = None,
) -> dict:
    """Remove outliers from a RegARIMA specification.

    Args:
        spec (dict): The specification dict.
        type (str | None): Filter by outlier type.
        date (str | None): Filter by date.
        name (str | None): Filter by name.

    Returns:
        dict: Modified specification dict.
    """
    spec = copy.deepcopy(spec)
    if "outliers" not in spec:
        return spec
    spec["outliers"] = [
        o
        for o in spec["outliers"]
        if not (
            (type is None or o.get("type") == type)
            and (date is None or o.get("date") == date)
            and (name is None or o.get("name") == name)
        )
    ]
    return spec


def add_ramp(
    spec: dict,
    start: str,
    end: str,
    name: str | None = None,
    coef: float = 0.0,
) -> dict:
    """Add a ramp to a RegARIMA specification.

    Args:
        spec (dict): The specification dict.
        start (str): Start date in ``"YYYY-MM-DD"`` format.
        end (str): End date in ``"YYYY-MM-DD"`` format.
        name (str | None): Name for the ramp.
        coef (float): Initial coefficient value.

    Returns:
        dict: Modified specification dict.
    """
    spec = copy.deepcopy(spec)
    if "ramps" not in spec:
        spec["ramps"] = []
    if name is None:
        name = f"ramp.{start}.{end}"
    spec["ramps"].append(
        {
            "name": name,
            "start": start,
            "end": end,
            "coef": coef,
        }
    )
    return spec


def remove_ramp(
    spec: dict,
    start: str | None = None,
    end: str | None = None,
    name: str | None = None,
) -> dict:
    """Remove ramps from a RegARIMA specification.

    Args:
        spec (dict): The specification dict.
        start (str | None): Filter by start date.
        end (str | None): Filter by end date.
        name (str | None): Filter by name.

    Returns:
        dict: Modified specification dict.
    """
    spec = copy.deepcopy(spec)
    if "ramps" not in spec:
        return spec
    spec["ramps"] = [
        r
        for r in spec["ramps"]
        if not (
            (start is None or r.get("start") == start)
            and (end is None or r.get("end") == end)
            and (name is None or r.get("name") == name)
        )
    ]
    return spec
