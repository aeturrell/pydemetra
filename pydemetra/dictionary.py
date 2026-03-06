from __future__ import annotations

from pydemetra._results import proc_dictionary


def dictionary(obj: object) -> list[str]:
    """Get the list of available result items from a Java processing object.

    Args:
        obj: A Java processing result object.

    Returns:
        List of item name strings.
    """
    return proc_dictionary(obj)


def result(obj: object, item_id: str) -> object:
    """Extract a specific result from a Java processing object.

    Args:
        obj: A Java processing result object.
        item_id: Name of the item to extract.

    Returns:
        The extracted value (type depends on the item).
    """
    from pydemetra._results import proc_numeric, proc_str, proc_ts, proc_vector

    for extractor in [proc_numeric, proc_vector, proc_ts, proc_str]:
        val = extractor(obj, item_id)
        if val is not None:
            return val
    return None


def user_defined(obj: object, items: list[str] | None = None) -> dict:
    """Extract multiple user-defined results.

    Args:
        obj: A Java processing result object.
        items: List of item names to extract.

    Returns:
        Dict mapping item names to their values.
    """
    if items is None:
        return {}
    return {item: result(obj, item) for item in items}
