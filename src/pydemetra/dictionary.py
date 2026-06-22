from __future__ import annotations

from pydemetra._results import proc_dictionary


def dictionary(obj: object) -> list[str]:
    """Get the list of available result items from a Java processing object.

    Args:
        obj (object): A Java processing result object.

    Returns:
        list[str]: List of item name strings.
    """
    return proc_dictionary(obj)


def result(obj: object, item_id: str) -> object:
    """Extract a specific result from a Java processing object.

    Args:
        obj (object): A Java processing result object.
        item_id (str): Name of the item to extract.

    Returns:
        object: The extracted value (type depends on the item).
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
        obj (object): A Java processing result object.
        items (list[str] | None): List of item names to extract.

    Returns:
        dict: Dict mapping item names to their values.
    """
    if items is None:
        return {}
    return {item: result(obj, item) for item in items}
