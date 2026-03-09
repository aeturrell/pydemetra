from __future__ import annotations

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper


def enum_extract(enum_type: EnumTypeWrapper, value: int) -> str:
    """Extract the enum name from an integer value, stripping the prefix before ``_``.

    Args:
        enum_type (EnumTypeWrapper): A protobuf enum type wrapper.
        value (int): The integer value of the enum.

    Returns:
        str: The portion of the name after the first underscore.
    """
    name = enum_type.Name(value)
    idx = name.index("_")
    return name[idx + 1 :]


def enum_of(enum_type: EnumTypeWrapper, code: str, prefix: str) -> int:
    """Look up the integer value of an enum from its short code and prefix.

    Args:
        enum_type (EnumTypeWrapper): A protobuf enum type wrapper.
        code (str): The short name (e.g. ``"FIXED"``).
        prefix (str): The prefix (e.g. ``"PARAMETER"``).

    Returns:
        int: The integer value of the enum.
    """
    return enum_type.Value(f"{prefix}_{code}")


def enum_sextract(enum_type: EnumTypeWrapper, value: int) -> str:
    """Extract the full enum name from an integer value."""
    return enum_type.Name(value)


def enum_sof(enum_type: EnumTypeWrapper, code: str) -> int:
    """Look up the integer value of an enum from its full name."""
    return enum_type.Value(code)
