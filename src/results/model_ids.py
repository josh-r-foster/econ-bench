"""Stable conversion between semantic model identifiers and path components."""

import re


_PORTABLE_LITERAL = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ENCODED_COMPONENT = re.compile(r"^~(?:[0-9a-f]{2})+$")
_WINDOWS_RESERVED_STEMS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _is_portable_literal(value: str) -> bool:
    if not _PORTABLE_LITERAL.fullmatch(value) or value.endswith("."):
        return False
    return value.split(".", 1)[0] not in _WINDOWS_RESERVED_STEMS


def model_id_to_path_component(model_id: str) -> str:
    """Return the canonical portable path component for a model identifier."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    if not model_id:
        raise ValueError("model_id must not be empty")

    if _is_portable_literal(model_id):
        return model_id

    return f"~{model_id.encode('utf-8').hex()}"


def model_id_from_path_component(component: str) -> str:
    """Recover a model identifier from its canonical path component."""
    if not isinstance(component, str):
        raise TypeError("component must be a string")
    if not component:
        raise ValueError("component must not be empty")

    if _is_portable_literal(component):
        return component

    if not _ENCODED_COMPONENT.fullmatch(component):
        raise ValueError("component is not a canonical model path component")

    try:
        model_id = bytes.fromhex(component[1:]).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("component does not contain a UTF-8 model identifier") from error

    if model_id_to_path_component(model_id) != component:
        raise ValueError("component is a noncanonical model path alias")
    return model_id
