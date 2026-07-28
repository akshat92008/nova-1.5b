from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_ALLOWLIST = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
    "0bsd",
}

_ALIASES = {
    "apache 2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache license 2.0": "apache-2.0",
    "mit license": "mit",
    "bsd 2-clause": "bsd-2-clause",
    "bsd 3-clause": "bsd-3-clause",
    "the unlicense": "unlicense",
    "cc0": "cc0-1.0",
}


def normalise_licence(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        pieces = re.split(r"[,;|]|\s+(?:OR|AND)\s+", value, flags=re.IGNORECASE)
    elif isinstance(value, Iterable):
        pieces = [str(item) for item in value]
    else:
        pieces = [str(value)]
    output: set[str] = set()
    for piece in pieces:
        item = re.sub(r"\s+", " ", piece.strip().lower())
        if not item:
            continue
        output.add(_ALIASES.get(item, item))
    return output


def allowed_licence(value: object, allowlist: set[str] | None = None) -> tuple[bool, str]:
    allowlist = {item.lower() for item in (allowlist or DEFAULT_ALLOWLIST)}
    values = normalise_licence(value)
    if not values:
        return False, ""
    accepted = sorted(values & allowlist)
    return (bool(accepted), accepted[0] if accepted else "")
