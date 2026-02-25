"""Stable hashing utilities used for deterministic shard assignment."""

from __future__ import annotations

import hashlib


def stable_hash_to_int(value: str) -> int:
    """Return a deterministic positive integer hash for a string value."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
