"""Domain models for test collection grouping and sharding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectedTest:
    """Immutable representation of a collected test case."""

    test_id: str
    group_key: str
