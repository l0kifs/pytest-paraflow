"""Map pytest items into domain test models."""

from __future__ import annotations

from collections.abc import Sequence

from _pytest.nodes import Item

from pytest_paraflow.domains.distribution.models import CollectedTest
from pytest_paraflow.infrastructure.pytest.group_key_resolver import MarkerGroupKeyResolver


class PytestItemMapper:
    """Maps pytest collected items to domain-level test models."""

    def __init__(self, group_key_resolver: MarkerGroupKeyResolver) -> None:
        self._group_key_resolver = group_key_resolver

    def to_collected_tests(
        self,
        items: Sequence[Item],
        group_markers: Sequence[str],
    ) -> list[CollectedTest]:
        """Convert pytest items to domain models."""
        return [
            CollectedTest(
                test_id=item.nodeid,
                group_key=self._group_key_resolver.resolve(item, group_markers),
            )
            for item in items
        ]
