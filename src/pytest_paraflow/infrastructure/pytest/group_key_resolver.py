"""Resolve domain grouping keys from pytest items and marker configuration."""

from __future__ import annotations

from collections.abc import Sequence

from _pytest.mark.structures import Mark
from _pytest.nodes import Item


class MarkerGroupKeyResolver:
    """Resolves grouping keys based on configured marker names."""

    _keyword_candidates = ("group", "id", "name", "key", "value")

    def resolve(self, item: Item, group_markers: Sequence[str]) -> str:
        """Return a grouping key for the given item."""
        for marker_name in group_markers:
            marker = item.get_closest_marker(marker_name)
            if marker is None:
                continue
            marker_value = self._extract_marker_value(marker)
            if marker_value is None:
                return f"marker:{marker_name}"
            return f"marker:{marker_name}:{marker_value}"

        return f"node:{item.nodeid}"

    def _extract_marker_value(self, marker: Mark) -> str | None:
        """Extract an optional marker value used as group identifier."""
        if marker.args:
            return str(marker.args[0])

        for key in self._keyword_candidates:
            if key in marker.kwargs:
                return str(marker.kwargs[key])

        return None
