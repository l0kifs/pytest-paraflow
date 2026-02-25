"""Domain services for sizing shards and assigning tests to shards."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from pytest_paraflow.domains.distribution.exceptions import (
    InvalidShardingConfigurationError,
    MissingShardingConfigurationError,
)
from pytest_paraflow.domains.distribution.models import CollectedTest


class GroupingService:
    """Groups tests by domain-provided grouping keys."""

    def group_tests(
        self,
        tests: Sequence[CollectedTest],
    ) -> dict[str, list[CollectedTest]]:
        """Return tests grouped by their group keys."""
        groups: dict[str, list[CollectedTest]] = {}
        for test in tests:
            groups.setdefault(test.group_key, []).append(test)
        return groups


class ShardSizingService:
    """Resolves total shard count from explicit or dynamic options."""

    def resolve_total_shards(
        self,
        explicit_num_shards: int | None,
        total_tests: int,
        target_shard_size: int | None,
    ) -> int:
        """Resolve shard count using explicit value or target shard size."""
        if explicit_num_shards is not None:
            if explicit_num_shards <= 0:
                raise InvalidShardingConfigurationError(
                    "--paraflow-num-shards must be greater than 0"
                )
            return explicit_num_shards

        if target_shard_size is None:
            raise MissingShardingConfigurationError(
                "Either --paraflow-num-shards or --paraflow-target-shard-size is required"
            )

        if target_shard_size <= 0:
            raise InvalidShardingConfigurationError(
                "--paraflow-target-shard-size must be greater than 0"
            )

        if total_tests <= 0:
            return 1

        return max(1, math.ceil(total_tests / target_shard_size))


class ShardAssignmentService:
    """Assigns grouped tests to shards with deterministic hashing."""

    def __init__(self, hash_provider: Callable[[str], int]) -> None:
        self._hash_provider = hash_provider

    def select_tests_for_shard(
        self,
        groups: Mapping[str, Sequence[CollectedTest]],
        shard_id: int,
        num_shards: int,
    ) -> set[str]:
        """Return test identifiers assigned to the requested shard."""
        selected_tests: set[str] = set()
        for group_key, grouped_tests in groups.items():
            group_shard = self._hash_provider(group_key) % num_shards
            if group_shard == shard_id:
                selected_tests.update(test.test_id for test in grouped_tests)
        return selected_tests
