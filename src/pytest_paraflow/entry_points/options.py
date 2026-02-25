"""Parse and validate pytest paraflow plugin options."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pytest_paraflow.domains.distribution.exceptions import (
    InvalidShardingConfigurationError,
    MissingShardingConfigurationError,
)


@dataclass(frozen=True)
class ParaflowOptions:
    """Validated plugin options extracted from pytest config."""

    shard_id: int
    num_shards: int | None
    target_shard_size: int | None
    group_markers: tuple[str, ...]


def resolve_paraflow_options(config: pytest.Config) -> ParaflowOptions | None:
    """Return validated plugin options or ``None`` when plugin is disabled."""
    shard_id = config.getoption("paraflow_shard_id")
    num_shards = config.getoption("paraflow_num_shards")
    target_shard_size = config.getoption("paraflow_target_shard_size")
    group_markers_raw = config.getoption("paraflow_group_marker") or []

    if shard_id is None and num_shards is None and target_shard_size is None:
        return None

    if shard_id is None:
        raise pytest.UsageError(
            "--paraflow-shard-id is required when sharding options are provided"
        )

    if num_shards is None and target_shard_size is None:
        raise pytest.UsageError(
            "Either --paraflow-num-shards or --paraflow-target-shard-size is required"
        )

    if shard_id < 0:
        raise pytest.UsageError("--paraflow-shard-id must be greater than or equal to 0")

    if num_shards is not None and num_shards <= 0:
        raise pytest.UsageError("--paraflow-num-shards must be greater than 0")

    if target_shard_size is not None and target_shard_size <= 0:
        raise pytest.UsageError("--paraflow-target-shard-size must be greater than 0")

    deduplicated_group_markers = tuple(dict.fromkeys(group_markers_raw))
    return ParaflowOptions(
        shard_id=shard_id,
        num_shards=num_shards,
        target_shard_size=target_shard_size,
        group_markers=deduplicated_group_markers,
    )


def validate_shard_range(shard_id: int, total_shards: int) -> None:
    """Validate that the configured shard id is inside shard range."""
    if shard_id >= total_shards:
        raise pytest.UsageError(
            f"--paraflow-shard-id must be in range [0, {total_shards - 1}]"
        )


def raise_usage_from_domain_error(error: Exception) -> None:
    """Translate domain-level validation errors into pytest usage errors."""
    if isinstance(
        error,
        (InvalidShardingConfigurationError, MissingShardingConfigurationError),
    ):
        raise pytest.UsageError(str(error)) from error
    raise error
