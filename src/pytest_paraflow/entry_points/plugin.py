"""Pytest plugin entry point for paraflow test sharding."""

from __future__ import annotations

import pytest
from _pytest.nodes import Item

from pytest_paraflow.config.settings import get_settings
from pytest_paraflow.domains.distribution.services import (
    GroupingService,
    ShardAssignmentService,
    ShardSizingService,
)
from pytest_paraflow.entry_points.options import (
    ParaflowOptions,
    raise_usage_from_domain_error,
    resolve_paraflow_options,
    validate_shard_range,
)
from pytest_paraflow.infrastructure.hash.hasher import stable_hash_to_int
from pytest_paraflow.infrastructure.pytest.group_key_resolver import MarkerGroupKeyResolver
from pytest_paraflow.infrastructure.pytest.item_mapper import PytestItemMapper


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register paraflow plugin command-line options."""
    settings = get_settings()
    group = parser.getgroup("paraflow")
    group.addoption(
        "--paraflow-shard-id",
        action="store",
        type=int,
        dest="paraflow_shard_id",
        default=settings.shard_id,
        help="Shard index to execute (zero-based)",
    )
    group.addoption(
        "--paraflow-num-shards",
        action="store",
        type=int,
        dest="paraflow_num_shards",
        default=settings.num_shards,
        help="Total number of shards",
    )
    group.addoption(
        "--paraflow-target-shard-size",
        action="store",
        type=int,
        dest="paraflow_target_shard_size",
        default=settings.target_shard_size,
        help=(
            "Desired tests per shard when total shard count is calculated dynamically"
        ),
    )
    group.addoption(
        "--paraflow-group-marker",
        action="append",
        dest="paraflow_group_marker",
        default=list(settings.group_marker),
        metavar="MARKER",
        help=(
            "Marker name used for grouping. Repeat option to support multiple markers. "
            "Tests with the same marker key run in the same shard."
        ),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[Item]) -> None:
    """Filter collected tests to only tests assigned to selected shard."""
    options = resolve_paraflow_options(config)
    if options is None:
        return

    mapper = PytestItemMapper(group_key_resolver=MarkerGroupKeyResolver())
    collected_tests = mapper.to_collected_tests(
        items=items,
        group_markers=options.group_markers,
    )

    sizing_service = ShardSizingService()
    try:
        total_shards = sizing_service.resolve_total_shards(
            explicit_num_shards=options.num_shards,
            total_tests=len(collected_tests),
            target_shard_size=options.target_shard_size,
        )
    except Exception as error:  # pragma: no cover - translated to usage error
        raise_usage_from_domain_error(error)
        return

    validate_shard_range(shard_id=options.shard_id, total_shards=total_shards)

    grouping_service = GroupingService()
    grouped_tests = grouping_service.group_tests(collected_tests)

    assignment_service = ShardAssignmentService(hash_provider=stable_hash_to_int)
    selected_test_ids = assignment_service.select_tests_for_shard(
        groups=grouped_tests,
        shard_id=options.shard_id,
        num_shards=total_shards,
    )

    selected_items = [item for item in items if item.nodeid in selected_test_ids]
    deselected_items = [item for item in items if item.nodeid not in selected_test_ids]

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)

    items[:] = selected_items
    setattr(
        config,
        "_paraflow_state",
        _ParaflowRuntimeState(
            shard_id=options.shard_id,
            total_shards=total_shards,
            selected=len(selected_items),
            deselected=len(deselected_items),
            options=options,
        ),
    )


def pytest_report_header(config: pytest.Config) -> str | None:
    """Display active paraflow sharding configuration in pytest header."""
    state = getattr(config, "_paraflow_state", None)
    if state is None:
        return None

    marker_info = ",".join(state.options.group_markers) or "none"
    return (
        "paraflow: "
        f"shard={state.shard_id}/{state.total_shards}, "
        f"selected={state.selected}, deselected={state.deselected}, "
        f"group_markers={marker_info}"
    )


class _ParaflowRuntimeState:
    """Runtime details used for reporting in pytest header."""

    def __init__(
        self,
        shard_id: int,
        total_shards: int,
        selected: int,
        deselected: int,
        options: ParaflowOptions,
    ) -> None:
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.selected = selected
        self.deselected = deselected
        self.options = options
