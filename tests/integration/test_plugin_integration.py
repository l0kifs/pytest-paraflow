"""Integration tests for paraflow pytest sharding behavior."""

from __future__ import annotations

import re

import pytest

from pytest_paraflow.infrastructure.hash.hasher import stable_hash_to_int

pytest_plugins = ("pytester",)


def _configure_test_environment(pytester: pytest.Pytester) -> None:
    """Configure pytester environment with marker declarations."""
    pytester.makeini(
        """
[pytest]
markers =
    paraflow_group(name): group tests that must run in the same shard.
    smoke(value): custom smoke marker used in integration tests.
    serial(value): custom serial marker used in integration tests.
"""
    )


def _executed_nodeids(result: pytest.RunResult) -> set[str]:
    """Extract executed node ids from verbose pytest output."""
    status_pattern = r"(PASSED|FAILED|SKIPPED|XFAIL|XPASS)"
    classic_pattern = re.compile(rf"^(?P<nodeid>\S+::\S+)\s+{status_pattern}\b")
    xdist_pattern = re.compile(
        rf"^(?:\[\w+\]\s+)?(?:\[\s*\d+%\]\s+)?{status_pattern}\s+(?P<nodeid>\S+::\S+)\b"
    )
    nodeids: set[str] = set()
    for line in result.stdout.str().splitlines():
        stripped_line = line.strip()
        for pattern in (classic_pattern, xdist_pattern):
            match = pattern.match(stripped_line)
            if match:
                nodeids.add(match.group("nodeid"))
                break
    return nodeids


def _find_distinct_shard_mapping(
    group_keys: tuple[str, ...],
) -> tuple[int, dict[str, int]]:
    """Pick a shard count where provided keys map to distinct shard ids."""
    for num_shards in (5, 7, 11, 13, 17, 19, 23, 29):
        shard_map = {key: stable_hash_to_int(key) % num_shards for key in group_keys}
        if len(set(shard_map.values())) == len(group_keys):
            return num_shards, shard_map
    raise AssertionError("Failed to find shard count with distinct group-key shard ids")


def test_static_sharding_partitions_suite(pytester: pytest.Pytester) -> None:
    """Static sharding should split tests into non-overlapping shard selections."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_static="""
def test_a():
    assert True

def test_b():
    assert True

def test_c():
    assert True

def test_d():
    assert True

def test_e():
    assert True

def test_f():
    assert True
"""
    )

    all_nodeids = {
        "test_static.py::test_a",
        "test_static.py::test_b",
        "test_static.py::test_c",
        "test_static.py::test_d",
        "test_static.py::test_e",
        "test_static.py::test_f",
    }

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
    )

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)

    assert shard_0_nodeids.isdisjoint(shard_1_nodeids)
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids


def test_group_marker_keeps_related_tests_together(pytester: pytest.Pytester) -> None:
    """Tests with the same group marker value should remain in one shard."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_grouping="""
import pytest

@pytest.mark.paraflow_group("db")
def test_grouped_one():
    assert True

@pytest.mark.paraflow_group("db")
def test_grouped_two():
    assert True

def test_regular_one():
    assert True

def test_regular_two():
    assert True
"""
    )

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=paraflow_group",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=paraflow_group",
    )

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)

    grouped_nodeids = {
        "test_grouping.py::test_grouped_one",
        "test_grouping.py::test_grouped_two",
    }
    all_nodeids = {
        "test_grouping.py::test_grouped_one",
        "test_grouping.py::test_grouped_two",
        "test_grouping.py::test_regular_one",
        "test_grouping.py::test_regular_two",
    }

    assert grouped_nodeids <= shard_0_nodeids or grouped_nodeids <= shard_1_nodeids
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids


def test_dynamic_sharding_uses_target_shard_size(pytester: pytest.Pytester) -> None:
    """Dynamic sharding should derive shard count from desired shard size."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_dynamic="""
def test_a():
    assert True

def test_b():
    assert True

def test_c():
    assert True

def test_d():
    assert True

def test_e():
    assert True
"""
    )

    all_nodeids = {
        "test_dynamic.py::test_a",
        "test_dynamic.py::test_b",
        "test_dynamic.py::test_c",
        "test_dynamic.py::test_d",
        "test_dynamic.py::test_e",
    }

    executed_union: set[str] = set()
    for shard_id in (0, 1, 2):
        result = pytester.runpytest_subprocess(
            "-vv",
            "--color=no",
            f"--paraflow-shard-id={shard_id}",
            "--paraflow-target-shard-size=2",
        )
        executed_union.update(_executed_nodeids(result))

    assert executed_union == all_nodeids

    invalid = pytester.runpytest_subprocess(
        "--color=no",
        "--paraflow-shard-id=3",
        "--paraflow-target-shard-size=2",
    )
    assert invalid.ret == pytest.ExitCode.USAGE_ERROR
    output = invalid.stdout.str() + invalid.stderr.str()
    assert "--paraflow-shard-id must be in range [0, 2]" in output


def test_sharding_options_require_paraflow_shard_id(pytester: pytest.Pytester) -> None:
    """Shard size options must not be accepted without paraflow shard id."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_validation="""
def test_smoke():
    assert True
"""
    )

    invalid = pytester.runpytest_subprocess("--paraflow-num-shards=2")
    assert invalid.ret == pytest.ExitCode.USAGE_ERROR
    output = invalid.stdout.str() + invalid.stderr.str()
    assert "--paraflow-shard-id is required" in output


def test_shard_id_requires_num_shards_or_target_size(pytester: pytest.Pytester) -> None:
    """Shard id alone is invalid without static or dynamic shard count options."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_validation="""
def test_smoke():
    assert True
"""
    )

    invalid = pytester.runpytest_subprocess("--paraflow-shard-id=0")
    assert invalid.ret == pytest.ExitCode.USAGE_ERROR
    output = invalid.stdout.str() + invalid.stderr.str()
    assert "Either --paraflow-num-shards or --paraflow-target-shard-size is required" in output


@pytest.mark.parametrize(
    ("args", "expected_message"),
    [
        (
            ("--paraflow-shard-id=-1", "--paraflow-num-shards=2"),
            "--paraflow-shard-id must be greater than or equal to 0",
        ),
        (
            ("--paraflow-shard-id=0", "--paraflow-num-shards=0"),
            "--paraflow-num-shards must be greater than 0",
        ),
        (
            ("--paraflow-shard-id=0", "--paraflow-target-shard-size=0"),
            "--paraflow-target-shard-size must be greater than 0",
        ),
    ],
)
def test_invalid_numeric_options_raise_usage_error(
    pytester: pytest.Pytester,
    args: tuple[str, ...],
    expected_message: str,
) -> None:
    """Invalid numeric sharding options should fail fast with usage errors."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_validation="""
def test_smoke():
    assert True
"""
    )

    invalid = pytester.runpytest_subprocess(*args)
    assert invalid.ret == pytest.ExitCode.USAGE_ERROR
    output = invalid.stdout.str() + invalid.stderr.str()
    assert expected_message in output


def test_plugin_disabled_without_paraflow_options(pytester: pytest.Pytester) -> None:
    """Plugin should leave collection unchanged when no paraflow options are passed."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_plain="""
def test_a():
    assert True

def test_b():
    assert True

def test_c():
    assert True
"""
    )

    result = pytester.runpytest_subprocess("--color=no")
    assert result.ret == pytest.ExitCode.OK
    output = result.stdout.str() + result.stderr.str()
    assert "3 passed" in output
    assert "paraflow:" not in output


def test_explicit_num_shards_takes_precedence_over_target_size(
    pytester: pytest.Pytester,
) -> None:
    """Explicit shard count should override dynamic shard-size calculation."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_precedence="""
def test_a():
    assert True

def test_b():
    assert True

def test_c():
    assert True

def test_d():
    assert True

def test_e():
    assert True
"""
    )

    all_nodeids = {
        "test_precedence.py::test_a",
        "test_precedence.py::test_b",
        "test_precedence.py::test_c",
        "test_precedence.py::test_d",
        "test_precedence.py::test_e",
    }

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-target-shard-size=1",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
        "--paraflow-target-shard-size=1",
    )

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)
    assert shard_0_nodeids.isdisjoint(shard_1_nodeids)
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids

    invalid = pytester.runpytest_subprocess(
        "--color=no",
        "--paraflow-shard-id=2",
        "--paraflow-num-shards=2",
        "--paraflow-target-shard-size=1",
    )
    assert invalid.ret == pytest.ExitCode.USAGE_ERROR
    output = invalid.stdout.str() + invalid.stderr.str()
    assert "--paraflow-shard-id must be in range [0, 1]" in output


def test_custom_marker_without_value_groups_all_matching_tests(
    pytester: pytest.Pytester,
) -> None:
    """Marker usage without value should still group tests by marker name."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_custom_marker="""
import pytest

@pytest.mark.smoke
def test_smoke_one():
    assert True

@pytest.mark.smoke
def test_smoke_two():
    assert True

def test_regular_one():
    assert True

def test_regular_two():
    assert True
"""
    )

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=smoke",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=smoke",
    )

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)
    grouped_nodeids = {
        "test_custom_marker.py::test_smoke_one",
        "test_custom_marker.py::test_smoke_two",
    }
    all_nodeids = {
        "test_custom_marker.py::test_smoke_one",
        "test_custom_marker.py::test_smoke_two",
        "test_custom_marker.py::test_regular_one",
        "test_custom_marker.py::test_regular_two",
    }

    assert grouped_nodeids <= shard_0_nodeids or grouped_nodeids <= shard_1_nodeids
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids


def test_custom_marker_keyword_value_groups_by_kwarg(pytester: pytest.Pytester) -> None:
    """Configured marker should use keyword values as group keys when provided."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_custom_kw="""
import pytest

@pytest.mark.smoke(group="db")
def test_db_one():
    assert True

@pytest.mark.smoke(group="db")
def test_db_two():
    assert True

@pytest.mark.smoke(group="api")
def test_api_one():
    assert True

@pytest.mark.smoke(group="api")
def test_api_two():
    assert True
"""
    )

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=smoke",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=smoke",
    )

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)
    db_nodeids = {"test_custom_kw.py::test_db_one", "test_custom_kw.py::test_db_two"}
    api_nodeids = {"test_custom_kw.py::test_api_one", "test_custom_kw.py::test_api_two"}
    all_nodeids = db_nodeids | api_nodeids

    assert db_nodeids <= shard_0_nodeids or db_nodeids <= shard_1_nodeids
    assert api_nodeids <= shard_0_nodeids or api_nodeids <= shard_1_nodeids
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids


def test_duplicate_group_markers_do_not_change_distribution(
    pytester: pytest.Pytester,
) -> None:
    """Duplicate --paraflow-group-marker values should not affect shard selection."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_header="""
import pytest

@pytest.mark.smoke
def test_smoke_one():
    assert True

@pytest.mark.smoke
def test_smoke_two():
    assert True

def test_regular_one():
    assert True

def test_regular_two():
    assert True
"""
    )

    single_flag = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=smoke",
    )
    duplicated_flag = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=smoke",
        "--paraflow-group-marker=smoke",
    )

    assert single_flag.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    assert duplicated_flag.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    assert _executed_nodeids(single_flag) == _executed_nodeids(duplicated_flag)


def test_first_configured_group_marker_wins(pytester: pytest.Pytester) -> None:
    """When multiple group markers are configured, first matching marker should win."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_marker_order="""
import pytest

@pytest.mark.smoke("A")
@pytest.mark.serial("shared")
def test_case_one():
    assert True

@pytest.mark.smoke("B")
@pytest.mark.serial("shared")
def test_case_two():
    assert True
"""
    )

    smoke_a = "marker:smoke:A"
    smoke_b = "marker:smoke:B"
    serial_shared = "marker:serial:shared"
    num_shards, shard_map = _find_distinct_shard_mapping((smoke_a, smoke_b, serial_shared))

    smoke_first_shard = shard_map[smoke_a]
    serial_first_shard = shard_map[serial_shared]

    smoke_first = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        f"--paraflow-shard-id={smoke_first_shard}",
        f"--paraflow-num-shards={num_shards}",
        "--paraflow-group-marker=smoke",
        "--paraflow-group-marker=serial",
    )
    serial_first = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        f"--paraflow-shard-id={serial_first_shard}",
        f"--paraflow-num-shards={num_shards}",
        "--paraflow-group-marker=serial",
        "--paraflow-group-marker=smoke",
    )

    smoke_first_nodeids = _executed_nodeids(smoke_first)
    serial_first_nodeids = _executed_nodeids(serial_first)
    assert smoke_first.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    assert serial_first.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    assert "test_marker_order.py::test_case_one" in smoke_first_nodeids
    assert "test_marker_order.py::test_case_two" not in smoke_first_nodeids
    assert {
        "test_marker_order.py::test_case_one",
        "test_marker_order.py::test_case_two",
    } <= serial_first_nodeids


def test_dynamic_sharding_with_empty_collection_handles_bounds(
    pytester: pytest.Pytester,
) -> None:
    """Empty collections should resolve to one shard and validate shard id range."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_empty="""
# no tests in this module
"""
    )

    shard_0 = pytester.runpytest_subprocess(
        "--color=no",
        "--paraflow-shard-id=0",
        "--paraflow-target-shard-size=2",
    )
    assert shard_0.ret == pytest.ExitCode.NO_TESTS_COLLECTED

    shard_1 = pytester.runpytest_subprocess(
        "--color=no",
        "--paraflow-shard-id=1",
        "--paraflow-target-shard-size=2",
    )
    assert shard_1.ret == pytest.ExitCode.USAGE_ERROR
    output = shard_1.stdout.str() + shard_1.stderr.str()
    assert "--paraflow-shard-id must be in range [0, 0]" in output


def test_static_sharding_partitions_suite_with_xdist(pytester: pytest.Pytester) -> None:
    """Static sharding should remain deterministic when xdist workers are enabled."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_xdist_static="""
def test_a():
    assert True

def test_b():
    assert True

def test_c():
    assert True

def test_d():
    assert True

def test_e():
    assert True

def test_f():
    assert True

def test_g():
    assert True

def test_h():
    assert True
"""
    )

    all_nodeids = {
        "test_xdist_static.py::test_a",
        "test_xdist_static.py::test_b",
        "test_xdist_static.py::test_c",
        "test_xdist_static.py::test_d",
        "test_xdist_static.py::test_e",
        "test_xdist_static.py::test_f",
        "test_xdist_static.py::test_g",
        "test_xdist_static.py::test_h",
    }

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "-n",
        "2",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "-n",
        "2",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
    )

    assert shard_0.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    assert shard_1.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)
    assert shard_0_nodeids.isdisjoint(shard_1_nodeids)
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids


def test_group_marker_keeps_related_tests_together_with_xdist(
    pytester: pytest.Pytester,
) -> None:
    """Group marker behavior should be preserved with xdist workers enabled."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_xdist_grouping="""
import pytest

@pytest.mark.paraflow_group("db")
def test_grouped_one():
    assert True

@pytest.mark.paraflow_group("db")
def test_grouped_two():
    assert True

def test_regular_one():
    assert True

def test_regular_two():
    assert True
"""
    )

    shard_0 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "-n",
        "2",
        "--paraflow-shard-id=0",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=paraflow_group",
    )
    shard_1 = pytester.runpytest_subprocess(
        "-vv",
        "--color=no",
        "-n",
        "2",
        "--paraflow-shard-id=1",
        "--paraflow-num-shards=2",
        "--paraflow-group-marker=paraflow_group",
    )

    assert shard_0.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    assert shard_1.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}

    shard_0_nodeids = _executed_nodeids(shard_0)
    shard_1_nodeids = _executed_nodeids(shard_1)
    grouped_nodeids = {
        "test_xdist_grouping.py::test_grouped_one",
        "test_xdist_grouping.py::test_grouped_two",
    }
    all_nodeids = {
        "test_xdist_grouping.py::test_grouped_one",
        "test_xdist_grouping.py::test_grouped_two",
        "test_xdist_grouping.py::test_regular_one",
        "test_xdist_grouping.py::test_regular_two",
    }

    assert grouped_nodeids <= shard_0_nodeids or grouped_nodeids <= shard_1_nodeids
    assert shard_0_nodeids | shard_1_nodeids == all_nodeids


def test_dynamic_sharding_partitions_suite_with_xdist(pytester: pytest.Pytester) -> None:
    """Dynamic shard sizing should remain correct with xdist worker distribution."""
    _configure_test_environment(pytester)
    pytester.makepyfile(
        test_xdist_dynamic="""
def test_a():
    assert True

def test_b():
    assert True

def test_c():
    assert True

def test_d():
    assert True

def test_e():
    assert True
"""
    )

    all_nodeids = {
        "test_xdist_dynamic.py::test_a",
        "test_xdist_dynamic.py::test_b",
        "test_xdist_dynamic.py::test_c",
        "test_xdist_dynamic.py::test_d",
        "test_xdist_dynamic.py::test_e",
    }

    executed_union: set[str] = set()
    for shard_id in (0, 1, 2):
        result = pytester.runpytest_subprocess(
            "-vv",
            "--color=no",
            "-n",
            "2",
            f"--paraflow-shard-id={shard_id}",
            "--paraflow-target-shard-size=2",
        )
        assert result.ret in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
        executed_union.update(_executed_nodeids(result))

    assert executed_union == all_nodeids
