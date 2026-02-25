"""Integration tests for paraflow pytest sharding behavior."""

from __future__ import annotations

import re

import pytest

pytest_plugins = ("pytester",)


def _configure_test_environment(pytester: pytest.Pytester) -> None:
    """Configure pytester environment with marker declarations."""
    pytester.makeini(
        """
[pytest]
markers =
    paraflow_group(name): group tests that must run in the same shard.
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
