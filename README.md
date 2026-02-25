# pytest-paraflow

`pytest-paraflow` is a pytest plugin for deterministic test sharding across machines.

## Features

- Deterministic sharding by test identity (stable hash-based assignment).
- Group-based sharding via markers so related tests stay together.
- Dynamic shard count derived from total test count and desired shard size.

## Install

```bash
uv add pytest-paraflow
```

## Usage

Enable sharding by passing `--paraflow-shard-id` and either `--paraflow-num-shards` or `--paraflow-target-shard-size`.

### Static shard count

```bash
pytest --paraflow-shard-id=0 --paraflow-num-shards=4
```

### Dynamic shard count

```bash
pytest --paraflow-shard-id=1 --paraflow-target-shard-size=200
```

For `N` collected tests, shard count is calculated as `ceil(N / target_shard_size)`.

### Group tests by marker

```python
import pytest

@pytest.mark.paraflow_group("db")
def test_a():
    ...

@pytest.mark.paraflow_group("db")
def test_b():
    ...
```

```bash
pytest \
  --paraflow-shard-id=0 \
  --paraflow-num-shards=3 \
  --paraflow-group-marker=paraflow_group
```

Marker behavior:

- Tests with the same marker value are assigned to the same shard.
- `@pytest.mark.paraflow_group(...)` is only used when `--paraflow-group-marker=paraflow_group` is provided.
- Without `--paraflow-group-marker`, paraflow falls back to per-test (`nodeid`) sharding.
- `--paraflow-group-marker=smoke` with `@pytest.mark.smoke` groups all `smoke` tests together (single group key).
- `--paraflow-group-marker=smoke` with `@pytest.mark.smoke("db")` groups by value (for example `db`, `api`).
- `--paraflow-group-marker` is repeatable. If multiple configured markers exist on one test, the first configured marker wins.

## CLI options

- `--paraflow-shard-id`: current shard index (zero-based).
- `--paraflow-num-shards`: total shard count.
- `--paraflow-target-shard-size`: desired tests per shard for dynamic sizing.
- `--paraflow-group-marker`: marker name used for grouping (repeatable).

## Validation rules

- `--paraflow-shard-id` is required whenever sharding is enabled.
- One of `--paraflow-num-shards` or `--paraflow-target-shard-size` is required.
- `--paraflow-shard-id` must be in `[0, total_shards - 1]`.
