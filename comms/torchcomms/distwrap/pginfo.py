# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from dataclasses import dataclass, field
from typing import Any

from torch.distributed import ProcessGroup


# =============================================================================
# Public API Functions
# =============================================================================


def pg_info_create(
    pg: ProcessGroup, global_ranks: list[int] | None, group_desc: str | None
) -> None:
    if pg in _PG_INFO_REGISTRY:
        raise AssertionError(f"Process group {pg} already registered")
    pg_info = _PG_INFO(
        global_ranks=global_ranks.copy() if global_ranks is not None else [],
        group_desc=group_desc or "",
    )
    _PG_INFO_REGISTRY[pg] = pg_info


def pg_info_destroy(pg: ProcessGroup) -> None:
    if pg not in _PG_INFO_REGISTRY:
        return
    _PG_INFO_REGISTRY.pop(pg)


def pg_info_get(pg: ProcessGroup) -> "_PG_INFO | None":
    return _PG_INFO_REGISTRY.get(pg, None)


def pg_info_assert_registered(pg: ProcessGroup) -> None:
    assert pg in _PG_INFO_REGISTRY, f"Process group {pg} not registered"


def pg_info_get_global_ranks(pg: ProcessGroup) -> list[int]:
    pg_info = _PG_INFO_REGISTRY.get(pg, None)
    if pg_info is None:
        raise AssertionError(f"Process group {pg} not registered")
    return pg_info.global_ranks


def pg_info_get_group_desc(pg: ProcessGroup) -> str:
    pg_info = _PG_INFO_REGISTRY.get(pg, None)
    if pg_info is None:
        raise AssertionError(f"Process group {pg} not registered")
    return pg_info.group_desc


def pg_info_set_data(pg: ProcessGroup, key: str, data: Any) -> None:
    pg_info = _PG_INFO_REGISTRY.get(pg, None)
    if pg_info is None:
        raise AssertionError(f"Process group {pg} not registered")
    pg_info.data[key] = data


def pg_info_get_data(pg: ProcessGroup, key: str) -> Any:
    pg_info = _PG_INFO_REGISTRY.get(pg, None)
    if pg_info is None:
        raise AssertionError(f"Process group {pg} not registered")
    assert key in pg_info.data, f"Key '{key}' not found in process group {pg}"
    return pg_info.data[key]


# =============================================================================
# Private Data Structures
# =============================================================================


@dataclass
class _PG_INFO:
    global_ranks: list[int]
    group_desc: str
    data: dict[str, Any] = field(default_factory=dict)


_PG_INFO_REGISTRY: dict[ProcessGroup, _PG_INFO] = {}
