import gc
import sys

import pytest
from nccl.bindings.nccl import Config

from nccl.bindings import ncclx_internal


def test_ncclx_surface_is_exported() -> None:
    expected = {
        "NcclxHints",
        "colltrace_get_comm_id",
        "colltrace_get_latest_coll_id",
        "colltrace_get_unread_events",
        "comm_dump",
        "comm_dump_all",
        "comm_set_config",
        "put",
        "win_get_attributes",
        "win_shared_query",
    }

    assert expected <= set(dir(ncclx_internal))


def test_hints_pointer_is_stable_and_nonzero() -> None:
    hints = ncclx_internal.NcclxHints({"enabled": True, "count": 3})
    pointer = hints.as_ptr()
    other = ncclx_internal.NcclxHints({"other": True})

    assert pointer != 0
    assert other.as_ptr() != 0
    assert hints.as_ptr() == pointer


def test_config_retains_and_replaces_hints_owner() -> None:
    config = Config()
    first = ncclx_internal.NcclxHints({"first": True})
    second = ncclx_internal.NcclxHints({"second": True})
    first_refs = sys.getrefcount(first)
    second_refs = sys.getrefcount(second)

    config.set_hints(first.as_ptr(), first)
    assert sys.getrefcount(first) == first_refs + 1

    config.set_hints(second.as_ptr(), second)
    assert sys.getrefcount(first) == first_refs
    assert sys.getrefcount(second) == second_refs + 1

    del config
    gc.collect()
    assert sys.getrefcount(second) == second_refs


def test_config_rejects_unowned_or_mismatched_hints() -> None:
    config = Config()
    hints = ncclx_internal.NcclxHints()

    with pytest.raises(ValueError, match="must own"):
        config.set_hints(hints.as_ptr(), None)

    with pytest.raises(ValueError, match="must point"):
        config.set_hints(hints.as_ptr() + 1, hints)
