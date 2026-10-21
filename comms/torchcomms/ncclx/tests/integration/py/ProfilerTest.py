#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
ncclx-specific profiler test using the common ProfilerTestBase class.
"""

import unittest

import torch
import torchcomms.tests.integration.helpers.ProfilerTest as profiler_test


def validate_ncclx_profiler_events(per_coll_meta):
    """ncclx-specific validation of profiler events."""
    assert len(per_coll_meta["barrier"]) == 1
    assert len(per_coll_meta["wait"]) == 1
    assert len(per_coll_meta["send"]) == 1
    assert len(per_coll_meta["recv"]) == 1
    assert len(per_coll_meta["all_reduce"]) == 1
    assert len(per_coll_meta["reduce"]) == 1
    assert len(per_coll_meta["all_gather_single"]) == 1
    assert len(per_coll_meta["all_gather"]) == 1
    assert len(per_coll_meta["gather"]) == 1
    assert len(per_coll_meta["reduce_scatter"]) == 1
    assert len(per_coll_meta["reduce_scatter_single"]) == 1
    assert len(per_coll_meta["scatter"]) == 1
    assert len(per_coll_meta["all_to_all"]) == 1
    assert len(per_coll_meta["all_to_all_single"]) == 1
    assert len(per_coll_meta["all_to_all_v_single"]) == 1
    assert len(per_coll_meta["broadcast"]) == 1


class ProfilerTest(profiler_test.ProfilerTestBase):
    """ncclx-specific profiler test."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            backend="ncclx",
            device=torch.device("cuda"),
            validation_func=validate_ncclx_profiler_events,
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
