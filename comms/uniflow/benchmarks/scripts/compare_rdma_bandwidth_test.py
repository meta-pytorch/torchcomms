# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from comms.uniflow.benchmarks.scripts import compare_rdma_bandwidth as compare


class CompareRdmaBandwidthTest(unittest.TestCase):
    def test_switching_benchmark_target_replaces_remote_binary(self) -> None:
        for old, new in [
            (compare.CORE_BENCH_TARGET, compare.EXTENSION_BENCH_TARGET),
            (compare.EXTENSION_BENCH_TARGET, compare.CORE_BENCH_TARGET),
        ]:
            with self.subTest(old=old, new=new):
                host = MagicMock(spec=compare.RemoteHost)
                benchmark = SimpleNamespace(
                    name="uniflow",
                    binary_name="uniflow_bench",
                    local_binary_path="/tmp/selected-benchmark",
                )
                args = SimpleNamespace(
                    force_copy="",
                    _has_remote_bins={"uniflow_bench": True},
                    _hosts={"host": host},
                    _remote_uniflow_linkage="static",
                    _remote_uniflow_target=old,
                    rdma_linkage="static",
                    bench_target=new,
                )

                compare._copy_binaries(args, [benchmark])

                host.copy_to.assert_called_once_with(
                    "/tmp/selected-benchmark", "uniflow_bench"
                )
                host.verify_and_chmod.assert_called_once_with(
                    ["uniflow_bench"],
                    uniflow_linkage="static",
                    uniflow_target=new,
                )

    def test_matching_benchmark_target_keeps_remote_binary(self) -> None:
        host = MagicMock(spec=compare.RemoteHost)
        args = SimpleNamespace(
            force_copy="",
            _has_remote_bins={"uniflow_bench": True},
            _hosts={"host": host},
            _remote_uniflow_linkage="static",
            _remote_uniflow_target=compare.CORE_BENCH_TARGET,
            rdma_linkage="static",
            bench_target=compare.CORE_BENCH_TARGET,
        )
        benchmark = SimpleNamespace(name="uniflow", binary_name="uniflow_bench")

        compare._copy_binaries(args, [benchmark])

        host.copy_to.assert_not_called()
