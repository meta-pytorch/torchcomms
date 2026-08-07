# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Host-only unit tests for the benchmark pure helpers + the AnyBench parser summary.

No GPU / distributed dependency: these exercise the pure sizing / bandwidth / SM-match /
result-aggregation logic that the multi-GPU benchmark itself cannot gate under CI.
"""

import unittest
from unittest import mock

from comms.dsl.tests._bench_common import (
    _bus_bytes,
    _bw_gbps,
    _fmt_size,
    _framework_num_blocks,
    _iters_for_size,
)
from comms.dsl.tests.anybench_a2a_cute_parser import _summarize


class FmtSizeTest(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(_fmt_size(32), "32B")
        self.assertEqual(_fmt_size(65536), "64KB")
        self.assertEqual(_fmt_size(50331648), "48MB")
        self.assertEqual(_fmt_size(2147483648), "2GB")


class BwGbpsTest(unittest.TestCase):
    def test_zero_latency_is_zero(self) -> None:
        # lat=0 must not divide-by-zero; the sentinel is 0.0 busbw.
        self.assertEqual(_bw_gbps(1 << 20, 8, 0.0), 0.0)

    def test_bus_bytes_excludes_diagonal(self) -> None:
        # chunk = numel // ws; bus bytes = chunk * (ws - 1) * elem (bf16 = 2 bytes).
        self.assertEqual(_bus_bytes(8 * 1024, 8), 1024 * 7 * 2)


class ItersForSizeTest(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(_iters_for_size(64 * 1024), 500)
        self.assertEqual(_iters_for_size(64 * 1024 * 1024), 200)
        self.assertEqual(_iters_for_size(1 << 30), 50)


class FrameworkNumBlocksTest(unittest.TestCase):
    # _max_num_blocks reads torch.cuda device props (GPU-only), so patch it to a fixed SM cap.
    @mock.patch("comms.dsl.tests._bench_common._max_num_blocks", return_value=16)
    def test_probed_matches_nccl_grid(self, _cap: mock.MagicMock) -> None:
        # (1 MB, cap=32) -> nccl_grid 16; nb = round(16 / 8) = 2, capped at 16.
        nb, grid = _framework_num_blocks(1048576, 32, 8, mock.MagicMock(), 32)
        self.assertEqual((nb, grid), (2, 16))

    @mock.patch("comms.dsl.tests._bench_common._max_num_blocks", return_value=16)
    def test_unprobed_falls_back_to_cap(self, _cap: mock.MagicMock) -> None:
        # An unprobed (msg, cap) returns the SM-budget cap + None (row flagged not-matched).
        nb, grid = _framework_num_blocks(999, 32, 8, mock.MagicMock(), 32)
        self.assertEqual((nb, grid), (16, None))


class SummarizeTest(unittest.TestCase):
    def test_zero_ratio_sentinel_is_kept(self) -> None:
        # A ratio of 0.0 (NCCL timing failed) must NOT be filtered out: it sinks min_ratio,
        # fails all_clear_nccl, and is surfaced as n_nccl_failed. This protects the
        # keep-the-0.0-sentinel logic from regressing to a truthy filter.
        rows = [
            {
                "backend": "cute",
                "variant": "copy",
                "size_bytes": 1024,
                "ratio": 0.0,
                "world_size": 8,
            },
            {
                "backend": "cute",
                "variant": "copy",
                "size_bytes": 2048,
                "ratio": 1.5,
                "world_size": 8,
            },
        ]
        out = _summarize(rows)
        self.assertFalse(out["all_clear_nccl"])
        self.assertEqual(out["n_nccl_failed"], 1)
        self.assertEqual(out["min_ratio_busbw"], 0.0)
        self.assertEqual(out["n_result_rows"], 2)

        clear = _summarize(
            [
                {
                    "backend": "cute",
                    "variant": "copy",
                    "size_bytes": 1024,
                    "ratio": 1.2,
                    "world_size": 8,
                },
            ]
        )
        self.assertTrue(clear["all_clear_nccl"])
        self.assertEqual(clear["n_nccl_failed"], 0)

    def test_error_rows_counted(self) -> None:
        rows = [
            {"backend": "cute_error", "size_bytes": 1024, "error": "boom"},
        ]
        out = _summarize(rows)
        self.assertEqual(out["n_errors"], 1)
        self.assertEqual(out["first_error"], "boom")

    def test_n_result_rows_counts_copy_only(self) -> None:
        # n_result_rows tracks the per-size (copy) arrays, not all cute rows, so it stays
        # consistent once non-copy variants (direct/ce) start being emitted alongside copy.
        rows = [
            {"backend": "cute", "variant": "copy", "size_bytes": 1024, "ratio": 1.1},
            {"backend": "cute", "variant": "direct", "size_bytes": 1024, "ratio": 1.4},
        ]
        out = _summarize(rows)
        self.assertEqual(out["n_result_rows"], 1)
        self.assertEqual(len(out["size_bytes"]), 1)
