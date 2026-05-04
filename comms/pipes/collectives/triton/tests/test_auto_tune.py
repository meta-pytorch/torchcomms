# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Unit tests for topology-aware auto-tuning in device_alltoallv_dynamic.py.

Tests the pure-function tuning logic (no GPU required):
- _tune_for_nvl(): NVLink-optimized parameter selection
- _tune_for_ib(): IB/RDMA-optimized parameter selection
- auto_tune_alltoallv_params(): topology-aware dispatch with per-peer blocks
"""

import unittest

from comms.pipes.collectives.triton.auto_tune_config import (
    _tune_for_ib,
    _tune_for_nvl,
    auto_tune_alltoallv_params,
)

KB: int = 1024
MB: int = 1024 * 1024


class TestTuneForNvl(unittest.TestCase):
    """Test NVLink tuning parameter selection at each message size tier."""

    def test_small_messages_single_block(self) -> None:
        """Messages ≤64KB should use 1 block/peer."""
        for size in [1 * KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB, 32 * KB, 64 * KB]:
            result = _tune_for_nvl(size)
            self.assertEqual(result["blocks_per_peer"], 1, f"Failed at {size}")

    def test_medium_messages_multi_block(self) -> None:
        """Messages 128KB-256KB should use 8 blocks/peer."""
        for size in [128 * KB, 256 * KB]:
            result = _tune_for_nvl(size)
            self.assertEqual(result["blocks_per_peer"], 8, f"Failed at {size}")

    def test_large_messages_max_blocks(self) -> None:
        """Messages >256KB should use 16 blocks/peer."""
        for size in [512 * KB, 1 * MB, 16 * MB]:
            result = _tune_for_nvl(size)
            self.assertEqual(result["blocks_per_peer"], 16, f"Failed at {size}")

    def test_chunk_size_always_64kb(self) -> None:
        """NVLink chunk size is always 64KB."""
        for size in [1 * KB, 64 * KB, 1 * MB, 16 * MB]:
            result = _tune_for_nvl(size)
            self.assertEqual(result["chunk_size"], 64 * KB, f"Failed at {size}")

    def test_warps_scaling(self) -> None:
        """Warps should increase with message size."""
        self.assertEqual(_tune_for_nvl(1 * KB)["num_warps"], 4)
        self.assertEqual(_tune_for_nvl(4 * KB)["num_warps"], 8)
        self.assertEqual(_tune_for_nvl(16 * KB)["num_warps"], 16)
        self.assertEqual(_tune_for_nvl(64 * KB)["num_warps"], 32)
        self.assertEqual(_tune_for_nvl(256 * KB)["num_warps"], 16)

    def test_returns_required_keys(self) -> None:
        """Result must contain blocks_per_peer, num_warps, chunk_size."""
        result = _tune_for_nvl(1 * MB)
        self.assertIn("blocks_per_peer", result)
        self.assertIn("num_warps", result)
        self.assertIn("chunk_size", result)


class TestTuneForIb(unittest.TestCase):
    """Test IB/RDMA tuning parameter selection at each message size tier."""

    def test_small_messages_single_block(self) -> None:
        """IB messages ≤1MB should use 1 block/peer."""
        for size in [1 * KB, 64 * KB, 256 * KB, 512 * KB, 1 * MB]:
            result = _tune_for_ib(size)
            self.assertEqual(result["blocks_per_peer"], 1, f"Failed at {size}")

    def test_medium_messages_4_blocks(self) -> None:
        """IB messages 2-4MB should use 4 blocks/peer."""
        for size in [2 * MB, 4 * MB]:
            result = _tune_for_ib(size)
            self.assertEqual(result["blocks_per_peer"], 4, f"Failed at {size}")

    def test_large_messages_8_blocks(self) -> None:
        """IB messages ≥8MB should use 8 blocks/peer."""
        for size in [8 * MB, 16 * MB]:
            result = _tune_for_ib(size)
            self.assertEqual(result["blocks_per_peer"], 8, f"Failed at {size}")

    def test_chunk_size_varies(self) -> None:
        """IB chunk size varies by message size (not fixed like NVL)."""
        self.assertEqual(_tune_for_ib(1 * KB)["chunk_size"], 256 * KB)
        self.assertEqual(_tune_for_ib(4 * KB)["chunk_size"], 128 * KB)
        self.assertEqual(_tune_for_ib(256 * KB)["chunk_size"], 512 * KB)
        self.assertEqual(_tune_for_ib(4 * MB)["chunk_size"], 512 * KB)

    def test_returns_required_keys(self) -> None:
        """Result must contain blocks_per_peer, num_warps, chunk_size."""
        result = _tune_for_ib(1 * MB)
        self.assertIn("blocks_per_peer", result)
        self.assertIn("num_warps", result)
        self.assertIn("chunk_size", result)


class TestAutoTuneAlltoallvParams(unittest.TestCase):
    """Test the top-level auto_tune_alltoallv_params() dispatch logic."""

    def test_peer_is_nvl_none_returns_nvl_config(self) -> None:
        """peer_is_nvl=None (default) should return NVL config with no per-peer blocks."""
        result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=None)
        nvl = _tune_for_nvl(1 * MB)
        self.assertEqual(result["blocks_per_peer"], nvl["blocks_per_peer"])
        self.assertEqual(result["num_warps"], nvl["num_warps"])
        self.assertEqual(result["chunk_size"], nvl["chunk_size"])
        self.assertIsNone(result["per_peer_blocks"])

    def test_all_nvl_peers_returns_nvl_config(self) -> None:
        """All-NVL peers should behave identically to peer_is_nvl=None."""
        result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=[True] * 8)
        self.assertIsNone(result["per_peer_blocks"])
        nvl = _tune_for_nvl(1 * MB)
        self.assertEqual(result["blocks_per_peer"], nvl["blocks_per_peer"])

    def test_mixed_peers_returns_per_peer_blocks(self) -> None:
        """Mixed NVL/IB peers should return per-peer block counts."""
        peer_is_nvl = [True, True, True, True, False, False, False, False]
        result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=peer_is_nvl)

        self.assertIsNotNone(result["per_peer_blocks"])
        per_peer = result["per_peer_blocks"]
        assert per_peer is not None
        self.assertEqual(len(per_peer), 8)

        nvl = _tune_for_nvl(1 * MB)
        ib = _tune_for_ib(1 * MB)

        # NVL peers get NVL block count
        for i in range(4):
            self.assertEqual(per_peer[i], nvl["blocks_per_peer"])
        # IB peers get IB block count
        for i in range(4, 8):
            self.assertEqual(per_peer[i], ib["blocks_per_peer"])

    def test_mixed_peers_blocks_per_peer_is_max(self) -> None:
        """blocks_per_peer (constexpr upper bound) should be max of NVL and IB."""
        peer_is_nvl = [True, False]
        result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=peer_is_nvl)
        nvl = _tune_for_nvl(1 * MB)
        ib = _tune_for_ib(1 * MB)
        self.assertEqual(
            result["blocks_per_peer"],
            max(nvl["blocks_per_peer"], ib["blocks_per_peer"]),
        )

    def test_mixed_peers_num_warps_is_max(self) -> None:
        """num_warps should be max of NVL and IB warps."""
        peer_is_nvl = [True, False]
        result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=peer_is_nvl)
        nvl = _tune_for_nvl(1 * MB)
        ib = _tune_for_ib(1 * MB)
        self.assertEqual(
            result["num_warps"],
            max(nvl["num_warps"], ib["num_warps"]),
        )

    def test_all_ib_peers_returns_per_peer_blocks(self) -> None:
        """All-IB peers should still return per-peer blocks (not None)."""
        result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=[False] * 8)
        self.assertIsNotNone(result["per_peer_blocks"])
        per_peer = result["per_peer_blocks"]
        assert per_peer is not None
        ib = _tune_for_ib(1 * MB)
        for bpp in per_peer:
            self.assertEqual(bpp, ib["blocks_per_peer"])

    def test_result_has_all_required_keys(self) -> None:
        """Result must always contain blocks_per_peer, num_warps, chunk_size, per_peer_blocks."""
        for peer_is_nvl in [None, [True] * 4, [False] * 4, [True, False, True, False]]:
            result = auto_tune_alltoallv_params(1 * MB, peer_is_nvl=peer_is_nvl)
            self.assertIn("blocks_per_peer", result)
            self.assertIn("num_warps", result)
            self.assertIn("chunk_size", result)
            self.assertIn("per_peer_blocks", result)

    def test_various_message_sizes_with_mixed_peers(self) -> None:
        """Mixed topology should work across all message size tiers."""
        peer_is_nvl = [True, True, False, False]
        for size in [1 * KB, 16 * KB, 256 * KB, 1 * MB, 4 * MB, 16 * MB]:
            result = auto_tune_alltoallv_params(size, peer_is_nvl=peer_is_nvl)
            self.assertIsNotNone(result["per_peer_blocks"])
            per_peer = result["per_peer_blocks"]
            assert per_peer is not None
            self.assertEqual(len(per_peer), 4)
            # blocks_per_peer should be >= any individual per_peer value
            for bpp in per_peer:
                self.assertLessEqual(bpp, result["blocks_per_peer"])
