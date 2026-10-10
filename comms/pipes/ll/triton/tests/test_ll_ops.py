# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
Unit tests for Triton LL constants, utility functions, and auto-tuning.

No GPU required — tests pure-Python logic only.
"""

from __future__ import annotations

import unittest

from comms.pipes.ll.triton.auto_tune import ll_auto_tune, ll_auto_tune_bidirectional
from comms.pipes.ll.triton.ll_ops import (
    can_use_ll,
    ll_buffer_size,
    LL_DATA_MASK,
    LL_FLAG_MASK,
    LL_LINE_SIZE,
    LL_MEMSET_INIT_BYTE,
    ll_num_lines,
    LL_PAYLOAD_PER_LINE,
    ll_peer_buffer_offset,
    ll_peer_index,
    LL_READY_TO_WRITE,
    ll_total_buffer_size,
)


class TestLlConstants(unittest.TestCase):
    def test_line_size(self) -> None:
        self.assertEqual(LL_LINE_SIZE, 16)

    def test_payload_per_line(self) -> None:
        self.assertEqual(LL_PAYLOAD_PER_LINE, 8)

    def test_ready_to_write(self) -> None:
        self.assertEqual(LL_READY_TO_WRITE, 0xFFFFFFFF)

    def test_memset_init_byte(self) -> None:
        self.assertEqual(LL_MEMSET_INIT_BYTE, 0xFF)

    def test_flag_mask(self) -> None:
        self.assertEqual(LL_FLAG_MASK, 0xFFFFFFFF00000000)

    def test_data_mask(self) -> None:
        self.assertEqual(LL_DATA_MASK, 0x00000000FFFFFFFF)

    def test_masks_are_complementary(self) -> None:
        self.assertEqual(LL_FLAG_MASK | LL_DATA_MASK, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(LL_FLAG_MASK & LL_DATA_MASK, 0)


class TestLlNumLines(unittest.TestCase):
    def test_zero_bytes(self) -> None:
        self.assertEqual(ll_num_lines(0), 0)

    def test_one_byte(self) -> None:
        self.assertEqual(ll_num_lines(1), 1)

    def test_seven_bytes(self) -> None:
        self.assertEqual(ll_num_lines(7), 1)

    def test_exact_one_line(self) -> None:
        self.assertEqual(ll_num_lines(8), 1)

    def test_nine_bytes(self) -> None:
        self.assertEqual(ll_num_lines(9), 2)

    def test_exact_two_lines(self) -> None:
        self.assertEqual(ll_num_lines(16), 2)

    def test_hundred_bytes(self) -> None:
        self.assertEqual(ll_num_lines(100), 13)


class TestLlBufferSize(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(ll_buffer_size(0), 16)

    def test_one_line(self) -> None:
        self.assertEqual(ll_buffer_size(8), 32)

    def test_two_lines(self) -> None:
        self.assertEqual(ll_buffer_size(16), 48)

    def test_partial_line(self) -> None:
        self.assertEqual(ll_buffer_size(9), 48)

    def test_large(self) -> None:
        self.assertEqual(ll_buffer_size(1024), (ll_num_lines(1024) + 1) * 16)


class TestLlTotalBufferSize(unittest.TestCase):
    def test_two_ranks(self) -> None:
        self.assertEqual(ll_total_buffer_size(8, 2), 32)

    def test_eight_ranks(self) -> None:
        self.assertEqual(ll_total_buffer_size(8, 8), 32 * 7)

    def test_gb200_max_ranks(self) -> None:
        per_peer = ll_buffer_size(65536)
        self.assertEqual(ll_total_buffer_size(65536, 72), per_peer * 71)


class TestCanUseLl(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertTrue(can_use_ll(0))

    def test_valid_multiples(self) -> None:
        self.assertTrue(can_use_ll(8))
        self.assertTrue(can_use_ll(16))
        self.assertTrue(can_use_ll(1024))

    def test_invalid_non_multiples(self) -> None:
        self.assertFalse(can_use_ll(1))
        self.assertFalse(can_use_ll(7))
        self.assertFalse(can_use_ll(15))
        self.assertFalse(can_use_ll(100))


class TestLlPeerIndex(unittest.TestCase):
    def test_peer_below_self(self) -> None:
        self.assertEqual(ll_peer_index(target_rank=0, self_rank=1), 0)
        self.assertEqual(ll_peer_index(target_rank=0, self_rank=3), 0)
        self.assertEqual(ll_peer_index(target_rank=2, self_rank=3), 2)

    def test_peer_above_self(self) -> None:
        self.assertEqual(ll_peer_index(target_rank=2, self_rank=1), 1)
        self.assertEqual(ll_peer_index(target_rank=3, self_rank=1), 2)
        self.assertEqual(ll_peer_index(target_rank=7, self_rank=0), 6)

    def test_world_size_4_from_rank_1(self) -> None:
        self.assertEqual(ll_peer_index(0, 1), 0)
        self.assertEqual(ll_peer_index(2, 1), 1)
        self.assertEqual(ll_peer_index(3, 1), 2)

    def test_contiguous_indices(self) -> None:
        world_size = 8
        self_rank = 3
        indices = [
            ll_peer_index(r, self_rank) for r in range(world_size) if r != self_rank
        ]
        self.assertEqual(indices, list(range(world_size - 1)))


class TestLlPeerBufferOffset(unittest.TestCase):
    def test_first_peer(self) -> None:
        self.assertEqual(
            ll_peer_buffer_offset(target_rank=0, self_rank=1, per_peer_size=256),
            0,
        )

    def test_second_peer(self) -> None:
        self.assertEqual(
            ll_peer_buffer_offset(target_rank=2, self_rank=1, per_peer_size=256),
            256,
        )

    def test_send_recv_symmetry(self) -> None:
        per_peer_size = 128
        my_rank = 1
        peer = 3
        send_offset = ll_peer_buffer_offset(my_rank, peer, per_peer_size)
        recv_offset = ll_peer_buffer_offset(peer, my_rank, per_peer_size)
        self.assertNotEqual(send_offset, recv_offset)
        self.assertEqual(send_offset, ll_peer_index(my_rank, peer) * per_peer_size)
        self.assertEqual(recv_offset, ll_peer_index(peer, my_rank) * per_peer_size)

    def test_offsets_partition_buffer(self) -> None:
        # All peer offsets from a given self_rank must be distinct, aligned to
        # per_peer_size, and exactly cover [0, (world_size-1)*per_peer_size).
        world_size = 8
        self_rank = 3
        per_peer_size = 256
        offsets = [
            ll_peer_buffer_offset(r, self_rank, per_peer_size)
            for r in range(world_size)
            if r != self_rank
        ]
        self.assertEqual(len(set(offsets)), world_size - 1)
        for off in offsets:
            self.assertEqual(off % per_peer_size, 0)
        self.assertEqual(
            sorted(offsets),
            [i * per_peer_size for i in range(world_size - 1)],
        )

    def test_offsets_within_total_buffer(self) -> None:
        # Each region [offset, offset + per_peer_size) must fit within the
        # total LL buffer allocated for the rank.
        world_size = 4
        self_rank = 2
        per_peer_size = ll_buffer_size(64)
        total = ll_total_buffer_size(64, world_size)
        for r in range(world_size):
            if r == self_rank:
                continue
            off = ll_peer_buffer_offset(r, self_rank, per_peer_size)
            self.assertGreaterEqual(off, 0)
            self.assertLessEqual(off + per_peer_size, total)


class TestLlAutoTune(unittest.TestCase):
    def test_small_message_uses_single_warp(self) -> None:
        config = ll_auto_tune(64)
        self.assertEqual(config["block_size"], 32)
        self.assertEqual(config["num_blocks"], 1)

    def test_boundary_256(self) -> None:
        config = ll_auto_tune(256)
        self.assertEqual(config["block_size"], 32)

    def test_medium_message(self) -> None:
        config = ll_auto_tune(1024)
        self.assertEqual(config["block_size"], 512)
        self.assertEqual(config["num_blocks"], 1)

    def test_large_message(self) -> None:
        config = ll_auto_tune(65536)
        self.assertEqual(config["block_size"], 512)
        self.assertEqual(config["num_blocks"], 32)

    def test_returns_valid_keys(self) -> None:
        config = ll_auto_tune(8)
        self.assertIn("num_blocks", config)
        self.assertIn("block_size", config)

    def test_monotonically_increasing_blocks(self) -> None:
        sizes = [8, 256, 512, 1024, 4096, 16384, 65536, 262144]
        block_counts = [ll_auto_tune(s)["num_blocks"] for s in sizes]
        for i in range(1, len(block_counts)):
            self.assertGreaterEqual(block_counts[i], block_counts[i - 1])

    def test_very_large_message(self) -> None:
        config = ll_auto_tune(1024 * 1024)
        self.assertEqual(config["num_blocks"], 256)


class TestLlAutoTuneBidirectional(unittest.TestCase):
    def test_doubles_blocks(self) -> None:
        uni = ll_auto_tune(16384)
        bidi = ll_auto_tune_bidirectional(16384)
        self.assertEqual(bidi["num_blocks"], uni["num_blocks"] * 2)
        self.assertEqual(bidi["block_size"], uni["block_size"])

    def test_capped_at_1024(self) -> None:
        bidi = ll_auto_tune_bidirectional(1024 * 1024)
        self.assertLessEqual(bidi["num_blocks"], 1024)
