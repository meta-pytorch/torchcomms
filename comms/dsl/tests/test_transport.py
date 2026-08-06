# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Host-side (CPU, no device) coverage for transport.check_transfer.

check_transfer is pure host validation -- it only reads per_peer_bytes /
max_blocks_per_peer off the transport and dtype.itemsize -- so all three ValueError
invariants (dtype divisibility, per-peer capacity, and signal-pad slot range) are
exercisable without a symmetric-memory PG. The device rendezvous path (nvl_rendezvous)
still needs a real symm-mem-capable group and is covered by the next-layer send/recv tests.
"""

import unittest
from dataclasses import dataclass
from typing import cast

import torch
from comms.dsl import check_transfer, P2pTransport


@dataclass
class _StubTransport:
    """Minimal P2pTransport-shaped stub: just the fields check_transfer reads."""

    world_size: int
    per_peer_bytes: int
    max_blocks_per_peer: int


class TransportCheckTransferTest(unittest.TestCase):
    def test_dtype_not_divisible_raises(self) -> None:
        # per_peer_bytes must be a whole multiple of the dtype itemsize (mirrors endpoint()):
        # 65 bytes is not divisible by bfloat16's 2-byte itemsize.
        t = cast(
            P2pTransport,
            _StubTransport(world_size=2, per_peer_bytes=65, max_blocks_per_peer=8),
        )
        with self.assertRaises(ValueError):
            check_transfer(t, numel=1, dtype=torch.bfloat16, num_blocks=1)

    def test_numel_overrun_raises(self) -> None:
        # per_peer_bytes=64 holds 64 uint8 / 32 bf16 elems; one past capacity raises.
        for dtype, cap in ((torch.uint8, 64), (torch.bfloat16, 32)):
            t = cast(
                P2pTransport,
                _StubTransport(world_size=2, per_peer_bytes=64, max_blocks_per_peer=8),
            )
            with self.assertRaises(ValueError):
                check_transfer(t, numel=cap + 1, dtype=dtype, num_blocks=1)

    def test_num_blocks_out_of_range_raises(self) -> None:
        t = cast(
            P2pTransport,
            _StubTransport(world_size=2, per_peer_bytes=64, max_blocks_per_peer=8),
        )
        with self.assertRaises(ValueError):
            check_transfer(t, numel=1, dtype=torch.uint8, num_blocks=0)
        with self.assertRaises(ValueError):
            check_transfer(
                t, numel=1, dtype=torch.uint8, num_blocks=t.max_blocks_per_peer + 1
            )

    def test_valid_transfer_passes(self) -> None:
        t = cast(
            P2pTransport,
            _StubTransport(world_size=2, per_peer_bytes=64, max_blocks_per_peer=8),
        )
        for dtype, cap in ((torch.uint8, 64), (torch.bfloat16, 32)):
            check_transfer(t, numel=cap, dtype=dtype, num_blocks=1)
            check_transfer(t, numel=cap, dtype=dtype, num_blocks=t.max_blocks_per_peer)
