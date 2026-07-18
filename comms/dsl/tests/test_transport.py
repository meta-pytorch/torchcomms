# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Host-side (CPU, no device) coverage for transport host validation.

``check_transfer`` and ``_require_signal_pad`` are pure host validation -- they
read only ints (per_peer_bytes / max_blocks_per_peer / dtype.itemsize / a pad
size) -- so their invariants are exercisable without a symmetric-memory process group.
The device rendezvous path requires a real symmetric-memory-capable group and is outside
this host-only test target.
"""

import unittest
from dataclasses import dataclass
from typing import cast

import torch
from comms.dsl import check_transfer, NvlTransport
from comms.dsl.transport import _require_signal_pad


@dataclass
class _StubTransport:
    """Minimal stub carrying only the fields ``check_transfer`` reads."""

    per_peer_bytes: int
    max_blocks_per_peer: int


def _stub(per_peer_bytes: int, max_blocks_per_peer: int = 8) -> NvlTransport:
    return cast(
        NvlTransport,
        _StubTransport(
            per_peer_bytes=per_peer_bytes, max_blocks_per_peer=max_blocks_per_peer
        ),
    )


class TransportCheckTransferTest(unittest.TestCase):
    def test_dtype_not_divisible_raises(self) -> None:
        # per_peer_bytes must be a whole multiple of the dtype itemsize:
        # 65 bytes is not divisible by bfloat16's 2-byte itemsize.
        with self.assertRaises(ValueError):
            check_transfer(_stub(65), numel=1, dtype=torch.bfloat16, num_blocks=1)

    def test_numel_overrun_raises(self) -> None:
        # per_peer_bytes=64 holds 64 uint8 / 32 bf16 elems; one past capacity raises.
        for dtype, cap in ((torch.uint8, 64), (torch.bfloat16, 32)):
            with self.assertRaises(ValueError):
                check_transfer(_stub(64), numel=cap + 1, dtype=dtype, num_blocks=1)

    def test_num_blocks_out_of_range_raises(self) -> None:
        t = _stub(64)
        with self.assertRaises(ValueError):
            check_transfer(t, numel=1, dtype=torch.uint8, num_blocks=0)
        with self.assertRaises(ValueError):
            check_transfer(
                t, numel=1, dtype=torch.uint8, num_blocks=t.max_blocks_per_peer + 1
            )

    def test_divisibility_checked_before_capacity(self) -> None:
        # When per_peer_bytes is both non-divisible AND too small for numel, the
        # divisibility error must win: cap_elems = per_peer_bytes // itemsize would be
        # computed from a garbage (floored) capacity otherwise. Kernel addressing relies
        # on this order.
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            check_transfer(
                _stub(65), numel=10_000_000, dtype=torch.bfloat16, num_blocks=1
            )

    def test_min_capacity_boundary(self) -> None:
        # per_peer_bytes == itemsize -> cap_elems == 1: exactly 1 elem fits, 2 overruns.
        t = _stub(2)  # 1 bf16 elem
        self.assertIsNone(
            check_transfer(t, numel=1, dtype=torch.bfloat16, num_blocks=1)
        )
        with self.assertRaises(ValueError):
            check_transfer(t, numel=2, dtype=torch.bfloat16, num_blocks=1)

    def test_valid_transfer_passes(self) -> None:
        # check_transfer returns None on success; assert that (and that it does not
        # raise) for every valid (dtype, numel, num_blocks) combination. numel == 0
        # (empty transfer) is allowed.
        t = _stub(64)
        for dtype, cap in ((torch.uint8, 64), (torch.bfloat16, 32)):
            self.assertIsNone(check_transfer(t, numel=0, dtype=dtype, num_blocks=1))
            self.assertIsNone(check_transfer(t, numel=cap, dtype=dtype, num_blocks=1))
            self.assertIsNone(
                check_transfer(
                    t, numel=cap, dtype=dtype, num_blocks=t.max_blocks_per_peer
                )
            )


class RequireSignalPadTest(unittest.TestCase):
    def test_pad_too_small_raises(self) -> None:
        # need = 2 * ws * mbp; one int64 short must raise.
        ws, mbp = 8, 32
        need = 2 * ws * mbp
        with self.assertRaisesRegex(RuntimeError, "signal pad too small"):
            _require_signal_pad(need - 1, ws, mbp)

    def test_pad_exact_and_larger_pass(self) -> None:
        ws, mbp = 8, 32
        need = 2 * ws * mbp
        self.assertIsNone(_require_signal_pad(need, ws, mbp))
        self.assertIsNone(_require_signal_pad(need + 1, ws, mbp))
