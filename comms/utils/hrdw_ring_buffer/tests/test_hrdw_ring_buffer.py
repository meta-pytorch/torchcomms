# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import datetime
import unittest
from typing import TYPE_CHECKING

import torch
from hrdw_ring_buffer._native import (
    DevicePollResult,
    Entry,
    Reader,
    RingBuffer,
    Scope,
    SystemPollResult,
)

# Mixin pattern: at type-check time we inherit TestCase so pyre sees
# `self.assert*` and `self.skipTest`; at runtime the mixin is a plain
# object so unittest's auto-discovery doesn't instantiate it as a test
# suite of its own.
if TYPE_CHECKING:
    _MixinBase = unittest.TestCase
else:
    _MixinBase = object


class RingBufferBasicTest(unittest.TestCase):
    """Tests for RingBuffer creation and properties."""

    def test_creation(self) -> None:
        ring = RingBuffer(1024)
        self.assertTrue(ring.valid())
        self.assertEqual(ring.size, 1024)
        self.assertEqual(ring.scope, Scope.DEVICE)

    def test_creation_system_scope(self) -> None:
        ring = RingBuffer(1024, Scope.SYSTEM)
        self.assertTrue(ring.valid())
        self.assertEqual(ring.size, 1024)
        self.assertEqual(ring.scope, Scope.SYSTEM)

    def test_rounds_up_to_power_of_two(self) -> None:
        ring = RingBuffer(1000)
        self.assertTrue(ring.valid())
        self.assertEqual(ring.size, 1024)

    def test_device_handle_returns_four_ints(self) -> None:
        ring = RingBuffer(512)
        handle = ring.device_handle()
        self.assertEqual(len(handle), 4)
        ring_ptr, write_idx_ptr, mask, shift = handle
        self.assertNotEqual(ring_ptr, 0)
        self.assertNotEqual(write_idx_ptr, 0)
        self.assertEqual(mask, 511)


class _ReaderPollMixin(_MixinBase):
    """Reader.poll tests parametrized by MemoryCoherenceScope.

    Subclasses set ``SCOPE`` and inherit from this mixin + TestCase. The
    helpers (`_make_ring`, `_poll`) abstract over the per-scope poll
    signature (Device takes a stream; System takes a timeout), so the
    test bodies stay scope-agnostic.
    """

    # Subclasses override.
    SCOPE: Scope = Scope.DEVICE

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("No CUDA device available")
        torch.cuda.set_device(0)

    def _make_ring(self, size: int) -> RingBuffer:
        return RingBuffer(size, self.SCOPE)

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def _poll(
        self, reader: Reader
    ) -> tuple[list[Entry], DevicePollResult | SystemPollResult]:
        # System reader doesn't need a stream — the writer's stream is
        # synchronized separately (below) and the host reads pinned
        # memory directly. Device reader requires the stream.
        if self.SCOPE == Scope.DEVICE:
            return reader.poll(stream=self._stream())
        return reader.poll()

    def _sync_writer(self) -> None:
        # System-scope: host polls pinned memory without going through
        # the CUDA stream, but the writes themselves are still issued on
        # a stream; sync explicitly so the test observes them. For
        # Device scope this is redundant with poll's internal sync.
        if self.SCOPE == Scope.SYSTEM:
            torch.cuda.synchronize()

    def test_poll_empty_ring(self) -> None:
        ring = self._make_ring(256)
        reader = Reader(ring)
        entries, result = self._poll(reader)
        self.assertEqual(len(entries), 0)
        self.assertEqual(result.entries_read, 0)
        self.assertEqual(result.entries_lost, 0)

    def test_poll_returns_entries(self) -> None:
        ring = self._make_ring(256)
        reader = Reader(ring)

        ring.write(self._stream(), 42)
        ring.write(self._stream(), 99)
        self._sync_writer()

        entries, result = self._poll(reader)
        self.assertEqual(result.entries_read, 2)
        self.assertEqual(result.entries_lost, 0)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].data, 42)
        self.assertEqual(entries[1].data, 99)
        self.assertGreater(entries[0].timestamp, 0)
        self.assertLessEqual(entries[0].timestamp, entries[1].timestamp)

    def test_poll_incremental(self) -> None:
        ring = self._make_ring(256)
        reader = Reader(ring)

        ring.write(self._stream(), 1)
        self._sync_writer()
        entries1, r1 = self._poll(reader)
        self.assertEqual(r1.entries_read, 1)
        self.assertEqual(entries1[0].data, 1)

        ring.write(self._stream(), 2)
        self._sync_writer()
        entries2, r2 = self._poll(reader)
        self.assertEqual(r2.entries_read, 1)
        self.assertEqual(entries2[0].data, 2)

    def test_entry_fields(self) -> None:
        ring = self._make_ring(256)
        reader = Reader(ring)

        ring.write(self._stream(), 12345)
        self._sync_writer()
        entries, _ = self._poll(reader)
        self.assertGreater(entries[0].timestamp, 0)
        self.assertGreater(entries[0].epoch, 0)
        self.assertEqual(entries[0].data, 12345)

    def test_entry_repr(self) -> None:
        ring = self._make_ring(256)
        reader = Reader(ring)

        ring.write(self._stream(), 42)
        self._sync_writer()
        entries, _ = self._poll(reader)
        r = repr(entries[0])
        self.assertIn("timestamp=", r)
        self.assertIn("epoch=", r)
        self.assertIn("data=42", r)

    def test_poll_result_repr(self) -> None:
        ring = self._make_ring(256)
        reader = Reader(ring)
        _, result = self._poll(reader)
        r = repr(result)
        self.assertIn("entries_read=0", r)
        self.assertIn("entries_lost=0", r)

    def test_packed_uint64(self) -> None:
        """Verify packing two uint32 fields into uint64 round-trips."""
        ring = self._make_ring(256)
        reader = Reader(ring)

        kernel_id, phase = 42, 1
        tag = (kernel_id << 32) | phase
        ring.write(self._stream(), tag)
        self._sync_writer()

        entries, _ = self._poll(reader)
        recovered_id = entries[0].data >> 32
        recovered_phase = entries[0].data & 0xFFFFFFFF
        self.assertEqual(recovered_id, 42)
        self.assertEqual(recovered_phase, 1)


class ReaderPollDeviceTest(_ReaderPollMixin, unittest.TestCase):
    """Reader.poll tests against a Device-scope ring (cudaMalloc + cudaMemcpy drain)."""

    SCOPE: Scope = Scope.DEVICE


class ReaderPollSystemTest(_ReaderPollMixin, unittest.TestCase):
    """Reader.poll tests against a System-scope ring (pinned mapped memory)."""

    SCOPE: Scope = Scope.SYSTEM

    def test_poll_timeout_returns_timed_out(self) -> None:
        """System-scope poll honors the timeout arg when the ring is empty."""
        ring = self._make_ring(256)
        reader = Reader(ring)

        # Empty ring + non-zero timeout: poll waits the full timeout
        # then returns timed_out=True with zero entries.
        start = datetime.datetime.now()
        entries, result = reader.poll(timeout=datetime.timedelta(milliseconds=20))
        elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
        self.assertEqual(result.entries_read, 0)
        self.assertEqual(len(entries), 0)
        # System-scope poll returns SystemPollResult — narrow for pyre.
        self.assertIsInstance(result, SystemPollResult)
        assert isinstance(result, SystemPollResult)
        self.assertTrue(result.timed_out)
        # Loose lower bound — gives us confidence the timeout was honored
        # without making the test flaky on slow systems.
        self.assertGreaterEqual(elapsed_ms, 15.0)
