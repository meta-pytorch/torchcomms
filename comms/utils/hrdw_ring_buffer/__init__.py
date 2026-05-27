# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast, Generic, NamedTuple, TypeVar

from hrdw_ring_buffer._native import (
    DevicePollResult,
    Reader as _Reader,
    RingBuffer as _RingBuffer,
    SystemPollResult,
)

T = TypeVar("T")


class RingHandle(NamedTuple):
    """Device handle for kernel launches: (ring_ptr, write_index_ptr, mask, shift).

    Unpacks via ``*handle`` for direct kernel-arg injection, or use named
    fields when you need to pass them individually.
    """

    ring_ptr: int
    write_index_ptr: int
    mask: int
    shift: int


@dataclass
class Entry(Generic[T]):
    """A ring buffer entry with decoded data."""

    timestamp: int
    epoch: int
    data: T


class RingBuffer(Generic[T]):
    """Device-scope GPU ring buffer with user-defined event types.

    The underlying ring stores (uint32 timestamp, uint32 epoch, uint64 data)
    16-byte entries. An optional ``unpack`` function decodes the raw uint64
    data field into a user-defined type on poll.

    A consumer Reader is auto-constructed and bound to this buffer; call
    ``poll(stream)`` to drain entries since the previous poll. The read
    cursor auto-advances per poll and the device writeIndex grows
    monotonically (slots wrap into the ring via ``& mask``), so no reset
    is needed between polls.

    Examples::

        # Raw uint64 tags (no unpack)
        ring: RingBuffer[int] = RingBuffer(8192)
        entries, result = ring.poll(stream)
        for e in entries:
            print(e.timestamp, e.data)  # data is int

        # Custom event type
        @dataclass
        class KernelEvent:
            kernel_id: int  # upper 32 bits
            op_type: int    # lower 32 bits

        ring: RingBuffer[KernelEvent] = RingBuffer(
            8192,
            unpack=lambda d: KernelEvent(d >> 32, d & 0xFFFFFFFF),
        )
        entries, result = ring.poll(stream)
        for e in entries:
            print(e.data.kernel_id, e.data.op_type)
    """

    def __init__(
        self,
        size: int,
        unpack: Callable[[int], T] | None = None,
    ) -> None:
        self._ring = _RingBuffer(size)
        self._reader = _Reader(self._ring)
        self._unpack = unpack

    @property
    def valid(self) -> bool:
        return self._ring.valid()

    @property
    def size(self) -> int:
        return self._ring.size

    def device_handle(self) -> RingHandle:
        """Returns the device handle for kernel launches. Unpack with ``*handle``."""
        return RingHandle(*self._ring.device_handle())

    def poll(self, stream: int) -> tuple[list[Entry[T]], DevicePollResult]:
        """Poll newly-written entries from the ring buffer.

        Auto-advances the internal read cursor on every call — successive
        polls only return entries written since the previous poll. The
        device writeIndex grows monotonically and slots wrap into the
        ring via `& mask`, so there's no need to reset between polls.

        Args:
            stream: raw cudaStream_t as int64.

        Returns:
            (entries, result) where each entry has .timestamp, .epoch, and
            .data (decoded via unpack if provided), and result is a
            DevicePollResult with entries_read / entries_lost / error.
        """
        raw_entries, result = self._reader.poll(stream)
        unpack = self._unpack
        # When unpack is None, T is bound to int by the caller; cast lets
        # pyre see both branches yielding T.
        decode: Callable[[int], T] = (
            unpack if unpack is not None else cast(Callable[[int], T], lambda d: d)
        )
        entries: list[Entry[T]] = [
            Entry(timestamp=e.timestamp, epoch=e.epoch, data=decode(e.data))
            for e in raw_entries
        ]
        # This wrapper is device-scope only, so the variant is always
        # DevicePollResult — assert + cast for type narrowing.
        assert isinstance(result, DevicePollResult)
        return entries, result

    def write(self, stream: int, data: int) -> None:
        """Write one entry from the host (launches a single-thread kernel)."""
        self._ring.write(stream, data)


__all__ = [
    "DevicePollResult",
    "Entry",
    "RingBuffer",
    "RingHandle",
    "SystemPollResult",
]
