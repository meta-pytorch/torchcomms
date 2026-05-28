# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import datetime
import enum

class Scope(enum.IntEnum):
    DEVICE = 0
    SYSTEM = 1

class Entry:
    timestamp: int
    epoch: int
    data: int

class DevicePollResult:
    entries_read: int
    entries_lost: int
    error: int

class SystemPollResult:
    entries_read: int
    entries_lost: int
    timed_out: bool

class RingBuffer:
    def __init__(self, size: int, scope: Scope = Scope.DEVICE) -> None: ...
    def valid(self) -> bool: ...
    @property
    def size(self) -> int: ...
    @property
    def scope(self) -> Scope: ...
    def device_handle(self) -> tuple[int, int, int, int]: ...
    def write(self, stream: int, data: int) -> None: ...

class Reader:
    def __init__(self, ring: RingBuffer) -> None: ...
    def poll(
        self,
        stream: int = 0,
        timeout: datetime.timedelta = ...,
    ) -> tuple[list[Entry], DevicePollResult | SystemPollResult]: ...

def refresh_clock() -> bool: ...
