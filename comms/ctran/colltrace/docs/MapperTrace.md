# MapperTrace

MapperTrace records mapper-level events during NCCL collective operations for diagnostic purposes. It is consumed by the CommsMonitor dump thread to produce a snapshot of the current collective's progress, including which requests are in-flight, which peers have been notified, and which puts have completed. It is part of "CollTrace" framework.

## Overview

Each `CtranMapper` owns a single `MapperTrace` instance. During a collective, the GPE thread records events (puts, copies, control messages, etc.) into `MapperTrace`. A separate thread (Usually from the Analyzer thrift service) periodically calls `dump()` to read those events and produce a diagnostic snapshot without blocking the GPE thread.

```
GPE Thread                              CommsMonitor Thread
    |                                           |
    |  recordMapperEvent(CollStart)             |
    |  recordMapperEvent(PutStart)              |
    |  recordMapperEvent(CopyStart)             |    dump()
    |  recordMapperEvent(RecvNotified)          |      |
    |  recordMapperEvent(MapperRequestEnd)      |      |-> reads events [0..N)
    |  recordMapperEvent(CollEnd)               |      |-> returns Dump snapshot
    |                                           |
```

## Event Types

Events are represented as a `std::variant` called `MapperEvent`:

| Event | Description | Recorded by |
|-------|-------------|-------------|
| `CollStart` | A collective operation begins. Carries the `ICollRecord` identifying it. | GPE thread (slow path, holds lock) |
| `CollEnd` | A collective operation ends. Resets all per-collective state. | GPE thread (slow path, holds lock) |
| `PutStart` | An RDMA put operation begins. | GPE thread (fast path) |
| `CopyStart` | A device memory copy begins. | GPE thread (fast path) |
| `SendCtrlStart` | A send control message begins. | GPE thread (fast path) |
| `RecvCtrlStart` | A receive control message begins. | GPE thread (fast path) |
| `SendSyncCtrlStart` | A synchronous send control message begins. | GPE thread (fast path) |
| `RecvSyncCtrlStart` | A synchronous receive control message begins. | GPE thread (fast path) |
| `RecvNotified` | A peer has signaled that data is ready. | GPE thread (fast path) |
| `MapperRequestEnd` | A previously started request has completed. | GPE thread (fast path) |

Events are split into two categories based on their recording path:

- **Slow path** (`CollStart`, `CollEnd`): These acquire `curCollInfoLocked_` and update `CurCollInfo` directly. They bracket the collective lifecycle.
- **Fast path** (all others): These are written directly into the `eventHistory_` buffer without acquiring any lock. They are performance-critical.

## Data Structures

### `eventHistory_` -- The Fixed-Size Event Buffer

```cpp
std::unique_ptr<MapperEvent[]> eventHistory_;  // pre-allocated buffer
std::atomic<int64_t> eventHistorySizeAtomic_;  // published write index
const uint64_t maxEventCount_;                 // buffer capacity
```

`eventHistory_` is a pre-allocated fixed-size array of `MapperEvent`. It is sized to `NCCL_MAPPERTRACE_EVENT_RECORD_MAX` (configurable via constructor) and **never reallocates**.

Only fast-path events are stored in this buffer. `CollStart` and `CollEnd` are handled through the locked `CurCollInfo` path and do not occupy slots in `eventHistory_`.

### `CurCollInfo` -- Per-Collective Aggregated State

```cpp
struct CurCollInfo {
    int64_t readIndex{0};              // next event to process from eventHistory_
    std::shared_ptr<ICollRecord> currentColl;  // active collective
    std::unordered_map<int, int> recvNotifiedByPeer;  // recv notification counts
    std::unordered_map<int, int> putFinishedByPeer;   // completed put counts
    std::unordered_map<const CtranMapperRequest*, uint64_t> unfinishedRequests;
};
```

Protected by `folly::Synchronized<CurCollInfo> curCollInfoLocked_`. Both `dump()` and the slow-path events (`CollStart`/`CollEnd`) hold this lock.

`readIndex` tracks how far `dump()` has consumed into `eventHistory_`. When `dump()` runs, it processes events from `readIndex` up to the current published `eventHistorySizeAtomic_` value, updating the aggregated maps.

## Concurrency Model

Two threads interact with `MapperTrace`:

1. **GPE thread** -- the sole writer of `eventHistory_` (enforced by `shouldMapperTraceCurrentThread`)
2. **CommsMonitor dump thread** -- reads `eventHistory_` via `dump()`

### Why No Mutex on the Fast Path

The fast path avoids locking because acquiring a mutex on every event would be prohibitively expensive during a collective (see [Performance](#performance)). Instead, thread safety relies on three properties:

**Property 1: Single writer.** Only the GPE thread writes to `eventHistory_`. The thread-local flag `shouldMapperTraceCurrentThread` ensures non-GPE threads early-return without writing.

**Property 2: Publish/consume via atomic.** The GPE thread writes the event to `eventHistory_[nextIndex]` *before* publishing `nextIndex + 1` with a release store. The dump thread loads `eventHistorySizeAtomic_` with acquire semantics, so it only sees fully-written events at indices below the published value.

```
GPE thread:                             Dump thread:
  eventHistory_[N] = event;  (write)      size = atomic.load(acquire);
  atomic.store(N+1, release);             // reads eventHistory_[0..size) safely
```

**Property 3: CollEnd and dump don't overlap.** Both `CollEnd` and `dump()` acquire `curCollInfoLocked_`. When `CollEnd` runs, it resets `eventHistorySizeAtomic_` to 0 and clears `CurCollInfo.readIndex` to 0. No dump can be in progress during this reset because both hold the same lock.

### Fixed-Size Buffer Design

`eventHistory_` is a `std::unique_ptr<MapperEvent[]>` rather than a `std::vector`. This is critical for safety:

- A `std::vector` can **reallocate** its internal storage when `emplace_back` exceeds capacity. If the dump thread is reading the old storage while the GPE thread triggers reallocation, the dump thread dereferences freed memory (SIGSEGV).
- The fixed-size buffer **never moves in memory**. The GPE thread overwrites slots by index, and the dump thread reads by index. Both are accessing the same stable memory region.

After `CollEnd` resets the atomic counter to 0, the next collective's events overwrite the buffer starting from index 0. This is safe because `CollEnd` also resets `CurCollInfo.readIndex` to 0 under the lock, so the dump thread will start fresh.

### Lifecycle of a Collective

```
                  curCollInfoLocked_ held
                        |
CollStart ──────────────┤
                        |
   PutStart ────────────┤──── eventHistory_[0] (no lock)
   CopyStart ───────────┤──── eventHistory_[1] (no lock)
   RecvNotified ────────┤──── eventHistory_[2] (no lock)
   MapperRequestEnd ────┤──── eventHistory_[3] (no lock)
   ...                  |
                        |
                        |     dump() may run here, reads [0..N), holds lock
                        |
CollEnd ────────────────┤     resets atomic to 0, clears CurCollInfo
                        |
```

### The `dump()` Method

`dump()` acquires `curCollInfoLocked_` and processes unread events from `eventHistory_`:

1. Loads `eventHistorySizeAtomic_` (acquire).
2. Iterates from `CurCollInfo.readIndex` to the loaded size, calling `recordMapperEventImpl()` on each event to update `CurCollInfo` (incrementing recv/put counts, tracking unfinished requests).
3. Advances `readIndex` to the current size.
4. Returns a `Dump` snapshot containing the current collective, per-peer counts, and unfinished requests.

Because `dump()` holds `curCollInfoLocked_`, it cannot overlap with `CollEnd`. This guarantees that `readIndex` and `eventHistorySizeAtomic_` are consistent: the dump thread never reads beyond what the GPE thread has written, and the GPE thread never resets the counter while a dump is in progress.

### Thread-Safety Summary

| Operation | Thread | Lock held? | Accesses |
|-----------|--------|------------|----------|
| `recordMapperEvent(CollStart)` | GPE | `curCollInfoLocked_` | `CurCollInfo` |
| `recordMapperEvent(CollEnd)` | GPE | `curCollInfoLocked_` | `CurCollInfo`, `eventHistorySizeAtomic_` |
| `recordMapperEvent(fast-path)` | GPE | None | `eventHistory_[N]`, `eventHistorySizeAtomic_` |
| `dump()` | CommsMonitor | `curCollInfoLocked_` | `eventHistory_[0..N)`, `eventHistorySizeAtomic_`, `CurCollInfo` |

## Performance

The fast-path `recordMapperEvent` is designed to have minimal overhead. Benchmark results on devserver (per-event amortized cost):

| Event Type | Time per Event |
|------------|---------------|
| `RecvNotified` | ~7 ns |
| `PutStart` | ~53 ns |
| `CopyStart` | ~80 ns |

`RecvNotified` is the cheapest because it carries only a single `int` field. `PutStart` and `CopyStart` are larger structs that include pointers, sizes, and (for `PutStart`) a `CtranMapperRemoteAccessKey` that must be copied.

The overhead of `CollStart` and `CollEnd` is higher due to the mutex acquisition, but they run only twice per collective (once each), so their cost is amortized across all events in the collective.

Run the benchmark:
```bash
buck run @mode/opt //comms/ctran/colltrace/benchmarks:mapper_trace_bench
```

## Files

| File | Description |
|------|-------------|
| `MapperTrace.h` | Class declaration, event type definitions, `recordMapperEvent` template |
| `MapperTrace.cc` | `dump()`, `recordMapperEventImpl` overloads, serialization |
| `tests/MapperTraceTest.cc` | Unit tests including concurrent stress test |
| `benchmarks/MapperTraceBench.cc` | folly benchmarks for fast-path event types |
