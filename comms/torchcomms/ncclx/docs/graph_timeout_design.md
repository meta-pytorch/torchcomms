# Graph Timeout Detection Design

## Overview

TorchCommNCCLX supports timeout detection for collectives during CUDA graph replay.
Unlike eager mode where the watchdog queries per-work events from a FIFO queue,
graph-captured collectives persist across replays and require a separate tracking
mechanism. The `GraphEventTracker` class provides this functionality.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        TorchCommNCCLX                           │
│                                                                 │
│  ┌────────────────────────┐    ┌─────────────────────────────┐  │
│  │ Event Pool             │    │ Timeout Watchdog Thread     │  │
│  │ (eager mode)           │    │                             │  │
│  │ getEvent()/            │    │ checkWorkQueue()            │  │
│  │ returnEvent()          │    │   → eager work FIFO GC      │  │
│  └───────────┬────────────┘    │                             │  │
│              │                 │ checkGraphEvents()          │  │
│              │                 │   → graph_event_tracker_    │  │
│              │                 │     .checkAll()             │  │
│              │                 └─────────────────────────────┘  │
│              │                                                  │
│  ┌───────────▼───────────────────────────────────────────────┐  │
│  │ TorchWorkNCCLX                                            │  │
│  │                                                           │  │
│  │ start_event_ — start detection (pool / ad-hoc)            │  │
│  │ end_event_   — completion detection (pool / ad-hoc)       │  │
│  │                                                           │  │
│  │ initEvents() / releaseEvents() — lifecycle management     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │ TorchWorkNCCLXQueue     │  │ GraphEventTracker            │  │
│  │ (eager mode)            │  │ (graph mode)                 │  │
│  │                         │  │                              │  │
│  │ Per-stream FIFO of      │  │ Per-graph GraphState:        │  │
│  │ intrusive_ptr<Work>     │  │   vector<GraphWork>          │  │
│  │                         │  │   atomic replay_counter      │  │
│  │                         │  │                              │  │
│  │ GC: pop when done       │  │ Cleanup via cudaUserObject   │  │
│  │ Work dtor returns       │  │ Replay detect via host node  │  │
│  │ events to pool          │  │                              │  │
│  └─────────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Event Design

### Events in Graph Mode

CUDA graph capture records `cudaEventRecord` calls as graph nodes. Regular-recorded
events become opaque during replay and cannot be queried from the host. To enable
host-side timeout detection, we use `cudaEventRecordExternal` for start/end events,
which creates EVENT_RECORD nodes that remain host-queryable during replay.

The `cudaEventRecordExternal` calls create explicit graph nodes on `internal_stream_`
(s1). These explicit nodes cause CUDA's fork/join checker to consider s1 as "active"
at `capture_end()`. To resolve this, `work.wait()` during graph capture performs a
two-step join:

1. `cudaStreamUpdateCaptureDependencies(s1, nullptr, 0, SET)` — clears s1's tracked
   fork state so the fork/join checker no longer considers it active.
2. `cudaStreamWaitEvent(s0, end_event_)` — creates a graph edge from s1's
   EVENT_RECORD_EXT node to s0, ensuring proper execution ordering during replay.

The next collective's `getOperationStream()` re-forks s1 from s0 as usual. All
async ops must be waited during graph capture to avoid
`cudaErrorStreamCaptureUnjoined` (error 904).

| Event | Recording API | Purpose | Eager | Graph |
|-------|--------------|---------|-------|-------|
| `start_event_` | Eager: `cudaEventRecord` / Graph: `cudaEventRecordExternal` | Detect collective start | Pool | Ad-hoc, transferred to GraphWork |
| `end_event_` | Eager: `cudaEventRecord` / Graph: `cudaEventRecordExternal` | Detect collective end, timeout detection | Pool | Ad-hoc, transferred to GraphWork |

### Event Lifecycle

**Eager mode:** Pool events, one-shot lifecycle.
```text
Pool.get() → Work ctor → record → watchdog query → GC → Work dtor → Pool.return()
```

**Graph mode:** Ad-hoc events, persistent across replays.
```text
Capture:  cudaEventCreate (start/end) → Work ctor → record
          → enqueueWork → copy event ptrs to GraphWork
          → work.wait() → SET(s1, empty) + streamWaitEvent(s0, end_event_)

Replay:   GPU replays EVENT_RECORD_EXT nodes → watchdog queries → timeout check

Cleanup:  Graph destruction → cudaUserObject callback sets released flag
          → watchdog checkAll() → cleanupReleasedGraphs() → destroyEvents()
```

## Stream Join Mechanism

### Problem

TorchCommNCCLX runs collectives on the internal stream `stream_` (s1), which is
forked from the user stream (s0) via `getOperationStream()`. The
`cudaEventRecordExternal` calls for `start_event_` and `end_event_` create explicit
`EVENT_RECORD` graph nodes on s1. If s1 is not joined back to s0 before
`cudaStreamEndCapture`, CUDA
detects the unjoined fork and returns `cudaErrorStreamCaptureUnjoined` (error 904).

We've previously avoided this by only using regular `cudaEventRecord` (which
don't materialize to explicit graph nodes). A forked stream with only implicit
nodes is transparent to the fork/join checker, thus we don't run into the error.

### Solution

When `work.wait()` is called during graph capture, it performs a two-step join:

```text
Per collective during graph capture:

  getOperationStream()
    eventRecord(dep_event, s0)       ← fork: s0 records, s1 waits
    streamWaitEvent(s1, dep_event)

  NCCL collective on s1
  cudaEventRecordExternal(start_event_, s1)   ← explicit node on s1
  cudaEventRecordExternal(end_event_, s1)     ← explicit node on s1

  enqueueWork()
    copy event ptrs to GraphEventTracker (for timeout monitoring)

  ... user compute on s0 can overlap with collective ...

  work.wait()
    streamUpdateCaptureDependencies( ← clear s1's tracked fork
        s1, {}, SET)
    streamWaitEvent(s0, end_event_)  ← graph edge: s0 depends on s1
```

The `SET(empty)` clears s1's tracked tail nodes so the fork/join checker at
`capture_end()` no longer considers s1 as having unjoined work. Clearing the
tracked deps does NOT remove the EVENT_RECORD_EXT graph nodes — they remain in
the graph. The `streamWaitEvent` creates the actual graph edge from the
EVENT_RECORD_EXT node to s0, ensuring correct execution ordering during replay.

This preserves async overlap: user compute on s0 between `enqueueWork()` and
`work.wait()` is not blocked by the collective. The join happens at the natural
synchronization point — the same place eager mode joins via `streamWaitEvent`.

**Important**: All async ops must be waited during graph capture. Unwaited ops
leave s1 unjoined, causing error 904 at `capture_end()`.

## GraphEventTracker Timeout Logic

### State Machine

During a single graph replay, the GPU executes nodes in this order:
```text
host_node (counter++) → start_event record → NCCL collective → end_event record
```

The watchdog polls `checkAll()` periodically, which queries each entry's events
and determines the current state:

```text
                                  replay_counter changed
                                 ┌────────────────────────┐
                                 │                        │
                                 ▼                        │
                       ┌──────────────────┐               │
            ┌─────────►│    COMPLETED     │───────────────┘
            │          │  (between replays │
            │          │   or coll done)   │
            │          └────────┬─────────┘
            │                   │ end = notReady
            │                   │ start = notReady
            │                   ▼
            │          ┌──────────────────┐
            │          │   NOT REACHED    │
            │          │  (replay started │
            │          │   but GPU before │
            │          │   this coll)     │
            │          └────────┬─────────┘
            │                   │ start = success
            │                   ▼
  end =     │          ┌──────────────────┐
  success   │          │   IN PROGRESS    │  elapsed > timeout
            └──────────│  (start done,    │──────► abort()
                       │   waiting end)   │
                       └──────────────────┘
```

State detection (no enum — derived from event queries each poll):
- **COMPLETED**: `end_event` query returns `cudaSuccess` → reset timer
- **NOT REACHED**: both `start_event` and `end_event` return `cudaErrorNotReady` → reset timer
- **IN PROGRESS**: `start_event` returns `cudaSuccess`, `end_event` returns `cudaErrorNotReady` → start/continue timer; if elapsed > timeout, return TIMEOUT

### Replay Boundary Detection

A CUDA host node (`launchHostFunc`) fires before any collective's start event in
each replay, incrementing an `atomic<uint64_t>` replay counter. If the counter
changes between polls, all timers reset — preventing false timeouts that span
multiple replays.

## Resource Cleanup

Event cleanup uses a **deferred model** to avoid CUDA API calls and lock
acquisition inside CUDA callbacks (which run on a shared internal thread per CUDA docs).

Three paths ensure events are always destroyed:

1. **Graph destruction → deferred cleanup**: The CUDA `cudaUserObject`
   destructor (`cleanupCallback`) does a single atomic store:
   `released.store(true)`. No lock, no CUDA calls. On the next watchdog poll,
   `cleanupReleasedGraphs()` (called at the top of `checkAll()`) finds entries
   with `released == true`, destroys their events, and erases the `GraphState`.

2. **Comm finalization** (`destroyAll()`): Called from `TorchCommNCCLX::finalize()`.
   Destroys all remaining events across all graphs unconditionally (regardless
   of `released` flag). Handles cases where the comm is finalized before graphs
   are destroyed.

3. **Watchdog periodic cleanup**: `cleanupReleasedGraphs()` runs on every
   `checkAll()` invocation, ensuring released graphs are cleaned up promptly
   even without explicit finalization.

All map/vector mutations happen under `mutex_`. The `cleanupCallback` is
fully lock-free — it only writes to an atomic in the static pool.
