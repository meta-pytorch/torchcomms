# Graph Timeout Detection Design

## Overview

TorchCommNCCLX supports timeout detection for collectives during CUDA graph replay.
Unlike eager mode where the watchdog queries per-work events from a FIFO queue,
graph-captured collectives persist across replays and require a separate tracking
mechanism. The `GraphEventTracker` class provides this functionality.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      TorchCommNCCLX                             │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │    Event Pool         │    │    Timeout Watchdog Thread    │  │
│  │  (eager mode only)    │    │                              │  │
│  │  getEvent()/          │    │  checkWorkQueue()            │  │
│  │  returnEvent()        │    │    → eager work FIFO GC      │  │
│  └──────────┬───────────┘    │                              │  │
│             │                │  checkGraphEvents()           │  │
│             │                │    → graph_event_tracker_     │  │
│             │                │      .checkAll()              │  │
│             │                └──────────────────────────────┘  │
│             │                                                   │
│  ┌──────────▼───────────────────────────────────────────────┐  │
│  │                   TorchWorkNCCLX                          │  │
│  │                                                           │  │
│  │  start_event_  — start detection (pool / ad-hoc)          │  │
│  │  end_event_    — completion detection (pool / ad-hoc)     │  │
│  │  sync_event_   — stream join, graph only (nullptr eager)  │  │
│  │                                                           │  │
│  │  initEvents() / releaseEvents() — lifecycle management    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  TorchWorkNCCLXQueue    │  │  GraphEventTracker           │  │
│  │  (eager mode)           │  │  (graph mode)                │  │
│  │                         │  │                               │  │
│  │  Per-stream FIFO of     │  │  Per-graph GraphState:        │  │
│  │  intrusive_ptr<Work>    │  │    vector<GraphWork>          │  │
│  │                         │  │    atomic replay_counter      │  │
│  │                         │  │                               │  │
│  │  GC: pop when done      │  │  Cleanup via cudaUserObject   │  │
│  │  Work dtor returns      │  │  Replay detect via host node  │  │
│  │  events to pool         │  │                               │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Event Design

### Why Three Events in Graph Mode

CUDA graph capture records `cudaEventRecord` calls as graph nodes. Regular-recorded
events become opaque during replay and cannot be queried from the host. To enable
host-side timeout detection, we use `cudaEventRecordExternal` for start/end events,
which creates EVENT_RECORD nodes that remain host-queryable during replay.

However, externally-recorded events are NOT recognized by `cudaStreamWaitEvent` as
valid stream join points (`cudaErrorStreamCaptureUnjoined`). So we need a third event
(`sync_event_`) recorded with regular `cudaEventRecord` purely for `work.wait()`.

| Event | Recording API | Purpose | Eager | Graph |
|-------|--------------|---------|-------|-------|
| `start_event_` | Eager: `cudaEventRecord` / Graph: `cudaEventRecordExternal` | Detect collective start | Pool | Ad-hoc, transferred to GraphWork |
| `end_event_` | Eager: `cudaEventRecord` / Graph: `cudaEventRecordExternal` | Detect collective end, timeout detection | Pool | Ad-hoc, transferred to GraphWork |
| `sync_event_` | `cudaEventRecord` (regular) | Stream join for `work.wait()` | N/A (nullptr) | Ad-hoc, destroyed in Work dtor |

### Event Lifecycle

**Eager mode:** Pool events, one-shot lifecycle.
```text
Pool.get() → Work ctor → record → watchdog query → GC → Work dtor → Pool.return()
```

**Graph mode:** Ad-hoc events, persistent across replays.
```text
Capture:  cudaEventCreate → Work ctor → record (External) → enqueueWork
          → transfer start/end to GraphWork → Work dtor (destroys sync_event_ only)

Replay:   GPU replays EVENT_RECORD_EXT nodes → watchdog queries → timeout check

Cleanup:  Graph destruction → cudaUserObject callback sets released flag
          → watchdog checkAll() → cleanupReleasedGraphs() → destroyEvents()
```

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
