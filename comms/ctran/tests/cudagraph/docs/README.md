# CTRAN CUDA Graph Test Framework

A builder-based framework for testing CTRAN collectives under CUDA graph
capture, replay, and lifecycle scenarios.

## Architecture

- **Test File (.cc)** — uses `DEFINE_CUDAGRAPH_PARAM_TEST(AllGather, algoDescriptor)`
  - Defines an `AlgoDescriptor`: `makeBuffers`, `capture`, `isSupported`
  - Parameterized over (algo, pattern, count, replayMultiplier)
- **CtranCudaGraphParamTest.h** — test harness that the test file uses
  - `GraphPattern` enum: Basic, MultipleSequential, MultiStream, ...
  - `AlgoDescriptor`: algorithm-agnostic interface
  - `runPattern()`: dispatches to pattern-specific runner
  - `DEFINE_CUDAGRAPH_PARAM_TEST` / `DEFINE_CUDAGRAPH_STRESS_TEST` macros
- **CudaGraphTestBuilder** — builder pattern for graph test execution, called by the harness
  - `.addCapture(fn)` — single graph capture
  - `.addSchedule(sched)` — multi-step eager/graph pipeline
  - `.withReset(fn)` — buffer reset between replays
  - `.withDeviceVerify(fn)` — GPU-side result comparison
  - `.withGraphAssertions()` — topology checks (host/kernel nodes)
  - `.withNumReplays(n)` — number of replay iterations
  - `.run()` — execute capture, replay, verify
- **CtranCudaGraphTestBase.h** — environment setup
  - `expectGraphNodes` — graph topology assertions
  - `GpePoolGuard` — RAII guard for GPE pool state
- **DeviceVerify (.cu / .h)** — GPU-side verification
  - `launchCompareBuffers()`: GPU kernel that compares actual vs expected buffers
  - `DeviceMismatchCounter`: RAII counter for mismatches

## Key Components

### AlgoDescriptor

Algorithm-agnostic interface that each collective test provides:

```cpp
AlgoDescriptor desc{
    .name = "AllGather",
    .isSupported = [](CtranComm* c, size_t count, int nRanks) { return true; },
    .makeBuffers = [](size_t count, int rank, int nRanks) {
        // Return shared_ptr<Buffers> with sendbuf/recvbuf/recvBytes
    },
    .capture = [](Buffers* bufs, size_t count, CaptureContext& ctx) {
        // Call the collective on ctx.comm / ctx.stream
    },
};
```

### Graph Patterns

Each pattern tests a different lifecycle or concurrency scenario:

| Pattern              | Edge Case                                                          |
|----------------------|--------------------------------------------------------------------|
| `Basic`              | Baseline correctness: GPE host-node callbacks, kernel flag/elem allocation, and pool reclamation work correctly across multiple replays. Catches regressions in the core capture/replay path. |
| `MultipleSequential` | Multiple collectives in a single graph share the same GPE submission pipeline. Tests that kernel flags, elems, checksums, and sync objects are allocated per-op and don't alias or leak when N ops coexist in one graph. |
| `MultiStream`        | Fork-join topology inside a captured graph. Tests that CTRAN's stream-ordered resources (GPE submissions, kernel flags) are correctly scoped per-stream and that cross-stream event dependencies survive graph replay. |
| `DestroyRecreate`    | Repeated graph destroy + re-capture cycles. Tests that GPE pool resources (kernel flags, elems, checksums, syncs) are fully reclaimed on graph destroy and can be re-allocated on the next capture without leaks or stale state. |
| `MixedEagerGraph`    | Eager → Graph → Eager execution sequence. Tests that graph capture/replay doesn't corrupt CTRAN's internal state (connection state, GPE queues, backend handles) for subsequent eager operations. |
| `MultiGraph`         | Two independent graphs captured from the same comm. Tests that per-graph resource isolation is correct — each graph gets its own kernel flags/elems and destroying one doesn't invalidate the other's resources. |
| `InPlace`            | Send and recv buffers alias (same pointer). Tests that CTRAN's buffer registration, NVL copies, and GPE kernel launches handle the in-place case without double-free or incorrect DMA source/dest during replay. |
| `Abort`              | Replay a graph after `comm->setAbort()`. Tests that the abort signal propagates through GPE host-node callbacks during replay and unblocks the collective promptly without deadlocking on kernel flags or sync objects. |

#### Topologies

Tests run across three simulated topologies (1 node, 8 GPUs each):

| Topology Suffix         | `NCCL_COMM_STATE_DEBUG_TOPO` | Description                                |
|-------------------------|------------------------------|--------------------------------------------|
| `1x8_init_none`         | *(default)*                  | Default NVL topology — all GPUs see each other as local |
| `1x8_nolocal_init_none` | `nolocal`                    | Simulated multi-node — no NVL, forces IB/network paths |
| `1x8_vnode_init_none`   | `vnode`                      | Virtual node topology (effectively 4x2) — tests hybrid local/remote splits |

All topologies use `NCCL_FASTINIT_MODE=none` to ensure full initialization
without fast-init shortcuts.

### Algorithm Coverage Matrix

Each algorithm has multiple sub-variants (e.g., different transport paths or
algorithmic strategies). All variants are parameterized across all patterns
except `Abort`, which is limited to algorithms with abort-path coverage.

| Algorithm\_Variant          | Topology                    | Basic | MultiSeq | MultiStream | DestroyRecreate | MixedEager | MultiGraph | InPlace | Abort |
|-----------------------------|-----------------------------|:-----:|:--------:|:-----------:|:---------------:|:----------:|:----------:|:-------:|:-----:|
| AllGather\_ctran             | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllGather\_ctdirect          | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllGather\_ctring            | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllGather\_ctrd              | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllGather\_ctbrucks          | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllGatherP\_ctdirect         | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllGatherP\_ctpipeline       | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllReduce\_ctran             | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |   x   |
| AllReduce\_ctdirect          | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |   x   |
| AllReduce\_ctring            | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |   x   |
| AllToAll\_ctran              | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllToAll\_alltoallv           | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllToAll\_alltoallv\_dynamic  | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| AllToAll\_alltoall\_dedup     | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| Broadcast\_ctran             | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| Broadcast\_ctdirect          | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| Broadcast\_ctbtree           | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| ReduceScatter\_ctran         | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| ReduceScatter\_ctdirect      | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| ReduceScatter\_ctring        | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| RMA\_put\_wait               | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |       |
| SendRecv\_ctp2p              | 1x8, nolocal, vnode         |   x   |    x     |      x      |        x        |     x      |     x      |    x    |   x   |

Each (algo, pattern) pair generates two test targets (one per topology):
- `<algo>_<pattern>_1x8_init_none`
- `<algo>_<pattern>_1x8_nolocal_init_none`

Within each target, all sub-variants are parameterized as GTest `Values`, so
a single target like `allgather_basic_1x8_init_none` runs the Basic pattern
across all 5 AllGather variants (ctran, ctdirect, ctring, ctrd, ctbrucks) x
2 message sizes x 1 replay multiplier.

#### Data Sizes

Each test is parameterized over two element counts (int32):
- **1024** (4 KB) — small message path, exercises control-plane-heavy code
- **8192** (32 KB) — larger message path, exercises pipelined data transfers

### CudaGraphTestBuilder

Fluent builder that handles the capture/replay/verify lifecycle:

1. **Capture** — wraps user lambda in `cudaStreamBeginCapture`/`EndCapture`
2. **Replay** — launches `cudaGraphExec` for `numReplays` iterations
3. **Reset** — calls user-provided reset function between replays
4. **Verify** — device-side comparison against eagerly-computed expected output
5. **Assertions** — validates graph topology (host nodes, kernel nodes)
6. **Cleanup** — destroys graphs, verifies no GPE resource leaks

### Verification Strategy

Tests compare graph replay output against eager execution output using a
GPU-side comparison kernel (`DeviceVerify.cu`).

**Why device-side verification matters for CUDA graph tests:**

Host-side verification (cudaMemcpy + CPU compare) requires a
`cudaDeviceSynchronize` or `cudaStreamSynchronize` before reading results. That
synchronization acts as a full barrier that masks synchronization bugs in the
algorithm itself — if the collective's internal signaling is broken but the data
happens to land before the host sync, the test passes despite the bug.

Device-side verification eliminates this blind spot. The comparison kernel is
enqueued on the same stream immediately after the graph replay, so it reads
buffers at exactly the point the algorithm considers them ready. If the
algorithm's synchronization logic has gaps — e.g., a missing stream-ordered
dependency, a kernel flag that isn't signaled, or a GPE callback that completes
out of order — the comparison kernel will observe partially-written or stale
data and report mismatches. No host sync covers up the race.

This is especially important for multi-replay tests: the comparison runs after
every replay iteration without any device sync between iterations (only the
first iteration syncs for graph assertions). A synchronization bug that only
manifests on replay N>1 — when the previous iteration's stale data is still
resident — will be caught.

```
Eager run (warmup) → expected buffers
    ↓
Graph capture → replay N times (no device sync between iterations)
    ↓ (each replay, fully async)
    Reset recvbuf → Replay graph → Device-compare actual vs expected
    ↓
Final sync → assert mismatch count == 0
```

## Adding a New Collective Test

1. Create `CtranCudaGraph<Algo>Test.cc` in `tests/cudagraph/`
2. Define an `AlgoDescriptor` with `makeBuffers`, `capture`, `isSupported`
3. Use the `DEFINE_CUDAGRAPH_PARAM_TEST` macro
4. Add the source file to `CUDAGRAPH_TEST_FILES` in `tests/cudagraph/BUCK`

Example:

```cpp
#include "comms/ctran/tests/cudagraph/CtranCudaGraphParamTest.h"

static AlgoDescriptor myAlgo() {
    return {
        .name = "MyAlgo",
        .isSupported = [](CtranComm*, size_t, int) { return true; },
        .makeBuffers = [](size_t count, int rank, int nRanks) {
            // allocate and initialize buffers
        },
        .capture = [](AlgoDescriptor::Buffers* bufs, size_t count,
                       ctran::testing::CaptureContext& ctx) {
            // call collective API
        },
    };
}

DEFINE_CUDAGRAPH_PARAM_TEST(MyAlgoTest, myAlgo());
```

## Adding a New Graph Pattern

1. Add enum value to `GraphPattern` in `CtranCudaGraphParamTest.h`
2. Add case to `patternToString()` and `baseReplays()`
3. Implement `run<Pattern>Pattern()` function
4. Add case to `runPattern()` switch
5. Add pattern flag to `CUDAGRAPH_PATTERN_FLAGS` in BUCK

## Running Tests

```bash
# Build the test builder library
buck2 build @fbcode//mode/opt -c hpc_comms.use_ncclx=stable \
    //comms/ctran/tests/cudagraph:cudagraph_test_builder

# Run all cudagraph tests
buck2 test @fbcode//mode/opt -c hpc_comms.use_ncclx=stable \
    //comms/ctran/tests/cudagraph/...

# Run a specific test target
# Format: //comms/ctran/tests/cudagraph:<algo>_<pattern>_<topology>_init_none
buck2 test @fbcode//mode/opt -c hpc_comms.use_ncclx=stable \
    //comms/ctran/tests/cudagraph:allgather_basic_1x8_init_none
```
