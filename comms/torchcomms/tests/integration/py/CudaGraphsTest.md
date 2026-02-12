# CUDA Graph Tests

## Overview

Integration tests for CUDA graph capture and replay of torchcomms collectives. Each test captures one or more collective operations into CUDA graphs, replays them multiple times, and verifies both correctness (output tensors match expected values) and graph structure (kernel types, counts, dependency ordering, absence of unexpected MEMCPYs).

## Testing Framework

### Definitions

- **`_Substep`** — A single unit of work in a pipeline step: a bare `CUDAGraph`, a `(CUDAGraph, Stream)` tuple, or a `(Callable, Stream)` tuple.
- **`PipelineStep`** — One step in a replay pipeline: a `_Substep`, a bare `Callable` (triggers full device sync), or a `list[_Substep]` (fork-join concurrency).
- **`CudaGraphNode`** — Parsed node from a CUDA graph's DOT representation (id, type, label, kernel_name).
- **`CudaGraphInfo`** — Structured representation of a captured CUDA graph's DAG. Provides methods for querying nodes by type/name, checking path existence (`has_path`), and determining whether two nodes are sequential or parallel.

### `GraphTestBuilder`

Fluent builder for CUDA graph capture-replay tests. Handles the common flow:

1. Create comms and streams
2. Capture operations into graphs (via `add_capture()`)
3. Analyze captured graphs (DOT dump → `CudaGraphInfo`)
4. For each replay: reset inputs to originals, run the pipeline, assert outputs match expected

Three replay modes:
- **`run_serial()`** — Replays all graphs sequentially on the default stream.
- **`run_concurrent()`** — Replays all graphs concurrently on their respective capture streams (fork-join).
- **`run_custom_schedule(pipeline_fn)`** — User-defined pipeline with arbitrary step ordering, event-based synchronization, and mixed graph/callable steps.

### `CudaGraphTestBase`

Base `unittest.TestCase` subclass providing:
- **Constants**: `NUM_REPLAYS=3`, `NUM_OPS=5`, `NUM_GRAPHS=3`
- **`create_comms(n)`** — Context manager creating `n` `TorchComm` objects, finalizes on exit.
- **`create_graphs(n)`** — Context manager creating `n` `CUDAGraph` objects with `keep_graph=True`, resets on exit.
- **`run_graph_pipeline(steps)`** — Executes a list of `PipelineStep`s with event-based synchronization between steps.

### `create_capture()`

Reusable complex capture pattern used across multiple test files:
- **Pattern**: `allreduce(sync, comm0)` → `sum` → `allgather(async, comm1)` with intra-graph stream dependencies (two local streams with explicit `wait_stream` ordering).
- **Parameterized** by tensor indices (`input_idx`, `intermediate_idx`, `output_idx`) and comm indices (`comm0_idx`, `comm1_idx`).
- Produces 3 tensors per capture (`TENSORS_PER_CAPTURE`): input (10×10), intermediate scalar (1,), output vector (world_size,).

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_GRAPH_SVG_DIR` | If set, captured graphs are rendered as SVG files in this directory for visual debugging |
| `TORCH_PROFILE_DIR` | If set, the replay phase is traced with `torch.profiler` and Chrome trace JSON files are saved to this directory |

## Test Classes

### `TestSingleGraph`

Tests capturing collectives into a single CUDA graph with varying async patterns, stream configurations, and communicator layouts. All tests assert graph structure via DOT analysis.

| Test | Edge case |
|------|-----------|
| `test_single_allreduce` | Baseline: single sync and async all_reduce; asserts 1 AllReduce kernel, EVENT_WAIT/RECORD nodes, no MEMCPYs |
| `test_multiple_allreduce` | N sequential all_reduce ops on separate tensors; asserts N AllReduce kernels are sequentially ordered in the graph DAG |
| `test_multiple_allreduce_async_wait_at_end` | N async all_reduce ops issued back-to-back with all waits deferred to the end; verifies kernels remain sequential despite deferred synchronization |
| `test_multiple_allreduce_mixed` | Alternating sync/async all_reduce ops within a single capture |
| `test_multiple_streams_single_comm` | All_reduce ops across N different streams sharing one comm; asserts single-comm serialization forces sequential kernel ordering despite multi-stream capture |
| `test_multiple_streams_multiple_comms` | Odd/even all_reduce pattern across two comms on separate streams; verifies no MEMCPYs with multi-comm multi-stream capture |
| `test_two_streams_two_comms_with_dependency` | Cross-stream dependency chain: allreduce(stream0, comm0) → sum(stream0) → allgather(stream1, comm1) with rank-dependent inputs; asserts AllReduce → reduce → AllGather DAG ordering |

### `TestMultipleGraphs`

Multiple separately-captured complex CUDA graphs (each using `create_capture`: allreduce → sum → allgather with intra-graph stream deps), replayed serially, concurrently, or with inter-graph dependencies.

| Test | Edge case |
|------|-----------|
| `test_multiple_graphs_serial` | N complex graphs sharing 2 comms, replayed serially; asserts AllReduce → reduce → AllGather ordering per graph, no MEMCPYs |
| `test_multiple_graphs_serial_different_comms` | N complex graphs each with dedicated comm pairs (2N comms total), replayed serially; tests comm isolation across graphs |
| `test_multiple_graphs_concurrent` | N complex graphs sharing 2 comms, replayed concurrently on separate streams; tests concurrent replay with shared comm resources |
| `test_multiple_graphs_concurrent_different_comms` | N complex graphs with dedicated comm pairs, replayed concurrently; tests fully isolated concurrent replay |
| `test_multiple_graphs_with_dependency` | Two complex graphs with inter-graph data dependency: graph0's allreduce output is copied into graph1's input between replays (host-side copy with full sync); tests sequential pipeline with data flow between graphs |
| `test_multiple_graphs_event_sync` | Three complex graphs: graphs 0 and 1 fork concurrently, then an event-synced copy transfers graph0's output into graph2's input, then graph2 runs — all chained via CUDA events (no full device sync); tests event-based inter-graph DAG scheduling |
| `test_multiple_graphs_external_event_sync` | Two complex graphs replayed concurrently with an external CUDA event captured into both graphs: graph0 records the event after its allreduce, graph1 waits on it before reading graph0's output; tests cross-graph on-device synchronization with no host-side sync |

### `TestGraphConcurrency`

Tests complex graph replay (using `create_capture`) running concurrently with non-graphable GPU work — compute kernels or collectives on separate comms — synchronized via CUDA events.

| Test | Edge case |
|------|-----------|
| `test_graph_parallel_with_nongraphable` | Complex graph replay runs concurrently with a non-graphable matmul on a separate stream; verifies both produce correct results without interference |
| `test_graph_parallel_with_nongraphable_collective` | Complex graph replay concurrently with a non-graphable all_reduce on a third comm (comms 0,1 for graph, comm 2 for non-graph); tests comm resource isolation under concurrency |
| `test_multiple_graphs_parallel_with_nongraphable` | Two complex graph replays plus a non-graphable matmul, all running concurrently on 3 streams; tests high stream concurrency with intra-graph streams nested inside |
| `test_graph_then_nongraphable_event_sync` | Complex graph replay followed by non-graphable work that reads the graph's allreduce output; synchronized via CUDA events (not full device sync), verifying event-based producer-consumer ordering between graph and non-graph work |
