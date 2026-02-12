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
