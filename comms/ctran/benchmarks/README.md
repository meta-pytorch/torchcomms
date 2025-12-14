# AllgatherP Performance Benchmark

This benchmark measures the performance of the AllgatherP algorithm
implementation in terms of bus bandwidth and algorithm bandwidth, allowing you
to compare it with standard NCCL Allgather baseline.

## Overview

The benchmark supports:

- **Three algorithms**: `AllGatherP_Direct`, `AllGatherP_Pipeline`, and
  `NCCL baseline AllGather`
- **Flexible message sizes**: Configurable min/max range
- **Multiple configurations**: In-place/out-of-place, different memory types
- **Comprehensive metrics**: Min/Max/Avg latency, Algo BW, Bus BW
- **Multi-host support**: Requires InfiniBand backend for AllgatherP algorithms

## Bandwidth Calculations

### Algorithm Bandwidth (AlgoBW)

The algorithm bandwidth represents the total amount of data moved across all
ranks:

```
AlgoBW = (sizeBytes * nRanks) / time / 1e9 GB/s
```

For AllGather, each rank contributes `sizeBytes`, resulting in total data of
`sizeBytes * nRanks`.

### Bus Bandwidth (BusBW)

The bus bandwidth accounts for the actual data movement on the network:

```
BusBW = AlgoBW * (nRanks - 1) / nRanks
```

This factor adjusts for the fact that each rank already has its own data and
only needs to receive data from `(nRanks - 1)` other ranks.

## Building

```bash
# Build the benchmark
buck2 build //comms/ctran/benchmarks:allgatherp_bench

# Or use build mode
cd fbsource/fbcode
buck2 build @mode/opt //comms/ctran/benchmarks:allgatherp_bench
```

## Running the Benchmark

### Basic Usage

Run with default settings (both algorithms, 32KB to 256MB):

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench
```

### Configuration Options

| Flag             | Default          | Description                                                                 |
| ---------------- | ---------------- | --------------------------------------------------------------------------- |
| `--min_bytes`    | 16384 (16KB)     | Minimum message size in bytes                                               |
| `--max_bytes`    | 1073741824 (1GB) | Maximum message size in bytes                                               |
| `--warmup_iters` | 5                | Number of warmup iterations                                                 |
| `--bench_iters`  | 20               | Number of benchmark iterations                                              |
| `--algo`         | "all"            | Algorithm to benchmark: "ctdirect", "ctpipeline", "nccl", or "all"          |
| `--in_place`     | false            | Run in-place allgather                                                      |
| `--mem_type`     | "cudaMalloc"     | Memory type: "cumem" or "cudaMalloc" (NCCL baseline always uses cudaMalloc) |

### Example Commands

**Benchmark all three algorithms (default):**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --algo=all
```

**Benchmark only NCCL baseline:**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --algo=nccl
```

**Benchmark only ctdirect algorithm:**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --algo=ctdirect
```

**Custom size range with more iterations:**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --min_bytes=65536 \
  --max_bytes=134217728 \
  --bench_iters=50
```

**In-place benchmark with cudaMalloc memory:**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --in_place=true \
  --mem_type=cudamalloc
```

**Full benchmark suite with large message sizes:**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --algo=all \
  --min_bytes=32768 \
  --max_bytes=536870912 \
  --bench_iters=30
```

**NCCL baseline only with custom parameters:**

```bash
buck2 run //comms/ctran/benchmarks:allgatherp_bench_1x8 -- \
  --algo=nccl \
  --min_bytes=16384 \
  --max_bytes=1073741824 \
  --bench_iters=30
```

## Multi-GPU and Multi-Node Execution

The benchmark uses the distributed test framework and can run on multiple GPUs
and nodes.

### Single Node, Multiple GPUs (1x8 configuration)

```bash
buck2 test //comms/ctran/benchmarks:allgatherp_bench-1x8 -- \
  --test-env=min_bytes=65536 \
  --test-env=max_bytes=134217728 \
  --test-env=algo=all
```

### Multi-Node (2x8 configuration)

```bash
buck2 test //comms/ctran/benchmarks:allgatherp_bench-2x8 -- \
  --test-env=min_bytes=65536 \
  --test-env=max_bytes=134217728 \
  --test-env=algo=all
```

## Output Format

The benchmark outputs a formatted table with the following columns:

```
==================================================================================
AllgatherP Performance Benchmark
==================================================================================
Configuration:
  Ranks: 8
  Min Size: 32768 bytes
  Max Size: 268435456 bytes
  Warmup Iterations: 5
  Benchmark Iterations: 20
  Memory Type: ncclmem
  In-Place: No
==================================================================================

Algorithm                    Size(B)       Count     Avg(ms)     Min(ms)     Max(ms) AlgoBW(GB/s)  BusBW(GB/s)
-------------------------------------------------------------------------------------------------------------------------------------
AllGatherP_Direct              32768       16384       0.125       0.118       0.132       2.104       1.841
AllGatherP_Pipeline            32768       16384       0.112       0.108       0.119       2.346       2.053
NCCL_AllGather                 32768       16384       0.118       0.115       0.125       2.221       1.944
...
```

**Note**: Min/Max show the fastest and slowest ranks, while Avg is the mean across all ranks.

### Column Descriptions

- **Algorithm**: Name of the algorithm being benchmarked
- **Size(B)**: Size of send buffer in bytes (per rank)
- **Count**: Number of elements (Size / sizeof(datatype))
- **Avg(ms)**: Average latency per iteration across all ranks (MPI_Allreduce with MPI_SUM / numRanks)
- **Min(ms)**: Minimum latency observed across all ranks (MPI_Allreduce with MPI_MIN)
- **Max(ms)**: Maximum latency observed across all ranks (MPI_Allreduce with MPI_MAX)
- **AlgoBW(GB/s)**: Algorithm bandwidth in GB/s (calculated from Avg latency)
- **BusBW(GB/s)**: Bus bandwidth in GB/s (calculated from Avg latency)

**Note**: The benchmark measures sustained throughput by launching all iterations back-to-back and then syncing once. Each rank's timing is aggregated using MPI_Allreduce to provide min/max/avg statistics across all ranks. This gives a comprehensive view of performance variation across the cluster.
per-iteration variance.

## Understanding the Results

### Performance Comparison

When comparing AllgatherP with NCCL:

- **Higher BusBW is better**: Indicates more efficient use of network bandwidth
- **Lower latency is better**: Especially important for small messages
- **Pipeline vs Direct**: Pipeline may show better performance for larger
  messages due to overlapping communication

### Expected Patterns

1. **Small Messages**: Direct algorithm may have lower overhead
2. **Large Messages**: Pipeline algorithm may achieve higher bandwidth through
   overlap
3. **In-Place vs Out-of-Place**: In-place may have slightly lower memory
   overhead
4. **Memory Types**: ncclmem (CuMem) may show better performance with IB backend

## Algorithm Details

### AllGatherP Direct (ctdirect)

- Each rank sends its data to all other ranks
- Uses IB backend for inter-node communication
- Simpler implementation with less overhead for small messages

### AllGatherP Pipeline (ctpipeline)

- Ring-based inter-node communication
- NVL-based intra-node broadcast
- Overlaps inter-node put with intra-node broadcast
- Better scalability for large messages

## Requirements

- **CTRAN enabled**: Set `NCCL_CTRAN_ENABLE=1` (handled automatically by
  benchmark)
- **AllGatherP Support**: AllgatherP algorithms must be supported (benchmark
  will fail if not)
- **CuMem Support**: Required when using `--mem_type=cuMem` (benchmark will fail
  if not supported)
- **GPU Memory**: Sufficient for test buffers (2x max_bytes per rank)
- **Multi-GPU**: For realistic performance testing, use 8+ GPUs

## Troubleshooting

### "allGatherP algo is not supported!"

- Ensure CTRAN is properly initialized
- Check that all peers support CTRAN backend
- Verify IB backend is available
- The benchmark will fail if AllgatherP is not supported
- To run only NCCL baseline (which doesn't require AllgatherP support), use
  `--algo=nccl`

### "CuMem is not supported!"

- CuMem (ncclmem) is not supported on this system
- Use `--mem_type=cudaMalloc` instead
- NCCL baseline always uses cudaMalloc regardless of this flag

## See Also

- AllgatherP implementation: `comms/ctran/algos/AllGatherP/`
- Unit tests: `comms/ctran/tests/CtranDistAllgatherPTests.cc`
- CTRAN documentation: `comms/ctran/README.md`
