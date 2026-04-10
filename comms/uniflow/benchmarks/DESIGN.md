# Uniflow Transport Benchmark Suite — Design Document

## 1. Motivation

Uniflow is a host-based point-to-point data transfer library for LLM workloads,
targeting open-source release. Transport-level benchmarks are needed to:

- **Validate performance** of NVLink, RDMA, and TCP transport backends
- **Track regressions** via CI and MAST-based periodic runs
- **Enable users** to evaluate uniflow on their hardware with minimal setup
- **Guide optimization** with detailed latency/bandwidth/scaling metrics

No transport benchmarks exist today — only executor microbenchmarks.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Launch Layer                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  torchrun    │  │  Shell script     │  │  MAST (TorchX spec)   │ │
│  │  (OSS)       │  │  (direct launch)  │  │  (internal clusters)  │ │
│  └──────┬───────┘  └────────┬─────────┘  └───────────┬───────────┘ │
│         │                   │                        │             │
│         └───────────────────┼────────────────────────┘             │
│                             │                                      │
│              Sets env vars: MASTER_ADDR, MASTER_PORT,              │
│              RANK, WORLD_SIZE, LOCAL_RANK                          │
└─────────────────────────────┼──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     uniflow_bench (C++ binary)                      │
│                                                                     │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ CLI Parser │─>│ Bootstrap  │─>│ Benchmark  │─>│   Reporter   │  │
│  │            │  │ +Rendezvous│  │ Runner     │  │ (table/CSV)  │  │
│  └───────────┘  └─────┬──────┘  └─────┬──────┘  └──────────────┘  │
│                       │               │                            │
│                       ▼               ▼                            │
│                ┌──────────────┐ ┌──────────────┐                   │
│                │ TcpController │ │  Transport   │                   │
│                │ (control)     │ │  (data)      │                   │
│                └──────────────┘ └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key property**: The benchmark binary is launcher-agnostic. It reads standard
env vars that any launcher sets. No torch or MPI dependency in the binary itself.

## 3. End-to-End Flow

### 3.1 Launch

The user picks any launcher — all three produce the same result:

- **torchrun** (OSS): `torchrun --nproc-per-node=8 --no-python ./uniflow_bench ...`
- **Shell script**: `bash run_benchmark.sh --nproc 8 --benchmark bandwidth`
- **MAST**: `torchx run -- mast.py --num_nodes 2 --nproc_per_node 8`

All launchers spawn N processes, each with environment variables set:
`MASTER_ADDR`, `MASTER_PORT`, `RANK` (0..N-1), `WORLD_SIZE` (N), `LOCAL_RANK` (0..ppn-1).

### 3.2 Process Startup (every rank)

```
main()
  │
  ├── 1. Parse CLI args (benchmark type, transport, message sizes, etc.)
  │
  ├── 2. Read env vars into BootstrapConfig
  │       (masterAddr, masterPort, rank, worldSize, localRank)
  │
  ├── 3. Create EventBase thread (needed by transports)
  │
  ├── 4. Create TransportFactory for the LOCAL_RANK device
  │
  ├── 5. Run Rendezvous protocol (see section 4)
  │       -> establishes transport connections to peer ranks
  │
  ├── 6. Run selected benchmarks (see section 5)
  │       -> collects BenchmarkResults
  │
  ├── 7. Report results (rank 0 only prints table/CSV)
  │
  └── 8. Cleanup: shutdown transports, close connections
```

## 4. Rendezvous Protocol

The rendezvous uses uniflow's own TcpController for peer discovery.
No external dependencies (no folly, no c10d, no MPI). Fully OSS-portable.

### 4.1 Star Topology

Rank 0 acts as the rendezvous server. All other ranks connect to it.
This gives us rank 0 <-> rank i connections for benchmarking.

```
                     Rank 0 (server)
                    ┌───────────┐
                    │ TcpServer │
                    │ masterAddr│
                    │ :masterPort│
                    └─────┬─────┘
                          │ accept x (N-1)
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │ Rank 1 │  │ Rank 2 │  │ Rank 3 │
         │  Client │  │  Client │  │  Client │
         └────────┘  └────────┘  └────────┘
```

### 4.2 Connection Establishment Sequence

For each rank 0 <-> rank i pair:

```
    Rank 0                                    Rank i
    ------                                    ------

    1. TCP CONTROL CHANNEL
    Server listens on masterAddr:masterPort
    Accepts connection          <------------ Client connects

    2. TOPOLOGY EXCHANGE
    Receives peer topology      <------------ Sends {rank, localRank, topology}
    Sends own topology          ------------>  Receives rank 0 topology

    3. TRANSPORT CREATION
    Creates transport for                      Creates transport for
    peer's device                              rank 0's device

    4. TRANSPORT BIND + INFO EXCHANGE
    Binds transport (allocates                 Binds transport (allocates
    local resources)                           local resources)
    Sends TransportInfo         ------------>  Receives TransportInfo
    Receives TransportInfo      <------------  Sends TransportInfo

    5. TRANSPORT CONNECT
    Connects to peer using                     Connects to peer using
    received TransportInfo                     received TransportInfo
    State: Connected                           State: Connected
```

### 4.3 Memory Registration (post-rendezvous)

After transport connections are established, each rank allocates and registers
GPU memory for benchmarking:

```
    Rank 0                                    Rank i
    ------                                    ------

    1. ALLOCATE GPU MEMORY
    Allocate buffer on                         Allocate buffer on
    LOCAL_RANK device                          LOCAL_RANK device

    2. REGISTER WITH TRANSPORT
    Register segment with                      Register segment with
    TransportFactory                           TransportFactory

    3. EXPORT + EXCHANGE REGISTRATION
    Export registration ID      ------------>  Receive peer's export ID
    Receive peer's export ID    <------------  Export registration ID

    4. IMPORT REMOTE REGISTRATION
    Import peer's segment                      Import peer's segment
    (maps remote memory locally)               (maps remote memory locally)

    RESULT: Both ranks can now do put() and get() to each other's GPU memory
```

### 4.4 All-to-All Rendezvous (future iteration)

For multi-pair benchmarks (e.g., bisection bandwidth), extend to mesh:
- Rank 0 collects all endpoints, broadcasts the full list
- Each rank pair establishes a dedicated control connection for transport info exchange
- Deferred to iteration 4 (multi-node RDMA)

## 5. Benchmark Execution

### 5.1 Benchmark Runner Flow

```
For each selected benchmark:
  │
  For each message size (powers of 2, from min-size to max-size):
    │
    ├── Setup: create transfer request with spans of current size
    │
    ├── Barrier: all ranks synchronize via control channel
    │
    ├── Warmup: run N warmup iterations, discard timing
    │
    ├── Barrier: synchronize again
    │
    ├── Timed run: execute iterations, record per-iteration latency
    │     For each iteration:
    │       record start time
    │       issue transport operation (put/get)
    │       wait for completion (future.get)
    │       record end time
    │       store sample
    │
    ├── Compute statistics: min, max, avg, p50, p99
    │   Derive: bandwidth = size / avg_latency
    │           message_rate = 1 / avg_latency
    │
    └── Append result
```

### 5.2 Benchmark Types

#### Bandwidth Benchmark

Measures sustained throughput for large transfers.

```
Direction:      put or get (configurable)
Message sizes:  1KB -> 1GB (powers of 2)
Metric:         GB/s
Pattern:        Unidirectional: rank 0 -> rank 1

  Rank 0                          Rank 1
  ------                          ------
  for each iteration:
    put(local -> remote)  --->    (passive: memory written by rank 0)
    wait for completion
    record latency
```

#### Latency Benchmark

Measures round-trip latency for small messages.

```
Direction:      Ping-pong
Message sizes:  1B -> 4KB (powers of 2)
Metric:         Half round-trip time in microseconds

  Rank 0                          Rank 1
  ------                          ------
  for each iteration:
    put to rank 1, wait    --->
                           <---   put to rank 0, wait
    record full round-trip
    latency = round_trip / 2
```

#### Message Rate Benchmark

Measures small-message throughput (operations per second).

```
Direction:      put
Message size:   64 bytes (fixed)
Metric:         Million operations/sec

  Rank 0:
    start timer
    fire N put operations (pipelining — don't wait between issues)
    wait for all completions
    stop timer
    rate = N / elapsed
```

#### Multi-Stream Benchmark

Measures throughput scaling with concurrent CUDA streams.

```
Direction:      put
Message size:   64MB (fixed, large)
Stream counts:  1, 2, 4, 8 (configurable)
Metric:         Aggregate GB/s across all streams

  For each stream count N:
    create N CUDA streams
    for each iteration:
      issue N puts in parallel (one per stream)
      wait for all completions
      record total elapsed
    aggregate_bw = N * message_size / avg_elapsed
    scaling = aggregate_bw / single_stream_bw
```

#### Connection Setup Benchmark

Measures transport connection establishment overhead.

```
Metric:         Microseconds per connection setup

  For each iteration:
    create transport from factory
    start timer
    bind (allocate local resources)
    exchange bind info via control channel
    connect (establish data path)
    stop timer
    shutdown transport
    record elapsed
```

### 5.3 Synchronization

Ranks synchronize using a lightweight barrier over the control channel:
- Rank 0 collects a token from each peer, then broadcasts a token back
- This ensures all ranks enter timed sections simultaneously
- No external barrier library needed — uses uniflow's own TcpConn send/recv

## 6. Result Reporting

### 6.1 Table Output (default, human-readable)

```
======================================================================
                    Uniflow Transport Benchmark
  Transport: NVLink    Ranks: 2    GPUs: cuda:0 <-> cuda:1
======================================================================

-- NVLink Bandwidth (put) -----------------------------------------------
   Size (B)    Iters    BW (GB/s)   Lat avg(us)   Lat p50(us)   Lat p99(us)
         64      100          0.8          0.08          0.07          0.15
       1024      100         12.3          0.08          0.07          0.15
      65536      100        145.6          0.43          0.42          0.52
    1048576      100        280.1          3.57          3.55          4.01
   67108864      100        580.2        110.34        109.89        115.21
 1073741824       20        612.5       1670.12       1668.45       1695.32

-- NVLink Bandwidth (get) -----------------------------------------------
   Size (B)    Iters    BW (GB/s)   Lat avg(us)   Lat p50(us)   Lat p99(us)
   ...

-- NVLink Latency -------------------------------------------------------
   Size (B)    Iters    Lat avg(us)   Lat p50(us)   Lat p99(us)   Lat min(us)
          1      1000          0.52          0.51          0.68          0.48
          8      1000          0.53          0.52          0.69          0.49
       4096      1000          0.71          0.70          0.85          0.65

-- NVLink Message Rate --------------------------------------------------
   Size (B)    Iters     Mops/s
         64    10000       1.92

-- NVLink Multi-Stream Scaling ------------------------------------------
   Streams    Size (B)    BW (GB/s)    Scaling
         1    67108864        580.2       1.00x
         2    67108864       1045.4       1.80x
         4    67108864       1820.7       3.14x
         8    67108864       2105.3       3.63x

-- Connection Setup -----------------------------------------------------
   Transport    Iters    Avg (us)    p50 (us)    p99 (us)
      NVLink      100       125.3       122.1       189.5
```

### 6.2 CSV Output (for analysis/plotting)

Columns: benchmark, transport, direction, size_bytes, iterations, bw_gbps,
lat_avg_us, lat_p50_us, lat_p99_us, lat_min_us, lat_max_us, msg_rate_mops

Written to file specified by `--output` flag. Enables easy import into
spreadsheets, Jupyter notebooks, or plotting tools.

## 7. File Structure

```
comms/uniflow/benchmarks/
├── BUCK                                 # Buck build targets
├── CMakeLists.txt                       # CMake build (OSS)
├── DESIGN.md                            # This document
├── main.cpp                             # CLI entry point
├── Bootstrap.h / .cpp                   # Env var parsing
├── Rendezvous.h / .cpp                  # TcpController-based peer discovery
├── Stats.h / .cpp                       # Statistical computations
├── Reporter.h / .cpp                    # Table + CSV formatters
├── BenchmarkRunner.h / .cpp             # Registry, size sweep, warmup, barrier
├── bench/
│   ├── NVLinkBandwidthBench.h / .cpp
│   ├── NVLinkLatencyBench.h / .cpp
│   ├── NVLinkMsgRateBench.h / .cpp
│   ├── NVLinkMultiStreamBench.h / .cpp
│   └── ConnectionSetupBench.h / .cpp
└── scripts/
    ├── run_benchmark.sh                 # torchrun-based local launcher
    ├── run_benchmark_direct.sh          # Direct process spawning (no torchrun)
    └── mast.py                          # TorchX MAST job specification
```

## 8. Components

### 8.1 Bootstrap

Reads standard distributed environment variables (MASTER_ADDR, MASTER_PORT,
RANK, WORLD_SIZE, LOCAL_RANK). Validates all required vars are present.
Throws a clear error message if any are missing, with guidance on how to
launch the binary correctly.

### 8.2 Rendezvous

Uses uniflow's TcpServer (on rank 0) and TcpClient (on other ranks) to
establish control connections. Orchestrates the topology exchange, transport
creation, bind/connect, and memory registration sequence described in section 4.

Returns a set of PeerConnections, each containing: the peer rank, a control
channel (TcpConn), and a connected Transport ready for data operations.

### 8.3 Stats

Computes min, max, average, p50, and p99 from a vector of latency samples.
Sorts in-place for percentile computation. Derives bandwidth (GB/s) and
message rate (Mops/s) from latency and message size.

### 8.4 Benchmark Runner

Maintains a registry of benchmark implementations. Iterates over message sizes
(powers of 2), runs warmup + timed iterations, collects results. Handles
barrier synchronization between ranks at each size point.

### 8.5 Reporter

Formats results as either a human-readable table (for terminal output) or
CSV (for file output / analysis). Only rank 0 produces output to avoid
interleaved console noise from multiple processes.

### 8.6 Individual Benchmarks

Each benchmark implements a common interface: given a configuration, peer
connections, and a transport factory, it runs the benchmark and returns
a list of results (one per message size / configuration point).

## 9. CLI

```
uniflow_bench -- Uniflow Transport Performance Benchmarks

OPTIONS:
  --benchmark <name>     bandwidth | latency | msgrate | multistream |
                         connsetup | all                    [default: all]
  --transport <type>     nvlink | rdma                      [default: nvlink]
  --min-size <bytes>     Minimum message size                [default: 1]
  --max-size <bytes>     Maximum message size                [default: 1073741824]
  --iterations <n>       Timed iterations per size point     [default: 100]
  --warmup <n>           Warmup iterations (discarded)       [default: 10]
  --direction <dir>      put | get | both                    [default: both]
  --num-streams <list>   Stream counts for multistream       [default: 1,2,4,8]
  --output <path>        CSV output file path                [default: none]
  --format <fmt>         table | csv | both                  [default: table]

ENVIRONMENT VARIABLES (set by torchrun/MAST/launcher):
  MASTER_ADDR            Rank 0 hostname/IP
  MASTER_PORT            Rank 0 port for rendezvous
  RANK                   Global rank of this process
  WORLD_SIZE             Total number of processes
  LOCAL_RANK             Local rank (maps to GPU device)
```

## 10. Launch Scripts

### 10.1 run_benchmark.sh (torchrun wrapper)

Wraps torchrun for convenient local runs. Builds the benchmark binary via
Buck, then invokes torchrun in standalone mode with `--no-python` to launch
the C++ binary directly. Accepts `--nproc` to set GPU count and passes all
remaining arguments to the benchmark binary.

### 10.2 run_benchmark_direct.sh (no torchrun dependency)

Spawns processes directly by setting env vars and backgrounding each process.
No torchrun or PyTorch dependency — useful for OSS environments without torch
installed, or for quick local testing. Waits for all processes and propagates
the first non-zero exit code.

### 10.3 mast.py (TorchX MAST job spec)

TorchX job specification following the established pattern (similar to MCCL
E2E tests). Configurable: num_nodes, nproc_per_node, hw_type, benchmark
type, transport type. Outputs results CSV to MAST's wsfuse output directory.

## 11. Build System

### Buck (internal)

Single binary target using `oss_cpp_binary` macro with dependencies on
uniflow controller, executor, transport, and segment libraries.
CI labels configured to exclude from default dev builds (GPU-dependent).

### CMake (OSS)

Guarded by `BUILD_BENCHMARKS` option. Produces `uniflow_bench` executable
and installs it along with launch scripts. Links against the unified
`uniflow` library target.

## 12. Iteration Plan

### Iteration 1: Foundation

**Scope**: Bootstrap + bandwidth benchmark + basic scripts

- CLI entry point with argument parsing
- Bootstrap (env var reading) and Rendezvous (TcpController-based)
- Stats computation and table Reporter
- BenchmarkRunner with size sweep and warmup
- NVLink bandwidth benchmark (put/get sweep 1KB-1GB)
- Buck and CMake build targets
- torchrun launch script

**Verification**: 2-GPU local run produces bandwidth table output.

### Iteration 2: Full NVLink Suite

**Scope**: Remaining 4 benchmark types

- NVLink latency benchmark (ping-pong, 1B-4KB)
- NVLink message rate benchmark (pipelined small puts)
- NVLink multi-stream scaling benchmark (1/2/4/8 streams)
- Connection setup benchmark (bind/connect cycle timing)

**Verification**: `--benchmark all` runs all 5 benchmarks.

### Iteration 3: Scripts and MAST

**Scope**: Production scripts, CSV output, MAST integration

- Direct launch script (no torchrun dependency)
- MAST TorchX job specification
- CSV output support in Reporter
- Enhanced torchrun script with more flags

**Verification**: MAST job produces results CSV.

### Iteration 4: RDMA and Multi-Node

**Scope**: RDMA benchmarks, mesh rendezvous (depends on RDMA data ops landing)

- RDMA bandwidth and latency benchmarks
- Extended Rendezvous for multi-node mesh topology
- Multi-node launch script support

**Verification**: 2-node MAST run with RDMA transport.

## 13. Design Decisions

**Single binary with CLI flags** — Follows the nccl-tests pattern, which is
the industry standard for transport benchmarking. A single binary simplifies
launcher integration and reduces build complexity.

**Env-var-based bootstrap** — Makes the binary launcher-agnostic. Works with
torchrun, MAST, MPI, or a simple shell script. The binary doesn't care who
set the environment variables.

**Uniflow's own TcpController for rendezvous** — No dependency on c10d
TCPStore, folly, or MPI for rendezvous. Keeps the benchmark fully OSS-portable,
consistent with uniflow's no-folly design principle.

**Custom benchmark framework (not Google Benchmark)** — Google Benchmark is
single-process. Transport benchmarks are inherently distributed (multi-rank,
multi-GPU) and need coordinated timing, barriers, and multi-process result
aggregation. A lightweight custom runner handles this without framework overhead.

**Rank 0 as sole reporter** — Avoids interleaved output from multiple processes.
Other ranks participate in benchmarks but stay silent on output.

**Powers-of-2 size sweep** — Standard practice in networking benchmarks.
Reveals performance characteristics at different transfer granularities:
small messages test latency overhead, large messages test bandwidth saturation.
