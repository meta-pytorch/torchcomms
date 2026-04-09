# Gloo vs UCC vs MPI: CPU Collective Performance Comparison

> **Disclaimer**: This report was written by Claude (Anthropic) with human review. While the benchmark numbers are real measurements, the analysis and platform support sections are based on Claude's training data and web research. Verify claims against upstream documentation before making architectural decisions.

## Background

### Gloo

Gloo is a collective communications library developed for multi-machine training, bundled with PyTorch [[1]]. It provides CPU-based implementations of common collectives (allreduce, allgather, broadcast, etc.) using TCP sockets. Gloo is the default CPU backend for `torch.distributed` and prioritizes portability and simplicity.

### UCC and UCP

**Unified Collective Communication (UCC)** is a collective communication library from the OpenUCX project [[2]]. UCC is built on top of **Unified Communication X (UCX)** [[3]], a production-grade communication framework originally developed for HPC workloads.

UCC uses a layered plugin architecture [[2]]:

- **MC (Memory Component)** — memory registration and copy operations
- **TL (Transport Layer)** — transport-specific collective implementations
- **CL (Collective Layer)** — algorithm selection and orchestration across transports
- **EC (Execution Component)** — offloaded execution engines

The transport layer most relevant to CPU benchmarks is **TL/UCP**, which uses UCX's **UCP (Unified Communication Protocol)** layer. UCP provides [[3]]:

- **Tag matching** — message matching by tag, used for point-to-point and collective rendezvous
- **Active messages** — lightweight RPC-style communication for small payloads
- **Multi-transport negotiation** — UCP automatically selects the best available transport at runtime (shared memory, TCP, RDMA, etc.)
- **Memory registration caching** — avoids redundant registration for frequently used buffers
- **Protocol selection** — eager (inline small messages) vs rendezvous (zero-copy large messages) based on message size thresholds

For intra-node CPU communication, UCP will typically negotiate **shared memory (sm)** transport via UCX's `posix` or `sysv` shared memory providers [[3]], bypassing the kernel's TCP/IP stack entirely. For inter-node or when shared memory is disabled (`UCX_TLS=tcp`), UCP falls back to its own TCP transport implementation.

### MPI (Open MPI)

**Message Passing Interface (MPI)** is the standard API for distributed-memory parallel computing [[4]], widely used in HPC. Open MPI is a widely-used open-source implementation [[5]].

Open MPI uses a modular architecture with pluggable transports [[5]]:

- **BTL (Byte Transfer Layer)** — point-to-point transports (TCP, shared memory, RDMA)
- **PML (Point-to-Point Messaging Layer)** — message matching and protocol selection
- **COLL (Collective)** — collective algorithm implementations with multiple modules (tuned, basic, han, etc.)

For intra-node communication, Open MPI can use its `sm` shared memory BTL [[6]], similar to UCX. The `coll/tuned` module provides algorithm selection based on message size and communicator size (linear, binomial tree, ring, etc.), comparable to UCC's TL/UCP layer.

## Platform & Transport Support

### Windows Support

| Capability | Gloo | UCC / UCX | Open MPI |
|------------|------|-----------|----------|
| Windows builds | Yes (bundled with PyTorch) | No (Linux-only, autotools build) | No (Linux/macOS/FreeBSD only) |
| TCP on Windows | Yes (`transport/tcp`) | N/A | N/A |
| Shared memory on Windows | No (POSIX shm APIs) | N/A | N/A |

Gloo is the **only** backend with Windows support. PyTorch ships Gloo on Windows using its TCP transport, but Gloo's shared memory transport relies on POSIX APIs (`shm_open`, `mmap`) that are unavailable on Windows, so all intra-node communication goes through the TCP/IP stack. Based on the benchmarks above, this means Windows CPU collectives would operate at the "Gloo TCP" performance tier — 2-265x slower than shared-memory backends on Linux.

UCX and UCC are **Linux-only** — both use autotools (`./autogen.sh`, `./configure`, `make`) with no MSVC or Windows CMake support [[2]] [[3]]. UCX's TCP transport (`uct/tcp`) uses POSIX socket APIs (`epoll`, `iovec`) and its shared memory transport uses `mmap`/`shmget`. While UCP's TCP transport is conceptually portable, there is no Windows build infrastructure or platform abstraction layer, so UCC+UCP over TCP is not available on Windows.

For Windows MPI workloads, **Microsoft MPI (MS-MPI)** is the standard alternative. MS-MPI is a Microsoft implementation of MPI designed for porting MPICH-based code [[7]]. It supports both TCP and shared memory on Windows, but it is not compatible with Open MPI and has no PyTorch `torch.distributed` integration.

### ibverbs (InfiniBand / RoCE) Support

All three backends support ibverbs for inter-node RDMA, but with very different depth:

| Capability | Gloo | UCC / UCX | Open MPI |
|------------|------|-----------|----------|
| ibverbs transport | Yes (`transport/ibverbs`) | Yes (native `rc`, `ud`, `dc`, `rc_mlx5`, etc.) | Yes (via UCX PML) [[6]] |
| QP types | RC only | RC, UD, DC (Dynamic Connected) | RC, UD, DC (via UCX) |
| RDMA write | Yes (`IBV_WR_RDMA_WRITE_WITH_IMM`) | Yes | Yes |
| RDMA read | Yes (`IBV_WR_RDMA_READ`) | Yes | Yes |
| Atomics | No | Yes (fetch-and-add, CAS) | Yes (via UCX) |
| Inline sends | No | Yes (configurable threshold) | Yes (via UCX) |
| Tag matching offload | No | Yes (ConnectX-5+ hardware TM) | Yes (via UCX) |
| Multi-rail | No | Yes (automatic lane selection) | Yes (via UCX) |
| On-demand paging (ODP) | No | Yes (implicit registration) | Yes (via UCX) |
| Memory registration caching | Manual (`ibv_reg_mr`) | Yes (`ucs_rcache`) | Yes (via UCX) |
| Adaptive routing | No | Yes (AR-aware) | Yes (via UCX) |
| GDR (GPU Direct RDMA) | No | Yes | Yes |

**Gloo** has a functional ibverbs transport using RC (Reliable Connection) queue pairs [[8]]. It supports send/recv, RDMA write with immediate data (`IBV_WR_RDMA_WRITE_WITH_IMM`), and RDMA read (`IBV_WR_RDMA_READ`), with manual memory registration via `ibv_reg_mr` [[8]]. It does not support atomics, dynamic connections, inline sends (`IBV_SEND_INLINE`), multi-rail, or hardware tag matching — verified by source code inspection [[8]].

**UCX** provides the deepest ibverbs integration. Its `rc_mlx5` transport drives Mellanox/NVIDIA ConnectX hardware directly via the `mlx5dv` devx interface (`#include <infiniband/mlx5dv.h>`) [[9]], bypassing the generic verbs layer for lower latency. The `dc_mlx5` (Dynamic Connected) transport uses DCI (Dynamic Connection Initiator) pools with configurable allocation policies [[10]]. UCX automatically selects between UD (for connection establishment and small messages), RC (for bulk transfers), and DC (for large-scale deployments). UCC inherits all of this through its TL/UCP transport layer.

**Open MPI 5.x** includes a UCX PML (Point-to-Point Messaging Layer) [[6]], giving it the same ibverbs capabilities as UCX. The legacy `openib` BTL has been **removed** in Open MPI 5.x (not just deprecated) [[6]] — ibverbs support now requires UCX or OFI.

### EFA (AWS Elastic Fabric Adapter) Support

EFA is AWS's custom network interface for HPC/ML workloads on EC2 [[11]]. Unlike InfiniBand, EFA exposes a **libfabric** provider (`efa`) [[12]] [[13]], using the **Scalable Reliable Datagram (SRD)** protocol [[14]]. EFA supports RDM (Reliable Datagram) endpoints, RDMA read (Nitro v4+), and RDMA write (Nitro v4+, varies by instance type) [[15]]. Atomics are not supported.

| Capability | Gloo | UCC / UCX | Open MPI |
|------------|------|-----------|----------|
| EFA support | No | Not natively; requires libfabric | Yes (via libfabric `efa` provider) |
| Transport path | N/A | N/A (UCX has no `efa` transport) | `mtl/ofi` with libfabric `efa` [[6]] |
| SRD protocol | N/A | N/A | Yes (via libfabric) [[14]] |
| Multi-rail EFA | N/A | N/A | Yes (via libfabric) |
| RDMA read | N/A | N/A | Yes (Nitro v4+) [[15]] |
| RDMA write | N/A | N/A | Yes (Nitro v4+, varies by instance) [[15]] |
| Tag matching | N/A | N/A | Yes (software) |

**Gloo** has **no EFA support**. It only implements TCP and ibverbs transports [[1]], with no libfabric integration. On AWS, Gloo falls back to TCP, forgoing EFA's SRD advantages entirely.

**UCX** does **not** have a native EFA transport. Inspection of UCX 1.20.0's transport list (`ucx_info -d`) shows only: `self`, `tcp`, `cma`, `posix`, `sysv`, `cuda_copy`, `cuda_ipc` — no `efa` transport [[16]]. ~~A previous version of this report incorrectly claimed UCX ≥1.12 included a native EFA transport.~~ UCX's ibverbs transports (`rc_mlx5`, `dc_mlx5`) do not work with EFA, which uses libfabric rather than ibverbs. To use EFA with UCC, an application would need to go through a libfabric-based path, which UCC does not currently support.

**Open MPI** supports EFA through its `mtl/ofi` framework with libfabric's `efa` provider [[6]]. AWS publishes optimized Open MPI builds for EFA through their `aws-efa-installer` package [[13]], which includes a pre-configured libfabric with the `efa` provider. Configuration: `--mca pml cm --mca mtl ofi`.

For **NCCL** (GPU collectives on AWS), the separate **aws-ofi-nccl** plugin [[17]] bridges NCCL to libfabric's EFA provider, but this is independent of the CPU backends discussed here.

### Fault Tolerance

A critical operational question: what happens when one worker crashes or hangs during a collective?

| Capability | Gloo | UCC / UCX | Open MPI |
|------------|------|-----------|----------|
| Crash detection (TCP) | Yes (socket EOF/error) | Yes (TCP keepalive, `UCX_TCP_KEEPIDLE=10s`) | Yes (socket error) |
| Crash detection (ibverbs) | N/A for CPU benchmarks | Yes (QP error + `UCX_KEEPALIVE_INTERVAL=20s`) | Yes (via UCX QP error) |
| Per-operation timeouts | Yes (configurable per op) | Partial (UCX transport-level, not per-collective) | No (MPI standard has no timeout concept) |
| Hang detection | Yes (operation timeout fires) | Partial (keepalive detects dead process, not hung collective) | No (collective hangs indefinitely if one rank stalls) |
| Abort communicator | Yes (`ProcessGroup::abort()`) | Yes (destroy UCC context + UCX worker) | Yes (`MPI_Comm_free`, but state may be corrupted) |
| Recover without full restart | Yes (tear down + recreate process group) | No | No (ULFM proposed but not in MPI standard) |
| Default error behavior | Throw C++ exception | Return error code / throw | `MPI_ERRORS_ARE_FATAL` — calls `abort()` on all ranks |

**Gloo** has the best fault tolerance story for CPU workloads. Each operation accepts a timeout, and Gloo's TCP transport detects peer crashes via socket EOF ("Connection closed by peer") [[18]]. When a timeout or error occurs, Gloo signals all pending operations with an exception, transitions the pair to CLOSED state, and wakes waiting threads [[18]]. Crucially, Gloo supports **in-process recovery**: the application can catch the timeout exception, tear down the failed process group, and create a new one without restarting the worker processes. This enables frameworks like `torchrun`'s elastic agent to replace a failed worker and re-form the group without a full job restart.

**UCC / UCX** provides crash detection but limited timeout control. UCX's TCP transport has kernel-level keepalive (`UCX_TCP_KEEPIDLE=10s`, `UCX_TCP_KEEPINTVL=2s`) to detect dead peers [[16]]. For ibverbs, UCX uses its own keepalive mechanism (`UCX_KEEPALIVE_INTERVAL=20s`, checking up to `UCX_KEEPALIVE_NUM_EPS=128` endpoints per round) to detect unresponsive peers [[16]]. RC/DC transport retries are bounded (`UCX_RC_MLX5_TIMEOUT=1s`, `UCX_RC_MLX5_RETRY_COUNT=7`) [[16]], so a crashed peer will eventually trigger a QP error rather than hanging forever. However, UCC does not expose per-collective timeouts — a hung (alive but stalled) peer will cause other ranks to block indefinitely in the collective, with no way to time out at the application level.

**Open MPI** has the weakest fault tolerance story. The MPI standard defines no timeout mechanism for any operation [[4]] — collectives and point-to-point calls block until completion or error. By default, MPI uses `MPI_ERRORS_ARE_FATAL` [[4]], which calls `abort()` on all ranks when any error is detected, taking down the entire job. Setting `MPI_ERRORS_RETURN` allows error codes to be returned instead, but the MPI standard leaves behavior after an error largely **undefined** [[4]]. **ULFM (User-Level Fault Mitigation)** is a proposed extension for structured recovery, but it is not part of the MPI standard and has only experimental support in Open MPI [[19]] [[20]].

#### ibverbs-Specific Concerns

RDMA introduces unique fault tolerance challenges beyond TCP:

- **QP error state is terminal**: When a peer crashes mid-RDMA operation, the local QP transitions to an error state. This state is **unrecoverable** — the QP cannot be reset or reused, and a new QP must be created (which effectively requires tearing down the communicator).
- **One-sided operations are invisible**: RDMA read/write operations bypass the remote CPU entirely. If the remote process crashes after registering memory but before deregistering it, the local side may still complete RDMA operations against stale or freed memory without any error indication.
- **No built-in keepalive**: ibverbs has no equivalent of TCP keepalive. A hung (but alive) remote process that stops posting receives will cause the sender to block on flow control credits. UCX addresses this with its own keepalive mechanism, but Gloo's ibverbs transport has no such layer — a hung peer can cause indefinite blocking.
- **Memory registration cleanup**: When a process crashes, its memory registrations are cleaned up by the kernel's ibverbs subsystem. However, there is a window between the crash and cleanup where remote RDMA operations may still succeed against the now-dead process's memory, potentially reading stale data.

#### Practical Implications

1. **Gloo (TCP)**: Best recovery story. Per-operation timeouts give bounded failure detection. Clean exception propagation allows the application to catch the error, destroy the process group, and recreate it in-process — no worker restart needed. This is the only backend that supports true elastic recovery.
2. **UCC/UCX (TCP or ibverbs)**: Detection within ~20-30 seconds via keepalive. Transport-level retries mask transient issues. No per-collective timeout, so a hung (not crashed) peer is harder to detect. Recovery requires full worker restart.
3. **MPI**: Detection depends on transport (usually via UCX in Open MPI 5.x). Default behavior is immediate process termination (`MPI_ERRORS_ARE_FATAL`). No standard recovery path — full job restart required.

PyTorch's `torchrun` (elastic agent) handles recovery at a higher level by monitoring worker processes and re-forming process groups. With Gloo, this can happen without restarting worker processes; with UCC or MPI, the failed worker (and typically all workers) must be restarted.

## Benchmark Setup

- **Platform**: CentOS Stream 9, Linux 6.16.1 (devvm, x86_64)
- **CPU**: AMD EPYC 9654 96-Core Processor
- **Ranks**: 8 (single node)
- **Device**: CPU
- **Dtype**: float32
- **Python**: 3.12.13
- **PyTorch**: 2.12.0.dev20260409+cpu (nightly)
- **Method**: Calibration phase (3 iterations) to estimate per-iteration cost, then all ranks agree on a fixed iteration count via allreduce(MAX) targeting ~0.3s per test point
- **Collectives**: all_reduce (SUM), all_gather, broadcast (root=0), barrier
- **Backends tested**:
  - **Gloo** — PyTorch's built-in CPU backend (TCP sockets)
  - **UCC (shm)** — UCC 1.7.0 / UCX 1.20.0 with default transports (shared memory for intra-node)
  - **UCC (tcp)** — UCC 1.7.0 / UCX 1.20.0 with `UCX_TLS=tcp` to force TCP-only, disabling shared memory
  - **MPI (shm)** — Open MPI 5.0.10 with default transports (shared memory + TCP)
  - **MPI (tcp)** — Open MPI 5.0.10 with `--mca pml ob1 --mca btl tcp,self` to force TCP-only, disabling shared memory

## Results

### all_reduce (SUM)

| Size | Gloo (us) | UCC shm (us) | UCC TCP (us) | MPI shm (us) | MPI TCP (us) | Gloo/shm | Gloo/MPI shm |
|------|----------:|-------------:|-------------:|--------------:|-------------:|---------:|-------------:|
| 64B | 1162.57 | 4.38 | 37.49 | 4.38 | 34.06 | **265x** | **265x** |
| 4KB | 1186.42 | 14.48 | 66.79 | 43.47 | 42.74 | **82x** | **27x** |
| 256KB | 1767.53 | 384.73 | 792.22 | 524.28 | 684.04 | **4.6x** | **3.4x** |
| 4MB | 10328.96 | 5323.21 | 7140.41 | 7029.73 | 9463.10 | **1.9x** | **1.5x** |
| 64MB | 136087.37 | 65721.39 | 102138.42 | 73083.98 | 150737.24 | **2.1x** | **1.9x** |
| 100MB | 205611.60 | 98008.67 | 155671.75 | 106005.20 | 236261.68 | **2.1x** | **1.9x** |

### all_gather

| Size | Gloo (us) | UCC shm (us) | UCC TCP (us) | MPI shm (us) | MPI TCP (us) | Gloo/shm | Gloo/MPI shm |
|------|----------:|-------------:|-------------:|--------------:|-------------:|---------:|-------------:|
| 64B | 630.90 | 16.74 | 43.85 | 16.36 | 45.59 | **38x** | **39x** |
| 4KB | 698.33 | 55.73 | 210.58 | 65.45 | 68.29 | **13x** | **11x** |
| 256KB | 3571.05 | 1489.68 | 2083.45 | 1652.15 | 2080.78 | **2.4x** | **2.2x** |
| 4MB | 43499.36 | 28126.84 | 44077.65 | 37458.29 | 44719.21 | **1.5x** | **1.2x** |
| 64MB | 578841.58 | 351086.83 | 540097.46 | 527391.27 | 780076.00 | **1.6x** | **1.1x** |
| 100MB | 938607.62 | 498710.01 | 825883.94 | 790732.49 | 1269653.75 | **1.9x** | **1.2x** |

### broadcast (root=0)

| Size | Gloo (us) | UCC shm (us) | UCC TCP (us) | MPI shm (us) | MPI TCP (us) | Gloo/shm | Gloo/MPI shm |
|------|----------:|-------------:|-------------:|--------------:|-------------:|---------:|-------------:|
| 64B | 80.84 | 2.04 | 17.95 | 2.16 | 14.99 | **40x** | **37x** |
| 4KB | 99.59 | 11.29 | 31.15 | 14.25 | 11.35 | **8.8x** | **7.0x** |
| 256KB | 708.84 | 215.35 | 487.86 | 157.80 | 1247.91 | **3.3x** | **4.5x** |
| 4MB | 8190.41 | 2351.52 | 4628.59 | 1254.38 | 13812.64 | **3.5x** | **6.5x** |
| 64MB | 78704.75 | 29628.26 | 60145.72 | 21044.99 | 286546.08 | **2.7x** | **3.7x** |
| 100MB | 169123.91 | 46591.16 | 92161.93 | 37325.11 | 326405.83 | **3.6x** | **4.5x** |

### barrier

| Size | Gloo (us) | UCC shm (us) | UCC TCP (us) | MPI shm (us) | MPI TCP (us) | Gloo/shm | Gloo/MPI shm |
|------|----------:|-------------:|-------------:|--------------:|-------------:|---------:|-------------:|
| N/A | 167.33 | 5.48 | 59.15 | 3.32 | 32.62 | **31x** | **50x** |

## Analysis

### Performance Tiers

The results reveal three clear performance tiers:

1. **UCC (shm) and MPI (shm)** — fastest, within 2x of each other on most collectives
2. **UCC (tcp) and MPI (tcp)** — intermediate, constrained by TCP overhead
3. **Gloo** — slowest, 2-265x behind depending on collective and message size

### MPI (shm) vs UCC (shm): Near-Parity with Shared Memory

MPI and UCC perform remarkably similarly when both use shared memory transports for intra-node communication. At the smallest sizes:

- **64B all_reduce**: MPI and UCC shm are **identical** at 4.38us — both bypass the kernel entirely via shared memory
- **Barrier**: MPI is slightly **faster** (3.32us vs 5.48us), likely due to MPI's optimized barrier algorithm for small communicators
- **Broadcast**: MPI is **faster** at large sizes (37.3s vs 46.6s for 100MB), suggesting superior pipelining in Open MPI's tree broadcast

At medium sizes (4KB-4MB), MPI trails UCC shm by 1.3-3x for all_reduce — UCC's allreduce algorithm selection (recursive halving-doubling for small, ring for large) appears to have tighter transitions than Open MPI's `coll/tuned`.

### MPI (tcp) vs UCC (tcp): TCP-Only Head-to-Head

With shared memory disabled on both sides, we can directly compare their TCP implementations:

- **Small messages (64B-4KB)**: MPI TCP and UCC TCP are **near-parity** — MPI TCP is slightly faster for all_reduce (34us vs 37us) and broadcast (15us vs 18us), while UCC TCP wins all_gather (44us vs 46us)
- **Medium messages (256KB-4MB)**: MPI TCP and UCC TCP remain close for all_reduce (684us vs 792us at 256KB) and all_gather (2081us vs 2083us at 256KB). UCC TCP has a clear advantage for broadcast at 4MB (4629us vs 13813us)
- **Large messages (64-100MB)**: MPI TCP degrades significantly — all_reduce at 100MB is 236ms (MPI TCP) vs 156ms (UCC TCP), and all_gather at 100MB is 1270ms vs 826ms. **UCC TCP is 1.5x faster** for large all_reduce and all_gather

MPI TCP's large-message weakness in all_gather (1270ms, the **slowest of all backends** including Gloo at 939ms) suggests Open MPI's `ob1` PML with TCP BTL has poor pipelining for TCP-only gather operations.

### Broadcast: Divergent Stories for shm vs tcp

MPI (shm) **outperforms even UCC shm** on broadcast at all sizes ≥256KB (157us vs 215us at 256KB, 37.3s vs 46.6s at 100MB). This is likely due to Open MPI's `coll/tuned` module using a **pipelined binary tree** broadcast.

However, MPI (tcp) is the **worst broadcast performer** at large sizes (326ms for 100MB, slower than Gloo's 169ms). The `ob1` PML's tree broadcast apparently relies heavily on shared memory for its pipelining advantage — without it, the algorithm's latency-sensitivity makes TCP round-trips devastating.

### Barrier: Transport Clearly Matters

Barrier latency cleanly separates shared memory from TCP:
- **MPI (shm)**: 3.32us — fastest overall
- **UCC (shm)**: 5.48us
- **MPI (tcp)**: 32.62us — ~10x slower than MPI (shm)
- **UCC (tcp)**: 59.15us
- **Gloo**: 167.33us

### Decomposing the Advantages

With five backends, we can separate transport and software stack effects:

1. **Shared memory vs TCP** (same software stack):
   - UCC: shm is 2-11x faster at small sizes, 1.5-2x at large
   - MPI: shm is 2-9x faster at small sizes, but MPI TCP can be **slower than Gloo** for large broadcast/all_gather
2. **UCC vs MPI** (same transport):
   - With shm: MPI wins broadcast by 1.2-1.9x; UCC wins all_reduce at medium sizes by 1.3-3x
   - With TCP: UCC wins large messages by 1.3-1.5x across all collectives; MPI wins small-message all_reduce/broadcast
3. **UCC/MPI TCP vs Gloo** (all TCP): UCC TCP is 1.3-31x faster than Gloo; MPI TCP is generally faster than Gloo except for large broadcast/all_gather

### Caveats

- These results are for **intra-node CPU communication only**. Inter-node benchmarks (where all backends would use TCP or RDMA to remote hosts) would show a different profile.
- UCC's shared memory transport benefits from co-located processes sharing a physical NUMA domain. Cross-NUMA performance may differ.
- The `UCX_TLS=tcp` override was set via environment variable before UCC initialization. UCP reads transport configuration at context creation time, so the TCP-only comm may share some internal UCX state with the default comm created earlier in the same process.
- MPI TCP-only mode was achieved via `--mca pml ob1 --mca btl tcp,self`, which forces Open MPI to use the OB1 PML with only TCP and self (loopback) BTLs, bypassing UCX and shared memory entirely.
- Gloo is battle-tested in production at Meta scale. UCC and MPI are less widely deployed in PyTorch contexts and may have edge cases in error handling, fault tolerance, or behavior under network partitions.
- MPI was tested with Open MPI 5.0.10 from conda-forge. Different MPI implementations (MPICH, Intel MPI) or versions may show different performance characteristics.
- MPI was launched via `mpirun`, while Gloo/UCC used `torchrun`. The different process launchers may affect startup/initialization overhead, but steady-state collective performance should be unaffected.

## Reproducing

```bash
# Gloo vs UCC (8 CPU ranks)
source .venv/bin/activate
LD_LIBRARY_PATH=/home/tristanr/.conda/envs/autotools/lib:$LD_LIBRARY_PATH \
  UCX_WARN_UNUSED_ENV_VARS=n \
  torchrun --nproc_per_node=8 \
  comms/torchcomms/tests/perf/py/gloo_vs_ucc_bench.py

# MPI with shared memory (8 CPU ranks)
source .venv/bin/activate
LD_LIBRARY_PATH=/home/tristanr/.conda/envs/autotools/lib:$LD_LIBRARY_PATH \
  GLOG_minloglevel=2 \
  /home/tristanr/.conda/envs/autotools/bin/mpirun --oversubscribe \
  --mca btl_tcp_if_include lo -np 8 \
  python comms/torchcomms/tests/perf/py/mpi_bench.py

# MPI TCP-only (8 CPU ranks)
source .venv/bin/activate
LD_LIBRARY_PATH=/home/tristanr/.conda/envs/autotools/lib:$LD_LIBRARY_PATH \
  GLOG_minloglevel=2 \
  /home/tristanr/.conda/envs/autotools/bin/mpirun --oversubscribe \
  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include lo -np 8 \
  python comms/torchcomms/tests/perf/py/mpi_bench.py
```

## References

- [[1]] Gloo GitHub repository — https://github.com/facebookincubator/gloo
- [[2]] UCC GitHub repository — https://github.com/openucx/ucc
- [[3]] UCX GitHub repository (README: features, platforms, transports) — https://github.com/openucx/ucx
- [[4]] MPI Forum — MPI Standard specification — https://www.mpi-forum.org/docs/
- [[5]] Open MPI project — https://www.open-mpi.org/
- [[6]] Open MPI 5.x source (BTL, PML, MTL module listings) — verified via GitHub `open-mpi/ompi` v5.0.x tree: `openib` BTL removed, UCX PML and OFI MTL present — https://github.com/open-mpi/ompi/tree/v5.0.x
- [[7]] Microsoft MPI documentation — https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- [[8]] Gloo ibverbs transport source — `IBV_WR_RDMA_WRITE_WITH_IMM`, `IBV_WR_RDMA_READ`, no `IBV_SEND_INLINE`, single QP/device, software tag management — https://github.com/facebookincubator/gloo/blob/main/gloo/transport/ibverbs/pair.cc
- [[9]] UCX mlx5 devx interface source — `#include <infiniband/mlx5dv.h>`, `HAVE_DEVX`, `uct_ib_mlx5_devx_create_qp()` — https://github.com/openucx/ucx/blob/master/src/uct/ib/mlx5/ib_mlx5.h
- [[10]] UCX DC transport source — DCI pool management, hardware tag matching support — https://github.com/openucx/ucx/blob/master/src/uct/ib/mlx5/dc/dc_mlx5.h
- [[11]] AWS EFA overview — https://aws.amazon.com/hpc/efa/
- [[12]] libfabric EFA provider source — https://github.com/ofiwg/libfabric/tree/main/prov/efa
- [[13]] AWS EFA setup guide (confirms libfabric `efa` provider, `fi_info -p efa` output) — https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html
- [[14]] AWS EFA kernel driver README (confirms SRD: "EFA supports unreliable datagrams (UD) as well as a new Scalable (unordered) Reliable Datagram protocol (SRD)") — https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/README
- [[15]] AWS EFA RDMA support by instance type (RDMA read Nitro v4+, RDMA write varies) — https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
- [[16]] UCX 1.20.0 runtime inspection — `ucx_info -d` (transport list), `ucx_info -f` (keepalive/timeout parameters: `UCX_KEEPALIVE_INTERVAL=20s`, `UCX_RC_MLX5_TIMEOUT=1s`, `UCX_RC_MLX5_RETRY_COUNT=7`, `UCX_TCP_KEEPIDLE=10s`) — measured locally
- [[17]] aws-ofi-nccl plugin — https://github.com/aws/aws-ofi-nccl
- [[18]] Gloo TCP transport error handling source — socket EOF detection ("Connection closed by peer"), timeout support, exception propagation, CLOSED state transition — https://github.com/facebookincubator/gloo/blob/main/gloo/transport/tcp/pair.cc
- [[19]] ULFM specification and Open MPI implementation — https://fault-tolerance.org/category/ulfm/
- [[20]] Open MPI fault tolerance FAQ — https://www.open-mpi.org/faq/?category=ft
