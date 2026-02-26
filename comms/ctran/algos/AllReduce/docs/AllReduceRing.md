# CTRING: AllReduce Ring Algorithm

This document describes the CTRING (Ctran Ring) AllReduce algorithm implementation, including its chunking strategy, resource tuning, and bi-directional AllGather optimization.

**Reference**: [D93668158](https://phabricator.intern.facebook.com/D93668158) - Bi-directional AllGather for AllReduceRing

## Overview

CTRING implements the classic ring AllReduce algorithm with optimizations for GPU execution. The algorithm consists of two phases:

1. **ReduceScatter (RS)**: N-1 steps where data is reduced across ranks
2. **AllGather (AG)**: N-1 steps where reduced shards are distributed to all ranks. With Bi-Directional AG we achieve this in half steps (~N/2)

Total steps in baseline: `2*(N-1)` where N is the number of ranks.

## Algorithm Structure

### Partitions, Shards, and Chunks

The data hierarchy is organized as follows:

```
┌─────────────────────────────────────────────────────────────┐
│                     Total Data (numElements)                 │
├─────────────────────────────┬───────────────────────────────┤
│         Partition 0         │         Partition 1           │ ...
├──────┬──────┬──────┬───────┼──────┬──────┬──────┬──────────┤
│Shard0│Shard1│Shard2│ShardN-1│Shard0│Shard1│...   │          │
├──┬──┬┴─┬──┬─┴─┬──┬─┴─┬──┬──┼──┬──┬┴─┬──┬─┴──────┴──────────┤
│C0│C1│C2│C3│C4 │C5│C6 │C7│...│C0│C1│C2│C3│...                │
└──┴──┴──┴──┴───┴──┴───┴──┴───┴──┴──┴──┴──┴───────────────────┘
```

- **Partition**: If data exceeds `chunkSize * numChunks`, it is split into multiple partitions transferred sequentially
- **Shard**: Each partition is divided into N shards (one per rank). Each rank handles one shard per ring step
- **Chunk**: Each shard is transferred in chunks via temporary buffers. Multiple rounds transfer each shard.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `partitionNumel` | Number of elements per partition |
| `shardNumel` | Number of elements per shard (`partitionNumel / nRanks`) |
| `chunkSize` | Size in bytes of each chunk transfer |
| `numChunks` | Number of chunk slots in temporary buffers |
| `numSteps` | Total ring steps per partition |

## Chunking Strategy

The chunking system implements pipelined data transfer with configurable chunk sizes and counts.

### Pipeline Depth Tiers

The auto-tuner selects pipeline depth based on per-rank message size:

**Default Architecture (GB200/Blackwell)**:
| Per-Rank Message Size | Pipeline Depth |
|----------------------|----------------|
| < 32KB               | 1              |
| 32KB - 1MB           | 2              |
| 1MB - 16MB           | 4              |
| 16MB - 32MB          | 2              |
| > 32MB               | 1              |

**Hopper Architecture (H100)**:
| Per-Rank Message Size | Pipeline Depth |
|----------------------|----------------|
| < 32KB               | 1              |
| 32KB - 1MB           | 2              |
| 1MB - 4MB            | 4              |
| 4MB - 8MB            | 2              |
| > 8MB                | 1              |

### Chunk Size Computation

```cpp
numChunks = pipelineDepth * nRanks
chunkSize = clamp(partitionBytes / numChunks, 1B, 16MB)
```

**Constraint**: `chunkSize * numChunks <= maxBDP`

### Maximum BDP (Bandwidth-Delay Product)

| Architecture | Max BDP |
|--------------|---------|
| GB200/Blackwell | 128 MB |
| Hopper (H100) | 32 MB |

### CVAR Overrides

| CVAR | Description |
|------|-------------|
| `NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE` | Override chunk size |
| `NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS` | Override number of chunks |
| `NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP` | Override max BDP |
| `NCCL_CTRAN_ALLREDUCE_RING_MIN_SHARD_SIZE` | Minimum shard size in bytes (default: 16) |

## Resource Tuning (Blocks/Threads)

The auto-tuner configures CUDA kernel launch parameters based on chunk size.

### Block Count Tiers

**Default Architecture**:
| Chunk Size | Num Blocks |
|------------|------------|
| < 8KB      | 1          |
| 8KB - 32KB | 2          |
| 32KB - 64KB| 4          |
| > 64KB     | 8          |

**Hopper Architecture**:
| Chunk Size  | Num Blocks | Block Size |
|-------------|------------|------------|
| < 16KB      | 1          | 384        |
| 16KB - 128KB| 1          | 512        |
| 128KB - 512KB| 2         | 512        |
| > 512KB     | 4          | 512        |

### CVAR Overrides

| CVAR | Description |
|------|-------------|
| `NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS` | Override block count |
| `NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE` | Override block size |

## Bi-Directional AllGather

### Motivation

In the standard ring AllReduce, the AllGather phase performs N-1 sequential steps where data flows in one direction (clockwise/rightward). Since AllGather is pure copy/forward with no reduction, it can be parallelized by sending data in **both directions simultaneously**.

### Enabling/Disabling

Bi-directional AllGather is controlled via the `NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE` CVAR:

| Value | Behavior |
|-------|----------|
| `0` | Disable bi-directional AG entirely (use simple kernel) |
| `-1` | Enable for all message sizes |
| `> 0` | Enable only for messages up to this size (default: 4MB) |

When disabled, the algorithm uses a simpler kernel (`EnableBidirAg=false`) with lower register usage, which may improve occupancy for certain workloads.

### Design

With bi-directional AllGather:
- **Forward direction**: Handles `ceil((N-1)/2)` AG steps (clockwise)
- **Reverse direction**: Handles `floor((N-1)/2)` AG steps (counter-clockwise)
- Bi-directional AllGather can be turned into standard AllGather by setting `NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE=0`

**Step Reduction**:
- Baseline: `2*(N-1)` steps = RS + AG
- Bi-directional: `(N-1) + ceil((N-1)/2)` steps = RS + Forward AG
- **~25% reduction** in total steps for large N

### Implementation Details

#### Buffer Allocation

Reverse direction uses separate buffers (64MB additional per rank):
- `tmpSendBufRev`: Staging buffer for outbound reverse data
- `tmpRecvBufRev`: Staging buffer for inbound reverse data

#### Shard Indexing

Forward and reverse directions use different shard index calculations:

```cpp
// Forward: shift = N - step (decreasing index per step)
int getStepShardIdx(step) {
    return (sendDataShardIdx + N - step) % N;
}

// Reverse: shift = revStep + 1 (increasing index per step)
// The +1 accounts for RS ending with rank r owning shard (r+1)%N
int getRevStepShardIdx(revStep) {
    return (revSendDataShardIdx + revStep + 1) % N;
}
```

#### Dependencies

Reverse AllGather starts only after ReduceScatter completes (rank's reduced shard must be in `recvbuff` first).

**Host-side dependency chain**:
```
kRevSendCopy -> kRevSendTrans -> (network) -> kRevRecvTrans -> kRevRecvFlush -> kRevRecvCopy -> kRevSendTrans
```

**Device-side kernel loop**:
```cpp
while (anyOpRemaining) {
    _progressSend();      // Forward send
    _progressRecv();      // Forward recv
    _progressRevSend();   // Reverse send
    _progressRevRecv();   // Reverse recv
}
```

### Data Flow Diagram

```
Ring with 4 ranks, bi-directional AllGather:

ReduceScatter (3 steps, clockwise only):
  Step 0: R0→R1→R2→R3→R0  (reduce shard 0,3,2,1 respectively)
  Step 1: R0→R1→R2→R3→R0  (reduce shard 3,0,1,2 respectively)
  Step 2: R0→R1→R2→R3→R0  (reduce shard 2,3,0,1 respectively)

AllGather (forward: 2 steps, reverse: 1 step):
  Forward Step 0: R0→R1→R2→R3→R0  (gather clockwise)
  Forward Step 1: R0→R1→R2→R3→R0  (gather clockwise)
  Reverse Step 0: R0←R1←R2←R3←R0  (gather counter-clockwise, concurrent)

Total: 3 RS + 2 AG = 5 steps (vs 6 without bi-directional)
```

## Performance Results

Bi-directional AllGather shows significant improvements for small-to-medium message sizes, see D93668158 for the relative improvement.

## Code Structure

| File | Description |
|------|-------------|
| `AllReduceRingCommon.cuh` | Shared data structures (`AlgoContext`, `KernArgs`), shard indexing, round tracking |
| `AllReduceRing.cc` | Host-side progress functions, resource setup, main loop, kernel dispatch |
| `AllReduceRing.cuh` | Device kernel with `EnableBidirAg` template: `_progressSend`, `_progressRecv`, `_progressRevSend`, `_progressRevRecv` |
| `AllReduceRingAutoTune.h` | Auto-tune interface: `GpuArch` enum, `AutoTuneParams`, `getAutoTunedParams()` |
| `AllReduceRingAutoTune.cc` | Auto-tune implementation: pipeline depth tiers, block/thread selection |
| `CtranAlgoConsts.h` | Architecture-specific constants (`kDefaultMaxBDP`, `kHopperMaxBDP`) |

## Kernel Variants

The ring kernel is templated on `EnableBidirAg`:

```cpp
template <typename T, commRedOp_t RedOp, bool EnableBidirAg>
__global__ void ncclKernelAllReduceCtranRing(...);
```

| Template Parameter | Kernel Behavior |
|--------------------|-----------------|
| `EnableBidirAg=true` | Full bi-directional AG with reverse direction buffers and progress functions |
| `EnableBidirAg=false` | Standard single-direction AG with lower register usage |

The kernel variant is selected at runtime based on message size and `NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE`.

## Future Optimizations

1. **Adaptive reverse step count**: Dynamically adjust reverse AG steps based on network conditions
2. **Fused kernels**: Combine forward and reverse copy operations for better memory bandwidth
3. **Topology-aware routing**: Optimize ring direction based on NVLink/IB topology
