# CTRAN AllReduce Algorithm Design

This document describes the CTRAN AllReduce algorithm shape used by the current stack and records the details of the implemented tree-based path. The implementation is selected with `NCCL_ALLREDUCE_ALGO=ctree` and uses Pipes `MultiPeerTransport` for `NVL_ONLY`, `IB_ONLY`, and `HYBRID` topologies.

## Goals

The CTRAN AllReduce path is structured as local reduction, cross-node reduction, and local publication phases. This keeps topology-specific pieces isolated enough that the cross-node phase can use a tree today and later grow ring or other implementations without changing the local phase contract.

The current `ctree` implementation targets lower latency than ring-style inter-node traversal by reducing the IB phase to tree depth. It keeps correctness independent of CUDA inter-block synchronization: a single-block launch is valid for every supported topology, and additional blocks only add independent tile parallelism.

The implementation currently supports `commSum` for `commFloat32` and `commFloat16`. Unsupported reduction ops and datatypes are rejected on the host.

## Algorithm Overview

1. **NVL ReduceScatter**: local GPUs reduce each owner segment into a per-owner Phase 2 buffer.
2. **IB AllReduce**: active segment owners run the cross-node implementation for their segment. The current implementation is dual-tree.
3. **NVL AllGather**: segment owners publish the globally reduced segments to local peers.

Degenerate topology paths are skipped:

- `nRanks == 1`: identity copy for out-of-place, no kernel launch for in-place.
- `nLocalRanks <= 1`: NVL phases and NVL phase-transition sync are no-ops.
- `nNodes == 1`: IB Phase 2 is a no-op.

## Partitioning and Topology

Let `pMin = min(nLocalRanks(node))` across nodes. The algorithm partitions the tensor into `pMin` owner segments. Local ranks `< pMin` own one segment and participate in the IB phase; extra local ranks on larger nodes contribute during NVL ReduceScatter and receive the final tensor during NVL AllGather, but do not join IB.

Each owner segment is split across `numBlocks` logical CUDA blocks. Each block owns a disjoint segment tile. Inside the block tile, the IB phase splits data into two lane halves:

- Lane 0: Tree-0 over the first half.
- Lane 1: Tree-1 over the second half.

No block owns data or staging used by another block.

### Binary Dual Trees

The implementation uses binary dual trees. Tree-0 is a BFS binary tree rooted at rank `0`. For even node counts, Tree-1 is the mirror of Tree-0 over rank space. For odd node counts, Tree-1 uses the shift-by-1 construction implemented by the topology builder.

For the 8-rank `IB_ONLY` case, the exact topology is:

```text
Tree-0, rooted at rank 0:
  0 -> {1, 2}
  1 -> {3, 4}
  2 -> {5, 6}
  3 -> {7}

Tree-1, mirrored and rooted at rank 7:
  7 -> {6, 5}
  6 -> {4, 3}
  5 -> {2, 1}
  4 -> {0}
```

The two trees process disjoint data halves and use disjoint staging slices and transport group IDs. They are intended to make concurrent progress, not to run Tree-0 to completion before Tree-1.

## Phase Behavior

### Phase 1: NVL ReduceScatter

When `nLocalRanks > 1`, each local GPU contributes the segment data needed by each owner. Segment owners reduce local contributions into `phase2Buf`. The current implementation uses Pipes NVL transport staging because user buffers are not assumed to be directly usable as NVL transport buffers.

When `nLocalRanks <= 1`, Phase 1 is a no-op and the IB tree reads directly from the local owner segment.

### Phase 2: IB AllReduce

The cross-node phase is the replaceable part of the design. Its contract is to reduce each owner segment across nodes and leave the globally reduced segment in the owner's Phase 2 buffer. The current implementation uses a dual-tree IB phase.

#### Current Tree Implementation

When `nNodes > 1`, each active owner rank runs the dual-tree phase. One full CUDA block services both tree lanes cooperatively. The block creates two virtual IB transport groups:

```text
groupId(lane) = blockIdx.x * 2 + lane
```

Each lane has its own data half, staging slice, and transport group ID. The scheduler calls bounded progress functions for lane 0 and lane 1 in a uniform loop. If a lane is waiting on `NIC_DONE`, `SLOT_FREE`, or `DATA_READY`, that progress attempt returns immediately and the block tries the other lane. This avoids serial tree-lane scheduling while keeping all threads in the block on a uniform control path.

Internal tree nodes receive from all children through transport-owned staging, reduce local and child inputs, and then forward the reduced data upward. Broadcast then flows from root to leaves.

### Phase 3: NVL AllGather

When `nLocalRanks > 1`, segment owners publish globally reduced segments to all local peers using Pipes NVL transport. Each block waits for both IB lanes for its tile before publishing that tile locally. When `nLocalRanks <= 1`, Phase 3 is skipped.

## Launch and Tiling Policy

The kernel uses `640` CUDA threads per block. `numBlocks` is capped at `16` for H100 and current GB200/GB300 use. The default policy starts at the cap and reduces blocks for small messages while:

```text
totalBytes < numBlocks * 640 * 64
```

`NCCL_CTRAN_MAX_NBLOCKS` controls the cap and defaults to `16`, which keeps the tree implementation below the NCCL Tree CTA count used on the current H100 and GB200/GB300 validation setups.

The implementation uses one topology-agnostic tile width for both NVL and IB reductions:

```text
kTileBytes = kBlockSize * kTileBytesPerThread = 640 * 96B = 60KB
```

`TileElems<T>` converts this byte width into a datatype-specific element count while preserving tile divisibility for the supported datatypes and the vector widths used by the reduction helpers.

## Staging and Memory Lifetime

The tree implementation does not allocate message-size-dependent AllReduce staging. NVL and IB receives use Pipes transport-owned staging buffers, and the tree path consumes those transient staging slices inside transport copy callbacks before the transport acknowledges and reuses the slots.

The IBGDA send/recv transport data buffer is configured by per-communicator hint or `NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE` when provided. The current validation and tuning use the Pipes default `32MiB` per-peer IBGDA data buffer.

## Correctness Invariants

- No inter-block synchronization is required or assumed.
- Each block owns disjoint data and transport group IDs.
- `group.sync()` is used only for block-local phase transitions.
- In-place and out-of-place reductions use the same block ownership model; a block finishes the reads needed for its tile before that tile can be overwritten by later local publication.
- Tail elements are handled by tile bounds and byte-count checks; only valid user elements are read or written.
- All remote lane progress uses device-visible signaling and memory ordering from Pipes transport primitives.

## Validation

The integration test is algorithm-agnostic and validates numerical correctness plus bitwise deterministic output across repeated runs, not trace shape. It covers `NVL_ONLY`, `IB_ONLY`, and `HYBRID` topologies; world sizes `1`, `2`, `3`, `4`, `7`, and `8` for local single-host topologies; and zero, small, medium, large, and tail message sizes in both in-place and out-of-place modes.

Latest local and RE validation, benchmark commands, and 8-rank performance tables are recorded in:

- https://www.internalfb.com/intern/paste/P2354738362/
- https://pxl.cl/9ZhCc

The primary benchmark comparison is forced NCCL Tree against CTREE using the third-party `nccl_allreduce_perf` target. The latest H100 8-rank `IB_ONLY` sweep covers `4B` through `1G` with `#wrong = 0` for CTREE.

## Further Work

- **Performance tuning**: continue tuning launch shape, tile sizing, transport progress, and tiny-message overhead.
- **LL protocol for IB**: add a lower-latency IB protocol for small payloads.
- **NVLS-based ReduceScatter**: replace the current Phase 1 implementation with an NVLS-based local ReduceScatter path.
- **Ring-based IB implementation**: *delivered* — the cross-node IB phase can now select a ring (`cthierarchical_ring`) as well as the tree. See [RingAllReduce.md](RingAllReduce.md).
- **Share ReduceScatter/AllGather with standalone collectives**: the ring's Phase 2 is factored into `reduceScatterRing` / `allGatherRing` halves (see [RingAllReduce.md](RingAllReduce.md)) that are structurally reusable for the standalone ReduceScatter / AllGather collectives; the tree's phases are analogous. Unifying the fused halves with the standalone collectives is possible future work — not yet pursued.
- **Wider reduction ops and datatypes**: add support beyond the current `commSum` scope, with correctness tests and benchmark coverage across all topologies.
- **Uneven `pMin` topology coverage**: add targeted validation for mixed local-rank-count topologies where extra local ranks contribute to NVL phases but do not participate in IB.
