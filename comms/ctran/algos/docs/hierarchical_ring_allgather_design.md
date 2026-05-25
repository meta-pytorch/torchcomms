# CTRAN Hierarchical Ring AllGather Design

## Status

Draft integration design for a no-GPE, device-driven hierarchical AllGather in ctran. The implementation strategy is a new Pipes collective launched from ctran on the caller's CUDA stream. The kernel uses device-side IBGDA send/recv for inter-node same-local-rank rings and device-side NVLink send/recv for intra-node fanout.

Primary source material:

- `D105039175`: standalone `Memcpy` CopyOp header
- `D105039177`: `P2pNvlTransportDevice` pipeline windowing and CopyOp hooks
- `D105039171`: IBGDA NIC routing cleanup
- `D104725448`: direct NVLink and hierarchical AllGather kernels
- `D105039174`: hierarchical AllGather tests and NCCL comparison benchmark
- `D105039172`: `recv_group` and `forward_group` CopyOp hooks
- `D105039176`: MPI local rank detection

Primary ctran and Pipes files:

- `ctran/algos/AllGather/AllGather.cc`
- `ctran/algos/AllGather/AllGatherImpl.h`
- `ctran/algos/AllGather/AllGatherDirect.cuh`
- `ctran/CtranPipes.cc`
- `ctran/def_build.bzl`
- `ctran/algos/CtranAlgo.h`
- `pipes/P2pIbgdaTransportDevice.cuh`
- `pipes/P2pNvlTransportDevice.cuh`
- `pipes/ThreadGroup.cuh`
- `pipes/collectives/RingAllgatherLauncher.h`
- `pipes/collectives/RingAllgather.cu`
- `pipes/collectives/RingUtils.h`
- `utils/cvars/nccl_cvars.yaml`

## Goals

The north star is to beat NCCL AllGather on multi-node GPU clusters, especially 64+ GPU configurations with 8 GPUs per node. The algorithm should reduce expensive inter-node traffic by running an inter-node ring per local-rank rail, then using high-bandwidth intra-node NVLink to fan out each received rail chunk to local GPUs.

Required properties:

1. Selectable with `NCCL_ALLGATHER_ALGO=cthierarchical_ring`.
2. Integrated into the ctran AllGather eager path without introducing a new ctran framework.
3. No ctran GPE thread in the data path.
4. Pipeline overlap between device-driven IBGDA ring progress and device-driven NVLink fanout.
5. Final-layout placement in `recvbuff`: IBGDA receive/forward copies place data into the final AllGather slot, and NVLink reads from that slot.
6. Graceful fallback when the topology, transport, or size does not benefit.
7. No regressions for single-node or one-GPU-per-node topologies.

## Non-Goals

- Do not replace existing `ctdirect`, `ctring`, `ctrd`, `ctbrucks`, `ctgraph*`, or flat Pipes ring algorithms.
- Do not use `CtranMapper`, `OpElem`, mapper notify objects, or ctran GPE progress for this algorithm.
- Do not require CUDA graph capture for the first production integration.
- Do not require NIC writes directly into user `recvbuff` in the first implementation. Pipes IBGDA send/recv uses registered staging and device copies into final layout.
- Do not add reduction or fused compute semantics in the first diff stack. Keep `CopyOp` hooks so future fused copies do not require changing the ctran API.

## Algorithm Summary

For a communicator with `nNodes` nodes and `nLocalRanks` GPUs per node, each `localRank` is an independent inter-node rail:

```text
rail(localRank) = ranks:
  rank for localRank on node 0,
  rank for localRank on node 1,
  ...,
  rank for localRank on node nNodes - 1
```

Within one rail, there is one GPU per node. All local-rank rails run concurrently, so an 8-GPU node has eight independent inter-node rings rather than one node leader bottleneck.

Each rank gathers the chunks for its own rail across nodes:

```text
rail chunk owner = rank(sourceNode, localRank)
```

After a rail chunk is received from IBGDA and copied into its final local `recvbuff` slot, the same kernel broadcasts that chunk over NVLink to the other local ranks. Conversely, it receives the other local ranks' rail chunks over NVLink. After all source nodes and local ranks finish, every rank has every global rank's AllGather chunk in final layout.

Rank identity must come from communicator metadata. Host setup builds a compact map:

```text
globalRankByNodeLocal[node][localRank]
nodeByGlobalRank[globalRank]
localRankByGlobalRank[globalRank]
```

Use `statex->localRankToRank(localRank, node)` to populate the forward map. If the first device implementation cannot consume this map and instead computes `node * nLocalRanks + localRank`, the support checker must verify that layout for every node/local-rank pair and reject the algorithm otherwise.

## Architecture

### Host Shape

The ctran entry point is a thin launcher:

1. validate topology and Pipes transport resources
2. build a `HierarchicalAllgatherLaunchParams` struct
3. allocate or reuse per-collective device scratch for ready slots and optional rank maps
4. enqueue scratch initialization on the caller stream
5. trigger colltrace and algo stats
6. launch the Pipes hierarchical kernel on the caller stream
7. update ctran op count after successful enqueue

There is no `CtranMapper` registration, no host put/notify loop, and no `KernelStartGpe()`.

### Device Shape

One CUDA kernel handles both phases. Blocks are partitioned into role-local lanes with contiguous group IDs:

```text
gridDim.x =
  numIbBlocks +
  2 * nvlActivePeerLanes * nvlGroupsPerPeer

IBGDA rail lane:
  group_id = 0..numIbBlocks-1
  activeBlocks = numIbBlocks

NVLink send lane for one active peer offset:
  group_id = 0..nvlGroupsPerPeer-1
  activeBlocks = nvlGroupsPerPeer

NVLink recv lane for one active peer offset:
  group_id = 0..nvlGroupsPerPeer-1
  activeBlocks = nvlGroupsPerPeer
```

Use `ThreadGroup::partition()`, `partition_interleaved()`, or equivalent role-local construction so each transport sees contiguous group IDs. Passing global block IDs to one lane while `activeBlocks` is lane-local can corrupt staging state or deadlock signal waits. Never pass `gridDim.x` to a transport unless every launched block calls that exact transport instance in the same call sequence.

The IBGDA lane runs the same-local-rank inter-node ring. After each tile/window is copied into final `recvbuff`, the producing IBGDA group publishes a device ready slot. NVLink send lanes consume those slots and send from final `recvbuff` to a deterministic peer offset; matching NVLink recv lanes compute the same deterministic sequence and receive peer rail chunks into final layout. If `nvlActivePeerLanes < nLocalRanks - 1`, the kernel processes peer offsets in rounds.

Because ready slots are bounded and producers may spin waiting for consumers before slot reuse, the implementation must choose one deadlock-prevention strategy before the production kernel lands:

1. Runtime resident-block guard: compute active blocks per SM for the compiled kernel with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, multiply by SM count, and reject configurations where `gridDim.x` can exceed the resident block budget.
2. Non-reused monotonic readiness: replace bounded ready slots with per-window counters or another scheme that does not require producers to wait for consumer lanes before reuse.

A bounded producer/consumer queue without a resident-capacity proof can deadlock if all resident blocks are producers waiting for consumer lanes that have not been scheduled.

### Why No GPE

The target performance range is 256 KB through 1 GB on multi-node, multi-GPU nodes. Host/GPE progress adds notify polling, flush completion, CPU scheduling jitter, and host/device handoff latency. A single device-driven kernel keeps progress on GPU, allows tile/window-level overlap, and matches the direction most likely to beat NCCL on large multi-node jobs.

### P0 Validation Prototype

The existing Pipes hierarchical fused kernel is not sufficient evidence for this design because it runs the inter-node phase and then the NVLink phase sequentially. This design's performance model requires a new concurrent multi-role kernel where IBGDA producer lanes and NVLink consumer lanes are resident at the same time and synchronize through device ready state.

Before implementing the full production algorithm, build a minimal prototype that proves the central overlap model:

- topology: start with 2x2 if available, then 4x2 on the GB200 testbed
- roles: one IBGDA rail lane, one NVLink send lane, one NVLink recv lane
- synchronization: ready slots or the chosen monotonic counter scheme
- validation: confirm IBGDA progresses later windows while NVLink consumes earlier windows
- metrics: compare measured time against both `T_ibgda + T_nvl` and `max(T_ibgda, T_nvl)`

This prototype is a P0 gate. If the prototype cannot demonstrate meaningful overlap without deadlock or severe backpressure, the full hierarchical implementation is unlikely to beat best NCCL.

## Proposed File Changes

Create:

- `ctran/algos/AllGather/AllGatherHierarchicalRing.cc`
- `pipes/collectives/HierarchicalAllgather.cuh`
- `pipes/collectives/HierarchicalAllgather.cu`
- `pipes/collectives/HierarchicalAllgatherLauncher.h`
- `pipes/collectives/HierarchicalAllgatherLauncher.cu`
- `ctran/benchmarks/AllGatherHierarchicalRingBench.cc`

Modify:

- `ctran/CtranComm.h`
- `ctran/algos/AllGather/AllGather.cc`
- `ctran/algos/AllGather/AllGatherImpl.h`
- `ctran/algos/CtranAlgo.cc`
- `ctran/CtranPipes.cc`
- `ctran/def_build.bzl`
- `pipes/collectives/BUCK` or the owning build definition for Pipes collectives
- `utils/cvars/nccl_cvars.yaml`
- generated cvar outputs, including `utils/cvars/nccl_cvars.h` and `utils/cvars/nccl_cvars.cc`
- `ncclx/meta/algoconf/AlgoConfig.cc`
- `ctran/tests/CtranDistAllgatherTests.cc`
- `ctran/tests/BUCK`
- `ctran/algos/AllGather/tests/AllGatherCtgraphSupportTest.cc` or a new support test

## Launch Parameters And Device Args

Add a Pipes launch params struct rather than extending ctran's device `KernelArgs` for the data path.

```cpp
namespace comms::pipes {

struct HierarchicalAllgatherLaunchParams {
  int myRank{0};
  int nRanks{0};
  int node{0};
  int nNodes{0};
  int localRank{0};
  int nLocalRanks{0};
  size_t sendBytes{0};
  size_t windowBytes{0};
  size_t ibSignalBytes{0};
  size_t nvlSignalBytes{0};
  int numIbBlocks{0};
  int nvlActivePeerLanes{0};
  int nvlGroupsPerPeer{0};
  int readyQueueDepth{0};
  float timeoutMs{0.0f};
  cudaStream_t stream{nullptr};

  const char* sendbuff{nullptr};
  char* recvbuff{nullptr};

  P2pIbgdaTransportDevice* ibPrev{nullptr};
  P2pIbgdaTransportDevice* ibNext{nullptr};
  P2pNvlTransportDevice* nvlPeers[kMaxLocalRanks]{};

  const int* globalRankByNodeLocal{nullptr};
  ReadySlot* readySlots{nullptr};
};

} // namespace comms::pipes
```

Device args can be a compact by-value copy of the launch params with fixed-size transport arrays. The rank map is indexed as:

```text
globalRankByNodeLocal[node * nLocalRanks + localRank]
```

The ready queue is device scratch with one bounded queue per IBGDA producer group:

```cpp
struct ReadySlot {
  uint64_t seq;
  int sourceNode;
  int sourceGlobalRank;
  size_t offset;
  size_t bytes;
  uint32_t consumedPeerMask;
};
```

The scratch layout is:

```text
readySlots[ibGroupId * readyQueueDepth + slotIdx]
```

The IBGDA group writes `sourceNode`, `sourceGlobalRank`, `offset`, `bytes`, resets `consumedPeerMask`, fences final `recvbuff` stores, then release-stores `seq`. One NVLink send lane per local peer acquire-waits for the sequence, sends from final `recvbuff`, and sets its peer bit in `consumedPeerMask`. The IBGDA producer may reuse the slot only after all local-peer bits are set.

NVLink receivers do not consume ready metadata from the sender. They compute the same deterministic sequence from `(sourceNode, windowIdx, ibGroupId, peerLocalRank)`, so every local rank must use identical node-ring orientation, window sizing, and group-to-tile mapping.

Scratch allocation should use a ctran/Pipes device scratch pool if one exists. A first implementation may use `cudaMallocAsync`, `cudaMemsetAsync`, and `cudaFreeAsync` on the collective stream, but allocator overhead must be measured before auto-selection is enabled.

## Host-Side Design

### Entry Point

Add:

```cpp
commResult_t ctranAllGatherHierarchicalRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream);
```

The entry point should:

1. call `CTRAN_COLL_INFO`
2. compute `sendBytes = sendcount * commTypeSize(datatype)`
3. validate support gates
4. build `KernelConfig(KernelType::ALLGATHER, stream, algoName, opCount)` for metadata
5. call `ctranKernelSetAllGatherArgs()` for colltrace metadata only
6. build `HierarchicalAllgatherLaunchParams`
7. initialize device scratch on `stream`
8. trigger colltrace `BeforeEnqueueKernel`
9. record algo stats manually
10. call `launch_hierarchical_allgather(params)`
11. trigger colltrace `AfterEnqueueKernel`
12. call `comm->ctran_->updateOpCount()`

Do not synchronize the stream. Ordering must come from enqueuing scratch initialization, kernel launch, and scratch free on the caller stream.

### Buffer Handling

Do not call existing `prepareAllGatherArgs()` for this no-GPE path. That helper is tied to mapper registration and ctran temporary registered buffers. Pipes send/recv owns its staging, so this algorithm should operate on the user `sendbuff` and `recvbuff` directly.

The device kernel must perform the local self-copy into:

```text
recvbuff[myRank * sendBytes : (myRank + 1) * sendBytes]
```

before sending local data into the inter-node ring. For in-place AllGather, the kernel should detect the existing call convention and use the local slot in `recvbuff` as the source. In-place and out-of-place paths need explicit correctness coverage.

Because this path bypasses `extraCopyBuff`, add a small-message test below `CTRAN_MIN_REGISTRATION_SIZE` to prove no temporary registered copyback is required.

### Pipes Transport Setup

Reuse existing ctran/Pipes initialization:

- `NCCL_CTRAN_USE_PIPES=1` must initialize `comm->multiPeerTransport_`
- `NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1` must configure IBGDA send/recv staging
- IBGDA send/recv staging memory must be positive, using the per-communicator override when set, otherwise `NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE`
- `NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS` must cover `numIbBlocks`
- NVLink tile transport resources must cover `nvlGroupsPerPeer`

### Per-Communicator IBGDA Memory Override

Add a per-communicator Pipes override for IBGDA send/recv staging memory, parallel to the existing per-communicator NVLink override:

```cpp
struct ctranPipesConfig {
  int64_t nvlChunkSize{-1};
  int useDualStateBuffer{-1};
  int64_t ibgdaDataBufferSize{-1};
};
```

`-1` means use the global cvar default. A positive value overrides `NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE` for this communicator only:

```cpp
config.ibgdaConfig.dataBufferSize =
    (comm->config_.pipesConfig.ibgdaDataBufferSize > 0)
    ? static_cast<size_t>(comm->config_.pipesConfig.ibgdaDataBufferSize)
    : static_cast<size_t>(NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE);
```

This is a communicator-construction-time override because `MultipeerIbgdaTransport` allocates send/recv staging buffers and device state during transport initialization. Do not model this as a per-collective or per-call option; changing it after `multiPeerTransport_` has been initialized would require rebuilding Pipes transports.

All ranks in a communicator should use the same effective IBGDA data-buffer size. The support path should log the effective value and reject the hierarchical algorithm if it is zero or if `effectiveDataBufferSize / numIbBlocks < 16`. If the bootstrap/config path can compare per-rank ctran configs, add a consistency check so asymmetric communicator overrides fail early rather than timing out inside transport waits.

For the IB rail:

```text
prevNode = (node + nNodes - 1) % nNodes
nextNode = (node + 1) % nNodes
prevGlobal = rank(prevNode, localRank)
nextGlobal = rank(nextNode, localRank)

ibPrev = mpt->get_p2p_ibgda_transport_device(prevGlobal)
ibNext = mpt->get_p2p_ibgda_transport_device(nextGlobal)
```

For NVLink peers:

```text
for peerLocalRank in 0..nLocalRanks-1:
  if peerLocalRank != localRank:
    peerGlobal = rank(node, peerLocalRank)
    nvlPeers[peerLocalRank] = mpt->get_p2p_nvl_transport_device(peerGlobal)
```

The support checker should validate these handles before launch. Missing handles should return unsupported rather than relying on device traps.

## Device Kernel Design

### Kernel Signature

```cpp
template <typename CopyOp = comms::pipes::Memcpy>
__global__ void hierarchical_allgather_kernel(
    const __grid_constant__ HierarchicalAllgatherArgs args,
    comms::pipes::Timeout timeout);
```

`timeout.start()` must run at kernel entry. The same timeout object must be passed to all IBGDA and NVLink `send`, `recv`, and `forward` calls. If Pipes timeouts currently trap rather than propagate a ctran-visible abort, keep this algorithm explicit-only until abort behavior is production-safe.

### Block Partitioning

Use role-local groups. Conceptually:

```text
lane 0:
  IBGDA rail groups

lanes 1..nvlActivePeerLanes:
  NVLink send groups, one lane per active peer offset

lanes nvlActivePeerLanes+1..2*nvlActivePeerLanes:
  NVLink recv groups, one lane per active peer offset
```

If a weighted partition API is used, ensure every lane has nonzero groups and that each lane-local `group_id` range starts at zero. Do not intermix IBGDA and NVLink calls on the same transport with different `activeBlocks` values inside one collective. Keep `activeBlocks` and `max_signal_bytes` stable for a transport's call sequence; changing slot mapping requires higher-level quiescence.

Default starting point for 8 GPUs per node:

```text
numIbBlocks = 8 or 16
nvlGroupsPerPeer = 2 or 4
nvlActivePeerLanes = 1, 2, 4, or nLocalRanks - 1
```

For `nLocalRanks = 8`, `numIbBlocks=16`, `nvlActivePeerLanes=4`, and `nvlGroupsPerPeer=4` gives `16 + 2*4*4 = 48` blocks. Full seven-peer concurrency with the same group count gives 72 blocks. Gate the chosen configuration against resident capacity or reduce the lane counts.

### Inter-Node IBGDA Rail

The IBGDA role runs a ring over nodes, not over all ranks:

```text
ibRank = node
ibSize = nNodes
railLocalRank = localRank
```

For each window/tile assigned to an IBGDA group:

1. self-copy local data from `sendbuff` or in-place local slot into final `recvbuff[rank(node, localRank)]`
2. publish a ready slot for `(sourceNode=node, windowIdx, ibGroupId)`
3. send the local rail chunk to `nextNode`
4. receive and forward remote rail chunks from `prevNode` through `nextNode`
5. receive the final remote rail chunk without forwarding

Use the existing IBGDA tiled APIs:

```cpp
next.send(group, src, bytes, numIbBlocks, ibSignalBytes, timeout);
prev.forward<CopyOp>(
    group, dst, next, bytes, numIbBlocks, ibSignalBytes, timeout);
prev.recv<CopyOp>(group, dst, bytes, numIbBlocks, ibSignalBytes, timeout);
```

`forward()` is preferred for intermediate ring hops because it preserves the Pipes signal ordering invariant: release predecessor staging before waiting on successor staging. Do not reimplement ring progress as independent send-all/recv-all loops.

The destination for every received source node is final layout:

```text
dst = recvbuff + rank(sourceNode, localRank) * sendBytes + windowOffset + tileOffset
```

After `recv` or `forward` copies the tile into `dst`, the IB group publishes readiness:

```text
__threadfence();
if (group.is_leader()) {
  slot.sourceNode = sourceNode;
  slot.sourceGlobalRank = sourceGlobalRank;
  slot.offset = windowOffset + tileOffset;
  slot.bytes = tileBytes;
  slot.consumedPeerMask = 0;
  releaseStore(slot.seq, expectedSeq);
}
```

Use device-scope release semantics if a local helper exists. NVLink consumers only need GPU visibility; host/system visibility is not required for the ready slot.

### NVLink Fanout

The NVLink role uses direction-split per-peer lanes. For each local peer, one send lane consumes local ready slots and calls `P2pNvlTransportDevice::send()`. The matching recv lane calls `recv<CopyOp>()` for that peer's deterministic stream. This avoids relying on a global all-ranks-send-first phase whose safety can depend on pipeline depth.

Recommended schedule:

```text
for sourceNode in ring order:
  for windowIdx in ascending order:
    for ibGroupId in ascending order:
      for peerRound in deterministic peer-offset rounds:
        send lanes:
          wait readySlot[ibGroupId].seq
          send this localRank's rail tile to assigned local peer
          mark peer bit consumed
        recv lanes:
          receive assigned peer localRank's rail tile into final layout
```

For source node `S`, this rank sends:

```text
src = recvbuff + rank(S, localRank) * sendBytes + windowOffset + tileOffset
```

and receives from peer local rank `P` into:

```text
dst = recvbuff + rank(S, P) * sendBytes + windowOffset + tileOffset
```

Use tiled NVLink APIs with lane-local active group counts:

```cpp
peer.send(
    sendLaneGroup,
    src,
    bytes,
    nvlGroupsPerPeer,
    nvlSignalBytes,
    timeout);
peer.recv<CopyOp>(
    recvLaneGroup,
    dst,
    bytes,
    nvlGroupsPerPeer,
    nvlSignalBytes,
    timeout,
    copyOpArgs...);
```

The local exchange should be deterministic pairwise allgather, not one owner block serially broadcasting to every peer. Pairwise lanes balance NVLink/NVSwitch traffic and make it easier to scale to 8 local GPUs.

The first implementation should enforce a symmetric NVLink group count: send and recv lanes for a peer use the same `nvlGroupsPerPeer`. The tiled NVLink protocol computes staging slot sizes from `dataBufferSize / activeBlocks`; asymmetric send/recv active group counts are not assumed safe unless the transport is explicitly proven to tolerate them.

### Pipeline Windows

Use a window size that is valid for both phases:

```text
ibWindow = min(ibPrev.pipeline_window(numIbBlocks),
               ibNext.pipeline_window(numIbBlocks))
nvlWindow = min(peer.pipeline_window(nvlGroupsPerPeer) for all local peers)
windowBytes = min(configuredWindowBytes, ibWindow, nvlWindow)
```

If any window is zero, support should reject the algorithm before launch. Host validation should also check configured block counts against the transport max-group settings. Device traps are acceptable for debug bugs, not for production support gating.

## Topology And Gating

### Explicit Algorithm Support

`ctranAllGatherSupport(comm, cthierarchical_ring, stream)` should return true only when all conditions hold:

- ctran initialized
- `ENABLE_PIPES` is built
- `comm->multiPeerTransport_` is initialized
- `NCCL_CTRAN_USE_PIPES=1`
- `NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1`
- effective IBGDA data buffer size is positive, using per-communicator override before cvar fallback
- `statex->nNodes() > 1`
- `statex->nLocalRanks() > 1`
- `statex->nRanks() == statex->nNodes() * statex->nLocalRanks()`
- initial production allowlist includes tested node/local-rank geometries, starting with 2x8, 4x8, and 8x8
- `statex->nLocalRanks()` is within the fixed NVLink peer array limit
- every `(node, localRank)` maps to a valid unique global rank
- the device rank map is available, or arithmetic layout is explicitly verified
- current rank has valid IBGDA predecessor and successor for the same `localRank`
- every same-node peer has a valid NVLink transport
- `numIbBlocks <= NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS`
- `nvlActivePeerLanes >= 1` and `nvlActivePeerLanes <= nLocalRanks - 1`
- `nvlGroupsPerPeer` is within the NVLink tile transport group limit
- IBGDA and NVLink pipeline windows are nonzero
- total launched blocks are within the resident-capacity guard when bounded ready queues are used

Do not require mapper backends for this algorithm. Mapper-backed algorithms can keep their existing checks, but Pipes algorithms should gate on `MultiPeerTransport`.

Do not include `NCCL_CTRAN_HIER_ALLGATHER_MIN_SIZE` in explicit support. An engineer forcing `NCCL_ALLGATHER_ALGO=cthierarchical_ring` should be able to test all valid message sizes. The min-size threshold is an auto-selection policy only.

### Auto-Selection

When `NCCL_ALLGATHER_ALGO=ctran`, use this order:

1. if `NCCL_CTRAN_HIER_ALLGATHER_AUTO_ENABLE=true`, hierarchical support passes, the topology is allowlisted, and the topology-specific size threshold passes, choose `cthierarchical_ring`
2. else if `nLocalRanks > 1`, choose existing `ctdirect`
3. else choose existing `ctring` for large messages or keep current behavior

Do not select hierarchical ring for single-node jobs. Single-node performance must remain on `ctdirect` or a future direct NVLink algorithm.

Initial auto-selection thresholds after benchmark signoff should be topology-specific:

```text
2x8: start at >= 1 MB or >= 2 MB, depending on NCCL default strength
4x8: start at >= 512 KB
8x8: start at >= 256 KB
```

Keep explicit `cthierarchical_ring` available for valid smaller messages so engineers can measure threshold behavior.

## CVARs

Add enum choice:

```yaml
- name        : NCCL_ALLGATHER_ALGO
  choices     : orig, ctran, ctdirect, ctring, ctrd, ctbrucks, cthierarchical_ring, ctgraph, ctgraph_pipeline, ctgraph_rdpipeline, ctgraph_ring, ctgraph_rd
```

Add tuning knobs:

```yaml
- name        : NCCL_CTRAN_HIER_ALLGATHER_AUTO_ENABLE
  type        : bool
  default     : false
  description : |-
    Enables automatic selection of ctran hierarchical ring AllGather when NCCL_ALLGATHER_ALGO=ctran and topology/size gates pass.

- name        : NCCL_CTRAN_HIER_ALLGATHER_MIN_SIZE
  type        : uint64_t
  default     : 262144
  description : |-
    Minimum per-rank send size in bytes for automatic selection. Explicit cthierarchical_ring ignores this threshold.

- name        : NCCL_CTRAN_HIER_ALLGATHER_WINDOW_BYTES
  type        : uint64_t
  default     : 1048576
  description : |-
    Requested inter-phase window size. The kernel uses the minimum of this value and the IBGDA/NVLink transport pipeline windows.

- name        : NCCL_CTRAN_HIER_ALLGATHER_NUM_IB_BLOCKS
  type        : int
  default     : 8
  description : |-
    Number of CUDA threadblocks assigned to the IBGDA rail role.

- name        : NCCL_CTRAN_HIER_ALLGATHER_NVL_GROUPS_PER_PEER
  type        : int
  default     : 4
  description : |-
    Number of CUDA threadblocks assigned to each per-peer NVLink send lane and each matching recv lane.

- name        : NCCL_CTRAN_HIER_ALLGATHER_NVL_ACTIVE_PEER_LANES
  type        : int
  default     : 4
  description : |-
    Number of local peer offsets processed concurrently by NVLink send/recv lanes. Values larger than nLocalRanks-1 are clamped or rejected by support.

- name        : NCCL_CTRAN_HIER_ALLGATHER_READY_QUEUE_DEPTH
  type        : int
  default     : 4
  description : |-
    Number of device ready slots per IBGDA producer group.

- name        : NCCL_CTRAN_HIER_ALLGATHER_THREAD_BLOCK_SIZE
  type        : int
  default     : 512
  description : |-
    Number of threads per CUDA threadblock used by the hierarchical kernel.

- name        : NCCL_CTRAN_HIER_ALLGATHER_IB_SIGNAL_BYTES
  type        : uint64_t
  default     : 0
  description : |-
    Maximum bytes between IBGDA transport signals. Zero uses the transport default.

- name        : NCCL_CTRAN_HIER_ALLGATHER_NVL_SIGNAL_BYTES
  type        : uint64_t
  default     : 0
  description : |-
    Maximum bytes between NVLink transport signals. Zero uses the transport default.

- name        : NCCL_CTRAN_HIER_ALLGATHER_TIMEOUT_MS
  type        : float
  default     : 30000.0
  description : |-
    Timeout in milliseconds for device-side Pipes waits. Zero means no timeout.
```

Reuse existing IBGDA send/recv cvars:

- `NCCL_CTRAN_IBGDA_SENDRECV_ENABLE`
- `NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE`, used only when the per-communicator `ibgdaDataBufferSize` override is unset
- `NCCL_CTRAN_IBGDA_SENDRECV_MAX_GROUPS`
- `NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH`

## Performance Plan

### Pipeline Depth

Pipeline depth is controlled by:

- requested window bytes
- IBGDA send/recv data buffer size
- IBGDA pipeline depth
- IBGDA role block count
- NVLink data buffer size
- NVLink pipeline depth
- active NVLink peer lanes
- NVLink groups per peer
- ready queue depth
- signal byte granularity

The target behavior is:

```text
IBGDA groups receive/forward source S window W
  -> copy into final recvbuff
  -> publish ready slot
NVLink send/recv lanes consume deterministic source S window W tile stream
  -> pairwise local-rank exchange over NVLink
IBGDA groups simultaneously progress later windows/source nodes
```

Any design that completes the full inter-node ring before starting NVLink fanout is not acceptable for the performance target.

Use a simple model for tuning:

```text
T_hier ~= max(T_ibgda_ring, T_nvl_fanout) + T_ready + T_kernel_sched

T_ibgda_ring  ~= (nNodes - 1) * (alpha_ibgda + bytes_per_rank / effective_ib_bw_per_rail)
T_nvl_fanout  ~= (nLocalRanks - 1) * bytes_per_rank / effective_nvl_bw
```

The hierarchy wins only when the overlapped max term plus device scheduling overhead is lower than the best NCCL baseline on the same topology.

### Starting Tuning Points

Window sweep:

- 64 KB
- 128 KB
- 256 KB
- 1 MB
- 2 MB
- 4 MB

Block sweep:

- `numIbBlocks`: 4, 8, 16
- `nvlActivePeerLanes`: 1, 2, 4, `nLocalRanks - 1`
- `nvlGroupsPerPeer`: 1, 2, 4
- total blocks: 16, 24, 32, 48, 64 when resident-capacity guard permits

IBGDA staging:

- start with effective `ibgdaDataBufferSize=32MB` if memory permits, preferably through the per-communicator override for targeted experiments
- start with pipeline depth 2
- increase pipeline depth only if IBGDA stalls on staging reuse

Expected behavior:

- small messages may remain on NCCL or `ctdirect`
- 256 KB to 1 MB needs small windows to expose overlap early
- 16 MB to 1 GB should favor 1 MB to 4 MB windows
- too-small windows increase signal overhead
- too-large windows delay first NVLink fanout

Initial profiles:

```text
small, < 256 KB:
  do not auto-select

mid, 256 KB-16 MB:
  ~32 resident blocks total
  numIbBlocks=8
  nvlActivePeerLanes=4
  nvlGroupsPerPeer=1-2
  IBGDA pipelineDepth=4
  IBGDA dataBufferSize=8 MB
  signal/window target=128 KB-256 KB

large, > 16 MB:
  48-64 resident blocks total
  numIbBlocks=16
  nvlActivePeerLanes=4, sweep 7 on 8-GPU nodes
  nvlGroupsPerPeer=2-4
  IBGDA pipelineDepth=4, sweep 8
  IBGDA dataBufferSize=16 MB or 32 MB
  signal/window target=256 KB-512 KB
```

### Multi-Rail And NIC Use

The first rail multiplier is local rank: 8 GPUs per node produce 8 same-local-rank rings. Within each rail, `P2pIbgdaTransportDevice` can round-robin groups over available NIC/QP resources. Start with one node ring per local rank. Do not add multiple logical ring strides until profiling shows one rail underuses fabric bandwidth.

Add instrumentation to record actual NIC/QP routing when the backend exposes it. If all local-rank rails collapse onto one injection path, keep auto-selection disabled for that topology.

### NVLink Saturation

The NVLink schedule should be deterministic pairwise exchange across local ranks. If it does not saturate NVSwitch/NVLink, increase peer-lane concurrency:

- split NVLink role groups by peer offset
- use symmetric send-to-`+d` / recv-from-`-d` rounds
- preserve role-local contiguous group IDs per transport call
- validate `activeBlocks` exactly for each lane
- sweep active peer lanes `{1, 2, 4, nLocalRanks - 1}`

Using all peer lanes can increase signal pressure. Enable it only when NVLink counters show headroom and no IBGDA or NVLink staging starvation.

### Final-Layout Semantics

The algorithm is zero host-copy, not strict NIC-to-user-buffer zero-copy. IBGDA send/recv writes transport staging, then device code copies into final `recvbuff`. NVLink reads final `recvbuff` and writes final peer slots. There is no ctran temporary registered buffer or host-visible copyback.

Strict NIC writes directly into `recvbuff` would require a separate explicit-buffer IBGDA design and is out of scope for the first integration.

## Error Handling And Resource Lifetime

Host-side code should:

- fail support before launch when required Pipes resources are missing
- validate cvar bounds before filling fixed-size arrays
- enqueue scratch init and kernel launch on the caller stream
- check synchronous launch errors
- release async scratch only after the kernel on the same stream
- avoid stream synchronization in the collective entry point

Device-side code should:

- start the Pipes timeout at kernel entry
- pass timeout to every IBGDA and NVLink `send`, `recv`, and `forward`
- avoid default infinite waits in production paths
- use deterministic call order so peers make matching transport calls
- publish ready slots only after final `recvbuff` stores are visible to local GPU consumers
- use timeout-aware waits for ready-slot sequence and consumed-peer-mask polling
- eventually bridge ctran host abort into Pipes transport waits; checking abort only between transport calls is insufficient
- do not enable auto-selection while transport timeout expiry traps the CUDA context instead of returning a recoverable collective error

The main failure mode is a kernel blocked in a transport wait after a peer exits or traps. This path should remain explicit-only until timeout and abort behavior are validated on failure-injection tests.

## Profiler Instrumentation

For the first integration:

- use `CTRAN_COLL_INFO`
- create a `KernelConfig` for AllGather metadata
- trigger colltrace before and after enqueue
- manually record algo stats because no GPE path records them
- record fallback/support failure reasons in logs

Do not fabricate mapper phases or mapper timestamps. Detailed device-side timing should be a follow-up Pipes profiler integration.

Useful counters:

- selected profile
- chosen window bytes
- number of windows
- `numIbBlocks`
- `nvlActivePeerLanes`
- `nvlGroupsPerPeer`
- ready queue depth
- IBGDA and NVLink pipeline windows
- per-rail IB bytes, signal waits, counter waits, and timeout count
- per-NIC bytes and QP distribution when exposed
- per-local-peer NVLink bytes and wait cycles
- pipeline occupancy: ready slots full, ready slots empty, backpressure waits
- first ready-slot latency
- last ready-slot latency
- NVLink fanout time when device instrumentation is available
- timeout count
- fallback reason

## Testing Strategy

### Unit And Support Tests

Add support tests for:

- explicit `cthierarchical_ring` rejects no Pipes
- explicit `cthierarchical_ring` rejects IBGDA send/recv disabled
- explicit `cthierarchical_ring` rejects zero IBGDA data buffer
- explicit `cthierarchical_ring` accepts a positive per-communicator `ibgdaDataBufferSize` override when the global cvar is unset or zero
- per-communicator `ibgdaDataBufferSize` participates in `ctranPipesConfig::operator==`
- explicit `cthierarchical_ring` rejects `nNodes == 1`
- explicit `cthierarchical_ring` rejects `nLocalRanks == 1`
- explicit `cthierarchical_ring` rejects non-rectangular rank geometry
- explicit `cthierarchical_ring` rejects missing NVLink peer transport
- explicit `cthierarchical_ring` rejects missing IBGDA prev/next transport
- explicit `cthierarchical_ring` rejects invalid block counts and zero pipeline windows
- explicit `cthierarchical_ring` ignores min-size threshold
- `ctran` auto-selects hierarchical only when auto-enable, support, allowlist, and min-size gates pass
- existing algorithms and graph-aware support remain unchanged

### Distributed Correctness

Extend `ctran/tests/CtranDistAllgatherTests.cc`:

- include `NCCL_ALLGATHER_ALGO::cthierarchical_ring`
- test in-place and out-of-place
- test byte-copy datatypes at least `commInt8`, `commFloat32`, and an aligned 16-byte-friendly type
- test messages below `CTRAN_MIN_REGISTRATION_SIZE`
- verify backend usage does not expect `CtranMapper` IB/NVL operations for this path
- verify colltrace/algo stats record `CtranAllGatherHierarchicalRing`
- verify stream ordering with kernels before and after the collective on the same stream
- add failure-injection coverage before auto-selection: kill or abort one rank mid-collective and verify peers do not hang indefinitely or leave the CUDA context unrecoverable

Real multi-node correctness:

- 2x8
- 4x8
- 8x8

### Build Targets

Do not guess target names. Query the owning build files before running new targets. Expected areas:

- `ctran/tests/BUCK`
- `ctran/algos/AllGather/tests/BUCK`
- `ctran/benchmarks/BUCK`
- Pipes collectives build definitions

Run formatting and lint before landing:

```bash
arc f
arc lint
```

## Benchmark Plan

Add `ctran/benchmarks/AllGatherHierarchicalRingBench.cc`, modeled after existing ctran and Pipes AllGather benchmarks.

Benchmark controls:

- algorithm: NCCL default/auto, NCCL Ring, NCCL Tree, best tuned NCCL configuration for the platform, ctran direct, ctran ring, ctran hierarchical ring
- message size: 8 KB to 1 GB
- warmup: 5 iterations for full runs
- measurement: 100 iterations for full runs
- topology: 2x8, 4x8, 8x8

Immediate GB200 validation target:

- host set: `rtptest2333.nha6.facebook.com`, `rtptest2335.nha6.facebook.com`, `rtptest2337.nha6.facebook.com`, `rtptest2339.nha6.facebook.com`
- first run shape: 4 nodes, `ppn=1`, one GPU per node, four ranks total
- purpose: isolate IBGDA ring performance before enabling the NVLink local fanout path
- primary baseline: NCCL default/auto with production-valid settings
- diagnostic baselines: `NCCL_ALGO=Ring`, `NCCL_ALGO=Tree`, and tuned NCCL settings on the same rank placement
- first ctran target: no-GPE `cthierarchical_ring` in 4x1 mode, where the NVLink fanout degenerates to `nvl_size=1` and the run isolates the IBGDA ring phase
- fixed-host validation target: `fbcode//comms/ctran/benchmarks:allgather_bench_4x1`, run with `NCCL_ALLGATHER_ALGO=cthierarchical_ring`, `NCCL_CTRAN_USE_PIPES=1`, `NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1`, and a positive effective IBGDA data-buffer size

- CVAR sweeps:
  - `NCCL_CTRAN_HIER_ALLGATHER_WINDOW_BYTES`
  - `NCCL_CTRAN_HIER_ALLGATHER_NUM_IB_BLOCKS`
  - `NCCL_CTRAN_HIER_ALLGATHER_NVL_ACTIVE_PEER_LANES`
  - `NCCL_CTRAN_HIER_ALLGATHER_NVL_GROUPS_PER_PEER`
  - `NCCL_CTRAN_HIER_ALLGATHER_READY_QUEUE_DEPTH`
  - `NCCL_CTRAN_HIER_ALLGATHER_IB_SIGNAL_BYTES`
  - `NCCL_CTRAN_HIER_ALLGATHER_NVL_SIGNAL_BYTES`
  - `NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE`
  - per-communicator `ibgdaDataBufferSize`
  - `NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH`

Record environment and fairness controls:

- NCCL protocol, channel count, net plugin, PXN setting, CollNet setting, NVLS setting, and any forced algorithm/protocol variables
- whether NCCL P2P, SHM, PXN, CollNet, and NVLS were left at production defaults or explicitly changed
- ctran hierarchical cvars used for the run
- effective per-communicator Pipes overrides, including `ibgdaDataBufferSize`
- CPU affinity, process-to-GPU mapping, and process-per-node count
- NIC affinity or IBGDA routing when exposed
- CUDA driver/runtime, GPU SKU, NVLink/NVSwitch generation, IB/RoCE link speed, and topology summary
- identical warmup/measurement counts and validation mode across NCCL and ctran runs

Report:

- latency in microseconds
- algorithmic bandwidth
- bus bandwidth or effective fabric bandwidth when available
- speedup over best NCCL baseline
- selected window bytes
- IB block count, active NVLink peer lanes, and NVLink groups per peer
- ready queue depth
- IBGDA data buffer size and pipeline depth
- node and local-rank geometry
- measured effective IB bandwidth per rail and NVLink broadcast bandwidth when available

Pass criteria:

- no single-node auto-selection regression
- no correctness failures for in-place/out-of-place paths
- `cthierarchical_ring` beats best NCCL baseline for 256 KB through 1 GB on 2x8, 4x8, and 8x8
- performance does not rely on disabling NCCL features in a way that makes the comparison invalid

Go/no-go for auto-selection:

- no hangs or timeout flakes across stress runs
- no accidental 1x8 selection
- median speedup is at least 1.10x over best NCCL default/tuned on 4x8 and 8x8 for most 256 KB-1 GB sizes
- 2x8 is at least parity below 1 MB and wins for large messages
- p95 latency is not worse than NCCL by more than 5% near thresholds
- staging memory, QP count, and device counter usage are production-safe
- no-go if wins require disabling NCCL features normally used in production
- no-go if NVLink fanout dominates, NIC use is imbalanced, or device abort cannot reliably unblock waits

## Rollout Plan

Initial rollout should be explicit-only:

- `NCCL_ALLGATHER_ALGO=cthierarchical_ring` enables the algorithm when support gates pass
- `NCCL_ALLGATHER_ALGO=ctran` does not auto-select it unless `NCCL_CTRAN_HIER_ALLGATHER_AUTO_ENABLE=true`
- keep a kill switch through the enum selection and the auto-enable cvar
- gate production auto-selection by topology allowlist after benchmark signoff
- emit counters for fallback reason, abort reason, first ready latency, last ready latency, timeout count, and support-gate failures

Auto-selection can be enabled only after correctness, timeout/abort, and performance data show no single-node regression and a multi-node win against the best NCCL baseline on the target fleet.

## Diff Stack

Recommended review stack:

1. **Registration and config**
   - add enum value and cvars
   - update ctran algo string maps
   - add `allGatherAlgoName()` case
   - add support-check helpers
   - wire IBGDA send/recv configuration through `CtranPipes.cc`
   - add per-communicator `ibgdaDataBufferSize` override before `MultiPeerTransport` construction
   - add auto-enable cvar defaulting false
   - no behavior change except recognizing the new value

2. **Concurrent-kernel prototype**
   - add a minimal 2x2/4x2 prototype kernel or test-only launcher
   - validate IBGDA producer and NVLink consumer lanes can overlap with ready synchronization
   - validate resident-capacity guard or monotonic readiness strategy
   - compare measured runtime against sequential and overlapped models

3. **Pipes hierarchical kernel**
   - add launch params and device args
   - add hierarchical AllGather kernel
   - implement IBGDA same-local-rank rail ring
   - implement device ready slots
   - implement deterministic NVLink fanout
   - wire abort-aware Pipes timeout into all transport calls

4. **ctran no-GPE entry point**
   - add `AllGatherHierarchicalRing.cc`
   - build rank maps and launch params
   - allocate/init/free device scratch on stream
   - wire explicit dispatch
   - add colltrace and algo stats
   - add auto-selection behind `NCCL_CTRAN_HIER_ALLGATHER_AUTO_ENABLE`

5. **Correctness and support tests**
   - add support tests
   - extend distributed AllGather tests
   - add stream-ordering and small-message coverage

6. **Benchmarks and tuning**
   - add benchmark target
   - document recommended sweeps
   - collect 2x8, 4x8, and 8x8 results against NCCL default/tuned baselines

## Implementation Checklist

- Add `cthierarchical_ring` to cvars and generated enum outputs.
- Add `ctranAllGatherHierarchicalRing()` declaration.
- Add `allGatherAlgoName(cthierarchical_ring)`.
- Add support checker with Pipes, topology, transport, rank-map, block-count, and pipeline-window gates.
- Split mapper support checks from Pipes support checks.
- Add per-communicator `ibgdaDataBufferSize` override to `ctranPipesConfig`, equality checks, and `CtranPipes.cc` transport construction.
- Add IBGDA send/recv enablement in `CtranPipes.cc` if it is not already present in the base revision.
- Add explicit dispatch for `cthierarchical_ring`.
- Add auto-selection branch under `ctran`, gated by `NCCL_CTRAN_HIER_ALLGATHER_AUTO_ENABLE` and min size.
- Build the P0 concurrent 2x2/4x2 prototype and verify overlap before the full kernel.
- Add Pipes launch params and device args.
- Add device rank map or arithmetic-layout verification.
- Add device ready slots and scratch allocation.
- Implement either a runtime resident-block guard or a non-reused monotonic readiness scheme before using bounded ready slots.
- Implement one-kernel IBGDA rail plus NVLink fanout.
- Implement active-peer NVLink lane scheduling with deterministic peer rounds.
- Use `P2pIbgdaTransportDevice` tiled `send`, `forward<CopyOp>`, and `recv<CopyOp>`.
- Use `P2pNvlTransportDevice` tiled `send` and `recv<CopyOp>`.
- Pass timeout objects to every Pipes transport call.
- Validate nonzero IBGDA and NVLink `pipeline_window()` before launch.
- Avoid `CtranMapper`, `OpElem`, `GpeKernelSync`, and `prepareAllGatherArgs()` in this path.
- Keep auto-selection disabled until Pipes waits can observe ctran abort or an equivalent production-safe soft-timeout path exists.
- Run `arc f`.
- Run `arc lint`.
- Run correctness tests.
- Run benchmark sweep and compare against best NCCL default/tuned baseline.
