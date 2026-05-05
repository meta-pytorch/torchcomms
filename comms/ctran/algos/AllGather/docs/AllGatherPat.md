# CTPAT: AllGather PAT Algorithm (NVL x IB)

This document describes the CTPAT AllGather algorithm, a rail-parallel butterfly (recursive-doubling) variant for multi-GPU-per-node topologies (H100 NVL x IB). CTPAT is implemented as a new `NCCL_ALLGATHER_P_ALGO` variant inside `AllGatherP/`, reusing the existing persistent infrastructure: `PersistArgs`, `allGatherCtrl` handle exchange, `copyToSelf`, `nvlCeBcast`, and the `PipeStart`/`PipeSync`/`PipeEnd` kernel pattern.

## Motivation

Existing persistent AllGatherP algorithms leave a performance gap at scale:

| Algorithm | Pattern | IB Steps | Latency Scaling |
|-----------|---------|----------|-----------------|
| `ctdirect` | All-to-all direct puts | nNodes-1 | O(N) |
| `ctpipeline` | Ring inter-node + NVL bcast | nNodes-1 | O(N) |
| `ctrdpipeline` | Butterfly inter-node + NVL bcast | log₂(nNodes) | O(log N) |

`ctrdpipeline` already has the right communication pattern. CTPAT is essentially `ctrdpipeline` with explicit naming as a PAT variant and a clean path for future extensions (non-power-of-two node counts, step aggregation). For v1, CTPAT is a narrow, power-of-two-only extension of `ctrdpipeline` — same execution model, same reusable machinery.

### Why a New Variant Instead of Renaming ctrdpipeline

The `ctrdpipeline` name describes the implementation technique (recursive doubling with pipelining). The `ctpat` name describes the algorithm family (Parallel Asynchronous Tiled), which has a broader scope:
- v1: power-of-two butterfly (same as ctrdpipeline)
- Future: non-power-of-two handling, step aggregation, auto-tuning

Introducing `ctpat` as a separate enum value avoids overloading `ctrdpipeline` with new semantics and provides a clean opt-in for users who want the PAT-family behavior.

## Architecture Decision: AllGatherP, Not AllGather

```
┌─────────────────────────────────────────────────────────────────┐
│                    User / TorchComms                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NCCLx collectives.cc                           │
│            ctranAllGatherSupport() / ctranAllGather()            │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────────────┐
│   AllGather/         │          │   AllGatherP/                   │
│   (non-persistent)   │          │   (persistent: init/exec/      │
│                      │          │    destroy lifecycle)           │
│  • ctdirect          │          │                                │
│  • ctring            │          │  • ctdirect                    │
│  • ctrd              │          │  • ctpipeline (ring)           │
│  • ctbrucks          │          │  • ctrdpipeline (butterfly)    │
│  • ctgraph_*         │          │  • ctpat (butterfly — NEW)     │
└──────────────────────┘          │                                │
                                  │  Reuses: PersistArgs,          │
                                  │  allGatherCtrl, PipeSync,      │
                                  │  nvlCeBcast, copyToSelf        │
                                  └────────────────────────────────┘
```

CTPAT lives in `AllGatherP/`, NOT `AllGather/`. The rationale:

1. The strongest reusable machinery is in `AllGatherP/`: persistent remote buffer exchange (`PersistArgs`, `allGatherCtrl`), `pipeSync`, and the exact NVL-after-IB execution model in `RecursiveDoublingImpl.cc`.
2. Re-implementing as an eager `AllGather/AllGatherPat.cc` would duplicate the hard part (handle exchange, PipeSync coordination, CE broadcast orchestration) and lose the fast path for replay/persistence.
3. CUDA graph support will come through the existing `AllGatherCudagraphAware.cc` path once the `ctgraph_pipeline` execute path is updated to respect `NCCL_ALLGATHER_P_ALGO`. This is deferred from v1. While the execute path change is small, it is a behavior change in graph capture that should be validated separately.

## Algorithm

### Communication Pattern

Rail-parallel butterfly (recursive doubling) with NVL CE broadcast:

```
H100 Node 0 (8 GPUs)                    H100 Node 1 (8 GPUs)
┌─────────────────────────┐              ┌─────────────────────────┐
│ G0  G1  G2  ...  G7     │              │ G0  G1  G2  ...  G7     │
│  │   │   │        │     │              │  │   │   │        │     │
│  └───┴───┴──┬─────┘     │              │  └───┴───┴──┬─────┘     │
│        NVSwitch          │              │        NVSwitch          │
│      (CE broadcast)      │              │      (CE broadcast)      │
└────────────┬─────────────┘              └────────────┬─────────────┘
             │                                         │
             │  IB: rail i ←→ rail i (butterfly)       │
             └─────────────────────────────────────────┘
```

Each GPU communicates only with its rail peer (same `localRank`) on remote nodes, using `log₂(nNodes)` butterfly steps. After each step, newly received chunks are broadcast to all local NVL peers via CE.

### Prerequisites

- `nNodes > 1` (multi-node)
- `nNodes` is a power of 2 (v1 constraint; future versions may relax this)
- `nRanks % nLocalRanks == 0`
- All local peers NVL-reachable (except when `nLocalRanks == 1`)

### Step Computation

Reuses the existing functions from `RecursiveDoublingImpl.cc`:

```cpp
// Node-level distance at step i (i=0 is largest distance)
int distNodesAtStep(int nNodes, int step) { return nNodes >> (step + 1); }

// Rail-peer rank at step i
int peerAtStep(int nodeId, int localRank, int nLocalRanks, int nNodes, int step) {
    int dist = distNodesAtStep(nNodes, step);
    int pos = (nodeId / dist) % 2;
    int peerNode = pos == 0 ? nodeId + dist : nodeId - dist;
    return peerNode * nLocalRanks + localRank;
}

// Chunk offset for j-th put at step i (anchorNode = myNode for send, peerNode for recv)
size_t rankChunkOffset(int anchorNode, int localRank, int nLocalRanks,
                       int nNodes, int step, int j) {
    int stride = nNodes >> step;
    int nodePos = j * stride + (anchorNode % stride);
    return static_cast<size_t>(nodePos) * nLocalRanks + localRank;
}
```

### Butterfly Communication Pattern (4 Nodes, localRank=0)

```
         Node 0       Node 1       Node 2       Node 3
         (rank 0)     (rank 8)     (rank 16)    (rank 24)
            │            │            │            │
 Initial:  [A]          [B]          [C]          [D]
            │            │            │            │
            │            │            │            │
 Step 0:    ├────────────────────────►│            │     dist = 2
 (dist=2)   │◄───────────────────────┤            │     Node 0 ↔ Node 2
            │            ├────────────────────────►│     Node 1 ↔ Node 3
            │            │◄───────────────────────┤
            │            │            │            │
 After 0:  [A,C]        [B,D]        [C,A]        [D,B]
            │            │            │            │
            │    NVL     │    NVL     │    NVL     │    NVL
            │   bcast    │   bcast    │   bcast    │   bcast
            │            │            │            │
 Step 1:    ├───────────►│            │            │     dist = 1
 (dist=1)   │◄──────────┤            │            │     Node 0 ↔ Node 1
            │            │            ├───────────►│     Node 2 ↔ Node 3
            │            │            │◄──────────┤
            │            │            │            │
 After 1:  [A,B,C,D]    [B,A,D,C]    [C,D,A,B]    [D,C,B,A]
            │            │            │            │
            │    NVL     │    NVL     │    NVL     │    NVL
            │   bcast    │   bcast    │   bcast    │   bcast
            │            │            │            │
 Final:    All GPUs on each node hold [A,B,C,D]
```

### Data Doubling Pattern

```
Step 0: Send 1 chunk, receive 1 chunk     (2 chunks held)
Step 1: Send 2 chunks, receive 2 chunks   (4 chunks held)
 ...
Step k: Send 2^k chunks, receive 2^k chunks

Data volume per step:
  ┌─────────────────────────────────────────────────┐
  │ Step   │ Chunks Sent │ Chunks Held │ IB Bytes   │
  ├────────┼─────────────┼─────────────┼────────────┤
  │   0    │     1       │     2       │  sendSize  │
  │   1    │     2       │     4       │ 2×sendSize │
  │   2    │     4       │     8       │ 4×sendSize │
  │  ...   │    ...      │    ...      │    ...     │
  │  k     │    2^k      │   2^(k+1)  │ 2^k×send   │
  └────────┴─────────────┴─────────────┴────────────┘
  Total IB per GPU = sendSize × (nNodes - 1)
```

After `log₂(nNodes)` steps, each GPU holds all `nNodes` chunks from its rail column. The NVL broadcasts after each step distribute received data to all local GPUs.

### Example: 4 Nodes x 8 GPUs (32 ranks)

GPU with localRank=0 on node 0 (rank 0):

```
Step 0 (dist=2): peer = node 2, rank 16
  Send: chunk[0]      (anchorNode=0)
  Recv: chunk[16]     (anchorNode=2)
  NVL bcast: chunk[16] to GPUs 1-7 on node 0

Step 1 (dist=1): peer = node 1, rank 8
  Send: chunk[0], chunk[16]   (anchorNode=0)
  Recv: chunk[8], chunk[24]   (anchorNode=1)
  NVL bcast: chunk[8], chunk[24] to GPUs 1-7 on node 0
```

### Rail-Parallel Execution (All 8 GPUs)

```
Node 0                                    Node 2
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ G0 ─── IB (rail 0) ───────────────────── G0                             │
│ G1 ─── IB (rail 1) ───────────────────── G1                             │
│ G2 ─── IB (rail 2) ───────────────────── G2                             │
│ G3 ─── IB (rail 3) ───────────────────── G3                             │
│ G4 ─── IB (rail 4) ───────────────────── G4                             │
│ G5 ─── IB (rail 5) ───────────────────── G5                             │
│ G6 ─── IB (rail 6) ───────────────────── G6                             │
│ G7 ─── IB (rail 7) ───────────────────── G7                             │
│                                │        │                                │
│  ◄── NVSwitch (CE bcast) ──►  │        │  ◄── NVSwitch (CE bcast) ──►  │
└────────────────────────────────┘        └────────────────────────────────┘

Each GPU uses its own NIC. 8 independent IB transfers run in parallel.
After IB completes, each GPU CE-broadcasts received data to 7 local peers.
```

## Execution Model

CTPAT reuses the proven GPE + PipeSync cooperative model from `ctrdpipeline`. The implementation is a new `execPat()` method on `AlgoImpl`, alongside the existing `execDirect`, `execPipeline`, and `execRecursiveDoubling`.

```
Main Thread              GPE Thread                  CUDA Stream
───────────              ──────────                  ───────────
allGatherPInit()
  → allGatherCtrl()
  → PersistArgs populated

allGatherPExec()
  → waitInit()
  → copyToSelf()                                     copyToSelf (CE)
  → submit(PipeStart)   ┌─dequeue                   PipeStart (→ exits)
                        │ initNotify() × nSteps
  → nvlCeBcast(own)     │ RTR handshake              nvlCeBcast(own chunk)
                        │
                        │ Step 0:
                        │   iput(1 chunk)
                        │   waitNotify()
                        │   pipeSync->post(0) ────→  PipeSync(0)
                        │                             nvlCeBcast(step 0)
                        │ Step 1:
                        │   iput(2 chunks)
                        │   waitNotify()
                        │   pipeSync->post(1) ────→  PipeSync(1)
                        │                             nvlCeBcast(step 1)
                        │ ...
                        └─GPE done                    PipeEnd (reset+barrier)
```

### PipeSync Overlap Timeline

The key performance advantage: IB step `i+1` runs concurrently with NVL broadcast of step `i`.

```
Time ──────────────────────────────────────────────────────────────────────►

GPE Thread:
  ┌──────────┐ ┌──────────────┐ ┌──────────────────┐ ┌──────────────────────┐
  │ RTR sync │ │ IB step 0    │ │ IB step 1        │ │ IB step 2            │
  │          │ │ iput(1 chunk)│ │ iput(2 chunks)   │ │ iput(4 chunks)       │
  │          │ │ waitNotify() │ │ waitNotify()     │ │ waitNotify()         │
  └──────────┘ └──────┬───────┘ └──────┬───────────┘ └──────┬───────────────┘
                      │ post(0)        │ post(1)            │ post(2)
                      ▼                ▼                    ▼
CUDA Stream:
  ┌──────────┐ ┌─────────────┐ ┌──────────────────┐ ┌──────────────────────┐ ┌──────┐
  │copyToSelf│ │nvlCeBcast   │ │PipeSync(0)       │ │PipeSync(1)           │ │Pipe  │
  │          │ │(own chunk)  │ │  wait...          │ │  wait...             │ │Sync  │
  │          │ │             │ │  ──► nvlCeBcast   │ │  ──► nvlCeBcast     │ │(2)   │
  │          │ │             │ │  (step 0: 1 chunk)│ │  (step 1: 2 chunks) │ │ ...  │
  └──────────┘ └─────────────┘ └──────────────────┘ └──────────────────────┘ └──────┘

                               ◄── overlap ──► ◄────── overlap ──────►

  Critical path = T_ib_0 + max(T_ib_1, T_nvl_0) + max(T_ib_2, T_nvl_1) + T_nvl_2
```

### Persistent Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ allGatherPInit(recvbuff, maxCount, hints, datatype, comm, stream)      │
│                                                                         │
│  1. Verify recvbuff is pre-registered (not dynamic)                    │
│  2. ►► PAT eligibility check (NEW for ctpat) ◄◄                       │
│     • nNodes > 1 && isPowerOf2(nNodes)                                 │
│     • nRanks % nLocalRanks == 0                                        │
│     • All local peers NVL-reachable                                    │
│  3. Create AlgoImpl, populate PersistArgs                              │
│  4. initResources() → allocate GpeKernelSync                          │
│  5. Submit exchangeMemHdl to GPE thread                                │
│     └─► GPE: allGatherCtrl() → populate remoteRecvBuffs/AccessKeys    │
│     └─► GPE: barrier() → all ranks ready                              │
│     └─► pArgs.initialized = true                                      │
│  6. Return CtranPersistentRequest                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (can be called many times)
┌─────────────────────────────────────────────────────────────────────────┐
│ allGatherPExec(sendbuff, count, datatype, request)                     │
│                                                                         │
│  switch (NCCL_ALLGATHER_P_ALGO):                                       │
│    ctdirect     → execDirect()                                         │
│    ctpipeline   → execPipeline()                                       │
│    ctrdpipeline → execRecursiveDoubling()                              │
│    ctpat        → execPat()           ◄◄ NEW                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (once, at end of lifetime)
┌─────────────────────────────────────────────────────────────────────────┐
│ allGatherPDestroy(request)                                             │
│                                                                         │
│  1. Free GpeKernelSync (pipeSync)                                      │
│  2. Delete AlgoImpl                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### When nLocalRanks == 1

NVL broadcasts are skipped. Uses blocking `ncclKernelAllGatherPPipe` kernel. The algorithm reduces to a pure IB recursive-doubling, equivalent to `ctrd` but through the persistent AllGatherP path.

```
nLocalRanks > 1 (full pipeline)          nLocalRanks == 1 (blocking)
─────────────────────────────            ──────────────────────────
                                         
Stream: copyToSelf                       Stream: copyToSelf
Stream: PipeStart ──► GPE starts         Stream: PipeBlocking ──► GPE starts
Stream: nvlCeBcast(own)                            │
Stream: PipeSync(0) ── waits...                    │ GPE runs ALL steps
Stream: nvlCeBcast(step 0)                         │ (iput + waitNotify) × nSteps
Stream: PipeSync(1) ── waits...                    │
Stream: nvlCeBcast(step 1)                         │ No NVL broadcasts
Stream: ...                                        │ No PipeSync
Stream: PipeEnd (reset + barrier)                  ▼
                                         Stream: PipeBlocking returns
                                         Stream: continues
```

Both paths unconditionally enqueue `copyToSelf` before the GPE kernel, matching `RecursiveDoublingImpl.cc:311`.

### GPE Callback Internal Flow (gpeFn)

```
gpeFn(opGroup) entry
        │
        ▼
  searchRegHandle(sendBuff)
  initNotify() for each of log₂(nNodes) peers
        │
        ▼
  RTR handshake: isendCtrl + irecvCtrl × nSteps
  waitRequest() for all recvCtrls
        │
        ▼
  ┌─────────────────────────────────────────┐
  │          Butterfly Loop                  │
  │  for i = 0 .. nSteps-1:                 │
  │    │                                     │
  │    │  peer = peerAtStep(nodeId, ..., i)  │
  │    │                                     │
  │    │  for j = 0 .. 2^i - 1:             │
  │    │    srcPtr = (i==0) ? sendBuff       │
  │    │           : recvbuff[chunkIdx]      │
  │    │    dstPtr = remoteRecvBuff[chunkIdx]│
  │    │    iput(src → dst, notify on last)  │
  │    │                                     │
  │    │  waitRequest(lastPut)               │
  │    │  waitNotify(peer)                   │
  │    │                                     │
  │    │  if nLocalRanks > 1:               │
  │    │    pipeSync->post(i) ──────► stream │
  │    │                                     │
  └─────────────────────────────────────────┘
        │
        ▼
  waitRequest(sendCtrls)
  return commSuccess
```

### Relationship to ctrdpipeline

For v1, `execPat()` is structurally identical to `execRecursiveDoubling()`. The step computation, GPE callback, and stream-side orchestration are the same. The value of introducing `ctpat` is:

1. A clean enum for users to opt in to the PAT family
2. A separate method that can diverge in future versions (non-power-of-two, step aggregation)
3. Avoids overloading `ctrdpipeline` semantics if the planner evolves

## Code Structure

### New Files

| File | Description |
|------|-------------|
| `AllGatherP/PatImpl.cc` | `execPat()` implementation — GPE callback + stream orchestration |

### Modified Files

| File | Changes |
|------|---------|
| `AllGatherP/AlgoImpl.h` | Add `execPat()` method declaration, add `ctpat` to `algoName()` |
| `AllGatherP/AllGatherP.cc` | Add `ctpat` case to `allGatherPExec()` dispatch switch |
| `utils/cvars/nccl_cvars.yaml` | Add `ctpat` to `NCCL_ALLGATHER_P_ALGO` choices |

**Not in v1**: Extracting step computation to a shared `RecursiveDoublingUtils.h` is deferred. The helper functions (`peerAtStep`, `rankChunkOffset`, etc.) are small — `PatImpl.cc` duplicates them in its own anonymous namespace for v1. Factoring to a shared header is a cleanup for after performance data validates the approach.

### Build System

BUCK: `ctran_lib` uses `glob(["**/*.cc"])` — new `.cc` files auto-included. No BUCK changes needed.

CMake: `file(GLOB_RECURSE)` auto-includes. No CMake changes needed.

After modifying `nccl_cvars.yaml`, regenerate:
```bash
NCCL_CVARS_OUTPUT_DIR=comms/utils/cvars buck2 run comms/utils/cvars:extractcvars
```

## Algorithm Selection

### Selection Flow

```
User sets NCCL_ALLGATHER_P_ALGO=ctpat
                    │
                    ▼
         allGatherPInit()
                    │
                    ├─── nNodes == 1? ──────────────► ERROR: ctpat requires multi-node
                    │
                    ├─── !isPowerOf2(nNodes)? ──────► ERROR: ctpat requires power-of-2 nodes
                    │
                    ├─── nRanks % nLocalRanks != 0? ► ERROR: uneven rank distribution
                    │
                    ├─── local peer not NVL? ────────► ERROR: NVL required for local peers
                    │     (when nLocalRanks > 1)
                    │
                    ▼
              Init succeeds
              PersistArgs created
              Handles exchanged
                    │
                    ▼
         allGatherPExec()
                    │
         switch(NCCL_ALLGATHER_P_ALGO)
                    │
      ┌─────────┬───┴────┬──────────────┐
      ▼         ▼        ▼              ▼
  ctdirect  ctpipeline  ctrdpipeline   ctpat
  (direct)  (ring)      (butterfly)    (butterfly)
                                        │
                                        ▼
                                    execPat()
```

### Persistent Path

`NCCL_ALLGATHER_P_ALGO=ctpat` selects the PAT variant in `allGatherPExec()`:

```cpp
case NCCL_ALLGATHER_P_ALGO::ctpat:
    return algo->execPat(sendbuff, count, datatype);
```

PAT eligibility is validated in `allGatherPInit()`, not at exec time. If prerequisites are not met (non-power-of-two nNodes, missing NVL connectivity, nRanks not divisible by nLocalRanks), init returns an error with a descriptive message. This matches the existing init path where registration and persistent setup already happen (`AllGatherP.cc:155`).

### CUDA Graph Path

Graph support for ctpat is **deferred from v1**. The `ctgraph_pipeline` path in `AllGatherCudagraphAware.cc` currently hardcodes `NCCL_ALLGATHER_P_ALGO = ctpipeline` at line 166 inside its execute path (`allGatherWinExec`), so updating `selectCtgraphAlgo()` alone would not make graph-captured ctpat work. Enabling it requires changing the `ctgraph_pipeline` execute path to respect the `NCCL_ALLGATHER_P_ALGO` CVAR, which is a broader change best done after v1 is validated.

### CVARs

| CVAR | Default | Description |
|------|---------|-------------|
| `NCCL_ALLGATHER_P_ALGO` | ctpipeline | Add `ctpat` to existing choices |

v1 uses explicit selection only (`NCCL_ALLGATHER_P_ALGO=ctpat`). There is no size-based auto-selection or fallback path — the user opts in to ctpat for their entire persistent AllGatherP lifetime. Size thresholds and auto-tuning CVARs are deferred to Phase 2 after benchmarking establishes where ctpat wins.

## Performance Model

### Latency

With PipeSync pipelining, IB step `i+1` overlaps with NVL broadcast of step `i`:

```
T_pat = T_ib_step_0
      + Σ_{i=1}^{nSteps-1} max(T_ib_step_i, T_nvl_step_{i-1})
      + T_nvl_step_{nSteps-1}

where:
  T_ib_step_i  = T_ib_rtt + 2^i × sendSize / BW_ib
  T_nvl_step_i = 2^i × (T_nvl_barrier + T_ce_copy(sendSize × (nLocalRanks-1)))
```

The butterfly sends increasing amounts per step — the final step transfers `nNodes/2 × sendSize`, which is half of all IB data. This asymmetry means the last step dominates IB time at large message sizes.

### Bandwidth

Total IB data per GPU = `sendSize × (nNodes - 1)`, same as ring and direct. Rail-parallel execution distributes traffic across all 8 NICs per node.

### Expected Advantage over ctpipeline (Ring)

| nNodes | Ring Steps | PAT Steps | Step Reduction |
|--------|-----------|-----------|---------------|
| 4 | 3 | 2 | 1.5x |
| 8 | 7 | 3 | 2.3x |
| 16 | 15 | 4 | 3.75x |
| 64 | 63 | 6 | 10.5x |

```
IB Steps vs Node Count:

Steps
  │
63│ ·                                                    Ring O(N)
  │
  │
32│ ·
  │
  │
16│ ·
  │
 8│ ·
 7│          ·
 6│                                                 ×    PAT O(log N)
 4│     ·              ×
 3│ ×        ·    ×
 2│ ×   ×
 1│
  └──────────────────────────────────────────────────
   2    4    8    16   32                   64    nNodes

  · = Ring (ctpipeline)    × = PAT (ctpat)
```

PAT wins increasingly at large node counts where `O(log N)` vs `O(N)` matters.

## Implementation Plan

```
Phase 1 (v1)              Phase 2                Phase 3              Phase 4
─────────────             ──────────             ─────────            ──────────
┌──────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────────┐
│ ctpat enum   │   │ Benchmark sweep  │   │ Non-power-of-2 │   │ Step         │
│ + PatImpl.cc │──►│ across H100      │──►│ node count     │──►│ aggregation  │
│ + tests      │   │ configs          │   │ support        │   │ + auto-tune  │
│              │   │                  │   │                │   │              │
│ Power-of-2   │   │ Crossover point  │   │ Bruck's or     │   │ PAT-style    │
│ only         │   │ analysis         │   │ hybrid planner │   │ batching     │
└──────────────┘   └──────────────────┘   └────────────────┘   └──────────────┘
    │                                          │
    │ Success bar:                             │ This is where ctpat
    │ Match ctrdpipeline                       │ diverges from ctrdpipeline
    │ Beat ctdirect (medium sizes)             │ in implementation
    │ 4x8, 8x8, 16x8 H100                     │
```

### Phase 1: PAT as AllGatherP Variant (v1)

Implement `ctpat` as a new `NCCL_ALLGATHER_P_ALGO` variant, structurally identical to `ctrdpipeline` with its own `execPat()` method.

**Steps**:
1. Add `ctpat` to `NCCL_ALLGATHER_P_ALGO` choices in `nccl_cvars.yaml` and regenerate
2. Create `AllGatherP/PatImpl.cc` with `execPat()` — same GPE callback and stream orchestration as `execRecursiveDoubling`, with step helper functions duplicated in a local anonymous namespace
3. Add `execPat()` to `AlgoImpl.h`, add `ctpat` case to `allGatherPExec()` and `algoName()`
4. Add PAT eligibility validation in `allGatherPInit()` (power-of-two nNodes, NVL connectivity for local peers, nRanks divisible by nLocalRanks). Failing at init is cleaner than letting request creation succeed and then erroring on replay — init is where registration and persistent setup already happen

**Success bar**: Match `ctrdpipeline` performance (since v1 is structurally identical), AND beat or match `ctdirect` on medium message sizes (32KB-8MB) for 4x8, 8x8, and 16x8 H100. A ctpat that is slower than ctrdpipeline is a regression, not a feature.

**Deliverables**:
- `AllGatherP/PatImpl.cc`
- Updated `AlgoImpl.h`, `AllGatherP.cc`, `nccl_cvars.yaml`
- Distributed tests with `nolocal` and `vnode` configurations
- Benchmark: ctpat vs ctdirect vs ctpipeline vs ctrdpipeline

### Phase 2: Benchmarking and Threshold Tuning

Comprehensive benchmark sweep to establish crossover points and determine whether ctpat should become the default `NCCL_ALLGATHER_P_ALGO`.

**Benchmark matrix**:

| Dimension | Values |
|-----------|--------|
| Message size | 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB, 1GB |
| Node count | 2x8, 4x8, 8x8, 16x8, 64x8 |
| Algorithms | ctpat, ctdirect, ctpipeline, ctrdpipeline |
| Memory type | cudaMalloc, ncclMemAlloc |

**Deliverables**:
- Benchmark results with analysis
- Recommendation on default algorithm selection
- Size threshold tuning (if ctpat is made default for certain ranges)

### Phase 3: Non-Power-of-Two Support (Future)

Extend the planner to handle arbitrary node counts. Options:
- Bruck's algorithm adaptation for non-power-of-two
- Padding to next power-of-two with dummy nodes
- Hybrid: power-of-two butterfly for the largest power-of-two subset, direct for remainder

This is where CTPAT diverges from `ctrdpipeline` in implementation, not just naming.

### Phase 4: Step Aggregation and Auto-Tuning (Future)

Add PAT-style step aggregation from baseline NCCL (batching multiple butterfly steps to reduce synchronization points). Add auto-tuned pipeline depth and block count tiers based on benchmark data from Phase 2.

## What is Cut from v1

- ~~New eager `AllGatherPat.cc` under `AllGather/`~~ → Use AllGatherP path
- ~~Separate `ctgraph_pat` enum~~ → Deferred; requires `ctgraph_pipeline` execute path change
- ~~Shared `RecursiveDoublingUtils.h`~~ → Duplicate helpers locally; factor later
- ~~Chunked staging buffers~~ → Defer to Phase 4 if benchmarks show a gap
- ~~Large CVAR/auto-tune surface~~ → Start with fixed settings, add CVARs as needed
- ~~Non-power-of-two handling~~ → Phase 3

## Testing

### Distributed Tests

Extend the existing distributed AllGatherP test suite at `ctran/tests/CtranDistAllgatherPTests.cc` (wired from `ctran/tests/BUCK:162`) with ctpat test cases. If the test file grows too large, add a sibling `ctran/tests/CtranDistAllgatherPPatTests.cc` in the same directory and BUCK target.

Test configurations reuse the existing patterns:

```python
# In ctran/tests/BUCK, extend ctran_dist_allgatherp or add:
"1x8_nolocal_pat": {  # 8 "nodes" with 1 GPU each (pure IB butterfly)
    "compiler_flags": ["-DNCCL_COMM_STATE_DEBUG_TOPO_NOLOCAL"],
    "nnodes": 1, "ppn": 8,
},
"1x8_vnode_pat": {  # Simulated multi-node with NVL
    "compiler_flags": ["-DNCCL_COMM_STATE_DEBUG_TOPO_VNODE"],
    "nnodes": 1, "ppn": 8,
},
```

### Correctness Test Cases

- nLocalRanks==1 (nolocal): pure IB butterfly, no NVL broadcasts
- 2-node, 4-node, 8-node power-of-two configurations
- Non-power-of-two nNodes: init returns error
- In-place and out-of-place operations
- Various datatypes and message sizes

### Performance Tests

Use existing `AllgatherPBench.cc` with `NCCL_ALLGATHER_P_ALGO=ctpat`:

```bash
NCCL_ALLGATHER_P_ALGO=ctpat buck2 run @fbcode//mode/opt \
    -c hpc_comms.use_ncclx=stable \
    //comms/ctran/benchmarks:AllgatherPBench -- \
    --minBytes 16K --maxBytes 1G --stepFactor 2
```

### Stress Tests

- Back-to-back allGatherPExec calls (persistent replay)
- Concurrent communicators
- Abort/timeout paths
