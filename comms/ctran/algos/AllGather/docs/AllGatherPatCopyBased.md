# CTPATCOPY: Step-Local Staged AllGather PAT

This document describes `ctpatcopy`, a staged-buffer variant of the AllGather PAT algorithm. It routes IB transfers through pre-registered staging buffers instead of putting directly to `recvbuff`, providing pre-registered buffer benefits and enabling future chunk-level pipelining. For v1, the staging pipeline operates at **step granularity**, reusing PipeStart/PipeSync/PipeEnd from the zero-copy `ctpat` but extending the model with a `stepDoneSync` signal (stream→GPE) to ensure staging→recvbuff copies complete before the next step. This makes v1 fully blocking per step.

**Prerequisite**: [AllGatherPat.md](AllGatherPat.md) — the zero-copy `ctpat` design.

## Motivation

The zero-copy `ctpat` puts directly from `recvbuff` to the peer's `recvbuff`. This works well but has two limitations:

1. **Registration dependency**: The user's `recvbuff` must be IB-registered (pre-registered at init or dynamically at exec). Dynamic registration is expensive (~50-200μs for large buffers). Pre-registration requires stable buffer addresses.

2. **No intra-step overlap**: All `2^i` puts at step `i` complete before any NVL CE broadcast begins. With staging, future versions can pipeline individual chunks within a step — but this requires chunk-granular completion tracking that the current transport doesn't cleanly support.

`ctpatcopy` addresses limitation 1 immediately and lays the groundwork for limitation 2.

### What v1 Does NOT Do

- **No chunk-level pipelining within a step.** v1 stages all `2^i` puts at step `i`, waits for the step to complete, then flushes, copies staging→recvbuff, and broadcasts. v1 is **fully blocking per step** — the GPE waits for the stream to finish staging→recvbuff copies via `stepDoneSync` before starting the next step. There is no inter-step IB/NVL overlap in v1.
- **No multi-round sub-chunking.** If `sendSize > stagingBufSize`, v1 falls back to zero-copy.
- **No auto-selection.** `ctpatcopy` is a separate `NCCL_ALLGATHER_P_ALGO` choice, not a dynamic branch inside `ctpat`.

### Expected Benefits (Hypotheses — To Be Validated by Benchmarking)

| Benefit | Mechanism | Expected Impact |
|---------|-----------|-----------------|
| No dynamic registration | Pre-registered staging buffers at init | Eliminate ~50-200μs per-call registration overhead |
| Bounded BAR1 usage | Fixed 32MB staging vs. registering full recvbuff | Reduced memory pressure at large rank counts |
| Foundation for chunk pipeline | Staging infrastructure in place | Future v2 can add chunk-level overlap when transport supports it |

We expect `ctpatcopy` to be **close to `ctpat` performance** for pre-registered buffers, with a small per-step overhead from the extra CE copies and `cudaStreamSynchronize` (v1 is fully blocking per step, so this overhead is not hidden by overlap). We expect `ctpatcopy` to **outperform `ctpat`** when dynamic registration is in play, since the pre-registered staging eliminates the ~50-200μs per-call registration cost. The performance crossover for intra-step pipelining benefits is deferred to v2 once we have benchmark data and transport support for chunk-granular completion.

## Architecture: ctpat vs ctpatcopy

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AllGatherP Persistent API                     │
│                   allGatherPInit / allGatherPExec / allGatherPDestroy│
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              switch(NCCL_ALLGATHER_P_ALGO)
                               │
     ┌─────────┬───────────┬───┴──────────┬──────────────┐
     ▼         ▼           ▼              ▼              ▼
 ctdirect  ctpipeline  ctrdpipeline     ctpat         ctpatcopy
 (direct)  (ring)      (butterfly)    (butterfly     (butterfly
                                      zero-copy)    staged) ← NEW
                                         │              │
                                         │              │
                                    ┌────┴────┐    ┌────┴────────────┐
                                    │ iput    │    │ CE→staging      │
                                    │ directly│    │ iput staging    │
                                    │ to peer │    │ iflush          │
                                    │ recvbuff│    │ CE staging→recv │
                                    └─────────┘    │ nvlCeBcast      │
                                                   │ stepDoneSync    │
                                                   └─────────────────┘
```

## Architecture Decision: Separate Algo, Not Dynamic Branch

`ctpatcopy` is a separate `NCCL_ALLGATHER_P_ALGO` enum value, NOT a dynamic branch inside `ctpat`. This avoids the init/exec divergence problem: staging buffers are allocated and exchanged at init time, so the algo choice must be fixed for the request lifetime.

```
NCCL_ALLGATHER_P_ALGO choices:
  ctdirect       — all-to-all direct puts
  ctpipeline     — ring + NVL CE bcast
  ctrdpipeline   — butterfly + NVL CE bcast (zero-copy)
  ctpat          — PAT butterfly + NVL CE bcast (zero-copy)
  ctpatcopy      — PAT butterfly + NVL CE bcast (staged)  ← NEW
```

## Data Flow

### Per-Step Flow (v1: Step Granularity)

```
SEND PATH — GPE thread, for each of 2^i chunks at step i:

  recvbuff[chunk_j] ──CE copy──► tmpSendBuf[slot]
  (step 0: sendbuff)                  │
                                      │ IB RDMA PUT
                                      ▼
                                peer's tmpRecvBuf[slot]


RECV PATH — GPE thread, after ALL 2^i puts complete + notify:

  tmpRecvBuf[slot] ──iflush──► tmpRecvBuf[slot]


STREAM SIDE — after GPE signals step completion via pipeSync:

  tmpRecvBuf[slot] ──CE copy──► recvbuff[chunk_j]
                                      │
                                      │ NVL CE broadcast
                                      ▼
                                local peers' recvbuff[chunk_j]
```

### Detailed Data Flow (Step i, Node A → Node B, localRank=0)

```
NODE A (sender)                                     NODE B (receiver)
┌────────────────────────────┐                      ┌────────────────────────────┐
│                            │                      │                            │
│  recvbuff (GPU)            │                      │  recvbuff (GPU)            │
│  ┌────┬────┬────┬────┐     │                      │  ┌────┬────┬────┬────┐     │
│  │ c0 │ c1 │ c2 │ c3 │     │                      │  │    │    │    │    │     │
│  └─┬──┴─┬──┴─┬──┴─┬──┘     │                      │  └────┴─▲──┴─▲──┴─▲──┘     │
│    │    │    │    │         │                      │         │    │    │         │
│    │ ① CE copy (batch)     │                      │      ⑤ CE copy (stream)    │
│    ▼    ▼    ▼    ▼         │                      │         │    │    │         │
│  tmpSendBuf (GPU, 32MB)    │                      │  tmpRecvBuf (GPU, 32MB)    │
│  ┌────┬────┬────┬────┐     │                      │  ┌────┬────┬────┬────┐     │
│  │ c0 │ c1 │ c2 │ c3 │     │                      │  │ c0 │ c1 │ c2 │ c3 │     │
│  └─┬──┴─┬──┴─┬──┴─┬──┘     │                      │  └─▲──┴─▲──┴─▲──┴─▲──┘     │
│    │    │    │    │         │                      │    │    │    │    │         │
└────┼────┼────┼────┼────────┘                      └────┼────┼────┼────┼────────┘
     │    │    │    │                                     │    │    │    │
     │    │    │    └──── ③ IB RDMA PUT (notify last) ───┘    │    │
     │    │    └───────── ③ IB RDMA PUT ──────────────────────┘    │
     │    └────────────── ③ IB RDMA PUT ───────────────────────────┘
     └─────────────────── ③ IB RDMA PUT ──────────────────────────────┘

② cudaStreamSynchronize(copyStream) — between ① and ③
④ iflush (after all puts + notify complete) — between ③ and ⑤
⑥ nvlCeBcast from recvbuff to local peers (after ⑤)
⑦ StepDone kernel signals GPE (after ⑥)
```

### Between-Step Forwarding

Step `i+1` reads from `recvbuff` (not staging) as its send source. This means the staging→recvbuff CE copies for step `i` must complete before step `i+1`'s GPE issues `icopy` from those positions.

**The existing PipeSync is one-way** (stream waits for GPE) and does not provide a signal back to the GPE. The zero-copy `ctpat` does not need this because IB puts land directly in recvbuff. But with staging, the GPE must wait for the stream-side CE copies to finish.

v1 uses a **`stepDoneSync` GpeKernelSync** for stream→GPE completion:

1. GPE: after `pipeSync->post(i)`, wait on `stepDoneSync->waitComplete(i)` before starting step `i+1`
2. Stream: after all CE copies and NVL broadcasts for step `i`, a 1-thread `StepDone` kernel signals `stepDoneSync->complete(i)`

This makes each step **fully blocking**: the GPE cannot start step `i+1` until the stream finishes step `i`'s staging→recvbuff copies. This is strictly correct. The cost is that the inter-step IB/NVL overlap from zero-copy `ctpat` is lost — but since v1 is about registration benefits, not overlap, this is acceptable.

```
New kernel:
__global__ void ncclKernelStepDone(
    int* flag, CtranAlgoDeviceState* devState, StepDoneKernArgs args) {
    ctran::device::devLoadAbortFlags(flag, devState);
    GpeKernelSyncDev::complete(args.stepDoneSync, 0, args.stepId);
}
```

**Reset between execs**: Both `pipeSync` and `stepDoneSync` must be reset at the end of each exec to prevent stale `completeFlag` values from causing early pass-through on the next persistent replay. `GpeKernelSync::waitComplete()` uses `>= step` comparison, so if `stepDoneSync` is left at a prior exec's terminal value, `waitComplete(0)` in the next exec passes immediately — breaking the staged dependency.

The reset is done in the `PipeEnd` kernel, which already owns end-of-exec cleanup. `PipeEnd` is extended to reset both sync objects:

```
// Extended PipeEnd kernel for ctpatcopy:
__global__ void ncclKernelAllGatherPPatCopyPipeEnd(
    int* flag, CtranAlgoDeviceState* devState, PatCopyPipeEndKernArgs args) {
    GpeKernelSyncDev::reset(args.pipeSync, 0);
    GpeKernelSyncDev::reset(args.stepDoneSync, 0);
    devStateLoadToShm(devState);
    const auto localRank = statex->localRank();
    const auto nLocalRanks = statex->nLocalRanks();
    barrier(localRank, nLocalRanks);
}
```

`stepDoneSync` is a stream-produced signal, so resetting it on the stream at the end of exec is the safest lifecycle point. Resetting from the GPE at the start of the next exec would be weaker because lingering stream work from the previous replay could race with the reset.

This is **step-local staging**, not full staged forwarding. The staging buffers decouple IB from user buffers within each step, but forwarding still reads from `recvbuff`.

## Staging Buffer Layout

v1 uses **contiguous per-step** layout, not round-robin slot reuse. Since the GPE issues all puts for a step before the stream consumes any of them, every chunk within a step must occupy a distinct staging position. Reusing slots within a step would overwrite unconsumed data.

```
tmpSendBuf (32MB default):
┌──────────────────────────────────────────────────────┐
│  chunk 0  │  chunk 1  │  chunk 2  │  ...  │ chunk N  │
│ sendSize  │ sendSize  │ sendSize  │       │ sendSize │
└──────────────────────────────────────────────────────┘

tmpRecvBuf (32MB default):
┌──────────────────────────────────────────────────────┐
│  chunk 0  │  chunk 1  │  chunk 2  │  ...  │ chunk N  │
│ sendSize  │ sendSize  │ sendSize  │       │ sendSize │
└──────────────────────────────────────────────────────┘

maxChunksPerStep = stagingBufSize / sendSize
```

At step `i` with `2^i` chunks: chunk `j` uses offset `j * sendSize` within the staging buffer. No slot reuse within a step.

**Capacity constraint**: If `2^i * sendSize > stagingBufSize` for any step, v1 falls back to zero-copy for that step (or the entire collective). The worst-case step is `i = nSteps - 1`, which needs `nNodes/2 * sendSize`. For 8 nodes with 32MB staging: `sendSize <= 8MB`. For 16 nodes: `sendSize <= 2MB`.

If `sendSize > stagingBufSize / (nNodes / 2)`: v1 falls back to zero-copy.

```
Capacity constraint (32MB staging, worst-case = last step):

  nNodes │ Max sendSize  │ Last step chunks │ Last step total
  ───────┼───────────────┼──────────────────┼────────────────
    4    │    16 MB      │        2         │     32 MB
    8    │     8 MB      │        4         │     32 MB
   16    │     4 MB      │        8         │     32 MB
   32    │     2 MB      │       16         │     32 MB
   64    │     1 MB      │       32         │     32 MB

  If sendSize exceeds the limit → fall back to zero-copy (ctpat)
```

Two buffers (send + recv). One pair total — butterfly has a single peer per step. The entire staging buffer is reused across steps (step `i+1` can overwrite step `i`'s staging because the stream has consumed it by then — enforced by the step completion signal).

## Execution Model

v1 uses the **same PipeSync model** as zero-copy `ctpat`, with staging inserted in the GPE callback:

```
GPE Thread                                     CUDA Stream
──────────                                     ───────────
                                               copyToSelf (CE)
PipeStart ───────────────────────────────────► PipeStart (releases GPE, exits)
                                               nvlCeBcast(own chunk)
Step 0 (1 chunk):
  CE: sendbuff → tmpSend[s0]
  cudaStreamSynchronize(copyStream)
  iput(tmpSend[s0] → peer tmpRecv[s0])
  waitRequest(lastPut)
  waitNotify()
  iflush(tmpRecv[s0])
  pipeSync→post(0) ─────────────────────────► PipeSync(0)
                                                CE: tmpRecv[s0] → recvbuff[c0]
                                                nvlCeBcast(recvbuff[c0])
                                                StepDone(0) ──► GPE
  stepDoneSync→waitComplete(0)
  (GPE blocked until stream finishes step 0)

Step 1 (2 chunks):
  CE: recvbuff → tmpSend[0..1]
  cudaStreamSynchronize(copyStream)
  iput(tmpSend[0] → peer tmpRecv[0])
  iput(tmpSend[1] → peer tmpRecv[1])
  waitRequest(lastPut)
  waitNotify()
  iflush(tmpRecv[0])
  iflush(tmpRecv[1])
  pipeSync→post(1) ─────────────────────────► PipeSync(1)
                                                CE: tmpRecv[0] → recvbuff[c1]
                                                CE: tmpRecv[1] → recvbuff[c2]
                                                nvlCeBcast(recvbuff[c1])
                                                nvlCeBcast(recvbuff[c2])
                                                StepDone(1) ──► GPE
  stepDoneSync→waitComplete(1)
  (GPE blocked until stream finishes step 1)

Step 2 (4 chunks):
  CE: recvbuff → tmpSend[0..3]
  cudaStreamSynchronize(copyStream)
  iput × 4
  waitNotify()
  iflush × 4
  pipeSync→post(2) ─────────────────────────► PipeSync(2)
                                                CE: tmpRecv[0..3] → recvbuff
                                                nvlCeBcast × 4
                                                StepDone(2) ──► GPE
 ...
                                               PipeEnd (reset + barrier)
```

### Step-Level Timeline (Fully Blocking, 4 Nodes = 2 Steps)

```
Time ──────────────────────────────────────────────────────────────────────────────►

GPE Thread:
  ┌──────────────────────────────┐   ┌─────────────────────────────────────────────┐
  │ Step 0 (1 chunk)             │   │ Step 1 (2 chunks)                           │
  │ CE: src→stg                 │   │ CE: src→stg[0..1]                           │
  │ sync(copyStream)            │   │ sync(copyStream)                            │
  │ iput(stg→peer)              │   │ iput(stg[0]→peer), iput(stg[1]→peer)       │
  │ waitRequest + waitNotify    │   │ waitRequest + waitNotify                    │
  │ iflush                      │   │ iflush[0], iflush[1]                        │
  │ pipeSync→post(0)            │   │ pipeSync→post(1)                            │
  └──────────────────┬──────────┘   └──────────────────────────────┬──────────────┘
                     │ blocked                                     │
                     │ stepDoneSync                                │
                     │ →waitComplete(0)                            ▼
                     │                                          done
                     ▼

CUDA Stream:
  ┌──────────┐ ┌────────────┐ ┌──────────────────────┐ ┌──────────────────────────┐
  │copyToSelf│ │PipeStart   │ │PipeSync(0)           │ │PipeSync(1)               │
  │          │ │(→exits)    │ │CE: stg[0]→recv[c0]   │ │CE: stg[0]→recv[c1]       │
  │          │ │            │ │nvlCeBcast(c0)        │ │CE: stg[1]→recv[c2]       │
  │          │ │nvlCeBcast  │ │StepDone(0)──►GPE     │ │nvlCeBcast(c1)            │
  │          │ │(own chunk) │ │                      │ │nvlCeBcast(c2)            │
  └──────────┘ └────────────┘ └──────────────────────┘ │StepDone(1)──►GPE         │
                                                       └──────────────────────────┘
                                                       ┌──────────────────────────┐
                                                       │PatCopyPipeEnd            │
                                                       │  reset(pipeSync)         │
                                                       │  reset(stepDoneSync)     │
                                                       │  barrier(localRanks)     │
                                                       └──────────────────────────┘

  ◄─── No overlap between steps: GPE blocked on stepDoneSync ───►
```

### Bidirectional Sync Flow

```
GPE Thread                          CUDA Stream
──────────                          ───────────

  pipeSync→post(i)  ──────────────►  PipeSyncKernel waits
  (GPE→stream:                       (stream unblocked)
   "IB done, staging                       │
    ready to read")                        │ CE copies + NVL bcasts
                                           │
  stepDoneSync                             │
  →waitComplete(i)  ◄────────────── StepDoneKernel signals
  (stream→GPE:                      (stream→GPE:
   GPE blocked until                 "recvbuff populated,
   stream finishes)                   staging safe to reuse")
         │
         ▼
  Start step i+1
```

### Key Differences from Zero-Copy ctpat

| Aspect | ctpat (zero-copy) | ctpatcopy (staged) |
|--------|-------------------|---------------------|
| IB source | recvbuff/sendbuff directly | tmpSendBuf (pre-registered) |
| IB destination | peer's recvbuff directly | peer's tmpRecvBuf |
| After IB arrival | Data already in recvbuff | CE copy: tmpRecvBuf → recvbuff on stream |
| iflush needed | No (ctrdpipeline doesn't use it) | Yes (IB→staging requires flush before CE reads) |
| User buffer registration | Required for iput source | Not needed — staging is pre-registered |
| Stream-side work per step | nvlCeBcast only | CE staging→recvbuff + nvlCeBcast |
| Overlap model | Inter-step via PipeSync | Fully blocking per step (stepDoneSync) |
| Sync direction | GPE→stream only | GPE→stream (pipeSync) + stream→GPE (stepDoneSync) |

### When nLocalRanks == 1

v1 `ctpatcopy` **requires nLocalRanks > 1**. The staged copy-back from `tmpRecvBuf` to `recvbuff` is driven by the CUDA stream after `PipeSync`, which only runs in the `nLocalRanks > 1` path. In the pure-IB case (`nLocalRanks == 1`), there is no stream-side copy-back, so step `i+1` would read stale data from `recvbuff`.

For `nLocalRanks == 1`, use `ctpat` (zero-copy) instead. The eligibility check in `execPatCopy()` rejects `nLocalRanks == 1` with an error.

## Buffer Management

### Allocation: Per-Persistent-Request via BufManager

```cpp
// In AllGatherP/Types.h:
namespace ctran::allgatherp {

enum class StagingBufId {
  kSendBuf,
  kRecvBuf,
  kNumBufs,
};

struct StagingInfo {
  ctran::algos::bufmanager::RegBuf sendBuf;   // local send staging
  ctran::algos::bufmanager::RegBuf recvBuf;   // local recv staging
  std::vector<ctran::algos::bufmanager::RemRegBuf> remRecvBufs;  // peers' recv staging
  size_t slotSize{0};
  size_t numSlots{0};
};

// Extended Resource struct:
struct Resource {
  GpeKernelSync* pipeSync{nullptr};       // GPE→stream: step IB complete
  GpeKernelSync* stepDoneSync{nullptr};   // stream→GPE: step CE copies complete
  std::unique_ptr<BufManager<StagingBufId, StagingBufId::kNumBufs>> stagingBufMgr;
  StagingInfo stagingInfo;
};

struct StepDoneKernArgs {
  int stepId;
  GpeKernelSync* stepDoneSync;
};

struct PatCopyPipeEndKernArgs {
  GpeKernelSync* pipeSync;
  GpeKernelSync* stepDoneSync;
};
}
```

### Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│ allGatherPInit                                                      │
│                                                                     │
│  1. initResources():                                               │
│     • allocate pipeSync (GpeKernelSync)                            │
│     • allocate stepDoneSync (GpeKernelSync)                        │
│     • BufManager: insert(kSendBuf, 32MB), insert(kRecvBuf, 32MB)  │
│     • BufManager: commit() → cudaMalloc 64MB total                │
│                                                                     │
│  2. exchangeMemHdl (GPE thread):                                   │
│     • allGatherCtrl(recvbuff) → populate PersistArgs               │
│     • BufManager: exchange(peerRanks) → IB-register + exchange    │
│     • pArgs.initialized = true                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼ (called many times)
┌─────────────────────────────────────────────────────────────────────┐
│ allGatherPExec → execPatCopy()                                     │
│                                                                     │
│  GPE: CE→staging → sync → iput → waitNotify → iflush → pipeSync   │
│  Stream: PipeSync → CE staging→recv → nvlCeBcast → StepDone       │
│  GPE: stepDoneSync→waitComplete (blocked until stream done)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼ (once)
┌─────────────────────────────────────────────────────────────────────┐
│ allGatherPDestroy                                                   │
│                                                                     │
│  1. BufManager: release() → deregister + cudaFree staging          │
│  2. cudaFreeHost(pipeSync)                                          │
│  3. cudaFreeHost(stepDoneSync)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Phase | What Happens |
|-------|-------------|
| `allGatherPInit` | BufManager creates staging buffers, `commit()` allocates GPU memory |
| `exchangeMemHdl` (GPE) | `bufManager.exchange()` IB-registers and exchanges addresses with butterfly peers |
| `allGatherPExec` | GPE uses staging for iput; stream uses staging for CE copies |
| `allGatherPDestroy` | `bufManager.release()` deregisters and frees staging |

Only **IB registration** needed. NVL reads from `recvbuff` (already IPC-exchanged via `PersistArgs`).

### Sizing

| Parameter | Default | CVAR |
|-----------|---------|------|
| Staging buffer size | 32MB per buffer | `NCCL_CTRAN_ALLGATHERP_PATCOPY_STAGING_BUF_SIZE` |
| Total memory | 64MB (32MB send + 32MB recv) | — |

When `sendSize > stagingBufSize / (nNodes / 2)`: `execPatCopy()` delegates to `execPat()` (zero-copy) rather than carrying an inline zero-copy duplicate. v1 does not support multi-round sub-chunking.

## GPE Callback

```cpp
commResult_t gpeFn(opGroup) {
    // Register sendbuff (step 0 source)
    mapper->searchRegHandle(sendBuff, sendSize, &sendHdl, &localReg);

    // Pre-init notifications and RTR for all steps
    for (int i = 0; i < nSteps; i++) {
        peers[i] = peerAtStep(...);
        mapper->initNotify(peers[i], stagingInfo.recvBuf.regHdl, &notifyVec[i]);
        mapper->isendCtrl(peers[i], &syncSreqs[i]);
        mapper->irecvCtrl(peers[i], &syncRreqs[i]);
    }
    // Wait all RTR
    for (int i = 0; i < nSteps; i++)
        mapper->waitRequest(&syncRreqs[i]);

    for (int step = 0; step < nSteps; step++) {
        int peer = peers[step];
        int nPuts = 1 << step;

        // Batch CE copy: source → staging for all chunks in this step
        for (int j = 0; j < nPuts; j++) {
            size_t chunkIdx = rankChunkOffset(myNode, localRank, nLocalRanks, nNodes, step, j);
            size_t byteOffset = chunkIdx * sendSize;

            const void* src = (step == 0 && j == 0) ? sendBuff
                            : getPtr(pArgs->recvbuff, byteOffset);

            mapper->icopy(stagingInfo.sendBuf.ptr + j * sendSize,
                          src, sendSize, copyStream);
        }

        // Sync: ensure ALL CE copies to staging are complete before IB reads
        cudaStreamSynchronize(copyStream);

        // Issue all IB puts for this step (staging is now populated)
        for (int j = 0; j < nPuts; j++) {
            bool isLast = (j == nPuts - 1);
            mapper->iput(stagingInfo.sendBuf.ptr + j * sendSize,
                         stagingInfo.remRecvBufs[step].ptr + j * sendSize,
                         sendSize, peer,
                         {stagingInfo.sendBuf.regHdl,
                          stagingInfo.remRecvBufs[step].rkey,
                          .notify_ = isLast},
                         isLast ? &lastPutReq : nullptr);
        }

        // Wait for all puts + peer notification
        mapper->waitRequest(&lastPutReq);
        mapper->waitNotify(notifyVec[step].get());

        // Flush all received staging
        for (int j = 0; j < nPuts; j++) {
            mapper->iflush(stagingInfo.recvBuf.ptr + j * sendSize,
                           stagingInfo.recvBuf.regHdl, &flushReq);
            mapper->waitRequest(&flushReq);
        }

        // Signal stream: step complete, staging ready for CE copies
        if (nLocalRanks > 1) {
            resource->pipeSync->post(step);
        }

        // Wait for stream to finish staging→recvbuff copies before next step
        // (also waited on for the last step to ensure staging is consumed
        //  before PipeEnd resets stepDoneSync)
        if (nLocalRanks > 1) {
            resource->stepDoneSync->waitComplete(step);
        }
    }
}
```

The `cudaStreamSynchronize(copyStream)` is issued **once per step** (not per chunk), after all CE copies for that step's chunks are enqueued. This ensures staging is fully populated before any IB put reads from it. The cost is one sync per step (log₂(nNodes) syncs total), which is acceptable for v1.

## Stream-Side

```cpp
commResult_t AlgoImpl::execPatCopy(sendbuff, count, datatype) {
    // ... validation, same as ctpat ...

    copyToSelf(comm_, sendbuff, sendSize, pArgs, stream_);

    // Submit GPE
    if (nNodes > 1) {
        submit(opGroup, gpeFn, config, ncclKernelAllGatherPPipeStart);
    }

    if (nLocalRanks > 1) {
        waitInit();
        nvlCeBcast(comm_, sendbuff, sendSize, myRank * sendSize, pArgs, stream_);

        for (int step = 0; step < nSteps; step++) {
            // Wait for GPE to signal step complete
            PipeSyncKernArgs syncArgs = {.stepId = step, .pipeSync = resource_.pipeSync};
            config.algoArgs = reinterpret_cast<void*>(&syncArgs);
            submit({}, nullptr, config, ncclKernelAllGatherPPipeSync);

            // CE copy: staging → recvbuff for each received chunk
            int nChunks = 1 << step;
            int peerNode = computePeerNode(myNode, nNodes, step);

            for (int j = 0; j < nChunks; j++) {
                size_t chunkIdx = rankChunkOffset(peerNode, localRank, nLocalRanks, nNodes, step, j);
                size_t byteOffset = chunkIdx * sendSize;

                // CE copy: staging[j] → recvbuff (contiguous layout, no slot reuse)
                cudaMemcpyAsync(
                    getPtr(pArgs.recvbuff, byteOffset),
                    stagingInfo.recvBuf.ptr + j * sendSize,
                    sendSize, cudaMemcpyDefault, stream_);

                bool needBarrier = (j == 0);
                nvlCeBcast(comm_, getPtr(pArgs.recvbuff, byteOffset),
                           sendSize, byteOffset, pArgs, stream_, needBarrier);
            }

            // Signal GPE: stream finished consuming staging for this step
            StepDoneKernArgs doneArgs = {
                .stepId = step, .stepDoneSync = resource_.stepDoneSync};
            config.algoArgs = reinterpret_cast<void*>(&doneArgs);
            submit({}, nullptr, config, ncclKernelStepDone);
        }

        PatCopyPipeEndKernArgs endArgs = {
            .pipeSync = resource_.pipeSync,
            .stepDoneSync = resource_.stepDoneSync};
        config.algoArgs = reinterpret_cast<void*>(&endArgs);
        submit({}, nullptr, config, ncclKernelAllGatherPPatCopyPipeEnd);
    }
}
```

Two new kernels: `ncclKernelStepDone` (stream→GPE completion signaling) and `ncclKernelAllGatherPPatCopyPipeEnd` (extended PipeEnd that resets both `pipeSync` and `stepDoneSync`). Reuses existing `PipeSync` and `PipeStart` from `ctpat`.

## Code Structure

### New Files

| File | Description |
|------|-------------|
| `AllGatherP/PatCopyImpl.cc` | `execPatCopy()` + staged `gpeFn` |
| `AllGatherP/PatCopyImpl.cu` | `ncclKernelStepDone` + `ncclKernelAllGatherPPatCopyPipeEnd` kernels |

### Modified Files

| File | Changes |
|------|---------|
| `AllGatherP/Types.h` | Add `StagingBufId`, `StagingInfo`; extend `Resource` with BufManager |
| `AllGatherP/AlgoImpl.h` | Add `execPatCopy()` declaration, `ctpatcopy` in `algoName()` |
| `AllGatherP/AllGatherP.cc` | Add `ctpatcopy` dispatch; init/destroy staging in lifecycle |
| `nccl_cvars.yaml` | Add `ctpatcopy` to `NCCL_ALLGATHER_P_ALGO`; add staging buf size CVAR |

### Build System

BUCK/CMake: `glob` auto-includes new `.cc` files. No build changes.

## Implementation Plan

### Phase 1: Staging Infrastructure + Step-Staged GPE

**Steps**:
1. Add `ctpatcopy` to `NCCL_ALLGATHER_P_ALGO` in `nccl_cvars.yaml` and regenerate
2. Add `StagingBufId`, `StagingInfo`, `StepDoneKernArgs`, `PatCopyPipeEndKernArgs` to `Types.h`; extend `Resource` with BufManager and `stepDoneSync`
3. Add staging allocation in `initResources()`, exchange in `exchangeMemHdl()`, release in `destroy()`
4. Create `PatCopyImpl.cu` with `ncclKernelStepDone` and `ncclKernelAllGatherPPatCopyPipeEnd` kernels
5. Create `PatCopyImpl.cc` with staged `gpeFn` + `execPatCopy()`
6. Wire dispatch in `AllGatherP.cc` and `AlgoImpl.h`

**Deliverables**:
- `AllGatherP/PatCopyImpl.cc`
- `AllGatherP/PatCopyImpl.cu`
- Updated `Types.h`, `AlgoImpl.h`, `AllGatherP.cc`, `nccl_cvars.yaml`
- Distributed tests: extend `ctran/tests/CtranDistAllgatherPTests.cc` with `ctpatcopy` in parameterized suite (nolocal + vnode configs)
- Negative tests: `nLocalRanks==1` rejection, staging capacity fallback (via small `NCCL_CTRAN_ALLGATHERP_PATCOPY_STAGING_BUF_SIZE`)

**Gate**: All tests pass — no crash, no data corruption, no stale `stepDoneSync` on persistent replay. Phase 2 does not start until Phase 1 tests are green.

### Phase 2: Benchmarking + Decision

**Deliverables**:
- Benchmark sweep on real H100 clusters (4x8, 8x8)
- Message sizes: 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB
- Algorithms compared: `ctpatcopy` vs `ctpat` vs `ctdirect` vs `ctpipeline`
- Two test modes: pre-registered buffers (ncclMemAlloc) and dynamic registration (cudaMalloc without commRegister)

**Gate**:
- `ctpatcopy` within **15% of `ctpat`** for pre-registered buffers across all message sizes (the staging overhead from per-step `cudaStreamSynchronize` + CE copies is the expected cost)
- `ctpatcopy` **faster than `ctpat`** when dynamic registration is forced (the pre-registered staging eliminates ~50-200μs per-call registration cost)
- Decision: ship `ctpatcopy` as-is, iterate on overhead, or deprioritize

**Deliverables if gate passes**:
- Recommendation on whether `ctpatcopy` should be promoted to default for specific workloads
- Documented crossover points (which nNodes × sendSize combinations favor `ctpatcopy`)

### Phase 3: Chunk-Level Pipelining (Future)

**Prerequisite gate**: Transport supports clean chunk-granular completion without per-chunk `cudaStreamSynchronize`. This requires either:
- A request-based CE copy completion API (post CE copy, poll for completion without stream sync)
- Or a transport-owned staging path that internally manages CE→IB ordering

**Scope** (once gate is met):
- Add `chunkSync` (GPE→stream per chunk) and `consumeSync` (stream→GPE per staging slot) coordination
- Enable intra-step overlap: IB transfer of chunk j+1 overlaps with CE copy + NVL bcast of chunk j
- Round-robin staging slot reuse within a step (safe because chunk-granular completion tracks consumption)
- Expected benefit: 5-54% improvement over step-granular staging at 4+ nodes (from earlier performance model)

**Deliverables**:
- Updated `PatCopyImpl.cc` with chunk-pipelined GPE callback
- New `PipeSyncChunk` and `StagingConsumed` kernels
- Benchmark comparison: chunk-pipelined vs step-granular vs zero-copy

### What Is Cut from v1

- ~~Chunk-level pipelining within a step~~ → requires transport-level completion support (Phase 3)
- ~~Multi-round sub-chunking~~ → fall back to `execPat()` for oversized messages
- ~~Auto-selection between zero-copy and staged~~ → separate `ctpatcopy` algo
- ~~Per-chunk cudaStreamSynchronize~~ → too expensive; not viable
- ~~nLocalRanks == 1 support~~ → use `ctpat` (zero-copy) instead

### Milestone Summary

```
Phase 1                          Phase 2                        Phase 3
────────                         ────────                       ────────
┌───────────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐
│ PatCopyImpl.cc + .cu  │  │ Benchmark sweep      │  │ Chunk-level pipelining  │
│ + Types.h + dispatch  │  │ 4x8, 8x8 H100       │  │ chunkSync + consumeSync │
│ + tests               │  │ 64KB → 256MB         │  │ intra-step IB/NVL      │
│                       │  │                      │  │ overlap                │
│ Gate:                 │  │ Gate:                 │  │                        │
│ All tests pass        │──►│ ≤15% overhead vs     │──►│ Gate:                  │
│ No data corruption    │  │ ctpat (pre-reg)      │  │ Transport supports     │
│ Persistent replay     │  │ Faster than ctpat    │  │ chunk-granular         │
│ safe (stepDoneSync    │  │ (dynamic reg)        │  │ completion             │
│ reset verified)       │  │                      │  │                        │
└───────────────────────┘  └──────────────────────┘  └─────────────────────────┘
```

## Testing

### Correctness Tests

Extend `ctran/tests/CtranDistAllgatherPTests.cc`:
- Add `ctpatcopy` to parameterized test instantiation and `algoToStr()`
- Power-of-2 nNodes enforcement
- nLocalRanks==1: verify `execPatCopy()` rejects with error (ctpatcopy requires nLocalRanks > 1)
- In-place and out-of-place
- Large message fallback: set `NCCL_CTRAN_ALLGATHERP_PATCOPY_STAGING_BUF_SIZE` to a small value (e.g., 4KB) to trigger the `execPatCopy() → execPat()` delegation path in CI without requiring enormous buffers

### Performance Tests

```bash
NCCL_ALLGATHER_P_ALGO=ctpatcopy buck2 run @fbcode//mode/opt \
    -c hpc_comms.use_ncclx=stable \
    //comms/ctran/benchmarks:AllgatherPBench -- \
    --algo ctpatcopy --min_bytes 64K --max_bytes 256M --bench_iters 20
```

Benchmark matrix:

| Dimension | Values |
|-----------|--------|
| Message size | 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB |
| Node count | 2x8, 4x8, 8x8, 16x8 |
| Algorithms | ctpatcopy, ctpat, ctdirect, ctpipeline |

### Stress Tests

- Back-to-back `allGatherPExec` calls (staging reuse across iterations)
- Concurrent communicators with separate staging buffers
- sendSize > stagingBufSize (verify fallback)
