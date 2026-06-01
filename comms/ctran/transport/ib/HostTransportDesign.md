# Host IB Transport Design

## Overview

The host-side IB P2P transport is a per-peer transport that moves
chunks of bytes between this rank and one peer over InfiniBand. It is
realized by two concrete classes that directly implement a common
abstract interface — `HostZcTransport` for pure zero-copy and
`HostCbTransport` for pure copy-based transfers. Both classes live in
`namespace ctran::transport::ib`.

**Abbreviations.** **ZC** = zero-copy (RDMA PUT directly to/from the
user buffer; no D2D copy, no staging). **CB** = copy-based (RDMA PUT
through a transport-owned staging buffer with a GPU D2D copy on each
side).

The transport contract is intentionally minimal. The caller picks
`vcIdx` per chunk and — when the chunk uses CB — picks the staging
slot the chunk uses. The transport runs the chunk on that VC / slot,
optionally fills a caller-provided `ChunkRequest`, and the caller
polls `testChunkDone(req)`. The transport does **not** model channels,
channel queues, channel state, or step counters. Mode (ZC vs. CB) is
a static property of the per-peer transport instance — every chunk
issued on the same instance uses the same mode.

The transport exposes:

- **Chunk** — one unit of bytes the caller wants moved. For CB,
  `len <= chunkSize_`. For ZC, `len` is unbounded (one chunk per
  caller-side slice).
- **Staging slot** — a CB-only resource. One of `pipelineDepth`
  per-direction records on `HostCbTransport`. Each owns a
  staging buffer, a `GpeKernelSync`, a flow-control counter pair
  `(remoteReady_[s], slotGeneration_[s])`, and per-slot
  `(issued, completed)` counters. Caller-addressable by index
  `[0, pipelineDepth)`. ZC chunks pass `kNoStagingSlot` and consume
  no slot.
- **VC** — one of `numVcs` per-peer `CtranIbVirtualConn`s, returned
  by `ctranIb_->connectVcs(peerRank_)`. Caller-addressable by index
  `[0, numVcs)`.
- **Per-VC ZC recv counters** — on `HostZcTransport` only, for
  each VC the transport keeps `(recvIssued, recvCompleted)`. Pure-ZC
  recvs use these to gate `testChunkDone`. ZC sends keep no
  per-VC state — the caller-owned `ChunkRequest::ibReq` is the
  source of truth.
- **ChunkRequest** — a handle written into a caller-provided
  `req` field of `SendChunkArgs` / `RecvChunkArgs`. Carries
  `(kind, vcIdx, stagingSlot, mySeq, ibReq)`. `testChunkDone(req)`
  forwards to either `req.ibReq.isComplete()` (ZC send) or a
  per-slot / per-VC counter compare against `mySeq`.

**Ordering invariant the transport guarantees.** Two chunks issued on
the same `(vcIdx, stagingSlot)` pair complete in issue order. Across
different VCs or across different staging slots within a VC, no
completion order is implied — the caller orders by choosing where it
places each chunk.

**Where instances come from.** The transport is never constructed
directly by algorithm code. `CtranMapper` owns the per-peer cache and
constructs the matching derived class on first
`mapper->getP2pTransport(peer, mode)` call. Each peer is bound to
exactly one mode for the lifetime of its cached instance.

## Terminology and resource model

**Mode is a class property.** Each `IP2pHostTransport*` is either pure
ZC (`HostZcTransport`) or pure CB (`HostCbTransport`). The caller asks
for a mode at cache lookup time
(`mapper->getP2pTransport(peer, HostTransportMode::kZeroCopy |
kCopyBased)`); a cache hit asserts the requested mode matches the
cached one.

**Staging-slot pool is flat and CB-only.** `HostCbTransport` owns a
flat pool of `pipelineDepth` per-direction staging-slot records. Each
slot is one staging buffer + one `GpeKernelSync` + one flow-control
counter pair + a `SendStagingRecord` / `RecvStagingRecord`. The
transport does not own any "channel descriptor" object. Channel ↔ slot
mapping is the caller's concern.

**ZC sends keep no per-chunk transport state.** A chunk on
`HostZcTransport::iSendChunk` posts `vcs_[vcIdx]->iput` and (if
`args.req != nullptr`) hands the caller's `ChunkRequest::ibReq` to the
IB backend as the iput's completion request.

**Every iput uses `notify=true`.** Both ZC and CB always issue
`vcs_[vcIdx]->iput(..., notify=true, ...)`. The QP delivers notifies
in iput-issue order on the receiver side, which is exactly the FIFO
ordering guarantee we rely on:

- ZC recv uses notifies to advance `vcRecvCompleted_[vcIdx]` in
  issue order (`pollRecvNotifications`).
- CB recv uses notifies as the WAIT_RECV → PROCESS_DATA trigger for
  the per-slot state machine.

Every chunk a sender issues will produce exactly one notify CQE on
the receiver, and the receiver must drain it (either via the ZC
counter loop or the CB record matcher) for the pipeline to advance.

**ZC recvs need a per-VC counter pair.** The only completion signal
`CtranIbVc` exposes is `checkNotify(bool*)`, which consumes one notify
per call. To support multiple in-flight ZC recvs and out-of-order
`testChunkDone` polling, the transport keeps
`(vcRecvIssued, vcRecvCompleted)` per VC. `iRecvChunk` always bumps
`recvIssued` (even when `args.req == nullptr`) because notifies are
FIFO-delivered on the QP — a later tracked recv's `mySeq` depends on
earlier notifies being drained.

**VC vector is flat.** Both transports cache the per-peer VC vector
returned by `ctranIb_->connectVcs(peerRank_)` eagerly in the ctor via
the shared `impl::connectAndPopulateVcs` helper. The caller picks
`vcIdx` directly per chunk.

**What the caller passes per chunk.** Chunk methods take a single
struct argument (`SendChunkArgs` / `RecvChunkArgs`) so callers fill
designated-initializer fields by name; the active transport uses what
its mode needs and asserts the wrong-mode fields are at their
defaults.

`SendChunkArgs` fields:

- *Common:* `userBuf`, `offset`, `len`, `vcIdx`, `req` (optional
  output handle; nullptr ⇒ fire-and-forget).
- *CB-specific:* `stagingSlot`, `round`, `hooks`. ZC asserts these
  are at defaults (`kNoStagingSlot`, `0`, empty).
- *ZC-specific:* `localMrHdl`, `remoteMrHdl`, `remoteKey`. CB asserts
  these are `nullptr`.

`RecvChunkArgs` fields: same shape minus the ZC-specific group
(receiver-side ZC has nothing per-chunk to address — data lands in
the buffer the receiver previously exported via `iSendCtrlMsg`).

**Completion is by request when the caller asks for one.** Every
`iSendChunk` / `iRecvChunk` writes through `args.req` only when it is
non-null. The handle is independent of the slot's "next in-flight
chunk" state, so out-of-order polling is permitted but yields no
useful ordering info beyond the per-`(vcIdx, stagingSlot)` guarantee.

**Within a staging slot**, a CB chunk runs to completion before that
slot can be reused. `isReadyForSend(vcIdx, slot)` returns true iff
the slot's status is `IDLE`.

## Architecture

The architecture is intentionally flat — algorithm code only ever
sees the abstract `IP2pHostTransport` interface returned by the
mapper. The two concrete classes implement it directly with no
intermediate base.

```
algorithm code
     │  mapper->getP2pTransport(peer, HostTransportMode)
     ▼
┌──────────────────────────────────────────────────────────────┐
│ CtranMapper                                                  │
│   getP2pTransport(peer, mode) → IP2pHostTransport*           │
│     - cache hit  → assert(cached.mode() == mode), reuse      │
│     - cache miss → construct ZC or CB transport              │
└────────────────────────┬─────────────────────────────────────┘
                         │  (single instance per peer)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ ctran::transport::IP2pHostTransport  (backend-neutral iface) │
│   ├─ peerRank() / mode() / pipelineDepth() / chunkSize()     │
│   ├─ progress()                                              │
│   ├─ iSendCtrlMsg(type, payload, len, CtrlRequest*)          │
│   ├─ iRecvCtrlMsg(payload, len, CtrlRequest*)                │
│   ├─ computeTotalChunks/Offset/Len(totalSize)                │
│   ├─ isReadyForSend / isReadyForRecv                         │
│   ├─ iSendChunk(const SendChunkArgs&)                        │
│   ├─ iRecvChunk(const RecvChunkArgs&)                        │
│   ├─ testChunkDone(req, bool* done)                          │
│   ├─ testCtrlMsgDone / waitCtrlMsgDone                       │
│   └─ lock() / unlock()  (per-transport caller lock)          │
└──────┬───────────────────────────────┬───────────────────────┘
       │  implements                   │  implements
       ▼                               ▼
┌─────────────────────────┐   ┌──────────────────────────────┐
│ ctran::transport::ib::  │   │ ctran::transport::ib::       │
│   HostZcTransport       │   │   HostCbTransport            │
│  (pure zero-copy)       │   │  (pure copy-based)           │
│                         │   │  Adds public API:            │
│                         │   │   + getDeviceTransport()     │
│                         │   │   + setKernelNumBlocks(…)    │
└─────────────────────────┘   └──────────────────────────────┘
```

Algorithm code never names the concrete classes directly — it only
holds an `IP2pHostTransport*` returned by
`mapper->getP2pTransport(peer, mode)`. The only place a concrete
type is named is the CB-only `getDeviceTransport()` /
`setKernelNumBlocks(…)` calls, which require a static_cast to
`HostCbTransport*` because they're CB-specific extensions outside
the abstract interface.

`iSendCtrlMsg` and `iRecvCtrlMsg` are deliberately pure ctrl-msg
primitives — they do NOT bundle mem export/import bookkeeping, and
they do NOT own the payload buffer. The caller owns the `ControlMsg`
(or any other byte buffer) and is responsible for building it and
keeping it alive for the duration of the request. See the
"Caller-side algorithm pattern" section below for the standard
ZC-receiver / ZC-sender / CB-receiver / CB-sender recipes.

## Caller-side algorithm pattern

The transport has no channel concept. Callers compose channels on top
of `(vcIdx, stagingSlot)` themselves. **The same loop drives ZC and
CB**: the caller fills both mode-specific arg groups (or only the
ones relevant to its mode); the active transport ignores what doesn't
apply.

**Loop shape: "while not finished, walk every channel once and make
at most one issue + one drain attempt per channel".** Do **not**
block-spin on `isReadyForSend` / `testChunkDone` for a single chunk
— that starves other channels (and, for CB, can deadlock against
backpressure from the receiver). The outer loop only exits when
every channel has issued all its chunks AND drained all its
outstanding `ChunkRequest`s. Recommended: `vcIdx = ch % numVcs`; for
CB, partition `pipelineDepth` slots across channels.

See `runChunkLoop` in
`comms/ctran/transport/ib/tests/HostTransportDistUT.cc` for the
reference driver; the algorithm-side skeleton is:

```cpp
// Caller picks mode and asks the mapper for the matching transport.
const auto mode = decideMode(/* algo policy */);
auto* transport = comm->ctran_->mapper->getP2pTransport(peer, mode);

// Per-transport caller-must-lock: every hot-path call below
// (iRecvCtrlMsg, waitCtrlMsgDone, progress, isReadyForSend,
// iSendChunk, testChunkDone, …) asserts that this thread holds the
// transport's mutex. Take it once for the whole driver scope; the
// guard releases it on scope exit. See "Concurrency model" below.
ctran::transport::P2pTransportLockGuard transportGuard(transport);

// Sender side: discover peer's recv-buf info. iRecvCtrlMsg is a pure
// ctrl-msg recv into the caller-owned `recvMsg` buffer; the caller
// resolves the bytes into a RemotePeerInfo themselves with
// importRemoteInfo() (CB ⇒ remote.isZeroCopy is false and
// memHdl/remoteKey are unused; ZC ⇒ filled in).
ControlMsg recvMsg;
CtrlRequest rcv;
transport->iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &rcv);
transport->waitCtrlMsgDone(rcv);
RemotePeerInfo remote{};
ctran::transport::ib::impl::importRemoteInfo(recvMsg, &remote);

const int slotsPerCh =
    (mode == HostTransportMode::kZeroCopy) ? 0 : pipelineDepth / numChannels;
const int totalChunks = transport->computeTotalChunks(totalSize);

std::vector<int> issued(numChannels, 0);
std::vector<int> drained(numChannels, 0);
std::vector<std::vector<ChunkRequest>> reqs(numChannels);
for (int ch = 0; ch < numChannels; ++ch) {
  reqs[ch].resize(stepsForCh(ch));   // 1 for ZC, ceil(slice/chunkSize) for CB
}

bool done = false;
while (!done) {
  // ONE progress() per outer pass drives the whole peer.
  transport->progress();
  done = true;

  for (int ch = 0; ch < numChannels; ++ch) {
    const int vcIdx = ch % transport->numVcs();
    const int slot  = (slotsPerCh == 0)
        ? kNoStagingSlot
        : (ch * slotsPerCh + issued[ch] % slotsPerCh);

    // (a) Issue at most one chunk if this channel is ready.
    if (issued[ch] < reqs[ch].size()) {
      if (transport->isReadyForSend(vcIdx, slot)) {
        transport->iSendChunk({
            .userBuf = buf,
            .offset  = transport->computeChunkOffset(issued[ch], totalSize),
            .len     = transport->computeChunkLen(issued[ch], totalSize),
            .vcIdx   = vcIdx,
            .req     = &reqs[ch][issued[ch]],
            // CB-specific (no-op in ZC)
            .stagingSlot = slot,
            .round       = (slotsPerCh == 0) ? 0 : (issued[ch] / slotsPerCh),
            // ZC-specific (no-op in CB)
            .localMrHdl  = myMr,
            .remoteMrHdl = remote.memHdl,
            .remoteKey   = &remote.remoteKey,
        });
        ++issued[ch];
      }
      done = false;
    }

    // (b) Drain at most one outstanding ChunkRequest on this channel.
    bool chunkDone = false;
    if (drained[ch] < issued[ch]) {
      transport->testChunkDone(reqs[ch][drained[ch]], &chunkDone);
      if (chunkDone) {
        ++drained[ch];
      }
    }
    if (drained[ch] < reqs[ch].size()) {
      done = false;
    }
  }
}
```

Key properties:

- **Hold the per-transport lock for the whole driver scope.** Every
  hot-path call (`progress`, `iSendCtrlMsg`, `iRecvCtrlMsg`,
  `waitCtrlMsgDone`, `isReadyForSend/Recv`, `iSendChunk`,
  `iRecvChunk`, `testChunkDone`, `testCtrlMsgDone`) aborts if the
  calling thread hasn't acquired the transport's mutex (typically via
  `P2pTransportLockGuard`). Trivial accessors (`peerRank`, `mode`,
  `pipelineDepth`, `chunkSize`, the `compute*` helpers) are exempt.
- **One issue + one drain attempt per channel per outer pass.** Never
  spin on a single channel — that starves the others, and for CB it
  can deadlock against credit return from the receiver.
- **`progress()` is called once per outer pass**, not once per chunk
  and not inside the per-channel branches.
- **ZC:** `slot = kNoStagingSlot`, `isReadyForSend` always returns
  true, so the issue branch never blocks on readiness. The sender
  additionally fills `(localMrHdl, remoteMrHdl, remoteKey)` in
  `SendChunkArgs`.
- **CB:** `slot = ch * slotsPerCh + step % slotsPerCh`,
  `isReadyForSend` true iff that slot is IDLE. If not ready, the
  outer pass moves on to the next channel and tries this one again
  on the next pass.
- **Caller can fire-and-forget** by leaving `args.req = nullptr` and
  not maintaining a per-channel drain counter. ZC send posts the iput
  without an attached completion request; CB still runs the slot
  state machine but provides no caller-side poll handle.

### Multi-peer callsite pattern

The same single-peer skeleton extends to a multi-peer collective by
materializing a small per-peer plan struct and looping over it. The
**outer** `progress()` is invoked once per pass (or once per NIC if the
algorithm prefers a NIC-affine drain — see below), and the per-channel
issue + drain block from the single-peer example is run inside an inner
loop over the plan. No new helper types are added to the transport;
`PeerPlan` is purely algorithm-side scaffolding.

```cpp
// One entry per peer this rank is talking to.
struct PeerPlan {
  int                                   peer;
  HostTransportMode                     mode;
  IP2pHostTransport*                    transport;
  int                                   numChannels;
  int                                   slotsPerCh;     // 0 for ZC
  size_t                                totalSize;
  void*                                 userBuf;
  void*                                 memHdl;         // ZC only
  RemotePeerInfo                        remote;         // filled by importRemoteInfo()
  std::vector<int>                      issued;         // [numChannels]
  std::vector<int>                      drained;        // [numChannels]
  std::vector<std::vector<ChunkRequest>> reqs;          // [ch][step]
};

std::vector<PeerPlan> plan = buildPeerPlans(/* algo input */);

// Per-transport caller-must-lock: every PeerPlan.transport must be
// held for the duration of the hot-path calls below. P2pTransportLockGuard
// is non-copyable/non-movable, so wrap each in a unique_ptr to put
// them in a container. The guards release in destruction order at
// scope exit. See "Concurrency model" below.
std::vector<std::unique_ptr<ctran::transport::P2pTransportLockGuard>>
    transportGuards;
transportGuards.reserve(plan.size());
for (auto& p : plan) {
  transportGuards.push_back(
      std::make_unique<ctran::transport::P2pTransportLockGuard>(p.transport));
}

// 1. Pre-issue ctrl + sized request vectors on every peer.
for (auto& p : plan) {
  ControlMsg recvMsg;
  CtrlRequest rcv;
  p.transport->iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &rcv);
  p.transport->waitCtrlMsgDone(rcv);
  ctran::transport::ib::impl::importRemoteInfo(recvMsg, &p.remote);
  p.issued.assign(p.numChannels, 0);
  p.drained.assign(p.numChannels, 0);
  p.reqs.assign(p.numChannels, {});
  for (int ch = 0; ch < p.numChannels; ++ch) {
    p.reqs[ch].resize(stepsForCh(p, ch));   // 1 for ZC, ceil for CB
  }
}

// 2. Main loop: walk every peer × channel once per outer pass.
bool done = false;
while (!done) {
  // (a) Single-NIC drain across all peers (uses the default full-CQ
  // overload). If the algorithm has its peers grouped by NIC, the
  // NIC-affine overload (see "Progress" below) is preferred — call
  // ctranIb_->progress(device) once per device per pass instead.
  for (auto& p : plan) {
    p.transport->progress();
  }

  done = true;
  for (auto& p : plan) {
    for (int ch = 0; ch < p.numChannels; ++ch) {
      const int vcIdx = ch % p.transport->numVcs();
      const int slot  = (p.slotsPerCh == 0)
          ? kNoStagingSlot
          : (ch * p.slotsPerCh + p.issued[ch] % p.slotsPerCh);

      // (b) Issue at most one chunk on this (peer, channel).
      if (p.issued[ch] < (int)p.reqs[ch].size()) {
        if (p.transport->isReadyForSend(vcIdx, slot)) {
          p.transport->iSendChunk({
              .userBuf = p.userBuf,
              .offset  = p.transport->computeChunkOffset(p.issued[ch], p.totalSize),
              .len     = p.transport->computeChunkLen (p.issued[ch], p.totalSize),
              .vcIdx   = vcIdx,
              .req     = &p.reqs[ch][p.issued[ch]],
              .stagingSlot = slot,
              .round       = (p.slotsPerCh == 0) ? 0 : (p.issued[ch] / p.slotsPerCh),
              .localMrHdl  = p.memHdl,
              .remoteMrHdl = p.remote.memHdl,
              .remoteKey   = &p.remote.remoteKey,
          });
          ++p.issued[ch];
        }
        done = false;
      }

      // (c) Drain at most one outstanding ChunkRequest.
      bool chunkDone = false;
      if (p.drained[ch] < p.issued[ch]) {
        p.transport->testChunkDone(p.reqs[ch][p.drained[ch]], &chunkDone);
        if (chunkDone) {
          ++p.drained[ch];
        }
      }
      if (p.drained[ch] < (int)p.reqs[ch].size()) {
        done = false;
      }
    }
  }
}
```

Key properties carried over from the single-peer case:

- **Hold the per-transport lock for every peer's transport for the
  whole driver scope.** Same caller contract as the single-peer
  pattern, applied per-peer; the `transportGuards` vector above
  acquires them all upfront and releases them at scope exit.
- **One issue + one drain attempt per (peer, channel) per outer
  pass.** Never spin on a single peer or channel.
- The done condition is "every `PeerPlan` has
  `drained[ch] == reqs[ch].size()` for every channel".
- The recv side is structurally identical (`iRecvChunk` instead of
  `iSendChunk`, no `localMrHdl/remoteMrHdl/remoteKey`).

## How ZC and CB are handled

The two derived classes share the `iSendChunk(SendChunkArgs) /
iRecvChunk(RecvChunkArgs) / testChunkDone(ChunkRequest)` API surface
defined on `IP2pHostTransport` but use disjoint internal state. Which
class is in play is fixed at cache lookup time.

### Pure-ZC fast path — `HostZcTransport`

Lowest-overhead path. No state machine, no hooks, no staging buffer,
no D2D copy, no flow control. Construction allocates nothing GPU-
side.

| Phase                       | Send                                                                                                                                                        | Recv                                                                                                                                |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `iSendChunk` / `iRecvChunk` | Posts `vcs_[vcIdx]->iput(notify=true)` synchronously; if `args.req != nullptr`, hands `args.req->ibReq` to the IB backend so `req.ibReq.isComplete()` works | Bumps `++vcRecvIssued_[vcIdx]` (always); if `args.req != nullptr`, fills `mySeq = vcRecvIssued_[vcIdx]`. No iput on the recv side.  |
| `progress()`                | Drains `ctranIb_->progress()` then drains per-VC ZC recv notifies via `pollRecvNotifications`.                                                              | Per VC, while `vcRecvCompleted_[i] < vcRecvIssued_[i]`, calls `vcs_[i]->checkNotify(&notified)` under the per-VC lock; `++completed`.|
| `testChunkDone(req)`        | `req.ibReq.isComplete()`                                                                                                                                    | `vcRecvCompleted_[req.vcIdx] >= req.mySeq`                                                                                          |

ZC does not pop `GpeKernelSync` from any pool; ZC ctor allocates no
GPU memory.

### Pure-CB slow path — `HostCbTransport`

Uses the per-slot state machine, hooks, staging buffers, GPU D2D
copy, and flow control. This is the path that handles all of:
acquiring a CB slot, running the prepare/iput state walk on the
sender, running the recv/process/signal-ready state walk on the
receiver, and returning the slot to IDLE.

The CB transport always uses `makeCopyBasedSendHooks` /
`makeCopyBasedRecvHooks` as its default hooks; non-default hooks may
be supplied for callers that need custom prepare/process behavior
(e.g., a chained recv-and-forward kernel).

### Cross-cutting properties

- **Each peer is bound to one mode.** The cached transport instance
  determines the path for every chunk issued on it.
- **`progress()`** on each transport drives both `ctranIb_->progress()`
  (CQE drain) — via `progressImpl` — and the per-class
  pipeline drain. Callers must invoke it periodically; it is not
  implicit in `iSendChunk` / `iRecvChunk`.
- **Issue-readiness is a pure read.** `isReadyForSend` /
  `isReadyForRecv` never touch the QP or the network.

## Lifecycle

### 1. Construction

The mapper constructs the matching derived class on first
`getP2pTransport(peer, mode)` call.

`HostZcTransport` ctor:
`(peerRank, CtranIb*, myRank, cudaDev, const CommLogData*)`. Allocates
nothing GPU-side. The `CommLogData*` is carried for future per-peer
ZC scratch regions and for memtrace attribution if such regions land.
The ctor eagerly:

1. Drives the per-peer VC rendezvous via
   `impl::connectAndPopulateVcs(ctranIb_, peerRank_, myRank_, vcs_)`.
   After this call `vcs_` is non-empty for the lifetime of the
   transport. Caller-blocking note: `ctranIb_->connectVcs(peer)` is
   two-sided (see `CtranIb.h:514-534`). The smaller-rank side
   initiates and returns immediately; the larger-rank side spins
   until the peer also calls `connectVcs` back. Algorithms calling
   `CtranMapper::getP2pTransport(peer, kZeroCopy)` therefore see
   construction (on first cache miss) potentially block on the
   larger-rank side.
2. Sizes the per-VC recv counters `vcRecvIssued_` and
   `vcRecvCompleted_` to `numVcs()`.

`HostCbTransport` ctor:
`(peerRank, CtranIb*, GpeKernelSyncPool*, pipelineDepth, chunkSize,
myRank, cudaDev, const CommLogData*)`. Allocates:

- `sendStaging_`, `recvStaging_` of `pipelineDepth_ * chunkSize_`
  bytes each via `ctran::utils::commCudaMalloc`; both are
  IB-registered.
- `pipelineDepth_` `GpeKernelSync*` entries each for `sendSyncs_` and
  `recvSyncs_`, popped from the borrowed `GpeKernelSyncPool*`. After
  pop, each entry's `nworkers` is set to
  `CTRAN_ALGO_MAX_THREAD_BLOCKS` and `resetStatus()` is called so
  the full `[0, nworkers)` `postFlag`/`completeFlag` range is
  initialized to `kUnset`. Callers whose kernel uses fewer blocks
  call `setKernelNumBlocks(sendN, recvN)` after the ctrl exchange to
  re-tune `nworkers` (and re-reset).
- `remoteReady_` of `pipelineDepth_ * sizeof(uint64_t)` bytes via
  `commCudaHostAlloc`; IB-registered.
- `fetchAddDiscardBuf_` of `sizeof(uint64_t)` bytes via
  `commCudaHostAlloc`; IB-registered.

All allocations pass the borrowed `CommLogData*` so NCCL memtrace
attributes them to the right call site.

### 2. VC-API rendezvous (eager, in constructor)

Each derived class drives the per-peer VC rendezvous in its ctor via
the shared `impl::connectAndPopulateVcs` helper, which calls
`ctranIb_->connectVcs(peerRank_)` and stores the resulting vector in
`vcs_`. The helper aborts if invoked with an already-populated
vector — it is the single producer of `vcs_` and is meant to run
exactly once per transport instance. Hot-path methods rely on
`vcs_` being non-empty and guard reads through `impl::checkValidVc`
as defense-in-depth.

### 3. Resource exchange (CB only, one-time, non-blocking)

`HostCbTransport::postResourceExchange()` runs once on first
`iSendChunk` / `iRecvChunk`. It exports `recvStaging_` and
`remoteReady_`, posts an `isendCtrlMsg` of the bundled
`ResourceExchangeMsg` on `vcs_[0]`, and posts a matching
`irecvCtrlMsg`. `testResourceExchange()` polls both halves and, on
completion, imports the peer's exported staging base + access key
and the peer's `remoteReady` address + access key into
`peerStagingBuf_` / `peerStagingKey_` and `peerRemoteReadyAddr_` /
`peerRemoteReadyKey_`.

**ZC chunks do not run this exchange.** The ZC transport never calls
`postResourceExchange` — pure-ZC writes go to caller-provided
`(remoteMrHdl, remoteKey)` from the `RemotePeerInfo` the sender built
from `iRecvCtrlMsg`'s received `ControlMsg` via `importRemoteInfo`.

### 4. Per-operation ctrl exchange

Each transport's `iSendCtrlMsg` / `iRecvCtrlMsg` is a thin wrapper
that calls into the shared `ctran::transport::ib::impl::*` helpers
(free functions in `HostTransportImpl.h` under the `impl`
sub-namespace).
The `iSendCtrlMsg` / `iRecvCtrlMsg` methods are pure ctrl-msg
primitives: they do NOT bundle mem export/import. The caller does
that explicitly with `exportRecvBuf` / `importRemoteInfo` (see
"Caller-side algorithm pattern" for the full example).

**Sender ZC + receiver ZC:**

- Receiver: build the EXPORT_MEM message in its own buffer and post.
  ```cpp
  ControlMsg msg;
  ctran::transport::ib::impl::exportRecvBuf(
      /*localMrHdl=*/recvBufRegElem, /*recvBuf=*/buf, msg);
  CtrlRequest ctrlReq;
  transport->iSendCtrlMsg(IB_EXPORT_MEM, &msg, sizeof(msg), &ctrlReq);
  transport->waitCtrlMsgDone(ctrlReq);  // caller must keep `msg` alive
  ```
  `HostZcTransport::iSendCtrlMsg` calls `iSendCtrlMsgImpl` which posts
  the caller-owned `msg` bytes on `vcs_[0]`.

- Sender: provide a buffer to receive into, then resolve the bytes.
  ```cpp
  ControlMsg recvMsg;
  CtrlRequest ctrlReq;
  transport->iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &ctrlReq);
  transport->waitCtrlMsgDone(ctrlReq);
  RemotePeerInfo remote{};
  ctran::transport::ib::impl::importRemoteInfo(
      recvMsg, &remote);  // populates memHdl + remoteKey
  ```
  The transport's `iRecvCtrlMsg` just posts the irecv into the
  caller-owned `recvMsg`; the `importRemoteInfo` call is the one
  place that turns the EXPORT_MEM payload into a `RemotePeerInfo`.

**Sender CB + receiver CB:**

- Receiver: build a SYNC msg directly in its own buffer and post.
  ```cpp
  ControlMsg msg;
  msg.setType(ControlMsgType::SYNC);
  CtrlRequest ctrlReq;
  transport->iSendCtrlMsg(SYNC, &msg, sizeof(msg), &ctrlReq);
  transport->waitCtrlMsgDone(ctrlReq);
  ```
- Sender: same `iRecvCtrlMsg(&recvMsg, sizeof(recvMsg), &ctrlReq)` +
  `waitCtrlMsgDone(ctrlReq)`. If the caller still wants a
  `RemotePeerInfo` for symmetry, `importRemoteInfo` returns
  `{isZeroCopy=false, memHdl=nullptr, remoteKey=zero}` for a SYNC
  payload. The remote address + key for CB iputs come from the
  one-time resource exchange (`peerStagingBuf_` / `peerStagingKey_`),
  not from this handshake.

### 5. Data transfer

ZC `iSendChunk` (sender side) computes
`dst = remoteMrHdl + offset` and posts
`vcs_[vcIdx]->iput(src, dst, len, regElem, *remoteKey, notify=true,
nullptr, ibReqPtr, false)` synchronously. `ibReqPtr` is
`&args.req->ibReq` if `args.req != nullptr`, else `nullptr`
(fire-and-forget). ZC `iRecvChunk` bumps `vcRecvIssued_[vcIdx]` and
optionally writes through `args.req`.

CB `iSendChunk` enqueues a chunk record on
`sendStagingSlots_[stagingSlot]` (status `PREPARE_DATA`), bumps
`cbSendActive_`, and writes through `args.req` if non-null. The
actual D2D copy + iput happens later in `progress()` via the
state machine. CB `iRecvChunk` enqueues a chunk record on
`recvStagingSlots_[stagingSlot]` (status `WAIT_RECV`) and bumps
`cbRecvActive_`.

### 6. Slot state machines (CB only)

**Send chunk states:**

```
PREPARE_DATA      → hooks.prepareData(ctx)
WAIT_PREPARE      → hooks.isDataReady(ctx)
WAIT_REMOTE_READY → hooks.isRemoteReady(ctx)
                    iput src=hooks.getLocalSrc(ctx),
                         dst=peerStagingBuf_ + slot*chunkSize,
                         remoteKey=peerStagingKey_
WAIT_IPUT         → iputReq.isComplete()
                    hooks.onSendDone(ctx)  (++slotGeneration[s])
                    ++sendStagingSlots_[s].completed
                    --cbSendActive_
                    status = IDLE
```

**Recv chunk states:**

```
WAIT_RECV     → pollRecvNotifications matches a notify on this VC,
                advances status to PROCESS_DATA
PROCESS_DATA  → hooks.processData(ctx)
WAIT_PROCESS  → hooks.isProcessDone(ctx)
SIGNAL_READY  → hooks.signalReady(ctx)  (ifetchAndAdd into peer's
                                          remoteReady_[s])
                hooks.onRecvDone(ctx)
                ++recvStagingSlots_[s].completed
                --cbRecvActive_
                status = IDLE
```

### 7. Progress

Each derived class implements its own `progress()` directly (no shared
base-class override):

```cpp
commResult_t HostZcTransport::progress() {
  FB_COMMCHECK(progressImpl(ctranIb_));
  FB_COMMCHECK(pollRecvNotifications());
  return commSuccess;
}

commResult_t HostCbTransport::progress() {
  FB_COMMCHECK(progressImpl(ctranIb_));
  progressSendPipeline();
  progressRecvPipeline();
  return commSuccess;
}
```

`progressImpl(ctranIb)` wraps `ctranIb->progress()` by
default. It also accepts an optional NIC index — the NIC-affine
overload `ctranIb_->progress(int device)` lets an algorithm drain a
single NIC's CQ instead of the global CQ:

```cpp
// NIC-affine drain, useful when the caller has its peers grouped per
// NIC and wants one progress loop per NIC.
progressImpl(ctranIb_, /*nicHint=*/device);
```

For the multi-peer pattern shown above, the recommended drain is
either the default `ctranIb_->progress()` invoked once per outer pass
(simple, drains everything) or, when peers are grouped by NIC,
`ctranIb_->progress(device)` invoked once per NIC per outer pass.

`HostZcTransport`'s recv loop runs unconditionally — `vcs_` is
non-empty post-ctor and `vcRecvCompleted_` is pre-sized in the ctor,
so a `progress()` call between `iSendCtrlMsg` and the first chunk is
a safe no-op.

`HostCbTransport`'s `progress()`:

1. `progressSendPipeline()` — early-out if `cbSendActive_ == 0` or
   resource exchange not yet done; otherwise walk every send slot
   and advance its state machine.
2. `progressRecvPipeline()` — `pollRecvNotifications()` first
   (matches each VC's pending notify to a `WAIT_RECV` slot,
   advancing it to `PROCESS_DATA`); then the same early-out + per-
   slot walk.

## Public API

### Constructors

```cpp
ctran::transport::ib::HostZcTransport(
    int peerRank, CtranIb* ctranIb,
    int myRank, int cudaDev,
    const CommLogData* logMetaData);

ctran::transport::ib::HostCbTransport(
    int peerRank, CtranIb* ctranIb,
    GpeKernelSyncPool* pool,
    int pipelineDepth, size_t chunkSize,
    int myRank, int cudaDev,
    const CommLogData* logMetaData);
```

Algorithm code never calls these directly; the mapper does.

### Accessors

```cpp
int   peerRank()      const;
HostTransportMode mode() const;
int   pipelineDepth() const;   // 0 for ZC
size_t chunkSize()    const;   // 0 for ZC
int   numVcs()        const;   // populated in the constructor
ctran::transport::ib::HostTransportDev* getDeviceTransport(); // CB only
```

### Static utilities

```cpp
void* ctran::transport::ib::impl::toIbRegElem(void* memHdl);
```

Casts an opaque mapper-style `RegElem*` to the `void*` the IB backend
expects.

### Resource lifecycle

```cpp
commResult_t progress();
```

Drains backend completions via `progressImpl(ctranIb_)` then
runs the per-class pipeline drain.

### Ctrl exchange

```cpp
commResult_t iSendCtrlMsg(
    ControlMsgType type, const void* payload, size_t len, CtrlRequest* out);
commResult_t iRecvCtrlMsg(void* payload, size_t len, CtrlRequest* out);
commResult_t testCtrlMsgDone(CtrlRequest& req, bool* done);
commResult_t waitCtrlMsgDone(CtrlRequest& req);
```

`iSendCtrlMsg` is a pure ctrl-msg primitive: it posts `len` bytes
from the caller-owned `payload` buffer on `vcs_[0]` with wire header
`type`. The buffer must outlive the request (the IB backend retains
a pointer to it until completion). The caller is responsible for
building `payload` — use
`ctran::transport::ib::impl::exportRecvBuf(localMrHdl, recvBuf, msg)`
to produce an `IB_EXPORT_MEM` payload for the ZC handshake, or
`msg.setType(ControlMsgType::SYNC)` for the CB handshake — typically
on a local `ControlMsg` variable that's then passed as
`(IB_EXPORT_MEM, &msg, sizeof(msg), &ctrlReq)` /
`(SYNC, &msg, sizeof(msg), &ctrlReq)`.

`iRecvCtrlMsg` posts an `irecvCtrlMsg` into the caller-owned
`payload` buffer of capacity `len`. After `testCtrlMsgDone` /
`waitCtrlMsgDone` returns, the received bytes are in the caller's
buffer. The caller decides what to do with them — typically
`ctran::transport::ib::impl::importRemoteInfo(*static_cast<ControlMsg*>(payload),
&remote)` for the ZC sender path, or `recvMsg.type ==
ControlMsgType::SYNC` checking for CB.

`testCtrlMsgDone(req, &done)` pumps `progress()` once and writes the
observed completion into `*done`. `waitCtrlMsgDone(req)` block-spins
on `testCtrlMsgDone` until the ctrl request completes — the
canonical usage is `iSendCtrlMsg/iRecvCtrlMsg` then immediately
`waitCtrlMsgDone`, since ctrl exchanges are 1–2 per op and not
performance-sensitive.

### CB-specific configuration

```cpp
void HostCbTransport::setKernelNumBlocks(
    int sendNumBlocks, int recvNumBlocks);
```

Sets the per-slot `GpeKernelSync.nworkers` for all send and / or
recv staging slots. Must be called before the first chunk issued
against a kernel that uses fewer blocks than the default
(`CTRAN_ALGO_MAX_THREAD_BLOCKS`). After updating, the per-slot
`postFlag` / `completeFlag` arrays are reset so the new
`[0, nworkers)` range is `kUnset`. Pass `<= 0` to leave that side
unchanged.

### Chunk addressing helpers

```cpp
int    computeTotalChunks(size_t totalSize) const;
size_t computeChunkOffset(int chunkIdx, size_t totalSize) const;
size_t computeChunkLen   (int chunkIdx, size_t totalSize) const;
```

ZC: returns `(1, 0, totalSize)`.
CB: returns the equivalent of `ceil(totalSize/chunkSize_)`,
`chunkIdx*chunkSize_`, `min(chunkSize_, totalSize - chunkIdx*chunkSize_)`.

### Issue-readiness — pure read, no progress()

```cpp
bool isReadyForSend(int vcIdx, int stagingSlot = kNoStagingSlot);
bool isReadyForRecv(int vcIdx, int stagingSlot = kNoStagingSlot);
```

ZC: `vcIdx ∈ [0, numVcs)` and `stagingSlot == kNoStagingSlot` ⇒ true.
CB: `vcIdx ∈ [0, numVcs)`, `stagingSlot ∈ [0, pipelineDepth)`, and
the slot is `IDLE`.

### Chunk-level data transfer

```cpp
commResult_t iSendChunk(const SendChunkArgs& args);
commResult_t iRecvChunk(const RecvChunkArgs& args);
commResult_t testChunkDone(const ChunkRequest& req, bool* done);
```

Single struct argument carries both common and mode-specific fields.
The active transport uses what its mode needs and asserts the
wrong-mode fields are at their defaults.

ZC requires `localMrHdl`, `remoteMrHdl`, `remoteKey` non-null and
`stagingSlot == kNoStagingSlot` for sends; recvs only need `vcIdx`.
CB requires `stagingSlot ∈ [0, pipelineDepth)` and the three
local/remote-info parameters to be `nullptr` (the destination + key
come from the one-time resource exchange).

`testChunkDone(req, &done)` pumps `progress()` once, then writes the
observed completion into `*done`. For ZC sends it forwards to
`req.ibReq.isComplete()`; for ZC recvs and CB it compares the
relevant counter against `req.mySeq`.

### Per-transport caller-must-lock

```cpp
void lock();
void unlock();
```

Every hot-path method on a concrete `IP2pHostTransport` asserts that
the calling thread first acquired this transport's mutex via
`lock()` (typically through the RAII guard `P2pTransportLockGuard`).
The check is always-on and aborts via `FB_CHECKABORT` when not held
— see the "Per-transport caller-lock contract" section under
"Concurrency model" below. Trivial accessors (`peerRank`, `mode`,
`pipelineDepth`, `chunkSize`, the `compute*` helpers) are exempt.

## Key Types

### ChunkRequest

```cpp
enum class ChunkKind : uint8_t { kInvalid = 0, kSend, kRecv };

struct ChunkRequest {
  ChunkKind kind{ChunkKind::kInvalid};
  int16_t  vcIdx{-1};
  int16_t  stagingSlot{kNoStagingSlot}; // kNoStagingSlot ⇔ ZC chunk
  uint64_t mySeq{0};                    // CB / ZC-recv: counter
  CtranIbRequest ibReq;                 // ZC-send only; CB / ZC-recv unused
};
```

Caller-owned. Written through the `req` field of `SendChunkArgs` /
`RecvChunkArgs`. Caller must keep it alive until `testChunkDone(req)`
returns true (for ZC send, the IB backend retains a pointer to the
embedded `ibReq`; for CB / ZC-recv, the embedded `ibReq` is unused
but the polling counters live on the transport).

### SendChunkArgs / RecvChunkArgs

```cpp
struct SendChunkArgs {
  // Common
  const void* userBuf{nullptr};
  size_t      offset{0};
  size_t      len{0};
  int         vcIdx{0};
  ChunkRequest* req{nullptr};   // optional output
  // CB-specific (ZC asserts these are at defaults)
  int            stagingSlot{kNoStagingSlot};
  int            round{0};
  SendChunkHooks hooks{};
  // ZC-specific (CB asserts these are nullptr)
  void* localMrHdl{nullptr};
  void* remoteMrHdl{nullptr};
  const CtranIbRemoteAccessKey* remoteKey{nullptr};
};

struct RecvChunkArgs {
  void* userBuf{nullptr};
  size_t offset{0};
  size_t len{0};
  int    vcIdx{0};
  ChunkRequest* req{nullptr};
  int            stagingSlot{kNoStagingSlot};
  int            round{0};
  RecvChunkHooks hooks{};
  // No ZC-specific recv args (data lands in the buffer the receiver
  // exported via iSendCtrlMsg).
};
```

Pass by const reference to `iSendChunk` / `iRecvChunk`. Designated
initializers make call sites self-documenting; defaults make
mode-irrelevant fields harmless to omit.

### SendChunkInfo / RecvChunkInfo (CB only)

Lives in `HostCbTransport.h`. Records the per-chunk addressing
+ hooks at issue time so the CB state machine can reconstruct
context on every progress pass.

```cpp
struct SendChunkInfo {
  const void* userBuf; size_t offset; size_t len;
  int vcIdx; int stagingSlot; uint64_t round;
  SendChunkHooks hooks;
};
struct RecvChunkInfo {
  void* userBuf; size_t offset; size_t len;
  int vcIdx; int stagingSlot; uint64_t round;
  RecvChunkHooks hooks;
};
```

### Send / Recv records (CB only)

```cpp
struct SendStagingRecord {
  SendChunkStatus status{SendChunkStatus::IDLE};
  CtranIbRequest iputReq;
  std::optional<SendChunkInfo> chunk;
  uint64_t issued{0};   // bumped on each iSendChunk acquiring this slot
  uint64_t completed{0};// bumped when state machine finishes this chunk
};
struct RecvStagingRecord {
  RecvChunkStatus status{RecvChunkStatus::IDLE};
  std::optional<RecvChunkInfo> chunk;
  uint64_t issued{0};
  uint64_t completed{0};
};
```

`pipelineDepth_` of each are stored as fixed-size arrays sized by the
compile-time `kMaxPipelineDepth = 8`.

### Per-VC ZC recv counters

```cpp
std::vector<uint64_t> vcRecvIssued_;
std::vector<uint64_t> vcRecvCompleted_;
```

Sized in the ctor to `numVcs()` once `impl::connectAndPopulateVcs`
has populated `vcs_`. Scalar `++` per recv-side issue; `++` per
drained notify in `pollRecvNotifications`.

### CB activity counters

`cbSendActive_` / `cbRecvActive_` count the number of CB
staging-slot records currently not in `IDLE`. Maintained by
`iSendChunk` / `iRecvChunk` (++ on issue) and the state machine
(-- on the final transition back to `IDLE`). When zero, the per-slot
state-machine pass is skipped entirely.

### CtrlRequest

Caller-owned poll handle for one ctrl-msg exchange (`iSendCtrlMsg` /
`iRecvCtrlMsg`). Pure inflight-request tracker — the wire-bytes
payload buffer is caller-owned and passed separately as a `void*`
to `iSendCtrlMsg` / `iRecvCtrlMsg`; CtrlRequest itself only holds
the underlying IB request needed to poll for completion.

```cpp
class CtrlRequest {
 public:
  bool isComplete() const { return complete_; }

 private:
  bool complete_{false};
  CtranIbRequest ctrlReq_;

  friend class ctran::transport::ib::HostZcTransport;
  friend class ctran::transport::ib::HostCbTransport;
};
```

There is no embedded `ControlMsg msg_` member — the bytes live in
the caller's buffer, which is passed to `iSendCtrlMsg` /
`iRecvCtrlMsg` as a `void*` and must outlive the request. There is
no `pendingOut_` either: the transport never auto-resolves a
`RemotePeerInfo`. That responsibility stays with the caller, who
reads from their own `ControlMsg` buffer and calls
`ctran::transport::ib::impl::importRemoteInfo` explicitly.

The caller writes through `&ctrlReq` on `iSendCtrlMsg` /
`iRecvCtrlMsg`, then polls with `testCtrlMsgDone(ctrlReq, &done)` or
block-waits with `waitCtrlMsgDone(ctrlReq)`. Lifetime: the caller
must keep both the request storage AND the payload buffer alive
until completion.

### HostTransportMode and RemotePeerInfo

```cpp
enum class HostTransportMode { kZeroCopy, kCopyBased };

struct RemotePeerInfo {
  bool isZeroCopy{false};
  void* memHdl{nullptr};                 // peer's exported recv buf (ZC only)
  CtranIbRemoteAccessKey remoteKey{};
};
```

`RemotePeerInfo` carries everything the sender needs to issue an
iput against the peer's recv buffer for the ZC path. CB ignores
`memHdl` / `remoteKey` because the destination is the peer's CB
staging buffer (set up via the one-time resource exchange) instead.

### ChunkContext

Hook context defined in `ChunkHooks.h`. Provided to every CB hook
call so hooks are stateless.

```cpp
struct ChunkContext {
  int slotIdx;            // staging buffer slot (0..D-1)
  int round;              // how many times this slot has been used
  size_t offset;          // byte offset into user buffer
  size_t len;             // chunk size
  void* stagingSlot;      // &staging[slotIdx * chunkSize]
  const void* userBuf;    // user buffer base pointer
  GpeKernelSync* sync;    // this chunk's per-chunk host↔device sync
  const uint64_t* remoteReady{nullptr};
  uint64_t* slotGeneration{nullptr};
  void (*signalSlotReady)(void*, int){nullptr};
  void* signalCtx{nullptr};
};
```

### SendChunkHooks / RecvChunkHooks

CB-only customization points. Each is a small struct of
`std::function`s the CB state machine calls at well-defined points.

```cpp
struct SendChunkHooks {
  std::function<void(ChunkContext&)>          prepareData;
  std::function<bool(ChunkContext&)>          isDataReady;
  std::function<const void*(ChunkContext&)>   getLocalSrc;
  std::function<void(ChunkContext&)>          onSendDone;
  std::function<bool(ChunkContext&)>          isRemoteReady;
};
struct RecvChunkHooks {
  std::function<void(ChunkContext&)> processData;
  std::function<bool(ChunkContext&)> isProcessDone;
  std::function<void(ChunkContext&)> onRecvDone;
  std::function<void(ChunkContext&)> signalReady;
};
```

**Built-in hook factories:**

- `makeCopyBasedSendHooks()` — host post / sync poll / staging-slot
  src / per-slot remoteReady credit. Used by every CB sender.
- `makeCopyBasedRecvHooks()` — host post / sync poll / per-slot
  ifetchAndAdd credit return. Used by every CB receiver.

Algorithms only override these for chained workflows. When the
caller leaves hooks empty, the CB transport substitutes the
canonical CB hooks; ZC ignores hooks entirely.

### HostTransportDev

Device-visible mirror used by the CB copy kernels. Holds per-slot
descriptors `(GpeKernelSync*, char* stagingSlot, size_t chunkSize)`
plus `pipelineDepth` and `chunkSize`. Built and `cudaMemcpy`'d to
device on first `getDeviceTransport()` call. Not used by ZC.

```cpp
struct DeviceChunkDesc {
  GpeKernelSync* sync; char* stagingSlot; size_t chunkSize;
};
struct HostTransportDev {
  DeviceChunkDesc sendChunks[kDeviceMaxPipelineDepth];
  DeviceChunkDesc recvChunks[kDeviceMaxPipelineDepth];
  int pipelineDepth; size_t chunkSize;
};
```

## Concrete transport class layouts

Each derived transport now owns its own copy of the shared per-peer
state (no base class). The layouts below correspond exactly to the
private/protected members in `HostZcTransport.h` and
`HostCbTransport.h`.

### HostZcTransport

```cpp
class HostZcTransport : public IP2pHostTransport {
  // ───── peer/runtime identity ─────
  int peerRank_;
  int myRank_;
  int cudaDev_;
  CtranIb* ctranIb_;

  // ───── per-peer VC vector (populated by the ctor) ─────
  // numVcs() returns vcs_.size(); vcs_ is the single source of truth.
  std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs_;

  // ───── pure-ZC recv-side counters (per VC) ─────
  // Sized in the ctor to numVcs(). Always bumped on every
  // iRecvChunk; vcRecvCompleted_ advances when notifies drain.
  std::vector<uint64_t> vcRecvIssued_;
  std::vector<uint64_t> vcRecvCompleted_;

  // ───── per-transport caller-must-lock mutex ─────
  // Algorithms must hold this (typically via P2pTransportLockGuard)
  // for the duration of every hot-path call into the transport — see
  // checkLocked() in IP2pHostTransport.h. Trivial accessors do not
  // require this lock.
  std::mutex transportMutex_;

  // ───── memtrace attribution (borrowed) ─────
  const CommLogData* logMetaData_{nullptr};
};
```

### HostCbTransport

```cpp
class HostCbTransport : public IP2pHostTransport {
  // ───── peer/runtime identity (was on the dropped base class) ─────
  int peerRank_;
  int myRank_;
  int cudaDev_;
  CtranIb* ctranIb_;

  // ───── per-peer VC vector (populated by the ctor) ─────
  std::vector<std::shared_ptr<CtranIbVirtualConn>> vcs_;

  // ───── CB pipeline geometry + borrowed pool ─────
  int                 pipelineDepth_;
  size_t              chunkSize_;
  const CommLogData*  logMetaData_{nullptr};
  GpeKernelSyncPool*  gpeKernelSyncPool_{nullptr};

  // ───── device-side mirror (lazy, getDeviceTransport()) ─────
  HostTransportDev*   devTransport_{nullptr};

  // ───── staging buffers (GPU, IB-registered) ─────
  char* sendStaging_{nullptr};
  char* recvStaging_{nullptr};
  void* sendStagingRegElem_{nullptr};
  void* recvStagingRegElem_{nullptr};

  // ───── per-slot GpeKernelSyncs (borrowed from gpeKernelSyncPool_) ─────
  std::vector<GpeKernelSync*> sendSyncs_;
  std::vector<GpeKernelSync*> recvSyncs_;

  // ───── per-slot flow-control counters (pinned host, IB-reg) ─────
  uint64_t* remoteReady_{nullptr};
  void*     remoteReadyRegElem_{nullptr};
  uint64_t  slotGeneration_[kMaxPipelineDepth]{};

  // ───── ifetchAndAdd return discard buf + per-slot atomic reqs ─────
  uint64_t*      fetchAddDiscardBuf_{nullptr};
  void*          fetchAddDiscardRegElem_{nullptr};
  CtranIbRequest atomicReqs_[kMaxPipelineDepth];

  // ───── one-time peer-resource exchange results ─────
  void*                   peerStagingBuf_{nullptr};
  CtranIbRemoteAccessKey  peerStagingKey_{};
  void*                   peerRemoteReadyAddr_{nullptr};
  CtranIbRemoteAccessKey  peerRemoteReadyKey_{};
  bool                    resourcesExchanged_{false};
  std::optional<ResourceExchangeState> resExchangeState_;

  // ───── CB staging-slot records (fixed-size arrays) ─────
  SendStagingRecord sendStagingSlots_[kMaxPipelineDepth];
  RecvStagingRecord recvStagingSlots_[kMaxPipelineDepth];

  // ───── active-slot counters (early-out for progress passes) ─────
  int cbSendActive_{0};
  int cbRecvActive_{0};
};
```

`HostCbTransport::ResourceExchangeState` (private inner type) carries
the in-flight resource-exchange wire state until both halves of the
handshake complete:

```cpp
struct ResourceExchangeMsg {
  ControlMsg staging;
  ControlMsg counter;
};
struct ResourceExchangeState {
  ResourceExchangeMsg outMsg{};
  ResourceExchangeMsg inMsg{};
  CtranIbRequest      sendReq{};
  CtranIbRequest      recvReq{};
};
```

Both classes also expose tiny accessors (`peerRank() / numVcs() /
myRank() / cudaDev() / mode() / pipelineDepth() / chunkSize()`); see
`HostZcTransport.h` and `HostCbTransport.h` for the full method list.

## Shared plumbing (HostTransportImpl)

Every operation that used to live on the deleted `P2pIbHostTransport`
base class now lives as a free function in
`ctran::transport::ib::impl` (the `impl` sub-namespace mirrors the
`Impl` suffix on the file name) in `HostTransportImpl.h`. Each
concrete transport member function (`progress`, `iSendCtrlMsg`,
`iRecvCtrlMsg`, `testCtrlMsgDone`, `waitCtrlMsgDone`) is a thin shim
that forwards to one of these helpers, passing in its own owned
state (`vcs_`, `ctranIb_`, etc.) by reference. All helpers are
defined `inline` in the header so every callsite sees the body.

A subset of the helpers — `exportRecvBuf`, `importRemoteInfo`,
`toIbRegElem` — is **public caller-facing API** (still in the `impl`
namespace; algorithm code refers to them as
`ctran::transport::ib::impl::exportRecvBuf` etc.): algorithm code
uses them to build / resolve the `ControlMsg` payloads that the
pure ctrl-msg `iSendCtrlMsg` / `iRecvCtrlMsg` primitives transport
on the wire.

### Helper signatures (`ctran::transport::ib::impl::*`)

```cpp
namespace ctran::transport::ib::impl {

// ─── Caller-facing helpers (used directly by algorithm code) ────────────

// RegElem* → ibRegElem* cast (single place the regcache abstraction
// is unwrapped).
void* toIbRegElem(void* memHdl);

// Build an IB_EXPORT_MEM ControlMsg describing the local recv buffer.
// ZC receivers call this and pass the result to iSendCtrlMsg().
commResult_t exportRecvBuf(
    void* localMrHdl,
    void* recvBuf,
    ControlMsg& outMsg);

// Resolve a received ControlMsg into a RemotePeerInfo
// (EXPORT_MEM ⇒ ZC peer, fills mr+key; SYNC ⇒ CB peer, all-zeros).
// ZC senders call this on the caller-owned ControlMsg buffer that
// they passed to iRecvCtrlMsg, AFTER waitCtrlMsgDone returns.
commResult_t importRemoteInfo(
    const ControlMsg& msg,
    RemotePeerInfo* out);

// ─── Internal helpers (used by HostZcTransport / HostCbTransport) ───────

// One-shot connectVcs() rendezvous + storage. Called exactly once
// per transport instance from its ctor. Aborts if invoked with an
// already-populated `vcs` (the helper is the single producer).
void connectAndPopulateVcs(
    CtranIb* ctranIb,
    int peerRank,
    int myRank,
    std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs);

// Defense-in-depth precondition for every hot-path access to vcs_.
// Aborts if vcs is empty OR vcIdx is out of range. Ctrl-msg paths
// pass kCtrlMsgVc explicitly to make the "VC 0 is the ctrl VC"
// assumption visible at the call site; chunk paths pass args.vcIdx.
void checkValidVc(
    const std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs,
    int vcIdx);

// Post `len` bytes from `payload` via vcs[kCtrlMsgVc]->isendCtrlMsg
// with wire header `type`. CtranIb retains a pointer to the
// caller-owned `payload` until the exchange completes; the caller
// must keep that buffer alive. `ctrlReq` is the caller-owned IB
// request slot (typically req->ctrlReq_, plucked out at the
// callsite by the friend transport class).
commResult_t iSendCtrlMsgImpl(
    std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs,
    ControlMsgType type,
    const void* payload,
    size_t len,
    CtranIbRequest& ctrlReq);

// Post matching irecvCtrlMsg via vcs[kCtrlMsgVc] into the
// caller-owned `payload` buffer of capacity `len`. After completion,
// the caller reads the bytes from its own buffer.
commResult_t iRecvCtrlMsgImpl(
    std::vector<std::shared_ptr<CtranIbVirtualConn>>& vcs,
    void* payload,
    size_t len,
    CtranIbRequest& ctrlReq);

// Pump progressFn once, then poll ctrlReq.isComplete(). Writes the
// observed completion into `complete` (sticky once set) and into
// *done. Pure ctrl-plane primitive — no remote-info resolution;
// that is the caller's responsibility via importRemoteInfo.
commResult_t testCtrlMsgDoneImpl(
    bool& complete,
    CtranIbRequest& ctrlReq,
    const std::function<commResult_t()>& progressFn,
    bool* done);
commResult_t waitCtrlMsgDoneImpl(
    bool& complete,
    CtranIbRequest& ctrlReq,
    const std::function<commResult_t()>& progressFn);

// Wraps ctranIb->progress() or the NIC-affine
// ctranIb->progress(device) overload.
commResult_t progressImpl(
    CtranIb* ctranIb,
    std::optional<int> nicHint = std::nullopt);

} // namespace ctran::transport::ib::impl
```

### `CtrlRequest` access — friendship is enough

`CtrlRequest` keeps its mutable members `private` and exposes only
the public `isComplete()` read. The two concrete transports
(`HostZcTransport`, `HostCbTransport`) are declared as `friend class`
on `CtrlRequest`, so they can pluck `complete_` / `ctrlReq_` out at
the callsite and pass them as primitive references to the helpers
below. The helpers themselves never see `CtrlRequest`'s shape —
they only take the `CtranIbRequest&` / `bool&` they actually mutate.

### How a concrete transport calls into the helpers

Both `HostZcTransport::iSendCtrlMsg` and `HostCbTransport::iSendCtrlMsg`
are identical thin shims — they don't build the payload themselves.
The caller does that BEFORE invoking the transport, and passes the
raw bytes by pointer:

```cpp
commResult_t HostZcTransport::iSendCtrlMsg(
    ControlMsgType type, const void* payload, size_t len, CtrlRequest* out) {
  FB_CHECKABORT(out != nullptr, "iSendCtrlMsg requires non-null out");
  ::ctran::transport::checkLocked(this);
  FB_COMMCHECK(checkEpochLock(ctranIb_));
  impl::checkValidVc(vcs_, impl::kCtrlMsgVc);
  // HostZcTransport is a friend of CtrlRequest, so it plucks the
  // primitive field out at the callsite. The helper itself only
  // sees CtranIbRequest&.
  return impl::iSendCtrlMsgImpl(vcs_, type, payload, len, out->ctrlReq_);
}
```

Callers build the payload themselves: ZC receivers via
`impl::exportRecvBuf(localMrHdl, recvBuf, msg)`, CB receivers via
`msg.setType(ControlMsgType::SYNC)`. Either way the buffer
(typically a local `ControlMsg`) must outlive the request.

`iRecvCtrlMsg` is similarly identical between the two concrete
classes: post the irecv into the caller-owned buffer.
`testCtrlMsgDone`, `waitCtrlMsgDone`, and `progress` follow the same
pattern — each concrete class forwards to an `impl::*` helper,
capturing `this->progress()` in a lambda where the helpers need a
progress callback:

```cpp
commResult_t HostZcTransport::testCtrlMsgDone(CtrlRequest& req, bool* done) {
  FB_CHECKABORT(done != nullptr, "testCtrlMsgDone requires non-null done");
  ::ctran::transport::checkLocked(this);
  return impl::testCtrlMsgDoneImpl(
      req.complete_, req.ctrlReq_, [this]() { return progress(); }, done);
}
```

`impl::connectAndPopulateVcs` is the single place that calls
`ctranIb_->connectVcs(peerRank_)`; the concrete transports invoke it
once from their ctor.

## memHdl and toIbRegElem

`memHdl` (the parameter type the algorithm side uses) is an opaque
handle into the registration cache (concretely a
`ctran::regcache::RegElem*`). The IB backend wants the inner
`ibRegElem` pointer. `ctran::transport::ib::impl::toIbRegElem(memHdl)`
does that cast — it is the single place where the layered
registration abstraction is unwrapped, so callers never have to
include `RegCache.h` to use the transport.

`ctran::transport::ib::impl::exportRecvBuf(localMrHdl, recvBuf, msg)`
calls `toIbRegElem` when building the IB_EXPORT_MEM ctrl payload. ZC
`iSendChunk` also calls `toIbRegElem` on `args.localMrHdl` to obtain
the IB regElem the iput needs as its source registration.

## Per-VC dispatch

Both transports dispatch per-chunk operations directly on the chosen
VC under the VC's per-object lock:

```cpp
CTRAN_IB_PER_OBJ_LOCK_GUARD(vcs_[vcIdx]->mutex, {
  vcs_[vcIdx]->iput(src, dst, len, regElem, remKey,
                    /*notify=*/true, /*config=*/nullptr,
                    rawReq, /*fast=*/false);
});
```

Notes:

- The shared ctrl helpers (`iSendCtrlMsgImpl`, `iRecvCtrlMsgImpl` in
  `ctran::transport::ib::impl`) use `vcs_[impl::kCtrlMsgVc]` (== 0)
  for ctrl exchange. The same applies to CB's resource exchange.
  The algorithm picks `vcIdx ∈ [0, numVcs)` for chunk dispatch.
- For ZC sends, `rawReq` is `&args.req->ibReq` (caller-owned) when
  the caller passed a non-null `args.req`, else `nullptr`
  (fire-and-forget).
- For CB receives, `pollRecvNotifications` calls
  `vcs_[i]->checkNotify` only when there is at least one
  `WAIT_RECV` record on that VC — unmatched notifies stay
  accumulated for a future `progress()` pass.

## Key invariants

- **Mode is bound at cache time.** A given `IP2pHostTransport*` only
  ever runs one mode. Re-asking the mapper for the same peer with a
  different mode is a programming error and aborts.
- **`progress()` is required.** `iSendChunk` / `iRecvChunk` issue
  state but do not advance it. Callers must invoke `progress()`
  between issue and `testChunkDone` polls. ZC sends are an exception
  — the caller's `req.ibReq.isComplete()` is signalled by
  `ctranIb_->progress()` (which `progress()` invokes via
  `progressImpl`), so the caller still ends up calling
  `progress()` to drain CQEs.
- **`testChunkDone` is a pure read** — does **not** call
  `progress()`. Calling it repeatedly without `progress()` will spin.
- **`isReadyForSend` / `isReadyForRecv` never call progress.**
  Callers sequence them explicitly.
- **Caller owns `*args.req`.** If the caller passes a non-null
  `args.req`, the storage must outlive the iput's CQE / notify. For
  vectors of `ChunkRequest`, pre-`reserve()` to avoid relocation
  (the IB backend retains a pointer to `req->ibReq` for ZC sends).
- **Within a `(vcIdx, stagingSlot)` pair, completion order matches
  issue order.** Across different pairs, no order is implied.
- **CB allocations go through `comms/ctran/utils/Alloc.h`.** Every
  GPU buffer the CB transport owns is allocated via
  `commCudaMalloc` / `commCudaHostAlloc` with the borrowed
  `CommLogData*`.
- **CB GpeKernelSyncs are borrowed.** The pool is owned by
  `CtranGpe`; the CB transport pops on construction and returns
  via `reset()` in its destructor. The mapper guarantees the
  destruction order: `setAtDestruction()` clears the per-peer
  cache before `gpe` is torn down.
- **`numVcs` is non-zero after construction.** A peer with zero VCs
  is a configuration error (`maxVcsPerPeer < 1` from the cvars
  `NCCL_CTRAN_IB_MAX_QPS` / `NCCL_CTRAN_IB_NUM_VCS_PER_RANK` /
  `NCCL_CTRAN_IB_DEVICES_PER_RANK`) — `impl::connectAndPopulateVcs`
  aborts in the ctor in that case.
- **CB `nworkers` matches the kernel's launch geometry.** Either the
  default (`CTRAN_ALGO_MAX_THREAD_BLOCKS`) is correct, or the caller
  calls `setKernelNumBlocks(sendN, recvN)` once after the ctrl
  exchange and before the first chunk issue.
- **ZC recv counters bump on every `iRecvChunk`.** Even when
  `args.req == nullptr`, `vcRecvIssued_` is incremented because
  notifies are FIFO-delivered and a later tracked recv's `mySeq`
  depends on earlier notifies being drained.

## Concurrency model

The transport surface has two layered locks. Every public hot-path
method on a concrete `IP2pHostTransport` enforces both at entry.

### 1. Per-transport caller-must-lock

Each `IP2pHostTransport` instance owns a `std::mutex transportMutex_`
exposed through the abstract interface:

```cpp
virtual void lock() = 0;
virtual void unlock() = 0;
```

Callers acquire it via the RAII guard
`ctran::transport::P2pTransportLockGuard guard(transport);` (modeled
on `CtranMapperEpochRAII`). The concrete `HostZcTransport::lock` /
`unlock` mirror the `CtranIb::epochLock` / `epochUnlock` pattern at
`CtranIb.cc:741-794`:

- `lock()` aborts if the calling thread already holds the lock
  (double-lock-from-same-thread), then takes `transportMutex_` and
  sets a `thread_local` flag for `this`.
- `unlock()` aborts if the flag is not set (unlock-without-lock),
  clears the flag, and releases the mutex.

The free function
`ctran::transport::checkLocked(IP2pHostTransport*)` consults that
`thread_local` flag and aborts via `FB_CHECKABORT` if the calling
thread is not the current owner. **The check is always-on — there
is no cvar gate.** Every public hot-path method on
`HostZcTransport` (and any future concrete transport) calls
`checkLocked(this)` immediately after its argument-null guards:
`progress`, `iSendCtrlMsg`, `iRecvCtrlMsg`, `testCtrlMsgDone`,
`waitCtrlMsgDone`, `iSendChunk`, `iRecvChunk`, `testChunkDone`,
`isReadyForSend`, `isReadyForRecv`.

Trivial accessors (`peerRank`, `mode`, `pipelineDepth`, `chunkSize`,
the `compute*` helpers) do NOT take the lock check — they are
read-only and lock-free, and may be called by setup code before
locking.

The thread-local flag table (`impl::p2pTransportLockedFlags`) lives
in `IP2pHostTransport.h` as an `inline thread_local`
`std::unordered_map<void*, std::atomic_bool>`, so the storage is
header-only and shared across translation units.

### 2. Layering on top of `CtranIb`'s epoch lock

The per-transport lock is the finer-grained guard for one peer's
`IP2pHostTransport*`. The existing `checkEpochLock(ctranIb_)` is
the coarser guard for the whole `CtranIb` instance and is required
on hot-path methods that directly issue IB primitives
(`iSendCtrlMsg`, `iRecvCtrlMsg`, `iSendChunk`). Callers must hold
both: the per-transport lock around the call and the epoch lock
around the surrounding critical section. Methods whose body does
not directly issue an IB primitive (`iRecvChunk` — counter bump
only; `testChunkDone` / `testCtrlMsgDone` / `waitCtrlMsgDone` —
which call `progress()` internally, and `progress()` itself —
which routes through `progressImpl → ctranIb->progress()` and
invokes `checkEpochLock` there) inherit the check transitively.

### 3. `CtranMapper::getP2pTransport` is serialized

`CtranMapper` caches transports in
`folly::Synchronized<std::unordered_map<int,
std::unique_ptr<IP2pHostTransport>>>`. `getP2pTransport` takes
`wlock()` for the entire function body, including the
`HostZcTransport` ctor (which drives the two-sided
`connectVcs(peer)` rendezvous). The cache is grow-only — entries
are never erased or rewritten except during teardown, so a raw
`IP2pHostTransport*` returned to a caller stays valid across
subsequent `getP2pTransport` calls.

## Testing

- `comms/ctran/transport/tests/CtranMapperHostTransportUT.cc` —
  exercises the mapper-owned cache: lazy construction per
  `(peer, mode)`, reuse, mode-mismatch abort, and the
  destruction-order guarantee that
  `gpe->numInUseGpeKernelSyncs() == 0` after `setAtDestruction()`.
- `comms/ctran/transport/ib/tests/HostTransportDistUT.cc` — end-to-
  end ZC RoundTrip, CB RoundTrip, and CB BidirectionalRoundTrip
  through real IB HCAs. The chunk-level driver is **mode-
  agnostic** — one `runChunkLoop` template body drives both modes,
  filling the unified `SendChunkArgs` / `RecvChunkArgs` struct with
  whatever fields apply. Every test wraps its transport in a
  `P2pTransportLockGuard` for the duration of the hot-path calls.
- `comms/ctran/transport/ib/tests/ChunkHooksTest.cc` — unit tests
  for the CB hook factories (`makeCopyBasedSendHooks` /
  `makeCopyBasedRecvHooks`) and the `ChunkContext` shape.
- `comms/ctran/transport/tests/MockP2pHostTransport.h` — gMock
  implementation of `IP2pHostTransport` for testing algorithms in
  isolation, against the same struct-arg API. `lock()` / `unlock()`
  are provided as no-op overrides.
