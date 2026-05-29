# CtranIb Multi-Channel Support via Per-Peer VC Management — Design

## 1. Overview

This document describes how `CtranIb` supports **multi-channel
communication** for its consumers (`p2pHostIbTransport`, RMA, ctran
algos) by managing a **vector of per-peer Virtual Connections (VCs)**.
The "channel" concept itself lives entirely in the **algo / RMA**
layer: the algo decides how many parallel logical channels it wants
and picks a VC per chunk. CtranIb owns the per-NIC IB resources (one
VC group per NIC) and exposes the per-peer VC vector via
`connectVcs(peer)`. `p2pHostIbTransport` is just the dispatch layer in
between — it caches the per-peer VC vector and forwards each chunk's
operations to `vcs[vcIdx]`, where `vcIdx` is supplied by the caller
per chunk.

**Why multiple VCs per peer?**

- **Multiple independent logical connections per peer.** Each VC owns
  its own ctrl/notify/atomic QPs, so each VC has its own ordering
  domain and its own notify stream. A slow consumer on one VC cannot
  back-pressure another VC to the same peer.
- **NIC-pinned VCs to avoid cross-NIC `iflush`.** Each VC's
  ctrl/notify/atomic + data QPs all sit on a single NIC. When a logical
  stream uses one VC, all of that stream's puts and its notify travel
  through one NIC's PCIe path, so the receiver gets the per-NIC
  ordering between data and notify that NIC hardware already provides
  — no cross-NIC RDMA-read flush required.

### Adoption flow — user → CtranIb

```
  +---------------------------------------------------------------+
  |  Algorithm / RMA layer (ctranAlgo, RMA collectives)           |
  |                                                               |
  |  Owns its own "logical channel" concept (if any). For each    |
  |  chunk picks vcIdx (and stagingSlot for CB) and issues:       |
  |     transport->iSendChunk(buf, off, len, config,              |
  |                           vcIdx, stagingSlot)                 |
  |     transport->iRecvChunk(buf, off, len, config,              |
  |                           vcIdx, stagingSlot)                 |
  |  Returns ChunkRequest; polled via testChunkDone(request).      |
  |  Common convention: vcIdx = channelId % numVcs (channel ↔ VC  |
  |  binding lives in the algo).                                  |
  +---------------------------------------------------------------+
                                |
                                |  per-chunk vcIdx (+ stagingSlot for CB)
                                v
  +---------------------------------------------------------------+
  |  p2pHostIbTransport      (one instance per peer)              |
  |                                                               |
  |  - Caches the per-peer VC vector returned by                  |
  |       ctranIb->connectVcs(peerRank)                               |
  |  - Owns a CB staging-slot pool of size pipelineDepth and a    |
  |    ZC chunk-record freelist.                                  |
  |  - For each chunk, dispatches data / ctrl / notify / atomic   |
  |    to vcs_[vcIdx] (vcIdx supplied by the caller).             |
  |  - No channel concept. No channel ↔ VC mapping. No queues.    |
  +---------------------------------------------------------------+
                                |
                                |  per-VC iput / isendCtrlMsg /
                                |  notify / checkNotify / ifetchAndAdd
                                v
  +---------------------------------------------------------------+
  |  CtranIb                 (one per comm)                       |
  |                                                               |
  |  - Bootstraps and owns the per-peer VC vector                 |
  |  - Owns per-NIC VCG[d] = { ib_device d, cq[d],                |
  |                            every per-peer VC pinned to NIC d }|
  |  - progress() polls each VCG's CQ and dispatches CQEs to      |
  |    the owning VC                                              |
  +---------------------------------------------------------------+
```

**Channel ↔ VC binding lives in the algo / RMA layer.** The algo
picks `vcIdx` per chunk and passes it through
`transport->iSendChunk(..., vcIdx, stagingSlot)`. A common
convention is `vcIdx = channelId % numVcs` so each logical channel
gets its own VC's ctrl/notify/atomic ordering domain and pins data
+ notify to one NIC. The transport does not enforce or interpret
any channel relationship — it just dispatches to `vcs_[vcIdx]`.

**No hard `numChannels <= maxVcsPerPeer` constraint.** The caller is
free to oversubscribe, picking a `vcIdx` that is shared across
multiple algo channels. The transport will dispatch correctly. The
tradeoff is that channels sharing a VC lose the per-channel
isolation properties that the per-VC design otherwise provides:

- A slow consumer on one channel will back-pressure another channel
  sharing the same VC's notify QP.
- Per-channel ctrl-message ordering folds together on that VC.
- Sender notifications on the shared VC arrive on a single notify
  stream and the algo must demultiplex them.

If a workload prefers full per-channel isolation, the operator can
grow `maxVcsPerPeer` by increasing `NCCL_CTRAN_IB_NUM_VCS_PER_RANK`
so that `numChannels <= numVcs` becomes achievable; both endpoints
must agree on the cvars (checked at first rendezvous via the
bus-card-swap loop count).

### Responsibilities — quick reference

| Layer | Owns | Calls into | Knows about |
|---|---|---|---|
| Algo / RMA | "logical channel" concept; channel ↔ VC binding; per-chunk `vcIdx` (and `stagingSlot` for CB) | `transport->iSendChunk(…, vcIdx, stagingSlot)` / `iRecvChunk(…, vcIdx, stagingSlot)`; `testChunkDone(request)` | "logical channel", "staging slot", "VC", "chunk request" |
| `p2pHostIbTransport` | CB staging-slot pool, ZC chunk-record freelist, per-VC dispatch, ctrl exchange on `vcs_[0]` | `ctranIb_->connectVcs`, `vcs_[i]->X(…)`, `ctranIb_->progress` | "chunk, staging slot, ZC record, VC" |
| `CtranIb` | per-peer VC vector, per-NIC VCG (device + CQ + QPs), bootstrap | per-VC `processCqe` | "per-peer VC + per-NIC VCG" |

`CtranIb` itself has three jobs in this layer cake:

1. **Bootstrap per-peer Virtual Connections (VCs).** A VC
   (`CtranIbVirtualConn`) is a single IB-connected, NIC-pinned bundle of
   QPs (one ctrl/notify/atomic triple plus a small data-QP pool) shared
   between this rank and a peer.
2. **Own per-NIC IB resources.** Exactly one Completion Queue (CQ) per
   physical IB device, the device handles, the listen thread, and the
   epoch lock all live inside `CtranIb`.
3. **Expose its per-peer VCs to consumers.** The primary consumer is
   `p2pHostIbTransport`, which obtains the per-peer VC vector from
   `connectVcs(peerRank)` and then calls `vc->iput(...)` /
   `vc->isendCtrlMsg(...)` etc. directly under `vc->mutex`. Algorithms
   that need a single per-peer VC keep using the legacy single-VC API
   (`getVc`, `iput`, …) — that surface is unchanged. `getVc(peer)` is
   now just a thin accessor that returns `rankToVcs[peer].front()` (see
   § 4), so legacy callers transparently land on `vcs[0]` of the same
   per-peer vector that `connectVcs` exposes.

The "channel" abstraction is **not** part of CtranIb's contract, and
it is **not** part of `p2pHostIbTransport`'s contract either. A
**channel is an algo / RMA concept**, owned entirely by the layer
above the transport. The algo decides how many channels it wants and
picks the VC for each chunk; `p2pHostIbTransport` just dispatches the
chunk's ops to `vcs_[vcIdx]`; CtranIb only knows about VCs.

**Channel ↔ VC binding is the algo's choice.** When the algo decides
it wants `C` channels to a peer, it asks the transport (which in
turn asks CtranIb) for the per-peer VC vector (sized
`maxVcsPerPeer`, see § 4) and decides per chunk which `vcIdx` to
pass. The natural convention is `vcIdx = channelId % numVcs` so
that:

- if `numChannels <= numVcs`, each channel gets its own VC and the
  full per-channel isolation properties (ordering domain, notify
  stream, NIC-local data+notify ordering);
- if `numChannels > numVcs`, multiple channels share a VC and the
  algo accepts the loss of isolation between those channels.

There is no remapping table on the CtranIb side, no `imm_data` tag
bits in the wire format, and no channel-aware code on the transport
side; the channel-to-VC binding is purely the algo's choice of
`vcIdx` per chunk. Operators who want to grow the available VC count
increase `NCCL_CTRAN_IB_NUM_VCS_PER_RANK` (and ensure
`NCCL_CTRAN_IB_MAX_QPS` is large enough to give each VC at least one
data QP).

## 2. Resource model — the VC group (VCG)

> The remainder of this document describes **CtranIb implementation
> details** of the multi-channel VC management. For
> `p2pHostIbTransport` implementation details (chunk pipeline, channel
> queues, dispatch, hooks, and risks), see
> `comms/ctran/transport/ib/p2pIBHostTransportDesign.md`.

A **VC group (VCG)** is the per-NIC bundle of resources CtranIb owns
for one IB device:

```
VCG[d] = {
  the IB device handle at devices[d],
  the Completion Queue cq[d]   // owned by VCG[d]
  every per-peer VC whose activeDevices contains d,
}
```

**Structural diagram** (one CtranIb instance, two NICs, two peer ranks
`P` and `Q`, `numVcsPerPeer = 4` → 2 VCs per NIC; per-VC data-QP count
is resolved per connection class — with the default cvar path and
`NCCL_CTRAN_IB_MAX_QPS = 8`, each VC ends up with 2 data QPs). Each VCG
owns its own NIC, its own CQ, and every per-peer VC pinned to that NIC.
Per-peer VCs from different peers cohabit the same VCG (their QPs share
the VCG's CQ).

```
                                CtranIb
+----------------------------------------------------------------------+
|                                                                      |
|  +---------------------------- VCG[0] ----------------------------+  |
|  |                                                                |  |
|  |   +-----------------+         cq[0]                            |  |
|  |   |  ib_device 0    |   (one CQ owned by VCG[0])               |  |
|  |   +-----------------+                                          |  |
|  |                                                                |  |
|  |   +-----------------------+    +-----------------------+       |  |
|  |   |  peer-P VC #0         |    |  peer-Q VC #0         |       |  |
|  |   |  activeDevices = [0]  |    |  activeDevices = [0]  |       |  |
|  |   |  ctrl/notify/atom QPs |    |  ctrl/notify/atom QPs |       |  |
|  |   |  data: [qp0, qp1]     |    |  data: [qp0, qp1]     |       |  |
|  |   +-----------------------+    +-----------------------+       |  |
|  +----------------------------------------------------------------+  |
|                                                                      |
|  +---------------------------- VCG[1] ----------------------------+  |
|  |                                                                |  |
|  |   +-----------------+         cq[1]                            |  |
|  |   |  ib_device 1    |   (one CQ owned by VCG[1])               |  |
|  |   +-----------------+                                          |  |
|  |                                                                |  |
|  |   +-----------------------+    +-----------------------+       |  |
|  |   |  peer-P VC #1         |    |  peer-Q VC #1         |       |  |
|  |   |  activeDevices = [1]  |    |  activeDevices = [1]  |       |  |
|  |   |  ctrl/notify/atom QPs |    |  ctrl/notify/atom QPs |       |  |
|  |   |  data: [qp0, qp1]     |    |  data: [qp0, qp1]     |       |  |
|  |   +-----------------------+    +-----------------------+       |  |
|  +----------------------------------------------------------------+  |
|                                                                      |
+----------------------------------------------------------------------+
```

Key properties of the diagram:

- VCG[d] owns NIC `d`, its CQ `cq[d]`, and every per-peer VC whose
  `activeDevices` contains `d`. Per-peer VCs from different peers (here
  `P` and `Q`) share the same VCG — all of their QPs land on the same
  `cq[d]`.
- Every QP is inserted into the single `qpToVcMap`, keyed by
  `(qpn, device)`. Progress dispatch is uniform — `progressInternal`
  polls each VCG's `cq[d]`, looks up the owning VC by `(qpn, d)`, and
  dispatches.
- Because each VC's data and notify travel through one NIC, the
  receiver of a notify on VC `i` does not need to flush across other
  NICs to make VC `i`'s data visible.

**Today's storage layout vs. the VCG concept.** The fields on
`CtranIb` are still flat — `devices[d]`, `cqs[d]` are arrays indexed
by NIC, and per-peer VCs are in a `peer → vector<VC>` map that is
*not* sliced by NIC. "VCG[d]" today is the **logical projection** of
all of those by NIC `d`. Phase 2 (§ 11) promotes `VcGroup` to a real
C++ struct that co-locates the per-NIC device + CQ + per-peer VC
projections, and exposes a per-VCG `progress()` so consumers can drive
one VCG independently of the others.

## 3. Per-peer VCs

A VC is a `CtranIbVirtualConn` instance whose ctrl/notify/atomic + data
QPs all live on a precomputed set of NICs (`activeDevices`). The number
of VCs per peer and the per-VC `activeDevices` set are derived by
CtranIb at init from cvars via the `ctran::ib::VcLayout(numNics,
maxVcsPerPeer)` ctor:

```
numNics       = NCCL_CTRAN_IB_DEVICES_PER_RANK
maxQps        = NCCL_CTRAN_IB_MAX_QPS
maxVcsPerPeer = NCCL_CTRAN_IB_NUM_VCS_PER_RANK   // normalised to 1 if <= 0
maxVcsPerNic  = vcLayout.maxVcsPerNic            // see below
```

`vcLayout_.vcToActiveDevices[v]` gives the NIC indices that VC `v`
owns. Two layout regimes are supported:

- **Pinned** (`maxVcsPerPeer >= numNics` and divisible): each NIC owns
  `maxVcsPerNic = maxVcsPerPeer / numNics` VCs (NIC-major); each VC's
  `activeDevices` is a single NIC. The legacy single-VC mode
  (`maxVcsPerPeer == 1`, `numNics == 1`) is the degenerate case.
- **Striped** (`maxVcsPerPeer < numNics` and `numNics % maxVcsPerPeer
  == 0`): `maxVcsPerNic = 1`; each VC's `activeDevices` is the
  contiguous range `[v*K, v*K+K-1]` with `K = numNics /
  maxVcsPerPeer`. The legacy multi-NIC single-VC mode
  (`maxVcsPerPeer == 1`, `numNics > 1`) is this regime's `K = numNics`
  case: one VC with `activeDevices = [0..numNics-1]`.

Any other `(numNics, maxVcsPerPeer)` pair is rejected at init by the
`VcLayout` ctor with `commInvalidArgument`.

**Per-VC data-QP count is computed at bootstrap by
`CtranIb::Bootstrap` via the static helper
`CtranIbVirtualConn::computeMaxQpsPerVc(comm, peer, numVcs)`** and passed
to the VC ctor as a plain `maxQpsPerVc` int. The helper resolves the
per-peer MAX_QPS for the peer's connection class (cvar default,
optionally overridden by
`NCCL_CTRAN_IB_QP_CONFIG_XRACK / XZONE / XDC / NCCL_CTRAN_EX_IB_QP_CONFIG`)
and divides evenly across `numVcs` (must divide exactly; checked with
`FB_CHECKABORT`). The VC itself has **no notion of "vcs per peer"** —
it just stores the per-VC budget and clamps / rounds it to the active
device count inside `setDefaultQPConfig`. For the default cvar path:

```
maxQpsPerVc = NCCL_CTRAN_IB_MAX_QPS / numVcs
```

For a peer that hits a connection-class override, `maxQpsPerVc` is the
overridden MAX_QPS divided by `numVcs`. When `numVcs == 1` this is a
no-op and the VC owns the full MAX_QPS.

Every peer for which the consumer calls `connectVcs(peerRank)` gets the
same `maxVcsPerPeer` VCs. The consumer treats its own channel id `c`
as a 1:1 index into this vector — channel `c` is `vcs[c]`. If the
consumer wants more channels than CtranIb provides, the operator
increases `NCCL_CTRAN_IB_NUM_VCS_PER_RANK` (and ensures
`NCCL_CTRAN_IB_MAX_QPS` is large enough to give each VC at least one
data QP); both endpoints must agree on the cvars (checked at first
rendezvous via the bus-card-swap loop count).

**Per-peer VC vector layout** (pinned example: `maxVcsPerPeer = 4`,
`numNics = 2`, `maxVcsPerNic = 2`):

```
vcs[0]                 vcs[1]                vcs[2]                 vcs[3]
└─ NIC 0 (VCG[0]) ──┘  └─ NIC 0 (VCG[0]) ─┘  └─ NIC 1 (VCG[1]) ──┘  └─ NIC 1 (VCG[1]) ─┘
```

Striped example: `maxVcsPerPeer = 2`, `numNics = 4` — each VC spans 2
contiguous NICs (`activeDevices = [0,1]` and `[2,3]` respectively).
Legacy single-VC, multi-NIC mode (`maxVcsPerPeer = 1`, `numNics > 1`)
collapses to one VC with `activeDevices = [0..numNics-1]`.

The consumer sees a flat `vcs_` vector and indexes into it by the
`vcIdx` supplied by the caller. CtranIb owns the layout, the QP
allocation, and the per-peer cardinality; the consumer owns the
channel-level routing decision.

Each VC owns its own `ibvControlQp_`, `ibvNotifyQp_`, `ibvAtomicQp_`,
and `ibvDataQps_`, all on the NICs listed in `activeDevices_`. The
first entry of `activeDevices_` (`ctrlDevice_`) hosts the
ctrl/notify/atomic triple; data QPs are distributed across all entries.
There is no shared ctrl/notify/atomic state across VCs.

## 4. Public API

### Per-peer VC entry point (the one and only data-path entry point)

```cpp
const std::vector<std::shared_ptr<CtranIbVirtualConn>>&
    connectVcs(int peerRank,
           std::optional<const SocketServerAddr*> peerAddr = std::nullopt);
```

- The number of returned VCs is `maxVcsPerPeer`, derived by CtranIb at
  init from cvars (§ 3). The consumer does not pass it. Both endpoints
  must run with matching `NCCL_CTRAN_IB_MAX_QPS`,
  `NCCL_CTRAN_IB_NUM_VCS_PER_RANK`, and `NCCL_CTRAN_IB_DEVICES_PER_RANK`;
  the bus-card-swap loop count carries the implied `numVcs` and the
  rendezvous aborts on mismatch.
- `peerAddr` is required on the smaller-rank side (it drives the TCP
  connect); the larger-rank side passes `std::nullopt` and the listen
  thread handles the accept.

**`connectVcs` is a blocking, init-once-per-peer API:**

- **Blocking on first call.** The call returns only after the per-peer
  rendezvous (TCP connect → bus-card swap loop → publish) has
  completed on both sides. The smaller-rank side does the work
  synchronously inside the call; the larger-rank side spins on
  `vcState_.tryGetVcs(peer)` (with `std::this_thread::yield()` and
  an `abortCtrl_->Test()` check) until the CtranIb listen thread
  publishes the vector. There is no async / future-returning variant.
- **Single-thread-per-peer contract.** `connectVcs(peer, ...)` is NOT
  safe to call concurrently from multiple threads for the same
  `peer`. Callers (algos / RMA / `p2pHostIbTransport`) must serialize
  per-peer invocations — e.g. call `connectVcs(peer)` once during
  connection setup and cache the returned vector. Concurrent calls
  for *different* peers are safe.
- **Deadlock-freedom despite blocking.** `connectVcs` is collective-style:
  every rank that participates in a transfer with peer `P` must
  eventually call `connectVcs(P, ...)` on both ends. The only blocking
  site is the larger-rank spinner, which polls
  `vcState_.tryGetVcs(peer)` until the listen thread observes the
  smaller-rank initiator's bootstrap socket and publishes the VC
  vector. The spin also honors `abortCtrl_`, so `releaseAll()` /
  explicit abort unblocks any spinner. For a deadlock to exist,
  every rank in some cycle would have to be blocked in `connectVcs`
  with no smaller-rank initiator running — which is impossible: in
  any pair `(a, b)` with `a < b`, rank `a` is by definition the
  smaller-rank side and runs the non-blocking initiator path,
  driving rank `b`'s listen thread to publish and unblock `b`'s
  spinner. Because this argument holds for every pair independently,
  `connectVcs` calls from a single rank can be issued in any order
  across peers without forming a wait-for cycle. See `CtranIb.h`
  for the full comment.
- **Init once, then cached.** Once published, the per-peer VC vector
  is cached in `vcStateMaps.rankToVcs[peer]`. Every subsequent call
  to `connectVcs(peer)` from either side returns the same cached
  `std::vector<std::shared_ptr<…>>` immediately — no TCP, no swap,
  no QP transition, no spin. The vector is shared across consumers:
  `p2pHostIbTransport`, RMA, algos all see the same `shared_ptr`s.
- **Single source of truth — fully backward compatible with
  `getVc`.** `vcStateMaps.rankToVcs[peer]` is the only per-peer VC
  storage. The legacy `getVc(peer)` is now a thin accessor that
  returns `rankToVcs[peer].front()` (i.e. `vcs[0]`); it does not
  maintain a separate map and does not own a separate VC. The
  bootstrap path is unified too: `Bootstrap::connect` (and the
  accept-side handshake) always loops over
  `getMaxVcsPerPeer()` VCs (always `>= 1` post-normalization, equal to
  `vcLayout_.maxVcsPerPeer`), so a single rendezvous publishes the
  full vector and the very first call from either entry point
  bootstraps it. Backward compatibility is preserved because legacy
  callers (`CtranMapper`, FTAR, the existing collective algos) only
  ever observe `vcs[0]`, which behaves exactly like the previous
  single per-peer VC: when `maxVcsPerPeer == 1` (the default), the VC
  spans all NICs (`activeDevices = [0..numNics-1]`) and is functionally
  the legacy non-pinned VC; when `maxVcsPerPeer > 1`, it is the
  NIC-0-pinned VC of the new vector. Either way, `qpToVcMap` is shared
  across all VCs for uniform CQE dispatch (§ 6).

### Bootstrap (rendezvous) — first `connectVcs(peer)` call only

Between two ranks `A < B`, both endpoints implicitly agree on
`numVcs == getMaxVcsPerPeer() == vcLayout_.maxVcsPerPeer` (always `>= 1`,
derived from cvars at init time):

- **Smaller rank A** runs the work synchronously inside `connectVcs`:
  resolves `peerAddr`, opens a single TCP connection to B's listen
  address, sends a magic + own-rank header, then runs `numVcs`
  bus-card swaps in vc-id order over that one socket. The bus card
  carries the VC's QP numbers and port info; the VC index is implicit
  by position. The swap-loop count itself is the cvar agreement —
  if A and B disagree the loop terminates early on one side and the
  rendezvous fails. After the swaps, A constructs the per-peer VC
  vector (each VC built with `activeDevices =
  vcLayout_.vcToActiveDevices[vcId]` and a `maxQpsPerVc` budget that
  `Bootstrap` computes once per peer via
  `CtranIbVirtualConn::computeMaxQpsPerVc(comm, peer, numVcs)`),
  inserts the QPs into `qpToVcMap`, and publishes the vector into
  `vcStateMaps.rankToVcs[B]`.
- **Larger rank B**: the CtranIb listen thread accepts the TCP
  connection from A, matches the magic + rank, runs the matching
  accept side of the bus-card swap loop, then publishes the resulting
  VC vector into `vcStateMaps.rankToVcs[A]`. The application thread
  that called `connectVcs(A)` is spinning on `tryGetVcs(A)` (with a
  yield and an abort check) and returns the cached vector as soon
  as the listen thread publishes.

The larger-rank spin is intentionally simple — the same pattern as
`preConnect` — and avoids per-peer waiter bookkeeping. `releaseAll()`
clears `rankToVcs` and the spin loop's `abortCtrl_->Test()` check is
what lets in-flight spinners exit cleanly during teardown.

After `connectVcs` returns, the consumer holds `shared_ptr`s to every VC and
calls `vc->iput(...)`, `vc->isendCtrlMsg(...)`, etc. directly **under
`vc->mutex`**. CtranIb does not wrap these per-VC calls.

## 5. VC public methods consumers call

After `connectVcs` returns, consumers dispatch directly to the VC. The
public methods are declared in `CtranIbVc.h`:

```cpp
class CtranIbVirtualConn {
 public:
  CtranIbRequest* iput(const void* buf, ..., bool notify, ...);
  CtranIbRequest* iget(...);
  CtranIbRequest* isendCtrlMsg(const void* buf, size_t len);
  CtranIbRequest* irecvCtrlMsg(void* buf, size_t* len);
  CtranIbRequest* ifetchAndAdd(...);
  CtranIbRequest* iatomicSet(...);
  void   notify(...);
  bool   checkNotify(uint64_t* notified);
  bool   checkNotifies(...);
  // ... see CtranIbVc.h for full list
  std::mutex mutex;
};
```

**Thread-safety contract:**

> Virtual-connection functions are not thread-safe. The caller MUST
> hold `vc->mutex` before calling any of the methods listed above.

This contract is **the** load-bearing invariant of the consumer-facing
contract. `progressInternal` (see § 6) takes `vc->mutex` per dispatch
when it consumes a CQE for that VC, so consumers must NOT hold
`vc->mutex` across `ctranIb->progress()` — doing so would self-deadlock
on the next CQE for the same VC.

## 6. CQE routing

`CtranIb::progressInternal` is unchanged. Each pass:

1. Polls every `cqs[d]` for `d ∈ [0, numNics)`.
2. For each CQE, looks up `qpToVcMap[(qpn, d)]` to find the owning VC.
3. Calls `vc->processCqe(...)` on the owning VC.

`qpToVcMap` keys every VC's QPs by `(qpn, device)`, regardless of
whether the VC was reached via `getVc` (`vcs[0]`) or via `connectVcs`;
`progressInternal` does not care.

**Per-device progress.** In addition to the all-NICs `progress()`,
`CtranIb` exposes a per-device variant:

```cpp
template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t progress(int device);
```

It polls only `cqs[device]` and runs the same dispatch logic
(`qpToVcMap[(qpn, device)]` → `vc->processCqe(...)`) on that one CQ.
Both overloads share a single `progressInternal(std::optional<int>
device)` implementation; passing `device` just narrows the per-NIC
loop to a single iteration.

This lets a consumer with a NIC-affine progress loop (e.g.,
`p2pHostIbTransport` driving VCs pinned to one NIC) drain its own
NIC's completions without contending on `cqMutex` for other NICs.
Callers that hold a `vcIdx` map to a device via the layout's
`vcToActiveDevices[vcIdx]` (or, in the pinned regime, via the
shortcut `device = vcIdx / getMaxVcsPerNic()`).

Two caveats follow directly from sharing a CQ across every VC pinned
to the same NIC:

- A `progress(device)` call dispatches CQEs for **every** VC whose
  `activeDevices` contains `device` (other peers' VCs as well), not
  just one caller's VC. The per-VC `vc->mutex` taken inside the
  dispatch loop still guarantees per-VC serialization with other
  consumers.
- A consumer that only ever calls `progress(device)` for a subset of
  devices will leave the other NICs' CQs un-drained. Either pair it
  with `progress()` on a separate thread, or call `progress(device)`
  for every NIC the workload touches.

## 7. Per-VC ctrl / notify / atomic QPs

Every VC owns its own `ibvControlQp_`, `ibvNotifyQp_`, and
`ibvAtomicQp_`, all hosted on `ctrlDevice_` (the first entry of
`activeDevices_`). They are not shared across VCs.

Consequences:

- **Per-VC ordering.** Ctrl messages, notifications, and atomics on
  VC `i` are ordered against other ops on VC `i` only. They are
  independent of other VCs (including `vcs[0]`, which legacy
  `getVc`-based callers see).
- **Atomic ordering is per VC.** Each VC's atomic op uses its own
  `ibvAtomicQp_`, so the per-peer ordering guarantee that the legacy
  API documents is now "per VC" — ordered within a single VC, not
  across VCs to the same peer. Consumers that need cross-VC atomic
  ordering must serialize at a higher level.
- **Notify-stream isolation.** A slow notify consumer on one VC cannot
  back-pressure another VC's notify stream. This is what gives the
  transport its per-channel notification stream when it dispatches to
  `vcs[i]->checkNotify(...)`.
- **NIC-local data + notify ordering.** Data and notify on VC `i`
  travel through the same NIC's PCIe path, so the receiver of a notify
  on VC `i` does not need to issue a cross-NIC `iflush` to make VC
  `i`'s data visible.
- **No special routing.** `vc->ifetchAndAdd` and `vc->iatomicSet` go
  through that VC's atomic QP — no special "atomic-on-VC-0" fallback.

## 8. Memory registration & wire format

No change. `regMem` / `deregMem` / `exportMem` / `importMem` produce
per-NIC `lkey` / `rkey` arrays; a VC pinned to NIC `d` consumes
`lkeys[d]` / `rkeys[d]`. There are no extra bits in `imm_data` for VC
demultiplexing — notifications are demultiplexed naturally by
`(qpn, device)` via `qpToVcMap` (§ 6).

## 9. Configuration

All sizing comes from operator cvars; the consumer does not pass
sizing parameters.

- `NCCL_CTRAN_IB_DEVICES_PER_RANK` (`numNics`) — number of physical IB
  devices CtranIb owns. One VCG per NIC.
- `NCCL_CTRAN_IB_NUM_VCS_PER_RANK` (`maxVcsPerPeer`) — number of
  per-peer VCs `connectVcs(peerRank)` returns. Normalised to `1` if unset
  or `<= 0`. Also acts as the divisor for the per-VC data-QP slice
  computed inside `CtranIbVirtualConn`. Must satisfy one of:
  `maxVcsPerPeer >= numNics` and `maxVcsPerPeer % numNics == 0`
  (pinned regime), or `maxVcsPerPeer < numNics` and `numNics %
  maxVcsPerPeer == 0` (striped regime). Otherwise CtranIb init aborts
  with `commInvalidArgument`.
- `NCCL_CTRAN_IB_MAX_QPS` — default per-peer data-QP budget. Each per-peer
  VC ends up with `MAX_QPS / maxVcsPerPeer` data QPs unless the connection
  class overrides MAX_QPS via the `NCCL_CTRAN_IB_QP_CONFIG_*` lists
  (see below). Must be `>= maxVcsPerPeer`.
- `NCCL_CTRAN_IB_QP_CONFIG_XRACK / XZONE / XDC / NCCL_CTRAN_EX_IB_QP_CONFIG`
  — per-connection-class tuning. The MAX_QPS field of the matching
  configList replaces `NCCL_CTRAN_IB_MAX_QPS` for that peer; the per-VC
  slice is then that value divided by `maxVcsPerPeer`. The override
  must be a multiple of `maxVcsPerPeer`.
- `NCCL_CTRAN_IB_QP_SCALING_THRESHOLD`, `NCCL_CTRAN_IB_QP_MAX_MSGS`,
  `NCCL_CTRAN_IB_VC_MODE` — apply per VC.
- `NCCL_CTRAN_IB_MAX_NUM_CQE` — sizes each NIC's shared CQ. Each
  VCG[d]'s CQ now hosts every VC whose `activeDevices` contains NIC
  `d`. Re-tune as VC counts grow.

Derived sizes (computed once at `CtranIb` init via the
`ctran::ib::VcLayout(numNics, maxVcsPerPeer)` ctor):

```
vcLayout_.maxVcsPerPeer       // always >= 1
vcLayout_.maxVcsPerNic        // 1 in striped regime; maxVcsPerPeer/numNics in pinned regime
vcLayout_.vcToActiveDevices   // per-VC NIC vector consumed by CtranIbVirtualConn
```

The per-VC data-QP count is resolved per peer at bootstrap time by
`CtranIb::Bootstrap` calling
`CtranIbVirtualConn::computeMaxQpsPerVc(comm, peer, numVcs)`. For the
default cvar path it is `NCCL_CTRAN_IB_MAX_QPS / maxVcsPerPeer`; for an
overridden connection class it is the overridden MAX_QPS divided by
`maxVcsPerPeer`. When `maxVcsPerPeer == 1` this division is a no-op
and the VC owns the full MAX_QPS.

Both endpoints must agree on the three input cvars; mismatch is
detected at the first `connectVcs(peer)` rendezvous (§ 4) by the
bus-card-swap loop count.

## 10. Lifetime, teardown, and threading

- The per-peer VC vector is owned by `CtranIb` until
  `releaseRemoteTransStates(peer)` runs. `releaseRemoteTransStates`
  drains `vcStateMaps.rankToVcs[peer]` (the single per-peer VC
  storage that backs both `connectVcs` and `getVc`), removes the
  corresponding QPs from `qpToVcMap`, and tears down the per-peer
  rendezvous state. Any in-flight `connectVcs(peer)` spinner exits
  via its `abortCtrl_->Test()` check on the next iteration.
- Consumers that cache `shared_ptr<CtranIbVirtualConn>` (notably
  `p2pHostIbTransport`) MUST drop their `shared_ptr`s before
  `releaseRemoteTransStates` runs. The underlying CQs and IB device
  handles are torn down during `CtranIb` shutdown — calling `vc->X(...)`
  on a "still-alive but under-dependencies-gone" VC will fault.
- A failed VC does not affect other VCs to the same peer. Consumers
  can recreate by re-calling `connectVcs(peer)` after the next
  `releaseRemoteTransStates`.
- Async events on a NIC affect every VC that owns QPs on that NIC; the
  existing `ibAsyncEventHandler` needs no schema change.
- Locking: the per-peer rendezvous is serialised by the `vcStateMaps`
  lock for map mutations. Per-VC ops are serialised by `vc->mutex`
  (held by the consumer; also taken by `progressInternal` per CQE
  dispatch). Consumers must NOT hold `vc->mutex` across
  `ctranIb_->progress()`.

## 11. Phased implementation

The design above is the **target** state. It is delivered in three
phases so that each phase ships independently and remains reviewable.

### Phase 1 — Multi-channel functionality ready (this round)

Goal: deliver a working multi-channel data path end to end so
algorithms / RMA / `p2pHostIbTransport` can issue per-channel
operations and rely on independent VCs underneath.

Functionality delivered in this phase:

- `CtranIb::connectVcs(peerRank, peerAddr?)` returns a per-peer vector of
  `maxVcsPerPeer` NIC-pinned VCs, lazily bootstrapped at first call.
  Cardinality is derived from cvars (§ 3, § 10).
- Each per-peer VC owns its own ctrl / notify / atomic + data QPs on
  its pinned NIC, providing the channel-independence properties
  documented in § 1.
- `CtranIb::progress()` drives CQE dispatch across every NIC's CQ via
  `qpToVcMap`; `getVc` callers (`vcs[0]`) and `connectVcs` callers share
  the same routing path (§ 6).
- `releaseRemoteTransStates(peer)` tears down the per-peer VC vector
  (the single source of truth that backs both `connectVcs` and `getVc`).
- The cvars `NCCL_CTRAN_IB_DEVICES_PER_RANK`,
  `NCCL_CTRAN_IB_NUM_VCS_PER_RANK`, and `NCCL_CTRAN_IB_MAX_QPS` size
  the per-peer VC vector.

What consumers can do after this phase:

- `p2pHostIbTransport` calls `connectVcs(peer)` and dispatches each chunk
  to `vcs[vcIdx]`, where `vcIdx` is supplied by the caller per chunk
  via `transport->iSendChunk(buf, off, len, config, vcIdx, stagingSlot)`
  (see `comms/ctran/transport/ib/p2pIBHostTransportDesign.md`).
- Algorithms / RMA above the transport pick `vcIdx = channelId %
  numVcs` (the recommended convention). When `numChannels <= numVcs`
  this gives each channel true per-channel ordering domains,
  independent notify streams, and NIC-local data+notify ordering. If
  the algo oversubscribes, the transport still dispatches
  correctly; the algo just gives up isolation between the
  oversubscribed channels.

Exit criteria: end-to-end multi-channel send/recv tests pass; the
legacy single-VC API (`getVc` and friends) is unchanged — it now
transparently observes `vcs[0]` of the per-peer vector.

### Phase 2 — `VcGroup` for clean per-NIC resource management & per-VCG progress

Goal: promote VCG from a logical concept to a real C++ struct so
per-NIC resources have one owner, **and expose a per-VCG `progress()`
entry point so consumers can drive one VCG (one NIC's CQ) independently
of the others**.

- Introduce
  `struct VcGroup { ibverbs::Device* device; CompletionQueue cq;
  std::unordered_map<int, std::vector<VcRef>> perPeerVcs;
  commResult_t progress(); };`
  and replace the flat arrays (`devices[]`, `cqs[]`) with
  `std::vector<VcGroup> vcgs_`.
- Add `CtranIb::getVcg(int nicIdx) -> VcGroup&` so the consumer can
  reach a specific VCG and call `vcg.progress()` on its own thread or
  cadence — useful when the consumer pins work to a NIC and wants to
  drive that NIC's CQ without contending on other NICs' progress.
  `CtranIb::progress()` becomes the simple loop
  `for (auto& vcg : vcgs_) vcg.progress();` so legacy callers are
  unaffected.
- `vcg.progress()` polls `vcg.cq`, looks up the owning VC via
  `qpToVcMap[(qpn, d)]`, takes `vc->mutex`, and dispatches to
  `vc->processCqe(...)` — same body as today's `progressInternal`,
  scoped to one VCG.
- Rendezvous and `connectVcs` keep the same public signature; internally
  they distribute new VCs into the right `vcgs_[d].perPeerVcs[peer]`.
- `releaseRemoteTransStates(peer)` becomes "erase peer from every
  `vcgs_[d].perPeerVcs`" — easier to audit than the current
  per-peer drain.

Exit criteria: `CtranIb` no longer carries flat per-NIC arrays; all
per-NIC state lives inside a `VcGroup`. `CtranIb::getVcg(d)` and
`vcg.progress()` are public; consumers (notably `p2pHostIbTransport`
in its NIC-affine progress loop) can opt in.

### Phase 3 — Migrate VCG and VC to `ibverbx`

Goal: move the **VCG** and **VC** abstractions out of `CtranIb` and
onto `ibverbx`. After this phase, `ibverbx` owns the per-NIC
resource group (device + CQ + per-peer VCs) and the per-peer
NIC-pinned virtual connection; `CtranIb`'s in-tree implementation of
these types goes away.

- Promote `VcGroup` and `CtranIbVirtualConn` (the per-peer VC) into
  first-class `ibverbx` types (e.g. `ibverbx::VcGroup`,
  `ibverbx::VirtualConn`). They take ownership of the underlying IB
  resources via `ibverbx`'s RAII wrappers (`ibverbx::Device`,
  `ibverbx::Cq`, `ibverbx::Qp`, `ibverbx::Pd`, `ibverbx::Mr`).
- `CtranIb` shrinks to a thin orchestration layer over the `ibverbx`
  types: bootstrap, listen thread, per-peer rendezvous, and the
  public `connectVcs`/`getVcg` surface. The `CtranIb`-side implementation
  of `CtranIbVirtualConn` and the in-tree `VcGroup` struct are
  deleted.
- The public consumer-facing API (`connectVcs`, `getVcg`, per-VC method
  signatures) stays compatible; only the underlying type origin
  moves from `CtranIb` to `ibverbx`.
- Async-event handling lives on the `ibverbx` event-channel surface,
  routed by VCG / VC ownership.

Exit criteria: `CtranIb` carries no in-tree implementation of `VcGroup`
or the per-peer VC; both come from `ibverbx`. Teardown is enforced by
`ibverbx` RAII rather than the explicit `releaseRemoteTransStates`
bookkeeping.
