# CtranIb External Bootstrap — Design & Readiness Analysis

## 1. Overview

`CtranIb` supports two ways to establish per-peer virtual connections
(VCs):

- **Internal bootstrap** (`BootstrapMode::kDefaultServer` /
  `kSpecifiedServer`): `CtranIb` owns a TCP listen thread and drives the
  full rendezvous itself (bus-card swap **plus an ack round-trip**). See
  `ctranib_multi_channel_vc_management.md` §4 and the "VC readiness"
  section there.
- **External bootstrap** (`BootstrapMode::kExternal`): the *caller*
  drives the rendezvous. `CtranIb` only exposes two primitives —
  `getLocalVcId()` and `connectVc()` — and the caller is responsible for
  carrying the bus cards between peers and for any cross-rank ordering.

This document describes the external path and analyzes why it is safe
for its current sole consumer even though it has **no ack / readiness
handshake** of its own.

**Who uses it.** The only production consumer today is
`comms/torchcomms/transport/RdmaTransport`, which is in turn driven by
the tensor-shard Thrift service in `msl/rl/tensor_transfer`:

```
msl/rl/tensor_transfer  (Thrift shard service, client-pull data plane)
        │
        ▼
comms/torchcomms/transport/RdmaTransport   (kExternal CtranIb)
        │
        ▼
comms/ctran/backends/ib  BootstrapExternal  (getLocalVcId / connectVc)
```

## 2. Control flow — no listen thread, no ack, synchronous on the caller

In `kExternal` mode `CtranIb` **never constructs the internal
`Bootstrap`** and **never starts the listen thread**
(`CtranIb.cc:655-687`): only `BootstrapExternal` is created, and it owns
no thread, no listen socket, and no accept loop. (Contrast: the internal
`Bootstrap::start()` launches `listenThread_ = std::thread{acceptLoop}`,
the `"CTranIbListen"` thread — `BootstrapInternal.cc:107,165,312,320`.)

Consequently both external primitives run **synchronously on the
caller's own thread**:

- `getLocalVcId(peerRank)` (`BootstrapExternal.cc:37-75`) creates the
  VC's QPs (INIT state) and returns the serialized local bus card.
- `connectVc(remoteVcId, peerRank)` (`BootstrapExternal.cc:77-116`) runs
  `setupVc` (prepost recv WQEs → QPs RTR → RTS → publish) via
  `setupAndPublishVc`, then marks the VC peer-ready.

Because there is no listen thread, external mode has **none** of the
internal-mode "listen thread publishes/acks while the user thread spins
on `getVc`" split. Structurally it behaves like the internal
*smaller-rank* path: a synchronous local setup on the calling thread.

## 3. Readiness in external mode

`CtranIb` uses a monotonic per-VC status `kNone → kLocalReady →
kPeerReady` (see `CtranIbVc.h`), and the issue path (`getVc` /
`tryGetVcs`) returns `nullptr` until a VC is `kPeerReady`.

- **Internal** bootstrap sets `kPeerReady` only *after* the ack
  round-trip, which proves the remote finished `setupVc` (recv WQEs
  posted) — a real cross-rank happens-before.
- **External** bootstrap has **no ack**, so `CtranIb` cannot prove the
  remote is ready. `connectVc` marks the VC `kPeerReady` right after the
  local `setupVc`/publish anyway (`BootstrapExternal.cc:114`). This is a
  **STOPGAP**: it makes the VC locally issuable and assumes the external
  caller established its own cross-rank barrier before issuing traffic.
  Removing the call would deadlock external mode under the `getVc`
  peer-ready gate.

So in external mode, `connectVc` returning (and `getVc()!=nullptr` /
`RdmaTransport::connected()`) proves only **local** readiness. The
cross-rank guarantee must come from the caller.

## 4. How the caller provides the cross-rank barrier

The caller stack supplies the missing barrier implicitly, through the
Thrift request/response ordering plus a **strictly one-directional,
client-pull data plane**.

Roles on a connection (`msl/rl/tensor_transfer/shard_service_thrift.cpp`):

- **server** = the *only* rank that issues RDMA writes
  (`transport->write()` → `ib_->iput`, `:541`), and only *inside* the
  `co_fetchSliceRdma` handler.
- **client** = the RDMA-write *target*; it sends its destination
  pointers + rkey in the fetch request and **never issues RDMA to the
  server**. Completion is delivered by the Thrift response, not by RDMA
  notify.

### Sequence

```
 role: SERVER = only issuer of RDMA writes (iput)
       CLIENT = RDMA-write target (its recv WQEs must be posted first)

===== PHASE 1: setup — co_connectRdmaTransport (no RDMA traffic) =====

 CLIENT (write target)                          SERVER (write issuer)
 ---------------------                          ---------------------
 bind() = getLocalVcId
   (QPs created, INIT)
 clientUrl ──── co_connectRdmaTransport(clientUrl) ────►
                                                bind() = getLocalVcId (QPs INIT)
                                                connect(clientUrl) = connectVc:
                                                  prepost recvs → RTR → RTS → publish
                                                  ★ SERVER VC ready
                       ◄──────── reply: serverUrl ────────
 connect(serverUrl) = connectVc:
   prepost recvs → RTR → RTS → publish
   ★ CLIENT VC ready  (recv WQEs POSTED, QP RTR/RTS)
 connect() == commSuccess  (checked)
 └─ getOrCreateConnection() returns ONLY now ─┘

===== PHASE 2: first transfer — co_fetchSliceRdma (only first-traffic RDMA) =====

 build fetch req (dst ptrs + rkey)
 ──────── co_fetchSliceRdma(dstPtrs, rkey) ────────►
                                                handler: write() = ib_->iput
                                                  ═══════ RDMA WRITE ═══════╗
 client QP already RTR/RTS,                                                 ║
 recv WQEs already posted   ◄══════════════════════════════════════════════╝
 → target provably ready: no unready-target write, no RNR
                       ◄──────── reply: done ────────
```

### Happens-before chain

```
 CLIENT connect(serverUrl) completes          (shard_service_thrift.cpp:1886, checks commSuccess)
        │  (getOrCreateConnection does not return until this completes, :1451)
        ▼
 CLIENT builds + sends co_fetchSliceRdma       (:1513-1527)
        │
        ▼
 SERVER issues iput to CLIENT                  (inside the fetch handler, :541)
```

Therefore, at the moment the server issues its first write, the target
(client) has provably finished `connectVc` — recv WQEs posted, QP in
RTR/RTS. The server likewise finished its own `connect(clientUrl)` back
in Phase 1. **A single Thrift request/response round-trip is the
cross-rank barrier** that `BootstrapExternal`'s STOPGAP assumes.

Key call sites:
- `RdmaTransport.cpp:206-217` (kExternal ctor), `:296-307`
  (`bind`/`connect`/`connected`), `:346` (`iput` in `write`).
- `shard_service_thrift.cpp:345-392` (server connect handler),
  `:541-550` (server write), `:1828-1905` (client connect handler),
  `:1451` / `:1513-1527` (client fetch path).

## 5. Why it is always safe — and the three preconditions

The barrier holds **only** because of three properties of the caller's
protocol. If any is violated, the RPC no longer orders the write after
the target's `connectVc`, and a proper external readiness handshake
would be required.

| # | Precondition | What breaks if violated |
|---|---|---|
| (a) | Data plane is **one-directional, server → client** | If the client also issues `iput`/`iget` to the server, the client's write is not ordered after any "server connected" RPC. |
| (b) | The server issues writes **only inside the fetch handler** | If the server pushes before receiving a client fetch, the write can precede the client's `connectVc`. |
| (c) | The client **never** issues RDMA to the server | Same as (a). |

Note the base "prepost before RTR" fix (D114139030) already guarantees
that any RC QP in RTR has recv WQEs, and a peer still in INIT cannot
raise RNR (an inbound write just triggers ordinary transport
retransmit). But relying on that alone would make first-transfer
correctness depend on the target reaching RTR within the RC retry
window; if it did not, the RC write would exhaust the retry counter and
surface as a **fatal** completion error (→ `broken_` + reconnect), not a
benign retry (`RdmaTransport.cpp:357-370,463-473`). The caller's barrier
(§4) is what keeps this from arising.

## 6. Conclusion & guidance

- **No external readiness handshake is needed for the current caller.**
  The tensor_transfer / RdmaTransport protocol provides the cross-rank
  barrier the STOPGAP assumes, so no rank ever issues an RDMA WQE to a
  peer that has not finished `connectVc`.
- **This safety is a property of the caller's protocol, not of the
  transport.** It is not enforced by `CtranIb`. If a future consumer
  makes the data plane bidirectional, has the server push before a
  client fetch, or has the client issue RDMA to the server, then
  external bootstrap must gain a real readiness handshake (an ack
  round-trip over the caller's channel, or promotion to `kPeerReady`
  only after an explicit post-barrier signal) before that consumer can
  be safe.
- The STOPGAP and this reasoning live at `BootstrapExternal.cc:105-114`;
  the internal-bootstrap ack contrast is at `BootstrapInternal.cc` and
  in `ctranib_multi_channel_vc_management.md`.
