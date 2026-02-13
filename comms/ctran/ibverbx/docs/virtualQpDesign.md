# VirtualQp / VirtualCq Design

## Overview

The VirtualQp / VirtualCq system provides message fragmentation and load balancing across multiple physical InfiniBand Queue Pairs (QPs). Users interact with two main classes:

- **`IbvVirtualQp`** — Accepts user work requests, fragments large RDMA messages across physical QPs, tracks completions, and reports aggregated results in posting order.
- **`IbvVirtualCq`** — Polls physical Completion Queues (CQs), routes completions using opcode-based logic, and returns `IbvVirtualWc` to the user.

**Source Files:**
- `IbvVirtualWr.h` — `IbvVirtualSendWr`, `IbvVirtualRecvWr`, `IbvVirtualWc`, `ActiveVirtualWr`, `WrTracker`
- `IbvVirtualQp.h` / `IbvVirtualQp.cc` — `IbvVirtualQp`, `IbvVirtualQpBusinessCard`
- `IbvVirtualCq.h` / `IbvVirtualCq.cc` — `IbvVirtualCq`, `QpId`, `RegisteredQpInfo`
- `IbvQp.h` — `IbvQp::PhysicalWrStatus`, `physicalSendWrStatus_`, `physicalRecvWrStatus_`
- `IbvCommon.h` — `Error`, `LoadBalancingScheme`, constants
- `DqplbSeqTracker.h` — `DqplbSeqTracker`

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          User Code                               │
│  • postSend() → IbvVirtualQp                                    │
│  • pollCq()   → IbvVirtualCq                                    │
└──────────┬──────────────────────────────────┬────────────────────┘
           │ postSend()                       │ pollCq()
           ▼                                  ▼
┌────────────────────────────────┐  ┌──────────────────────────────┐
│      IbvVirtualQp              │  │      IbvVirtualCq            │
│  • Fragments large messages    │  │  • Polls physical CQ         │
│  • Load balancing (SPRAY/DQPLB)│  │  • Routes based on opcode    │
│  • Tracks pending virtual WRs  │  │  • Returns IbvVirtualWc      │
│  • Registers with VirtualCq    │◄─┤                              │
│  • processCompletion(wc)       │  │  Routing logic:              │
│    (multi-QP RDMA only)        │  │  • RDMA+multi-QP: aggregate  │
│                                │  │  • Everything else: pass-thru│
└──────────┬─────────────────────┘  └──────────┬───────────────────┘
           │                                   │
           │ Registration at construction      │ Poll loop
           │ physicalQpNum → {this, isMultiQp}  │
           └───────────────────────────────────┘
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        Physical    Physical    Physical
         QP[0]       QP[1]       QP[N]
        (registered) (registered) (registered)
             │           │           │
             └─────────┬─┴───────────┘
                       │
                       ▼
                 Physical CQ
           (1 per NIC, typically 1 total;
            2 for multi-NIC setups)
                       │
                       ▼
             VirtualCq polls and routes:
             • RDMA opcode + isMultiQp=true → VirtualQp.processCompletion()
             • Everything else              → Pass through directly
```

**Key design points:**
1. VirtualQp **registers** its physical QPs with VirtualCq at construction
2. Registration includes an `isMultiQp` flag indicating whether load balancing is active
3. `pollCq()` uses **opcode-based routing**: RDMA opcodes in multi-QP mode route to VirtualQp for fragment aggregation; everything else (single-QP, SEND, RECV, atomics) passes through directly
4. Every QP is managed by a VirtualQp — even single-QP configurations
5. Typically 1 physical CQ shared by all QPs (2 for multi-NIC setups)

---

## 2. Data Structures

All data structures are defined in `IbvVirtualWr.h` unless otherwise noted.

### 2.1 User-Facing Types

- **`IbvVirtualSendWr`** — Send work request. Contains `wrId`, `localAddr`, `length`, `remoteAddr`, `opcode`, `sendFlags`, `immData`, atomic fields (`compareAdd`, `swap`), and per-device memory keys (`deviceKeys`).

- **`IbvVirtualRecvWr`** — Recv work request. Contains `wrId`, `localAddr`, `length` (0 = notification recv, >0 = data recv), and `deviceKeys`.

- **`IbvVirtualWc`** — Work completion returned by `pollCq()`. Contains `wrId`, `status`, `opcode`, `qpNum` (virtual, not physical), `immData`, and `byteLen`. For single-QP, constructed directly from the physical `ibv_wc`. For multi-QP, constructed by `buildVirtualWc()` from the completed `ActiveVirtualWr`.

### 2.2 Internal Tracking Types

- **`ActiveVirtualWr`** — Internal WR tracking state for fragmentation, notify, and completion aggregation:
  - `userWrId` — original wrId for completion reporting
  - `remainingMsgCnt` — counts all expected CQEs (data + notify if applicable); 0 = complete
  - `aggregatedStatus` — first error wins across all fragments
  - `offset` — fragmentation progress; `offset >= length` means all fragments posted
  - `needsNotify` / `notifyPosted` — SPRAY notify tracking booleans

- **`WrTracker<ActiveVirtualWrT>`** — Three-structure tracker that manages a virtual WR through its lifecycle:

```
┌───────────────────────────────────────────────────────────────────┐
│ WrTracker<ActiveVirtualWr>                                        │
│                                                                   │
│ activeVirtualWrs_: F14FastMap<uint64_t, ActiveVirtualWr>           │
│   Source of truth for all in-flight virtual WRs.                  │
│   Key = internalWrId (monotonically increasing, unique).          │
│   Lifecycle: inserted by add(), erased by remove() after          │
│   reporting.                                                      │
│                                                                   │
│ pendingPostQue_: std::deque<uint64_t>                             │
│   FIFO queue of internalWrIds whose fragments have NOT all been   │
│   posted to physical QPs yet. Pushed by add(), popped once        │
│   offset >= length. dispatchPendingSends() always processes       │
│   the front WR first.                                             │
│                                                                   │
│ pendingCompletionQue_: std::deque<uint64_t>                       │
│   FIFO queue of internalWrIds in posting order, used to enforce   │
│   in-order completion reporting. reportSendCompletions() only     │
│   reports the FRONT WR when its remainingMsgCnt == 0              │
│   (head-of-line blocking by design).                              │
│                                                                   │
│ nextInternalVirtualWrId_: uint64_t                                │
│   Monotonic counter; each add() assigns the next value.           │
└───────────────────────────────────────────────────────────────────┘
```

- **`IbvQp::PhysicalWrStatus`** — Per-physical-QP tracking (defined in `IbvQp.h`). Each entry maps `physicalWrId → virtualWrId` (internal WR ID). Maintained in `physicalSendWrStatus_` and `physicalRecvWrStatus_` deques. Used for completion correlation and backpressure (size >= `maxMsgCntPerQp_` means QP full).

```
┌───────────────────────────────────────────────────────────────────┐
│ IbvQp::physicalSendWrStatus_: std::deque<PhysicalWrStatus>        │
│ IbvQp::physicalRecvWrStatus_: std::deque<PhysicalWrStatus>        │
│                                                                   │
│ Each entry = { physicalWrId, virtualWrId (= internalWrId) }       │
│                                                                   │
│ • Pushed when a fragment is posted to this physical QP.           │
│ • Popped when the corresponding CQE arrives (FIFO guaranteed     │
│   by IB verbs per-QP completion ordering).                        │
│ • The popped virtualWrId is used to look up ActiveVirtualWr       │
│   in the tracker and decrement remainingMsgCnt.                   │
│ • Also used for backpressure: size() >= maxMsgCntPerQp_           │
│   means QP full.                                                  │
└───────────────────────────────────────────────────────────────────┘
```

### 2.3 VirtualQp Members

| Member | Type | Purpose |
|--------|------|---------|
| `sendTracker_` | `WrTracker<ActiveVirtualWr>` | Send WR tracking |
| `recvTracker_` | `WrTracker<ActiveVirtualWr>` | Recv WR tracking |
| `pendingSendNotifyQue_` | `std::deque<uint64_t>` | SPRAY send notify backpressure queue |
| `pendingRecvNotifyQue_` | `std::deque<...>` | SPRAY recv notify backpressure queue |
| `notifyQp_` | `std::optional<IbvQp>` | Separate QP for SPRAY notify messages |
| `dqplbSeqTracker_` | `DqplbSeqTracker` | DQPLB sequence number tracking |

Send and recv use separate trackers so their completion ordering is independent — recv completions are never blocked by pending sends.

---

## 3. IbvVirtualCq: Registration and Routing

### 3.1 Registration

VirtualQp registers its physical QPs with VirtualCq at construction. The VirtualCq maintains a registration table (`registeredQps_`: `F14FastMap<QpId, RegisteredQpInfo>`) mapping physical QP identifiers to VirtualQp pointers. `QpId` uses both `deviceId` and `qpNum` to uniquely identify physical QPs (different NICs can have QPs with the same number).

```
VirtualQp Construction
         │
         │ Determine isMultiQp = (physicalQps_.size() > 1)
         ▼
┌────────────────────────────────────────┐
│ For each physical QP in physicalQps_:  │
│   virtualCq->registerPhysicalQp(       │
│       qp.qpNum, qp.deviceId, this,    │
│       isMultiQp, virtualQpNum_)        │
│                                        │
│ Also register notifyQp_ (if multi-QP) │
└────────────────────────────────────────┘
         │
         ▼
VirtualCq.registeredQps_[{deviceId, qpNum}] = {this, isMultiQp}
```

Multiple VirtualQps can share the same VirtualCq:

```
VirtualCq (shared)
  └── registeredQps_
        ├── {device0, qp100} → VirtualQp_A
        ├── {device0, qp101} → VirtualQp_A
        ├── {device0, qp102} → VirtualQp_A
        ├── {device1, qp200} → VirtualQp_B
        ├── {device1, qp201} → VirtualQp_B
        └── {device1, qp202} → VirtualQp_B
```

### 3.2 Routing in pollCq()

| VirtualQp Type | Physical QPs | Opcode | pollCq() Handling |
|----------------|--------------|--------|-------------------|
| Single-QP | 1 | Any | Direct pass-through |
| Multi-QP | >1 | RDMA_WRITE, RDMA_READ, RECV_RDMA_WITH_IMM | Route to VirtualQp for fragment aggregation |
| Multi-QP | >1 | SEND, RECV, atomics | Direct pass-through (no fragmentation) |

`pollCq()` drains all available CQEs from all physical CQs (no `numEntries` limit). Results are returned directly as `std::vector<IbvVirtualWc>`.

### 3.3 Lifetime and Thread Safety

VirtualCq must outlive VirtualQp. VirtualCq and its associated VirtualQps should be used from the same thread. If multi-threaded access is needed, external synchronization is required.

### 3.4 Move Semantics

When VirtualQp is moved, the registration is updated: the old `this` pointer in `registeredQps_` is replaced with the new one. The move constructor sets `other.virtualCq_ = nullptr` to prevent double-unregister, then re-registers. VirtualCq move assignment updates all registered VirtualQps' `virtualCq_` pointers to the new `this`.

---

## 4. postSend() Workflow

### 4.1 Opcode-Based Routing

| Opcode | Path | Description |
|--------|------|-------------|
| `IBV_WR_SEND` | Single QP | Always uses `postSendSingleQp()`. No fragmentation. |
| `IBV_WR_ATOMIC_FETCH_AND_ADD` | Single QP | Atomic pass-through to QP[0]. Always 8-byte. |
| `IBV_WR_ATOMIC_CMP_AND_SWP` | Single QP | Atomic pass-through to QP[0]. Always 8-byte. |
| `IBV_WR_RDMA_WRITE` | Multi-QP if `isMultiQp_` | Fragments across multiple QPs for bandwidth. |
| `IBV_WR_RDMA_WRITE_WITH_IMM` | Multi-QP if `isMultiQp_` | Fragments + SPRAY notify for receiver notification. |
| `IBV_WR_RDMA_READ` | Multi-QP if `isMultiQp_` | Fragments across multiple QPs. |

**Why SEND and atomics always use single QP:**
- `IBV_WR_SEND` requires matching recv buffers; fragments could arrive out-of-order with load balancing
- `IBV_WR_ATOMIC_*` operations are inherently 8-byte, single-address operations
- RDMA operations write/read directly to known remote addresses and are order-independent

### 4.2 Single-QP Fast Path (postSendSingleQp)

For single-QP VirtualQps, SEND, and atomic operations:

```
postSendSingleQp():
  [1] Build physical ibv_send_wr with user's wrId directly
  [2] Post to physicalQps_[0]
  [3] No tracking needed — VirtualCq returns physical WC directly

Benefits: ~70 ns overhead, no fragmentation, no memory allocation
```

### 4.3 Multi-QP RDMA Path

```
postSend() — multi-QP RDMA:
  [1] Validate: length != 0, signaled
  [2] Calculate fragment count:
        expectedMsgCnt = ceil(length / maxMsgSize_)
        If SPRAY + WRITE_WITH_IMM: remainingMsgCnt = expectedMsgCnt + 1
        Otherwise: remainingMsgCnt = expectedMsgCnt
  [3] Generate unique internalWrId
  [4] sendTracker_.add(ActiveVirtualWr{...})
        → Insert into activeVirtualWrs_, pendingPostQue_, pendingCompletionQue_
  [5] Call dispatchPendingSends()
```

### 4.4 Fragmentation (dispatchPendingSends)

`dispatchPendingSends()` sends fragments from the front of `sendTracker_.pendingPostQue_` to available physical QPs. Called from `postSend()` and `processCompletion()`.

```
dispatchPendingSends():
  while (sendTracker_ has pending WRs):
    Get front WR from pendingPostQue_
    while (offset < length):
      qpIdx = findAvailableSendQp()
      if (qpIdx == -1) return          // All QPs full, resume later
      fragLen = min(maxMsgSize_, length - offset)
      Build physical ibv_send_wr for this fragment
      physicalQps_[qpIdx].postSend()
      Track: physicalSendWrStatus_.push({physWrId, internalId})
      offset += fragLen
    Pop from pendingPostQue_ (all fragments sent)
```

| Aspect | Description |
|--------|-------------|
| **Processes in order** | Always takes from front of `pendingPostQue_` |
| **Load balancing** | `findAvailableSendQp()` selects QP with capacity |
| **Partial send** | Returns early if no QP slots available; resumes on next completion |
| **Fragment tracking** | Each physical WR tracked with `(physicalWrId, internalWrId)` |

### 4.5 postRecv() Routing

- Single-QP (`!isMultiQp_`): passthrough via `postRecvSingleQp()`
- Multi-QP with `length > 0`: data recv via `postRecvSingleQp()`
- Multi-QP with `length == 0`: notification recv
  - DQPLB: initializes receiver with pre-posted zero-length recvs on first call
  - SPRAY: posts zero-length recv to `notifyQp_` (with backpressure queuing in `pendingRecvNotifyQue_`)

---

## 5. pollCq() Workflow

```
virtualCq.pollCq():
  results = []
  For each physical CQ in physicalCqs_:
    Drain CQ until empty (batch 32 at a time):
      For each physical WC:
        Lookup: info = registeredQps_.find({deviceId, wc.qp_num})
        if (isMultiQp && RDMA opcode):
          virtualWcs = vqp->processCompletion(wc, deviceId)
          Append virtualWcs to results
        else:
          Build IbvVirtualWc directly from ibv_wc (passthrough)
          Append to results
  return results
```

---

## 6. Completion Processing (processCompletion)

`processCompletion()` uses a **2x2 matrix dispatch** based on QP type and direction:

```
                    Send                         Recv
           ┌─────────────────────────┬─────────────────────────┐
NotifyQp   │ processNotifyQpSend-    │ processNotifyQpRecv-    │
           │   Completion()          │   Completion()          │
           │ (SPRAY sender's         │ (SPRAY receiver's       │
           │  notify done)           │  notify arrived)        │
           ├─────────────────────────┼─────────────────────────┤
DataQp     │ processDataQpSend-      │ processDataQpRecv-      │
           │   Completion()          │   Completion()          │
           │ (Data fragment          │ (DQPLB recv with        │
           │  completed)             │  seq# in IMM)           │
           └─────────────────────────┴─────────────────────────┘
```

Each handler follows the same pattern:
1. Pop `physicalSendWrStatus_` / `physicalRecvWrStatus_` to get `internalWrId`
2. Update WR state via `updateWrState()` (decrement `remainingMsgCnt`, aggregate errors)
3. Report completed WRs in order via `reportSendCompletions()` / `reportRecvCompletions()`
4. Schedule follow-up work (dispatch pending sends, flush pending notifies, replenish DQPLB recvs)

### 6.1 In-Order Completion Guarantee

Physical completions may arrive out-of-order (across multiple NICs), but virtual completions are reported strictly in posting order. `reportSendCompletions()` only reports the front WR from `pendingCompletionQue_` when its `remainingMsgCnt == 0`, then drains consecutive completed WRs.

```
Example: User posts WR_A, WR_B, WR_C in order

1. WR_B fragment completes → No results (WR_A at front, not complete)
2. WR_C completes fully    → No results (WR_A still at front)
3. WR_A last fragment      → Returns [WR_A]
                              Check WR_B: not complete → stop
4. WR_B last fragment      → Returns [WR_B, WR_C]
                              Both were complete, drained together

Final order reported: WR_A, WR_B, WR_C (posting order preserved)
```

### 6.2 SPRAY Notify Flow

SPRAY mode with `RDMA_WRITE_WITH_IMM` requires a separate notify message after all data fragments complete. The notify provides a receiver-side ordering guarantee: when the receiver gets WR_B's notify, all prior WRs are guaranteed complete.

```
SPRAY WR Lifecycle:

  postSend():
    remainingMsgCnt = numFragments + 1
    needsNotify = true, notifyPosted = false
           │
           ▼
  Data CQEs arrive:
    remainingMsgCnt decrements (e.g., 4 → 3 → 2 → 1)
           │
           ▼
  reportSendCompletions() — WR at front, remainingMsgCnt == 1:
    All data done → post notify to notifyQp_
    notifyPosted = true
    break (wait for notify CQE)
           │
           ▼
  Notify CQE arrives:
    remainingMsgCnt: 1 → 0
    Report to user, remove WR
```

**Backpressure:** If `notifyQp_` is at capacity when posting notify, the WR is queued in `pendingSendNotifyQue_` and `notifyPosted` is set to `true` to prevent re-queuing. When backpressure clears, `flushPendingSendNotifies()` processes the queue.

### 6.3 DQPLB Recv Tracking

DQPLB mode embeds sequence numbers in IMM data for in-order recv completion tracking. `DqplbSeqTracker` buffers out-of-order arrivals and reports a `notifyCount` indicating how many front WRs in the recv tracker can be completed.

```
Example: 3 recvs posted, messages arrive out of order

T1: msg with seq=0 arrives on QP[2]
    DqplbSeqTracker: nextExpected=0 → match → notifyCount=1
    Complete WR at front → return [{wrId=100}]
    Replenish recv on QP[2]

T2: msg with seq=2 arrives on QP[0]  (seq=1 missing!)
    DqplbSeqTracker: nextExpected=1 → mismatch → buffer seq=2
    notifyCount=0 → return []
    Replenish recv on QP[0]

T3: msg with seq=1 arrives on QP[1]  (fills gap)
    DqplbSeqTracker: nextExpected=1 → match, also finds seq=2 buffered
    notifyCount=2 → complete 2 front WRs
    return [{wrId=101}, {wrId=102}]
    Replenish recv on QP[1]
```

---

## 7. End-to-End Lifecycle

```
USER                    IbvVirtualQp                  Physical QPs       IbvVirtualCq
 │                           │                             │                   │
 │  postSend(wr)             │                             │                   │
 ├──────────────────────────►│                             │                   │
 │                           │                             │                   │
 │  ┌────────────────────────┤                             │                   │
 │  │ [A] Opcode Routing     │                             │                   │
 │  │ SEND/atomics → single  │                             │                   │
 │  │ RDMA+!multi  → single  │                             │                   │
 │  │ RDMA+multi   → below   │                             │                   │
 │  └────────────────────────┤                             │                   │
 │                           │                             │                   │
 │  ┌────────────────────────┤                             │                   │
 │  │ [B] sendTracker_.add() │                             │                   │
 │  │ • Insert ActiveVirtualWr into activeVirtualWrs_      │                   │
 │  │ • Push internalId → pendingPostQue_                  │                   │
 │  │ • Push internalId → pendingCompletionQue_            │                   │
 │  └────────────────────────┤                             │                   │
 │                           │                             │                   │
 │  ┌────────────────────────┤                             │                   │
 │  │ [C] dispatchPendingSends()                           │                   │
 │  │ • For each fragment:                                 │                   │
 │  │   ├─ findAvailableSendQp() → qpIdx                  │                   │
 │  │   ├─ buildPhysicalSendWr()                           │                   │
 │  │   ├─ physicalQps_[qpIdx].postSend()─────────────────►│                   │
 │  │   ├─ Track {physWrId, internalId}                    │                   │
 │  │   └─ offset += fragLen                               │                   │
 │  └────────────────────────┤                             │                   │
 │                           │                             │                   │
 │◄──────────────────────────┤ return success              │                   │
 │                           │                             │                   │
 │        ═══════════ TIME PASSES (HW processes RDMA) ═══════════             │
 │                           │                             │                   │
 │                           │                             │  pollCq()         │
 │  pollCq()                 │                             │◄──────────────────┤
 │  ┌────────────────────────┼─────────────────────────────┼──────────────────►│
 │  │                        │                             │                   │
 │  │  ┌─────────────────────┤                             │                   │
 │  │  │ [D] ibv_poll_cq drains physical CQEs              │                   │
 │  │  │ For each CQE:                                     │                   │
 │  │  │  • Lookup registeredQps_[{deviceId, qpNum}]       │                   │
 │  │  │  • isMultiQp && RDMA opcode?                      │                   │
 │  │  │    YES → vqp->processCompletion(wc)               │                   │
 │  │  │    NO  → passthrough (IbvVirtualWc directly)      │                   │
 │  │  └─────────────────────┤                             │                   │
 │  │                        │                             │                   │
 │  │  ┌─────────────────────┤  processCompletion()        │                   │
 │  │  │ [E] Pop physicalSendWrStatus_ → internalWrId      │                   │
 │  │  │ [F] updateWrState(): remainingMsgCnt--, aggregate │                   │
 │  │  │ [G] reportSendCompletions(): drain front WRs      │                   │
 │  │  │ [H] dispatchPendingSends(): resume pending WRs    │                   │
 │  │  └─────────────────────┤                             │                   │
 │  │                        │                             │                   │
 │◄─┴────────────────────────┤ return vector<IbvVirtualWc> │                   │
 │                           │                             │                   │
 │ User receives completions │                             │                   │
 │ in posting order          │                             │                   │
```

### WrTracker Queue Evolution

```
Phase              activeVirtualWrs_    pendingPostQue_      pendingCompletionQue_
═════════════════  ═══════════════════  ═══════════════════  ═════════════════════
After add()        [id] → ActiveWr     [..., id]            [..., id]
                   (offset=0,           (needs fragments      (waiting for
                    remainingMsgCnt=N)   posted)               completion)

After dispatch     [id] → ActiveWr     id removed           [..., id]
(all fragments     (offset=length,      (all posted)          (still waiting)
 posted)            remainingMsgCnt=N)

During pollCq      [id] → ActiveWr     —                    [..., id]
(partial CQEs)     (remainingMsgCnt                          (behind incomplete
                    decreasing)                                WR or at front)

After final CQE    id removed          —                    id removed
(reported)         (erased)                                  (popped)
```

---

## 8. Key Invariants

| Invariant | Description |
|-----------|-------------|
| **Posting order = completion order** | `pendingCompletionQue_` preserves insertion order. `reportSendCompletions()` only reports the front WR, enforcing head-of-line blocking. |
| **Unique internalWrId** | Monotonically increasing counter. Even if the user reuses `wrId`, the internal ID is always distinct. |
| **Per-QP FIFO** | Each physical QP's `physicalSendWrStatus_` is a FIFO deque. IB verbs guarantees per-QP completion ordering, so the front always matches the next CQE. |
| **Fragment count = remainingMsgCnt** | Starts at `ceil(length / maxMsgSize)` (+ 1 for SPRAY notify). Each CQE decrements it. When 0, the WR is complete. |
| **Implicit backpressure** | If all QPs are at capacity, the WR stays in `pendingPostQue_` and `dispatchPendingSends()` returns. The next completion retries. |
| **Single-QP bypasses WrTracker** | `postSendSingleQp()` posts directly with the user's `wrId`. No tracker entry. VirtualCq passes the CQE through as-is. |

---

## 9. Usage

The typical usage pattern:

1. Create a `VirtualCq` with physical CQs
2. Create a `VirtualQp` with physical QPs and a pointer to the VirtualCq (auto-registers)
3. Prepare `IbvVirtualSendWr` with `wrId`, `localAddr`, `length`, `remoteAddr`, `opcode`, `sendFlags`, and `deviceKeys`
4. Call `vqp.postSend(sendWr)`
5. Poll completions: `auto wcs = virtualCq.pollCq()` returns `vector<IbvVirtualWc>`
6. Consume completions via `wc.wrId`, `wc.status`, etc.

See `IbvVirtualQp.h` for the full API and `tests/IbverbxDistributedVirtualQpTest.cc` for usage examples.

### Connection Setup (BusinessCard)

`IbvVirtualQpBusinessCard` carries physical QP numbers for connection setup exchange. It is a self-contained struct with `folly::dynamic` + JSON serialization. The ith QP connects to the ith remote QP. `notifyQp_` access is guarded by `std::optional` — `BusinessCard::notifyQpNum_` defaults to 0 when no notifyQp exists.

---

## 10. Performance Characteristics

| VirtualQp Type | postSend() | pollCq() Overhead |
|----------------|------------|-------------------|
| Single-QP (any opcode) | ~70 ns (raw ibverbs + validation) | ~5 ns (pass-through) |
| Multi-QP SEND/RECV/atomics | ~70 ns (same as single-QP) | ~5 ns (pass-through) |
| Multi-QP RDMA | ~155 ns (fragmentation + tracking) | ~25-28 ns (aggregation) |

See [Performance Analysis](./simplified_ibverbx_performance_analysis.md) for detailed nanosecond-level breakdowns.
