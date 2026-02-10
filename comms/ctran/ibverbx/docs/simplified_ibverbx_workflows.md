# Ibverbx Complete Workflows

This document describes the complete workflows for `postSend()` and `pollCq()` in the simplified Ibverbx architecture with essential VirtualCq and registration-based routing.

**Related Documents:**
- [Simplified Ibverbx Design](./simplified_ibverbx_design.md) - Main design document

---

## 1. postSend() Complete Workflow

This shows the complete flow from user calling `postSend()` to physical fragments being posted.

```
User Code
    │
    │ vqp.postSend(IbvVirtualSendWr wr)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ IbvVirtualQp::postSend()                                     │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ FAST PATH CHECK: Single-QP VirtualQp                     ││
│ │                                                          ││
│ │ if (!isMultiQp_) {                                       ││
│ │   return postSendSingleQp(wr);  // Pure passthrough      ││
│ │ }                                                        ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ ══════════════════ MULTI-QP PATH ══════════════════════════ │
│                                                              │
│ [1] Parameter Validation                                     │
│     • Check deviceKeys for multi-NIC                         │
│     • Check length != 0                                      │
│     • Validate opcode (reject SEND_WITH_IMM, atomics)        │
│                                                              │
│ [2] Calculate Fragment Count and Notify Requirement           │
│     • expectedMsgCnt = ceil(length / maxMsgSize_)            │
│     • Determine if SPRAY notify needed:                       │
│       - needsNotify = (SPRAY + WRITE_WITH_IMM)               │
│       - If needsNotify: remainingMsgCnt = expectedMsgCnt + 1 │
│       - Otherwise: remainingMsgCnt = expectedMsgCnt          │
│                                                              │
│ [3] Generate Internal Unique ID                              │
│     • internalId = sendTracker_.nextInternalVirtualWrId_++   │
│     • Guarantees uniqueness even if user reuses wr.wrId      │
│                                                              │
│ [4] Add to Tracker using sendTracker_.add()                  │
│     • sendTracker_.add(                                      │
│         ActiveVirtualWr {                                │
│           .userWrId = wr.wrId,  // User's original ID        │
│           .remainingMsgCnt = remainingMsgCnt,  // +1 if SPRAY│
│           .aggregatedStatus = IBV_WC_SUCCESS,                │
│           .localAddr, .length, .remoteAddr,                  │
│           .opcode, .immData, .deviceKeys,  // Cached         │
│           .offset = 0,                                       │
│           .needsNotify = needsNotify,                        │
│           .notifyPosted = false                              │
│         },                                                   │
│         /*addToOutstanding=*/true);                          │
│                                                              │
│     • Adds to: activeVirtualWrs_, pendingQue_, outstandingQue_│
│                                                              │
│ [5] Call dispatchPendingSends()                        │
│     │                                                        │
│     ▼                                                        │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ dispatchPendingSends()                             ││
│ │                                                          ││
│ │ while (sendTracker_.hasPending()) {                      ││
│ │                                                          ││
│ │   internalId = sendTracker_.frontPending()               ││
│ │   pending = sendTracker_.find(internalId)                ││
│ │                                                          ││
│ │   while (pending->offset < pending->length) {            ││
│ │                                                          ││
│ │     // Find available physical QP                        ││
│ │     qpIdx = findAvailableSendQp()                        ││
│ │     if (qpIdx == -1) return  // All QPs full             ││
│ │                                                          ││
│ │     // Build physical WR fragment                        ││
│ │     fragLen = min(maxMsgSize_, length - offset)          ││
│ │     physicalWr.sg_list.addr = localAddr + offset         ││
│ │     physicalWr.sg_list.length = fragLen                  ││
│ │     physicalWr.wr.rdma.remote_addr = remoteAddr + offset ││
│ │     physicalWr.wr_id = nextPhysicalWrId_++               ││
│ │                                                          ││
│ │     // Post to physical QP (RAW IBVERBS)                 ││
│ │     physicalQps_[qpIdx].postSend(&physicalWr)            ││
│ │                           │                              ││
│ │                           ▼                              ││
│ │                    ┌──────────────┐                      ││
│ │                    │ ibv_post_send│ (70 ns - HW limit)   ││
│ │                    └──────────────┘                      ││
│ │                                                          ││
│ │     // Track for completion correlation                  ││
│ │     physicalQps_[qpIdx].physicalSendQueStatus_            ││
│ │       .push_back({physicalWrId, internalId})             ││
│ │                                                          ││
│ │     // Advance offset                                    ││
│ │     pending->offset += fragLen                           ││
│ │   }                                                      ││
│ │                                                          ││
│ │   // All fragments sent for this WR                      ││
│ │   // (Derived: offset >= length)                         ││
│ │   sendTracker_.popPending()                              ││
│ │   // Note: SPRAY notify handled in reportSendCompletions ││
│ │ }                                                        ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ return success                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 Single-QP Fast Path (postSendSingleQp)

For single-QP VirtualQps, the fast path provides pure passthrough with minimal overhead:

```
User Code
    │
    │ vqp.postSend(IbvVirtualSendWr wr)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ IbvVirtualQp::postSendSingleQp()                             │
│                                                              │
│ [1] Minimal Validation                                       │
│     • Check length != 0                                      │
│                                                              │
│ [2] Build Physical WR Directly                               │
│     • sendWr.wr_id = wr.wrId  // User's ID directly          │
│     • sendSge.addr = wr.localAddr                            │
│     • sendSge.length = wr.length                             │
│     • sendWr.wr.rdma.remote_addr = wr.remoteAddr             │
│                                                              │
│ [3] Post to Physical QP 0                                    │
│     physicalQps_[0].postSend(&sendWr)                        │
│                                                              │
│ [4] NO TRACKING NEEDED                                       │
│     • VirtualCq returns physical WC directly to user         │
│     • No internal ID, no activeVirtualWrs_ entry             │
│                                                              │
│ return success                                               │
└─────────────────────────────────────────────────────────────┘

Benefits:
• ~70 ns overhead (raw ibverbs + minimal validation)
• No fragmentation logic
• No completion tracking
• No memory allocation
```

### 1.2 Fragmentation Logic (dispatchPendingSends)

The `dispatchPendingSends()` function sends fragments from the front of `sendTracker_.pendingQue_` to available physical QPs. It's called from `postSend()` and `processCompletion()`.

**Processing Flow:**

```
dispatchPendingSends()
         │
         ▼
┌─────────────────────────────────────────┐
│ while (!sendTracker_.pendingQue_.empty())       │
│         │                               │
│         ▼                               │
│ ┌─────────────────────────────────────┐ │
│ │ Get front WR from queue             │ │
│ │ • internalId = sendTracker_.pendingQue_.front()│
│ │ • Lookup in sendTracker_.activeVirtualWrs_  │ │
│ │ • DCHECK: WR must exist (never cancelled)   │ │
│ └─────────────────────────────────────┘ │
│         │                               │
│         ▼                               │
│ ┌─────────────────────────────────────┐ │
│ │ All fragments already sent?         │ │
│ │ • If offset >= length → pop & cont  │ │
│ └─────────────────────────────────────┘ │
│         │                               │
│         ▼                               │
│ ┌─────────────────────────────────────┐ │
│ │ while (offset < length)             │ │
│ │   │                                 │ │
│ │   ▼                                 │ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │ Find available QP             │   │ │
│ │ │ • If freedQpIdx hint provided │   │ │
│ │ │   and hasQpCapacity: use it   │   │ │
│ │ │ • Else: findAvailableSendQp() │   │ │
│ │ │ • If none available → return  │   │ │
│ │ │   (will resume when slot frees)│  │ │
│ │ └───────────────────────────────┘   │ │
│ │   │                                 │ │
│ │   ▼                                 │ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │ Calculate fragment            │   │ │
│ │ │ • fragLen = min(maxMsgSize,   │   │ │
│ │ │              length - offset) │   │ │
│ │ └───────────────────────────────┘   │ │
│ │   │                                 │ │
│ │   ▼                                 │ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │ Build physical ibv_send_wr    │   │ │
│ │ │ • buildPhysicalSendWr(        │   │ │
│ │ │     pending, deviceId, fragLen)│  │ │
│ │ │ • Returns {sendWr, sendSge}   │   │ │
│ │ └───────────────────────────────┘   │ │
│ │   │                                 │ │
│ │   ▼                                 │ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │ Post to physical QP           │   │ │
│ │ │ • physicalQps_[qpIdx].postSend()│ │ │
│ │ └───────────────────────────────┘   │ │
│ │   │                                 │ │
│ │   ▼                                 │ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │ Track for completion          │   │ │
│ │ │ • physicalSendQueStatus_.push( │   │ │
│ │ │     physicalWrId, internalId) │   │ │
│ │ └───────────────────────────────┘   │ │
│ │   │                                 │ │
│ │   ▼                                 │ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │ Advance offset                │   │ │
│ │ │ • offset += fragLen           │   │ │
│ │ └───────────────────────────────┘   │ │
│ └─────────────────────────────────────┘ │
│         │                               │
│         ▼                               │
│ ┌─────────────────────────────────────┐ │
│ │ All fragments sent for this WR      │ │
│ │ • offset >= length (derived)        │ │
│ │ • sendTracker_.popPending()         │ │
│ │ • (notify handled in drainOutstanding)│
│ └─────────────────────────────────────┘ │
│         │                               │
│         └──────── next WR ──────────────┘
│
└─────────────────────────────────────────┘
         │
         ▼
    Return success
```

**Key Characteristics:**

| Aspect | Description |
|--------|-------------|
| **Processes in order** | Always takes from front of `sendTracker_.pendingQue_` |
| **Load balancing** | `findAvailableSendQp()` selects QP with capacity |
| **Partial send** | Returns early if no QP slots available; resumes later |
| **Fragment tracking** | Each physical WR tracked with `(physicalWrId, internalWrId)` |
| **SPRAY notify** | Handled in `reportSendCompletions()` when WR is at front and data complete |

### 1.3 State After postSend()

**Multi-QP: Normal case (all fragments sent):**
```
┌──────────────────────────────────────────────────────────────┐
│ sendTracker_.activeVirtualWrs_[internalId] = ActiveVirtualWr {│
│   .userWrId = 42,                                            │
│   .remainingMsgCnt = 4,  // 3 data + 1 notify (if SPRAY)     │
│   .aggregatedStatus = IBV_WC_SUCCESS,                        │
│   .offset = length,      // All fragments sent               │
│   .needsNotify = true,   // true if SPRAY mode               │
│   .notifyPosted = false  // Notify not yet posted            │
│ }                                                            │
│                                                              │
│ sendTracker_.pendingQue_ = []  // Empty, all sent            │
│ sendTracker_.outstandingQue_ = [internalId, ...]  // Waiting │
│                                                              │
│ physicalQps_[0].physicalSendQueStatus_ =                      │
│   [{physWrId=100, internalId}, {physWrId=101, internalId}]   │
│ physicalQps_[1].physicalSendQueStatus_ =                      │
│   [{physWrId=102, internalId}]                               │
└──────────────────────────────────────────────────────────────┘
```

**Multi-QP: If QPs were full (partial progress):**
```
┌──────────────────────────────────────────────────────────────┐
│ sendTracker_.activeVirtualWrs_[internalId] = ActiveVirtualWr {│
│   .offset = 50000,       // Partial progress                 │
│   // offset < length → still has fragments to send           │
│ }                                                            │
│                                                              │
│ sendTracker_.pendingQue_ = [internalId, ...]  // Still in queue│
│                                                              │
│ (Will resume when completion frees a QP slot)                │
└──────────────────────────────────────────────────────────────┘
```

**Single-QP: No tracking state:**
```
┌──────────────────────────────────────────────────────────────┐
│ sendTracker_.activeVirtualWrs_ = {}  // Empty                │
│ sendTracker_.pendingQue_ = []        // Empty                │
│ sendTracker_.outstandingQue_ = []    // Empty                │
│                                                              │
│ physicalQps_[0].physicalSendQueStatus_ = []  // No tracking   │
│                                                              │
│ (Physical WC returned directly to user via VirtualCq)        │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. pollCq() Complete Workflow

This shows the complete flow from user calling `pollCq()` to receiving virtual completions.

**Key Optimization**: The pollCq() has two paths based on the `isMultiQp` flag set during registration:
- **Single-QP VirtualQp** (isMultiQp=false): No fragmentation/aggregation → pass-through directly
- **Multi-QP VirtualQp** (isMultiQp=true): Load balancing with fragmentation → route to VirtualQp

```
User Code
    │
    │ auto wcs = virtualCq.pollCq(32);
    ▼
┌─────────────────────────────────────────────────────────────┐
│ IbvVirtualCq::pollCq(numEntries)                             │
│                                                              │
│ results = []                                                 │
│                                                              │
│ For each physical CQ in physicalCqs_:                        │
│     │                                                        │
│     ▼                                                        │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Poll Physical CQ (RAW IBVERBS)                           ││
│ │   ibv_poll_cq(physicalCq, 32, physicalWcs)               ││
│ │         │                                                ││
│ │         ▼                                                ││
│ │   ┌──────────────┐                                       ││
│ │   │ibv_poll_cq() │ (90 ns base cost)                     ││
│ │   └──────────────┘                                       ││
│ │         │                                                ││
│ │         │ Returns nwcs physical completions              ││
│ │         ▼                                                ││
│ │   For each physical WC:                                  ││
│ │         │                                                ││
│ │         ▼                                                ││
│ │   ┌──────────────────────────────────────┐               ││
│ │   │ Lookup registration info             │               ││
│ │   │ info = registeredQps_.find(          │               ││
│ │   │   {deviceId, wc.qp_num})             │               ││
│ │   └──────────────────────────────────────┘               ││
│ │         │                                                ││
│ │    ┌────┴─────────┐                                      ││
│ │    │ isMultiQp?   │                                      ││
│ │    ▼              ▼                                      ││
│ │  FALSE           TRUE                                    ││
│ │    │              │                                      ││
│ │    │              │                                      ││
│ │ ┌──▼────────────┐ │                                      ││
│ │ │SINGLE-QP      │ │                                      ││
│ │ │FAST PATH      │ │                                      ││
│ │ │               │ │                                      ││
│ │ │Pass through   │ │                                      ││
│ │ │directly       │ │                                      ││
│ │ │               │ │                                      ││
│ │ │results.       │ │                                      ││
│ │ │ push_back     │ │                                      ││
│ │ │ (wc)          │ │                                      ││
│ │ └───────────────┘ │                                      ││
│ │                   │                                      ││
│ │                   ▼                                      ││
│ │         ┌─────────────────────────────────────────┐      ││
│ │         │ MULTI-QP: Route to VirtualQp            │      ││
│ │         │                                         │      ││
│ │         │ virtualWcs = vqp->processCompletion(wc) │      ││
│ │         │                │                        │      ││
│ │         │                ▼                        │      ││
│ │         │ ┌─────────────────────────────────────┐ │      ││
│ │         │ │ IbvVirtualQp::processCompletion()   │ │      ││
│ │         │ │                                     │ │      ││
│ │         │ │ 2x2 Matrix Dispatch:                │ │      ││
│ │         │ │ ┌──────────┬──────────┬──────────┐  │ │      ││
│ │         │ │ │          │  Send    │  Recv    │  │ │      ││
│ │         │ │ ├──────────┼──────────┼──────────┤  │ │      ││
│ │         │ │ │ NotifyQp │ process  │ process  │  │ │      ││
│ │         │ │ │          │ NotifyQp │ NotifyQp │  │ │      ││
│ │         │ │ │          │ Send-    │ Recv-    │  │ │      ││
│ │         │ │ │          │ Compl..  │ Compl..  │  │ │      ││
│ │         │ │ ├──────────┼──────────┼──────────┤  │ │      ││
│ │         │ │ │ DataQp   │ process  │ process  │  │ │      ││
│ │         │ │ │          │ DataQp   │ DataQp   │  │ │      ││
│ │         │ │ │          │ Send-    │ Recv-    │  │ │      ││
│ │         │ │ │          │ Compl..  │ Compl..  │  │ │      ││
│ │         │ │ └──────────┴──────────┴──────────┘  │ │      ││
│ │         │ │                                     │ │      ││
│ │         │ │ Each processor follows:             │ │      ││
│ │         │ │  [1] Pop physical queue status      │ │      ││
│ │         │ │  [2] Update WR state                │ │      ││
│ │         │ │  [3] Report completed WRs in-order  │ │      ││
│ │         │ │  [4] Schedule more work / cleanup   │ │      ││
│ │         │ │                                     │ │      ││
│ │         │ │ Note: processDataQpRecvCompletion   │ │      ││
│ │         │ │       is DQPLB only (uses           │ │      ││
│ │         │ │       DqplbSeqTracker)              │ │      ││
│ │         │ │                                     │ │      ││
│ │         │ │ return virtualWcs (may be empty)    │ │      ││
│ │         │ └─────────────────────────────────────┘ │      ││
│ │         │                │                        │      ││
│ │         │                ▼                        │      ││
│ │         │   For each returned virtual wc:         │      ││
│ │         │     results.push_back(virtualWc)        │      ││
│ │         └─────────────────────────────────────────┘      ││
│ │                                                          ││
│ │    (Both paths converge to results)                      ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ return results  // vector<IbvVirtualWc>                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
User Code receives:
  - Virtual CQEs from registered VirtualQps (aggregated, in-order)
  - Physical CQEs from unregistered QPs (pass-through)
```

**SPRAY WR Completion Flow (using needsNotify and notifyPosted booleans):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPRAY Send: remainingMsgCnt = numFragments + 1, needsNotify = true          │
│ Non-SPRAY:  remainingMsgCnt = numFragments, needsNotify = false             │
└─────────────────────────────────────────────────────────────────────────────┘

SPRAY WR Lifecycle:

    postSend():
      remainingMsgCnt = 4 (3 data + 1 notify)
      needsNotify = true
      notifyPosted = false
                │
                ▼
    Data CQEs arrive (in any order):
      remainingMsgCnt: 4 → 3 → 2 → 1
                │
                ▼
    reportSendCompletions() (WR at front, remainingMsgCnt == 1):
      needsNotify = true, notifyPosted = false
      → Post notify to notifyQp
      → notifyPosted = true
      → break (wait for notify CQE)
                │
                ▼
    Notify CQE arrives:
      remainingMsgCnt: 1 → 0
                │
                ▼
    reportSendCompletions() (WR at front, remainingMsgCnt == 0):
      → Report to user, remove WR, pop outstandingQue_

Backpressure case:
    If notifyQp full when posting notify:
      → Queue in pendingNotifyQue_
      → notifyPosted = true
      → When backpressure clears: flushPendingSendNotifies() posts it

Non-SPRAY operations:
  - needsNotify = false
  - isComplete() = (remainingMsgCnt == 0)
  - No notify handling needed
```

---

## 3. Example Scenario: Multi-Fragment Message

**User posts 300KB message with maxMsgSize=100KB:**

```
Time  Event                         State
════  ═════════════════════════     ══════════════════════════════════════
T0    vqp.postSend(wr)              sendTracker_.activeVirtualWrs_[1] = {
                                      .userWrId = 42,
                                      .remainingMsgCnt = 3,
                                      .offset = 0,
                                      .needsNotify = false  // non-SPRAY
                                    }
                                    sendTracker_.pendingQue_ = [1]
                                    sendTracker_.outstandingQue_ = [1]

T1    dispatchPendingSends()  Posted 3 physical fragments:
                                    - Fragment 0: 100KB → physWrId=100
                                    - Fragment 1: 100KB → physWrId=101
                                    - Fragment 2: 100KB → physWrId=102

                                    activeVirtualWrs_[1].offset = 300KB
                                    sendTracker_.pendingQue_ = []  // Removed

─────────────────────────────────────────────────────────────────────────

T2    virtualCq.pollCq(32)          ibv_poll_cq() returns:
                                      physicalWc{wr_id=101, status=SUCCESS}

                                    VirtualCq routes to VirtualQp:
                                      internalId = 1
                                      activeVirtualWrs_[1]
                                        .remainingMsgCnt = 2

                                    reportSendCompletions():
                                      Front = 1, needsNotify = false
                                      isComplete() = false (remainingMsgCnt > 0)
                                      → Stop (head-of-line blocking)

                                    Return: []

T3    virtualCq.pollCq(32)          ibv_poll_cq() returns:
                                      physicalWc{wr_id=102, status=SUCCESS}

                                    VirtualQp updates:
                                      activeVirtualWrs_[1]
                                        .remainingMsgCnt = 1

                                    Still not complete → return []

T4    virtualCq.pollCq(32)          ibv_poll_cq() returns:
                                      physicalWc{wr_id=100, status=SUCCESS}

                                    VirtualQp updates:
                                      activeVirtualWrs_[1]
                                        .remainingMsgCnt = 0
                                        isComplete() = true (remainingMsgCnt == 0)

                                    reportSendCompletions():
                                      ✓ Front is complete!
                                      → Build IbvVirtualWc:
                                        {wrId=42, status=SUCCESS,
                                         byteLen=300KB, qpNum=virtualQpNum}
                                      → sendTracker_.remove(1)
                                      → sendTracker_.popOutstanding()

                                    Return [{wrId=42, ...}]

═════  ═════════════════════════════  ════════════════════════════════════
User receives: wcs[0].wrId == 42  (original userWrId)
               wcs[0].byteLen == 300KB
               wcs[0].status == SUCCESS
```

---

## 4. Example Scenario: Out-of-Order Completions with In-Order Reporting

This shows how `sendTracker_.outstandingQue_` ensures in-order reporting even when physical completions arrive out of order.

**User posts two messages: WR-A (2 fragments) then WR-B (1 fragment):**

```
Time  Event                         Internal State
════  ═════════════════════════     ══════════════════════════════════════
T0    vqp.postSend(wrA)             sendTracker_.activeVirtualWrs_ = {
      (userWrId=100, 200KB)           1: {userWrId=100, remainingMsgCnt=2,
                                          needsNotify=false}
                                    }
                                    sendTracker_.pendingQue_ = [1]
                                    sendTracker_.outstandingQue_ = [1]

T1    vqp.postSend(wrB)             sendTracker_.activeVirtualWrs_ = {
      (userWrId=200, 80KB)            1: {userWrId=100, remainingMsgCnt=2},
                                      2: {userWrId=200, remainingMsgCnt=1,
                                          needsNotify=false}
                                    }
                                    sendTracker_.pendingQue_ = [1, 2]
                                    sendTracker_.outstandingQue_ = [1, 2]

T2    mapPendingSend...             Posted all fragments:
                                    - WR-A frag0 → physWrId=10 → QP[0]
                                    - WR-A frag1 → physWrId=11 → QP[1]
                                    - WR-B frag0 → physWrId=12 → QP[0]

─────────────────────────────────────────────────────────────────────────

T3    pollCq()                      Physical completion: physWrId=12 (WR-B)

                                    VirtualQp updates:
                                      activeVirtualWrs_[2].remainingMsgCnt = 0
                                      activeVirtualWrs_[2].isComplete() = true

                                    reportSendCompletions():
                                      Front = 1 (WR-A)
                                      WR-A.isComplete() = false
                                      → STOP (cannot report WR-B yet!)

                                    Return: []  ← WR-B complete but NOT reported

T4    pollCq()                      Physical completion: physWrId=10 (WR-A frag0)

                                    VirtualQp updates:
                                      activeVirtualWrs_[1].remainingMsgCnt = 1

                                    reportSendCompletions():
                                      Front = 1, not complete
                                    Return: []

T5    pollCq()                      Physical completion: physWrId=11 (WR-A frag1)

                                    VirtualQp updates:
                                      activeVirtualWrs_[1].remainingMsgCnt = 0
                                      activeVirtualWrs_[1].isComplete() = true

                                    reportSendCompletions():
                                      ✓ Front = 1 (WR-A), complete!
                                        → Report {wrId=100, status=SUCCESS}
                                        → remove [1], popOutstanding()
                                      ✓ Front = 2 (WR-B), complete!
                                        → Report {wrId=200, status=SUCCESS}
                                        → remove [2], popOutstanding()

                                    Return: [{wrId=100}, {wrId=200}]
                                            ↑ WR-A first, then WR-B (in posting order!)

═════════════════════════════════════════════════════════════════════════
Key Insight: Even though WR-B completed first (T3), it was not reported
             until WR-A completed (T5). User sees completions in the same
             order they called postSend().
```

---

## 5. Example Scenario: SPRAY Mode with Notify Tracking

**User posts message with RDMA_WRITE_WITH_IMM in SPRAY mode:**

```
Time  Event                         State
════  ═════════════════════════     ══════════════════════════════════════
T0    vqp.postSend(wr)              SPRAY mode + WRITE_WITH_IMM detected
      opcode=WRITE_WITH_IMM         remainingMsgCnt = 3 (2 data + 1 notify)
      length=200KB
      maxMsgSize=100KB              sendTracker_.activeVirtualWrs_[1] = {
                                      .userWrId = 42,
                                      .remainingMsgCnt = 3,  // data + notify!
                                      .needsNotify = true,
                                      .notifyPosted = false
                                    }
                                    sendTracker_.outstandingQue_ = [1]

T1    mapPendingSend...             Posted 2 DATA fragments:
                                    - Fragment 0: 100KB → physWrId=100 → QP[0]
                                    - Fragment 1: 100KB → physWrId=101 → QP[1]

                                    offset = 200KB (all data sent)
                                    sendTracker_.pendingQue_ = []

─────────────────────────────────────────────────────────────────────────

T2    pollCq()                      Completion: physWrId=100
                                    remainingMsgCnt = 2

                                    reportSendCompletions():
                                      needsNotify = true, notifyPosted = false
                                      remainingMsgCnt > 1 → data not complete
                                      → Head-of-line blocking

                                    Return: []

T3    pollCq()                      Completion: physWrId=101
                                    remainingMsgCnt = 1

                                    reportSendCompletions():
                                      needsNotify = true, notifyPosted = false
                                      remainingMsgCnt == 1 → all data done!
                                      → Check backpressure: OK
                                      → postNotifyForWr(1)
                                        Posts zero-length WRITE_WITH_IMM:
                                        physWrId = 102 → notifyQp_
                                      → notifyPosted = true
                                      → break (wait for notify CQE)

                                    Return: [] (waiting for notify CQE)

T4    pollCq()                      Completion: physWrId=102 (from notifyQp_)

                                    isNotifyQp = true
                                    → remainingMsgCnt = 0

                                    reportSendCompletions():
                                      remainingMsgCnt == 0
                                      → Report + remove + pop outstandingQue_
                                      results.push_back(buildVirtualWc(*pending))
                                      sendTracker_.remove(internalId)

                                    flushPendingSendNotifies() (no-op, queue empty)

                                    Return: [{wrId=42, status=SUCCESS}]

═════════════════════════════════════════════════════════════════════════
Receiver Side:
  - Receives 2 RDMA_WRITE completions (no CQE, just data)
  - Receives 1 RDMA_WRITE_WITH_IMM completion (CQE with immediate data)
  - Immediate data signals "all data fragments received"
```

---

## 6. Example Scenario: SPRAY Notify with Backpressure

**Multiple SPRAY WRs hit notify backpressure:**

```
Time  Event                         State
════  ═════════════════════════     ══════════════════════════════════════
T0    Many SPRAY WRs posted         notifyQp_->physicalSendQueStatus_.size()
                                      >= kMaxOutstandingNotifies (256)

                                    WR-A data complete, WR-B data complete...
                                    outstandingQue_ = [A, B, ...]

T1    reportSendCompletions()       WR-A at front:
      for WR-A                        needsNotify = true, notifyPosted = false
                                      remainingMsgCnt == 1 → all data done

                                    Check backpressure:
                                      notifyQp_->physicalSendQueStatus_.size() >= 256
                                      → BLOCKED!

                                    → pendingNotifyQue_.push_back(internalId_A)
                                    → notifyPosted = true (to prevent re-queuing)
                                    → WR stays in outstandingQue_ (for in-order)
                                    → break (head-of-line blocking)

                                    Note: WR-B cannot be processed yet because
                                    WR-A is still at front of outstandingQue_

                                    pendingNotifyQue_ = [A]
                                    outstandingQue_ = [A, B, ...]

─────────────────────────────────────────────────────────────────────────

T2    pollCq()                      Notify CQE received (from earlier WR)
                                    → notifyQp_->physicalSendQueStatus_.pop_front()
                                    → Size now < 256

                                    flushPendingSendNotifies():
                                      Front = A
                                      Backpressure check: OK (size < 256)
                                      → postNotifyForWr(A)
                                      → pendingNotifyQue_.pop_front()

                                    pendingNotifyQue_ = []
                                    outstandingQue_ = [A, B, ...] (A still waiting for CQE)

T3    pollCq()                      Notify CQE for WR-A received
                                    → remainingMsgCnt = 0

                                    reportSendCompletions():
                                      WR-A at front, remainingMsgCnt == 0
                                      → Report + remove + pop outstandingQue_

                                      WR-B now at front:
                                        needsNotify = true, notifyPosted = false
                                        remainingMsgCnt == 1 → all data done
                                      → Check backpressure: OK
                                      → postNotifyForWr(B)
                                      → notifyPosted = true
                                      → break (wait for notify CQE)

                                    Return: [{WR-A completion}]
                                    outstandingQue_ = [B, ...]

═════════════════════════════════════════════════════════════════════════
Key Insight: WRs stay in outstandingQue_ until remainingMsgCnt == 0, ensuring
             in-order completion even when notify CQEs arrive out-of-order.
             pendingNotifyQue_ acts as overflow buffer when backpressure hits.
```

---

## 6.1. Example Scenario: DQPLB Recv Tracking

**DQPLB mode uses sequence numbers in IMM data for in-order completion tracking:**

```
Setup:
  - LoadBalancingScheme::DQPLB enabled
  - 4 physical QPs for load balancing
  - DqplbSeqTracker tracks sequence numbers
  - Pre-posted zero-length recvs on all physical QPs (virtualWrId = -1)

User posts 3 recvs:
  recvTracker_.activeVirtualWrs_:
    [id=1] = {userWrId=100, remainingMsgCnt=1}
    [id=2] = {userWrId=101, remainingMsgCnt=1}
    [id=3] = {userWrId=102, remainingMsgCnt=1}
  recvTracker_.outstandingQue_ = [1, 2, 3]

Time  Event                         State
════  ═════════════════════════     ══════════════════════════════════════
T0    Sender sends msg 0            IMM data contains seq=0
      arrives on QP[2]              Pre-posted recv on QP[2] gets CQE

T1    pollCq()                      Completion: physWrId on QP[2]
      → processCompletion()         freedQpIdx = 2

                                    dqplbSeqTracker_.processReceivedImm(seq=0):
                                      nextExpectedSeq_ = 0 → match!
                                      nextExpectedSeq_ = 1
                                      return notifyCount = 1

                                    Loop notifyCount times:
                                      frontId = 1
                                      activeVirtualWrs_[1].remainingMsgCnt = 0

                                    reportRecvCompletions():
                                      front WR (id=1) isComplete() = true
                                      results.push_back({wrId=100, ...})
                                      remove(1), pop outstandingQue_

                                    replenishDqplbRecv(qpIdx=2):
                                      Post new zero-length recv on QP[2]

                                    Return: [{wrId=100, status=SUCCESS}]

                                    recvTracker_.outstandingQue_ = [2, 3]

─────────────────────────────────────────────────────────────────────────

T2    Sender sends msg 2            IMM data contains seq=2
      arrives on QP[0]              (msg 1 not yet arrived - out of order!)

T3    pollCq()                      Completion: physWrId on QP[0]
      → processCompletion()         freedQpIdx = 0

                                    dqplbSeqTracker_.processReceivedImm(seq=2):
                                      nextExpectedSeq_ = 1 → mismatch!
                                      Buffer seq=2 in pendingSeqs_
                                      return notifyCount = 0

                                    Loop 0 times (no-op)

                                    reportRecvCompletions():
                                      front WR (id=2) remainingMsgCnt = 1
                                      → not complete, break

                                    replenishDqplbRecv(qpIdx=0):
                                      Post new zero-length recv on QP[0]

                                    Return: [] (waiting for in-order seq)

─────────────────────────────────────────────────────────────────────────

T4    Sender sends msg 1            IMM data contains seq=1
      arrives on QP[1]              (fills the gap!)

T5    pollCq()                      Completion: physWrId on QP[1]
      → processCompletion()         freedQpIdx = 1

                                    dqplbSeqTracker_.processReceivedImm(seq=1):
                                      nextExpectedSeq_ = 1 → match!
                                      nextExpectedSeq_ = 2
                                      Check pendingSeqs_: seq=2 found!
                                      nextExpectedSeq_ = 3
                                      return notifyCount = 2 (msgs 1 and 2)

                                    Loop 2 times:
                                      i=0: frontId = 2, remainingMsgCnt = 0
                                      i=1: frontId = 3, remainingMsgCnt = 0

                                    reportRecvCompletions():
                                      front WR (id=2) isComplete() = true
                                      results.push_back({wrId=101, ...})
                                      remove(2), pop outstandingQue_

                                      front WR (id=3) isComplete() = true
                                      results.push_back({wrId=102, ...})
                                      remove(3), pop outstandingQue_

                                    replenishDqplbRecv(qpIdx=1):
                                      Post new zero-length recv on QP[1]

                                    Return: [{wrId=101}, {wrId=102}]

═════════════════════════════════════════════════════════════════════════
Key Insight:
  - User postRecv() calls are tracked in recvTracker_ with remainingMsgCnt=1
  - DqplbSeqTracker handles out-of-order physical arrivals
  - notifyCount determines how many front WRs to complete
  - Virtual completions are always reported in posting order
  - replenishDqplbRecv() maintains the pre-posted recv pool
```

---

## 7. Example Scenario: Single-QP vs Multi-QP VirtualQp

**User has both single-QP and multi-QP VirtualQps sharing the same VirtualCq:**

```
Setup:
┌────────────────────────────────────────────────────────────────┐
│ IbvVirtualCq virtualCq;                                        │
│                                                                │
│ // Multi-QP VirtualQp (load balancing)                         │
│ IbvVirtualQp multiQpVqp(qps, &virtualCq, ...);                 │
│   → physicalQps_.size() = 4                                    │
│   → isMultiQp_ = true                                          │
│   → Registers: QP[10], QP[11], QP[12], QP[13], notifyQP[20]    │
│                                                                │
│ // Single-QP VirtualQp (simple, no load balancing)             │
│ IbvVirtualQp singleQpVqp(singleQp, &virtualCq, ...);           │
│   → physicalQps_.size() = 1                                    │
│   → isMultiQp_ = false                                         │
│   → Registers: QP[30]                                          │
│                                                                │
│ virtualCq.registeredQps_ = {                                   │
│   {dev0, QP10} → {&multiQpVqp, isMultiQp=true},                │
│   {dev0, QP11} → {&multiQpVqp, isMultiQp=true},                │
│   {dev0, QP12} → {&multiQpVqp, isMultiQp=true},                │
│   {dev0, QP13} → {&multiQpVqp, isMultiQp=true},                │
│   {dev0, QP20} → {&multiQpVqp, isMultiQp=true},  // notifyQp   │
│   {dev0, QP30} → {&singleQpVqp, isMultiQp=false}               │
│ }                                                              │
└────────────────────────────────────────────────────────────────┘

Runtime:
┌────────────────────────────────────────────────────────────────┐
│ multiQpVqp.postSend(largeMsg);   // Fragments across QP[10-13] │
│                                  // Uses sendTracker_ for tracking│
│                                                                │
│ singleQpVqp.postSend(smallMsg);  // Uses QP[30] directly       │
│                                  // NO tracking, postSendSingleQp()│
└────────────────────────────────────────────────────────────────┘

Poll Results:
┌────────────────────────────────────────────────────────────────┐
│ auto wcs = virtualCq.pollCq(32);                               │
│                                                                │
│ // Physical CQEs received:                                     │
│ // 1. {qp_num=10, wr_id=100}  ← Multi-QP fragment              │
│ // 2. {qp_num=30, wr_id=999}  ← Single-QP completion           │
│ // 3. {qp_num=11, wr_id=101}  ← Multi-QP fragment              │
│                                                                │
│ Processing:                                                    │
│ 1. QP[10] → isMultiQp=true → route to multiQpVqp               │
│    → processCompletion() returns [] (intermediate fragment)    │
│                                                                │
│ 2. QP[30] → isMultiQp=false → FAST PATH (pass-through)         │
│    → results.push_back({qp_num=30, wr_id=999})                 │
│                                                                │
│ 3. QP[11] → isMultiQp=true → route to multiQpVqp               │
│    → WR complete, returns [{wrId=userWrId, ...}]               │
│                                                                │
│ Final results = [                                              │
│   {qpNum=30, wrId=999},             // Single-QP (fast path)   │
│   {qpNum=virtualQpNum, wrId=userWrId}  // Multi-QP (aggregated)│
│ ]                                                              │
└────────────────────────────────────────────────────────────────┘

User receives:
  - Single-QP completions: passed through directly (minimal overhead)
  - Multi-QP completions: aggregated with userWrId restored
```

**Performance Benefit:**

| VirtualQp Type | pollCq() Overhead | Why |
|----------------|-------------------|-----|
| Single-QP (isMultiQp=false) | ~5 ns | Direct pass-through, no aggregation |
| Multi-QP (isMultiQp=true) | ~25-28 ns | Fragment aggregation, in-order tracking |

---

## 8. Key Data Structure States Summary

### 8.1 WrTracker Three-Structure Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ sendTracker_.activeVirtualWrs_: F14FastMap<uint64_t, ActiveVirtualWr>│
│   • Key: internalWrId (unique, monotonically increasing)            │
│   • Value: All state for a virtual WR                               │
│   • Lifecycle: Created at postSend(), erased when reported          │
├─────────────────────────────────────────────────────────────────────┤
│ sendTracker_.pendingQue_: std::deque<uint64_t>                      │
│   • Contains: internalWrIds of WRs with unsent fragments            │
│   • FIFO: Front WR gets fragments sent first                        │
│   • Empty when: All fragments of all WRs have been posted           │
├─────────────────────────────────────────────────────────────────────┤
│ sendTracker_.outstandingQue_: std::deque<uint64_t>                  │
│   • Contains: internalWrIds in posting order                        │
│   • FIFO: Front WR must complete before later WRs are reported      │
│   • Purpose: Ensures in-order completion reporting                  │
├─────────────────────────────────────────────────────────────────────┤
│ pendingNotifyQue_: std::deque<uint64_t>  (SPRAY mode only)          │
│   • Contains: internalWrIds of WRs with notifyPosted=true but queued│
│   • Added when: All data done + backpressure prevents notify post   │
│   • Processed by: flushPendingSendNotifies() when backpressure clears   │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 SPRAY Notify Tracking (Boolean-based)

```
┌───────────────────────────────────────────────────────────────────┐
│ Non-SPRAY Operation:                                              │
│   needsNotify = false                                             │
│   remainingMsgCnt = numFragments (data only)                      │
│   isComplete() = (remainingMsgCnt == 0)                           │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ SPRAY + WRITE_WITH_IMM:                                           │
│   needsNotify = true                                              │
│   notifyPosted = false                                            │
│   remainingMsgCnt = numFragments + 1  (data + notify)             │
└───────────────────────────────────────────────────────────────────┘

SPRAY WR Lifecycle:

    ┌────────────────────────────────────────┐
    │ postSend():                            │
    │   remainingMsgCnt = numFragments + 1   │
    │   needsNotify = true                   │
    │   notifyPosted = false                 │
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │ Data CQEs arrive:                      │
    │   remainingMsgCnt decrements           │
    │   (e.g., 4 → 3 → 2 → 1)                │
    └───────────────────┬────────────────────┘
                        │ remainingMsgCnt == 1
                        │ AND WR at front of outstandingQue_
                        ▼
    ┌────────────────────────────────────────┐
    │ reportSendCompletions():               │
    │   needsNotify=true, notifyPosted=false │
    │   All data done (remainingMsgCnt == 1) │
    └───────────┬────────────────────────────┘
                │
       ┌────────┴─────────┐
       ▼ (capacity)       ▼ (backpressure)
    ┌─────────────────┐  ┌───────────────────────┐
    │ Post notify     │  │ Queue in              │
    │ notifyPosted=   │  │   pendingNotifyQue_   │
    │   true          │  │ notifyPosted = true   │
    └────────┬────────┘  └───────────┬───────────┘
             │                       │ flushPendingSendNotifies()
             └───────────┬───────────┘
                         ▼
    ┌────────────────────────────────────────┐
    │ Notify CQE received:                   │
    │   remainingMsgCnt decrements (1 → 0)   │
    │   isComplete() = true                  │
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │ COMPLETION                             │
    │   WR is reported via buildVirtualWc()  │
    │   WR removed from sendTracker_         │
    │   WR popped from outstandingQue_       │
    └────────────────────────────────────────┘
```

### 8.3 State Transitions

```
postSend() called (multi-QP):
  → Create entry in sendTracker_.activeVirtualWrs_[internalId]
  → Push internalId to sendTracker_.pendingQue_
  → Push internalId to sendTracker_.outstandingQue_
  → Set remainingMsgCnt = numFragments + 1 (if SPRAY), else numFragments
  → Set needsNotify = true (if SPRAY), else false
  → Set notifyPosted = false

postSend() called (single-QP):
  → NO tracking state created
  → Direct post to physicalQps_[0]

mapPendingSend... sends all fragments:
  → Update activeVirtualWrs_[id].offset = length
  → Remove internalId from sendTracker_.pendingQue_
  → (SPRAY notify handled in reportSendCompletions)

processCompletion() for data CQE:
  → Decrement activeVirtualWrs_[id].remainingMsgCnt
  → Aggregate error status if needed
  → Call reportSendCompletions() to drain completed WRs

processCompletion() for notify CQE:
  → Decrement activeVirtualWrs_[id].remainingMsgCnt (1 → 0)
  → Call reportSendCompletions() to report WRs in order
  → Call flushPendingSendNotifies() to process backpressure queue

reportSendCompletions() processes front of outstandingQue_:
  → Check needsNotify and notifyPosted:
    - needsNotify=false + remainingMsgCnt==0: report, remove, pop
    - needsNotify=true + notifyPosted=false + remainingMsgCnt==1: post notify, set notifyPosted=true, break
    - needsNotify=true + notifyPosted=true + remainingMsgCnt>0: break (wait for notify CQE)
    - needsNotify=true + remainingMsgCnt==0: report, remove, pop
  → Returns completions in posting order
```
