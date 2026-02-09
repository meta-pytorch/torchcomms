# IBVerbX Simplified Design — Code Review Guide

## Change Summary

Replace the Coordinator singleton with a registration-based direct routing architecture between `IbvVirtualQp` and `IbvVirtualCq`. Introduce new flat WR/WC types at the virtual boundary, a `WrTracker` template for unified send/recv tracking, and a two-phase `pollCq()` drain pattern.

**Diff stats**: 22 files changed, ~6000 insertions, ~2100 deletions (includes design docs and `.claude/CLAUDE.md`).

**External callers**: None. The IB backend (`comms/ctran/backends/ib/`) uses raw `IbvQp` objects, not VirtualQp/VirtualCq. No caller migration is needed.

**Test results**: Unit tests 20/20 pass. Distributed tests 441/441 pass (SPRAY + DQPLB modes, various buffer sizes 64MB–1GB, 16–1024 QPs, INT8/INT32/FLOAT data types).

---

## Files Changed

| File | Lines | What changed |
|------|-------|-------------|
| `IbvCommon.h` | +20 | Added `MemoryRegionKeys`, `IbvVirtualWc` |
| `IbvVirtualWr.h` | +160/−20 | Added `IbvVirtualSendWr`, `IbvVirtualRecvWr`, `ActiveVirtualWr`, `WrTracker<T>`, unified `PhysicalWrStatus`. Kept old types removed in later diff. |
| `DqplbSeqTracker.h` | +3/−3 | `getSendImm(int remainingMsgCnt)` → `getSendImm(bool isLastFragment)` |
| `IbvQp.h` | +8/−9 | Unified `PhysicalWrStatus`. Renamed deques to `physicalSendQueStatus_`/`physicalRecvQueStatus_`. |
| `IbvQp.cc` | +10/−10 | Updated method names matching renamed deques. |
| `IbvVirtualCq.h` | +200/−220 | **Major rewrite.** Registration table, two-phase `pollCq()`, removed Coordinator-based routing. |
| `IbvVirtualCq.cc` | +50/−30 | Registration methods, move semantics with pointer updates, removed Coordinator calls. |
| `IbvVirtualQp.h` | +870/−560 | **Major rewrite.** New `postSend`/`postRecv` accepting flat WR types, `WrTracker`, `processCompletion()` with 2×2 dispatch, all inline helpers. |
| `IbvVirtualQp.cc` | +160/−60 | Constructor registers with VirtualCq, destructor unregisters, move ops, `notifyQp_` as `optional<IbvQp>`. |
| `IbvPd.h` | +3/−2 | `createVirtualQp()`: single `IbvVirtualCq*` param (was two CQ pointers). |
| `IbvPd.cc` | +25/−15 | Conditional notifyQp creation (SPRAY only). Single CQ for send/recv. |
| `Coordinator.h` | −179 | **Deleted.** |
| `Coordinator.cc` | −187 | **Deleted.** |
| `BUCK` | +5/−4 | Removed `Coordinator.cc`, `//folly:singleton` dep. Added `//folly/container:f14_hash`. |
| `Ibverbx.h` | −1 | Removed `Coordinator.h` include. |
| `tests/IbverbxTest.cc` | +300/−185 | Removed Coordinator tests, updated VirtualQp/VirtualCq tests for new API. |
| `tests/IbverbxDistributedVirtualQpTest.cc` | +100/−120 | Migrated to `IbvVirtualSendWr`/`IbvVirtualRecvWr`/`IbvVirtualWc`. |
| `tests/IbverbxDistributedVirtualQpGb200Test.cc` | +70/−75 | Same migration for multi-NIC tests. |
| `tests/IbverbxDistributedVirtualQpTestDqplb.cc` | (new target) | Added to `tests/BUCK`. |
| `tests/BUCK` | +15 | Added DQPLB distributed test target. |

---

## Architecture: Before vs After

### Before (Coordinator pattern)

```
VirtualQp ──request──► Coordinator (singleton) ──response──► VirtualCq
                           │
                    global lookup maps
                    (VirtualQp* by wrId)
```

- `VirtualQp::postSend()` builds `VirtualQpRequest`, sends to Coordinator.
- Coordinator maintains `registeredVirtualQps_` map, routes requests.
- `VirtualCq::pollCq()` calls Coordinator to translate physical WCs.
- `VirtualSendWr` / `VirtualRecvWr` deep-copy `ibv_send_wr` + SGE lists.
- Two separate CQ pointers (`sendCq`, `recvCq`) in `createVirtualQp()`.

### After (Registration pattern)

```
VirtualQp ──register──► VirtualCq (registration table)
                              │
VirtualQp ◄──processCompletion()── VirtualCq::pollCq()
```

- Constructor registers each physical QP with VirtualCq.
- Destructor unregisters.
- `pollCq()` uses F14FastMap lookup to route CQEs directly.
- For multi-QP RDMA ops: calls `vqp->processCompletion()` inline.
- Flat WR types (`IbvVirtualSendWr`, `IbvVirtualRecvWr`) replace deep copies.
- Single CQ for both send/recv.

---

## Key Components to Review

### 1. New Type Definitions (`IbvCommon.h`, `IbvVirtualWr.h`)

**`IbvVirtualWc`** — Custom completion struct at the virtual boundary:
```cpp
struct IbvVirtualWc {
  uint64_t wrId;              // User's original WR ID
  ibv_wc_status status;       // Aggregated (first error wins)
  ibv_wc_opcode opcode;       // From physical WC
  uint32_t qpNum;             // Virtual QP number
  uint32_t immData;           // Immediate data
  uint32_t byteLen;           // Total byte length
};
```

**`IbvVirtualSendWr` / `IbvVirtualRecvWr`** — Flat WR types replacing `ibv_send_wr` deep copies. Use `F14FastMap<int32_t, MemoryRegionKeys> deviceKeys` for multi-NIC MR key lookup instead of `ibv_sge` arrays.

**`ActiveVirtualWr`** — Unified internal tracking for both send and recv. Key fields: `remainingMsgCnt` (decremented per CQE), `aggregatedStatus` (first error wins), `offset` (fragmentation progress), `needsNotify` / `notifyPosted` (SPRAY state).

**`WrTracker<T>`** — Three-structure pattern:
- `activeVirtualWrs_` (F14FastMap) — O(1) lookup by internal ID
- `pendingQue_` (deque) — WRs not yet fully posted to physical QPs
- `outstandingQue_` (deque) — WRs posted, awaiting completion

**`PhysicalWrStatus`** — Unified struct (was separate `PhysicalSendWrStatus`/`PhysicalRecvWrStatus`). Correlates physical WR IDs to internal virtual WR IDs.

### 2. VirtualCq Registration and Two-Phase pollCq (`IbvVirtualCq.h/.cc`)

**Registration table**: `F14FastMap<QpId, RegisteredQpInfo>` where `QpId = {deviceId, qpNum}`.

**`pollCq(int numEntries)`** — Two phases:

| Phase | What happens |
|-------|-------------|
| **Phase 1 (Drain)** | For each physical CQ, batch-poll 32 CQEs in a while loop until empty. Route each CQE by looking up the registration table. |
| **Phase 2 (Return)** | Pop up to `numEntries` from `completedVirtualWcs_` deque. |

**CQE routing in Phase 1** (three paths):

| Condition | Action |
|-----------|--------|
| `!isMultiQp` | Direct passthrough → `IbvVirtualWc` |
| `isMultiQp` && `SEND`/`RECV` opcode | Passthrough (no aggregation needed) |
| `isMultiQp` && RDMA opcode | Call `vqp->processCompletion()` for fragment aggregation |

**Review focus**: Move semantics update `vqp->virtualCq_` pointers for all registered QPs (lines 29–33 of `.cc`).

### 3. VirtualQp Rewrite (`IbvVirtualQp.h/.cc`)

**Constructor** (`IbvVirtualQp.cc:14–48`):
- Takes `vector<IbvQp>&&` (data QPs) + `optional<IbvQp>&&` (notifyQp, SPRAY only) + single `IbvVirtualCq*`
- Builds `qpNumToIdx_` map, computes `deviceCnt_`, calls `registerWithVirtualCq()`

**`postSend()` routing** (`IbvVirtualQp.h:275–349`):

| Condition | Path |
|-----------|------|
| `!isMultiQp_` | `postSendSingleQp()` — pure passthrough |
| `IBV_WR_SEND` | `postSendSingleQp()` — no fragmentation |
| RDMA ops | Add to `sendTracker_`, call `dispatchPendingSends()` |

**`postRecv()` routing** (`IbvVirtualQp.h:380–438`):

| Condition | Path |
|-----------|------|
| `!isMultiQp_` | `postRecvSingleQp()` — passthrough |
| `length > 0` | `postRecvSingleQp()` — passthrough |
| Zero-length + DQPLB | Add to `recvTracker_`, initialize DQPLB receiver on first call |
| Zero-length + SPRAY | Add to `recvTracker_`, post to notifyQp (with backpressure) |

**Fragmentation** (`dispatchPendingSends`, `IbvVirtualQp.h:547–599`):
- Iterates `sendTracker_.pendingQue_`, fragments each WR into `maxMsgSize_`-byte chunks
- Round-robin QP selection via `findAvailableSendQp()` with capacity check
- `buildPhysicalSendWr()` handles SPRAY (converts WRITE_WITH_IMM → WRITE per fragment) and DQPLB (embeds seq# via `dqplbSeqTracker_.getSendImm()`)

**processCompletion** (`IbvVirtualQp.h:888–918`):
Called by VirtualCq for multi-QP RDMA CQEs. 2×2 dispatch matrix:

|  | NotifyQp | DataQp |
|--|----------|--------|
| **Send** | Notify sent done → report send completion, flush pending notifies | Fragment done → update WR state, resume `dispatchPendingSends()` |
| **Recv** | SPRAY notify received → report recv completion, flush pending recv notifies | DQPLB seq# received → process seq#, decrement outstanding, replenish recv |

**SPRAY notify flow**:
1. All data fragments posted as `IBV_WR_RDMA_WRITE` (stripped of IMM)
2. When all data CQEs arrive (`remainingMsgCnt == 1`): post zero-length `IBV_WR_RDMA_WRITE_WITH_IMM` on notifyQp
3. Backpressure: if notifyQp has ≥ `maxMsgCntPerQp_` pending, queue in `pendingNotifyQue_`

**DQPLB recv flow**:
1. `initializeDqplbReceiver()` pre-posts `maxMsgCntPerQp` × `numQps` zero-length recvs
2. Each DQPLB data CQE: extract seq# via `dqplbSeqTracker_.processReceivedImm()`, which returns count of consecutive completed notifies
3. `replenishDqplbRecv()` posts one new zero-length recv after consuming one

### 4. Factory Changes (`IbvPd.h/.cc`)

```cpp
// Before:
createVirtualQp(int totalQps, ibv_qp_init_attr*, IbvVirtualCq* sendCq, IbvVirtualCq* recvCq, ...);

// After:
createVirtualQp(int totalQps, ibv_qp_init_attr*, IbvVirtualCq* virtualCq, ...);
```

- NotifyQp created only for SPRAY mode (was always created).
- `initAttr->send_cq` and `initAttr->recv_cq` both point to the single VirtualCq's physical CQ.

### 5. Coordinator Deletion

`Coordinator.h` and `Coordinator.cc` fully deleted. All global state eliminated. BUCK target removes `Coordinator.cc` source and `//folly:singleton` dependency.

---

## Test Migration

### Unit Tests (`IbverbxTest.cc`)

| Removed | Reason |
|---------|--------|
| `Coordinator` test | Coordinator deleted |
| `CoordinatorRegisterUnregisterUpdateApis` test | Coordinator deleted |
| `IbvVirtualQpUpdatePhysicalSendWrFromVirtualSendWr` test | Function removed |

| Updated | Change |
|---------|--------|
| `IbvVirtualCq` test | Verifies registration table instead of Coordinator maps |
| `IbvVirtualQp` test | Uses single-CQ `createVirtualQp()` |

### Distributed Tests (all three files)

Mechanical migration applied uniformly:

| Before | After |
|--------|-------|
| `ibv_send_wr` construction with SGE list | `IbvVirtualSendWr{.wrId=, .localAddr=, .length=, .remoteAddr=, .opcode=, .sendFlags=, .immData=, .deviceKeys=}` |
| `ibv_recv_wr` construction | `IbvVirtualRecvWr{.wrId=, .localAddr=, .length=, .lkey=}` |
| `vqp.postSend(&sendWr, &badWr)` | `vqp.postSend(virtualSendWr)` |
| `vqp.postRecv(&recvWr, &badWr)` | `vqp.postRecv(virtualRecvWr)` |
| `vcq.pollCq(n)` → `vector<ibv_wc>` | `vcq.pollCq(n)` → `vector<IbvVirtualWc>` |
| `wc.wr_id`, `wc.status`, `wc.opcode` | `wc.wrId`, `wc.status`, `wc.opcode` |
| `createVirtualQp(n, attr, &sendCq, &recvCq, ...)` | `createVirtualQp(n, attr, &virtualCq, ...)` |
| `ASSERT_EQ(wc.immData, expected)` for SPRAY | Removed — IMM consumed internally by VirtualQp layer |

**immData note**: `IbvVirtualWc.immData` is NOT propagated from physical completions in multi-QP mode. SPRAY uses IMM for notify signaling, DQPLB uses it for sequence tracking. Tests no longer assert on `immData` values for multi-QP completions.

---

## Data Flow Diagrams

### SPRAY Send Path (multi-QP, WRITE_WITH_IMM)

```
User calls postSend(IbvVirtualSendWr{opcode=WRITE_WITH_IMM, immData=X})
  │
  ▼
sendTracker_.add(ActiveVirtualWr{remainingMsgCnt = numFragments + 1, needsNotify=true})
  │
  ▼
dispatchPendingSends()
  │  For each fragment:
  │    buildPhysicalSendWr() → opcode changed to WRITE (no IMM)
  │    postSend on round-robin physical QP
  │    Record in physicalSendQueStatus_
  │
  ▼
[CQEs arrive via pollCq() → processCompletion() → processDataQpSendCompletion()]
  │  For each data CQE:
  │    popPhysicalQueueStatus()
  │    updateWrState() → remainingMsgCnt--
  │    dispatchPendingSends() (resume on freed QP)
  │
  ▼
When remainingMsgCnt == 1 (all data done):
  reportSendCompletions() → postNotifyForWr()
    │  Post zero-length WRITE_WITH_IMM on notifyQp with immData=X
    │
    ▼
[Notify CQE arrives → processNotifyQpSendCompletion()]
  │  remainingMsgCnt-- → now 0
  │  reportSendCompletions() → buildVirtualWc() → push to results
  │
  ▼
User sees IbvVirtualWc{wrId=original, status=SUCCESS, opcode=IBV_WC_RDMA_WRITE}
```

### SPRAY Recv Path (multi-QP)

```
User calls postRecv(IbvVirtualRecvWr{wrId=Y, length=0})
  │
  ▼
recvTracker_.add(ActiveVirtualWr{remainingMsgCnt=1})
  │
  ▼
postRecvToNotifyQp() → zero-length recv on notifyQp
  │
  ▼
[Notify CQE arrives via pollCq() → processCompletion() → processNotifyQpRecvCompletion()]
  │  popPhysicalQueueStatus()
  │  updateWrState() → remainingMsgCnt-- → now 0
  │  reportRecvCompletions() → buildVirtualWc() → push to results
  │
  ▼
User sees IbvVirtualWc{wrId=Y, status=SUCCESS}
```

### DQPLB Recv Path

```
First postRecv() call:
  initializeDqplbReceiver()
    │  Pre-post maxMsgCntPerQp × numQps zero-length recvs
    │  physicalRecvQueStatus_ entries with virtualWrId = -1 (not tied to user WR)
    │
    ▼
[DQPLB CQE arrives → processDataQpRecvCompletion()]
  │  popPhysicalQueueStatus()
  │  dqplbSeqTracker_.processReceivedImm(imm_data)
  │    → returns notifyCount (consecutive completed messages)
  │  For each notify:
  │    recvTracker_.frontOutstanding() → decrement remainingMsgCnt
  │    reportRecvCompletions()
  │  replenishDqplbRecv() → post one new zero-length recv
  │
  ▼
User sees IbvVirtualWc for each completed message
```

---

## Known Issues and Future Work

Tracked in `docs/ibverbx_gaps.md`. Top items:

| ID | Severity | Issue |
|----|----------|-------|
| T1 | High | Error WC routing: `isSendOpcode()` reads undefined opcode field on error CQEs |
| T2 | High | No dead-QP exclusion after QP enters ERROR state |
| T3 | High | Partial fragment dispatch can leave zombie WRs |
| T4 | High | `dispatchPendingSends` discards original error, uses stale `errno` |
| T5 | Medium | `maxMsgSize_` still `int` (design doc says `uint32_t`) |
| T6–T7 | Medium | `flushPendingSendNotifies`/`flushPendingRecvNotifies` swallow errors |
| T15 | Medium | Unsignaled `IBV_WR_RDMA_WRITE_WITH_IMM` allowed (skips completion tracking) |

Full list: 23 TODO items + 9 implementation notes in `docs/ibverbx_gaps.md`.

---

## Build and Test Commands

```bash
# Build (both variants)
buck build @mode/opt fbcode//comms/ctran/ibverbx:ibverbx
buck build @mode/opt fbcode//comms/ctran/ibverbx:ibverbx-rdma-core

# Unit tests
buck test @mode/opt fbcode//comms/ctran/ibverbx/tests:ibverbx_test

# Distributed tests (2-node)
buck test @mode/opt fbcode//comms/ctran/ibverbx/tests:ibverbx_distributed_virtualqp_test
buck test @mode/opt fbcode//comms/ctran/ibverbx/tests:ibverbx_distributed_virtualqp_test_dqplb
buck test @mode/opt fbcode//comms/ctran/ibverbx/tests:ibverbx_distributed_virtualqp_gb200_test
```

---

## Review Checklist

- [ ] **Type definitions** (`IbvCommon.h`, `IbvVirtualWr.h`): `IbvVirtualWc` fields, `WrTracker` three-structure pattern, `ActiveVirtualWr` field completeness
- [ ] **Registration lifecycle** (`IbvVirtualCq.cc`, `IbvVirtualQp.cc`): constructor registers, destructor unregisters, move ops re-register with correct `this` pointer
- [ ] **Two-phase pollCq** (`IbvVirtualCq.h:119–213`): three routing paths, error propagation, batch-32 drain loop
- [ ] **postSend multi-QP path** (`IbvVirtualQp.h:275–349`): opcode routing, `remainingMsgCnt` calculation (fragments + 1 for SPRAY notify), `sendTracker_.add()` correctness
- [ ] **Fragmentation** (`IbvVirtualQp.h:547–634`): `dispatchPendingSends` loop, `buildPhysicalSendWr` SPRAY/DQPLB branching, offset arithmetic, round-robin QP selection
- [ ] **Completion aggregation** (`IbvVirtualQp.h:669–713`, `716–738`): `reportSendCompletions` notify posting logic, `reportRecvCompletions` FIFO ordering
- [ ] **processCompletion 2×2 matrix** (`IbvVirtualQp.h:888–1092`): notify vs data QP dispatch, `isSendOpcode()` correctness, `popPhysicalQueueStatus` ordering verification
- [ ] **DQPLB receiver** (`IbvVirtualQp.h:443–496`): `initializeDqplbReceiver` pre-posting, `replenishDqplbRecv` per-CQE, `processDataQpRecvCompletion` seq# processing
- [ ] **Backpressure** (`IbvVirtualQp.h:427–430, 682–693, 782–784`): `maxMsgCntPerQp_` threshold on notifyQp (same as data QPs), pending queues for send and recv notifies
- [ ] **Factory** (`IbvPd.cc:118–165`): single CQ assignment, conditional notifyQp creation
- [ ] **Move semantics** (`IbvVirtualQp.cc:50–106`, `IbvVirtualCq.cc:21–52`): pointer update loops, unregister-before-move in assignment
- [ ] **BUCK** changes: `Coordinator.cc` removed, `folly:singleton` dep removed, `f14_hash` added
- [ ] **Test migration**: `ibv_send_wr` → `IbvVirtualSendWr`, `ibv_wc` → `IbvVirtualWc`, immData assertions removed
