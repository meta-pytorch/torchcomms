# IbvVirtualQp/IbvVirtualCq Design Gaps

Gaps identified by comparing `simplified_ibverbx_design.md` against production code (`IbvVirtualQp.h/.cc`, `IbvVirtualCq.h/.cc`, `IbvVirtualWr.h`, `IbvCommon.h`). Multiple rounds of review were conducted, including two independent agent-based code reviews.

---

## TODO: Issues to address

### High Severity

| ID | Issue | File / Line | Notes |
|----|-------|-------------|-------|
| T1 | **Error WC routing with undefined opcode** — When `physicalWc.status != IBV_WC_SUCCESS`, the `opcode` field is undefined per IB spec. Both `isSendOpcode()` (IbvVirtualQp.h:810) and VirtualCq passthrough (IbvVirtualCq.h:168-169) rely on opcode for routing. Error completions can be misrouted: silently passed through as SEND/RECV instead of going through `processCompletion()`, or dispatched to the wrong send/recv handler. | IbvVirtualQp.h:810-813, IbvVirtualCq.h:167-180 | Check `physicalWc.status` first. For error WCs in multi-QP mode, route to `processCompletion()`. In `processCompletion()`, use `PhysicalWrStatus` deques (not opcode) to determine send vs recv direction. |
| T2 | **No dead-QP exclusion after error state** — When a physical QP enters IB ERROR state (after any error CQE), all subsequent WRs on that QP are flushed. `findAvailableSendQp()` keeps selecting it via round-robin, causing cascading failures. | IbvVirtualQp.h:644-663 | Add `bool isHealthy_` per physical QP. `updateWrState()` marks QP unhealthy on error. `findAvailableSendQp()` skips unhealthy QPs. Return error if all QPs are unhealthy. |
| T3 | **Partial fragment dispatch leaves zombie WR** — When `dispatchPendingSends()` fails mid-fragmentation (e.g., QP full or error state), already-posted fragments generate completions decrementing `remainingMsgCnt`, but the count was set for the *total* expected fragments. The WR can never complete. | IbvVirtualQp.h:584-587 | On dispatch failure, adjust `remainingMsgCnt` to match the number of fragments actually posted. Mark the WR as failed so it completes (with error) once in-flight fragments drain. |
| T4 | **`dispatchPendingSends` error at line 344 discards original error, uses stale errno** — `Error(errno)` loses the descriptive error from `dispatchPendingSends()` and uses whatever errno happens to be set. | IbvVirtualQp.h:344-346 | Propagate the original error: `auto result = dispatchPendingSends(); if (result.hasError()) return folly::makeUnexpected(result.error());` |

### Medium Severity

| ID | Issue | File / Line | Notes |
|----|-------|-------------|-------|
| T5 | **`maxMsgSize_` declared as `int`, causes signed/unsigned mixing in fragmentation** — Design doc (D8) fixed this to `uint32_t`, but implementation still uses `int` (line 156). `static_cast<int>(pending->length - pending->offset)` at line 577 is UB for >2GB remaining. Constructor params also `int` (line 17 of .cc). | IbvVirtualQp.h:156, IbvVirtualQp.h:576-577, IbvVirtualQp.cc:17-18 | Change `maxMsgSize_` and constructor param to `uint32_t` per design doc D8/D15. Fix fragmentation arithmetic. |
| T6 | **`flushPendingSendNotifies` silently swallows `postNotifyForWr` errors** — Returns `folly::unit` (success) even when the post failed. The WR stays in the pending queue but the caller thinks everything is fine. Receiver may hang waiting for a notify never sent. | IbvVirtualQp.h:797-803 | Return the error from the failed post instead of `break`ing and returning success. |
| T7 | **`flushPendingRecvNotifies` silently swallows `postRecvToNotifyQp` errors** — Same pattern as T6. | IbvVirtualQp.h:533-534 | Same fix: propagate error instead of `break` + return success. |
| T8 | **DQPLB sequence number wraparound — no overflow detection** — `sendNext_` wraps at 16M (kSeqNumMask). If `receivedSeqNums_` is large when wraparound occurs, entries silently collide, losing `isLastFragment` flags. | DqplbSeqTracker.h:34, 43 | Add XCHECK that `receivedSeqNums_.size()` stays well below `kSeqNumMask`. Log warning on excessive growth. |
| T9 | **VirtualCq Phase 1 drains ALL CQEs before returning any** — Under burst load, `pollCq()` processes potentially thousands of CQEs before returning a single result, causing head-of-line blocking latency. | IbvVirtualCq.h:129 | Consider early exit from Phase 1 once `completedVirtualWcs_.size() >= numEntries`, or add a `maxPhysicalPollsPerCall` bound. |
| T10 | **SPRAY cross-QP data ordering not guaranteed for multi-NIC** — Notify on `notifyQp_` (possibly different NIC) has no ordering guarantee with data RDMA WRITEs on other NICs. Receiver may see notify before data arrives. Single-NIC is safe due to NIC-internal ordering. | IbvVirtualQp.h:682-699 | For multi-NIC SPRAY, ensure notify QP is on same NIC as data QPs, or add a fence mechanism. Document single-NIC assumption. |
| T11 | **VirtualQp destructor does not check for in-flight operations** — Destroying VirtualQp with active WRs causes stale CQ entries. `unregisterFromVirtualCq()` removes the registration, then QP destruction flushes WRs, generating completions for unregistered QP numbers. | IbvVirtualQp.cc:108-110 | Add XCHECK in destructor: `sendTracker_.activeCount() == 0 && recvTracker_.activeCount() == 0`. Or log warning. |
| T12 | **`IBV_WR_SEND` in multi-QP always goes to QP[0]** — `postSend()` routes SEND to `postSendSingleQp()` (line 307), which always uses `physicalQps_[0]`. No load balancing, no backpressure tracking. QP 0 send queue can overflow if many SENDs are posted. | IbvVirtualQp.h:305-307 | Either load-balance SENDs across physical QPs, or reject SEND in multi-QP mode with an error, or document as intentional limitation. |
| T13 | **DQPLB with `IBV_WR_RDMA_WRITE` (no IMM) silently drops sequence numbers** — `buildPhysicalSendWr()` sets `imm_data` on the send WR (line 631), but plain `IBV_WR_RDMA_WRITE` ignores `imm_data`. Receiver never sees the seq#, causing permanent hang. | IbvVirtualQp.h:624-633 | Either reject DQPLB with non-IMM opcodes (return EINVAL), or force all DQPLB fragments to use `IBV_WR_RDMA_WRITE_WITH_IMM`. |
| T14 | **`processCompletion()` ignores `deviceId` parameter; `qpNumToIdx_` keyed only on `qpNum`** — Two QPs on different devices with the same QP number would collide in `qpNumToIdx_`. The `deviceId` parameter is accepted but unused. | IbvVirtualQp.h:887-918, line 126 | Key `qpNumToIdx_` on `QpId{deviceId, qpNum}` instead of just `qpNum`. Use the `deviceId` parameter in the lookup. |
| T15 | **Unsignaled `IBV_WR_RDMA_WRITE_WITH_IMM` allowed in multi-QP** — Line 298 exempts `IBV_WR_RDMA_WRITE_WITH_IMM` from the signaling check. Unsignaled ops don't generate send CQEs, but `remainingMsgCnt` expects them, causing permanent hang. | IbvVirtualQp.h:297-302 | Remove the exception. All multi-QP operations must be signaled for completion tracking. |

### Low Severity

| ID | Issue | File / Line | Notes |
|----|-------|-------------|-------|
| T16 | **`processCompletions()` creates temp vector per WC** — Each `processCompletion()` call allocates a fresh vector. Most are empty (intermediate fragments). | IbvVirtualQp.h:920-939 | Pass `allResults` as reference to `processCompletion()` and append directly. Tracked as I11 previously. |
| T17 | **Physical CQ polling allocates `std::vector<ibv_wc>` per call in hot path** — `IbvCq::pollCq(32)` allocates 32*sizeof(ibv_wc) on heap each call inside the drain loop. | IbvCq.h, IbvVirtualCq.h:130 | Pre-allocate reusable buffer or change to caller-provided buffer pattern. |
| T18 | **`F14FastMap::at()` throws on missing `deviceKeys` entry** — If caller provides `deviceKeys` but misses a specific `deviceId`, `.at()` throws `std::out_of_range`. Violates no-exceptions convention. | IbvVirtualQp.h:251-252, 260-261, 610-612, 620-622 | Use `find()` and return `Error` on missing key, or document that callers must provide all device keys. |
| T19 | **`kIbMaxMsgSizeByte = 100` impractically small default** — 1MB message → 10,000 fragments. 1GB → 10M fragments. Likely a testing placeholder. | IbvCommon.h:20 | Set to a production-appropriate value (e.g., 1MB). |
| T20 | **`uint32_t length` in IbvVirtualSendWr limits single transfer to 4GB** — GPU collectives can exceed this on modern hardware (80+ GB HBM). | IbvVirtualWr.h:25 | Change to `uint64_t` if >4GB single-message support is needed. Also update `ActiveVirtualWr::offset`. |
| T21 | **Public IbvVirtualQp constructor violates RAII pattern** — Per CLAUDE.md, constructors should be private with parent as friend. Factory is `IbvPd::createVirtualQp()`. | IbvVirtualQp.h:61 | Make constructor private, keep `IbvPd` as friend. |
| T22 | **Wasted notify QP with `totalQps=1` in SPRAY mode** — `isMultiQp_=false` means notify QP is created but never used. Wastes a kernel QP resource. | IbvPd.cc, IbvVirtualQp.cc:30 | Skip creating notify QP when `totalQps == 1`. |
| T23 | **`kIbMaxCqe_` has trailing underscore in constant name** — Convention for member variables, not constants. | IbvCommon.h:21 | Rename to `kIbMaxCqe`. |

---

## Issues from ibverbx_gaps.md that are still relevant during implementation

These are minor issues from the original gaps doc that should be addressed when convenient.

| ID | Issue | Status | Notes |
|----|-------|--------|-------|
| I3 | `lkey`/`rkey` for single-NIC in `IbvVirtualSendWr` | Fixed | `deviceKeys` is now mandatory (always populated): 1 entry for single-NIC, N for multi-NIC. Always use `deviceKeys.at(deviceId)`. No empty-check branching needed. |
| I7 | `IBV_WR_SEND` fragmentation not supported | Open | Intentional limitation. SEND routed to single QP. Matches CtranIb usage. |
| I8 | `postSendSingleQp()` rejects zero-length WRs | Open | May be overly restrictive for `IBV_WR_RDMA_WRITE_WITH_IMM`. |
| I9 | `postRecvSingleQp()` does not use `deviceKeys` for lkey | Fixed | `IbvVirtualRecvWr` no longer has a separate `lkey` field. Use `wr.deviceKeys.at(deviceId).lkey` consistently with send path. |
| I11 | `processCompletions()` batch API allocates per-call vectors | Open | Same as T16. |
| I13 | `findRegisteredQpInfo()` returns pointer into F14FastMap | Open | Safe under single-threaded usage. |
| I14 | Multi-SGE support not documented | Open | Intentional: flat `localAddr` + `length` enforces single-SGE. |
| I15 | `deviceKeys` map copied on every `postSend()` | Open | Consider move semantics. |
| I23 | `maxMsgCntPerQp_==-1` skips `initializeDqplbReceiver()` pre-posting | Open | DQPLB requires positive value. Add validation in constructor. |

---

## Appendix: Fixed Issues

All issues below have been resolved in the implementation and/or design doc.

### Design Doc Issues (Fixed)

| ID | Issue | Resolution |
|----|-------|------------|
| C1 | Return type inconsistency in `pollCq()` | Fixed — `IbvVirtualWc` struct with 6 camelCase fields |
| C2 | DQPLB send IMM uses wrong field | Fixed — `getSendImm(bool isLastFragment)` |
| C3 | `deviceKeys` vector indexing assumes contiguous device IDs | Fixed — Changed to `F14FastMap<int32_t, MemoryRegionKeys>` |
| C4 | Missing `buildVirtualWc()` implementation | Fixed — Implemented |
| M1 | `modifyVirtualQp()` and `BusinessCard` not mentioned | Fixed — Reused with `std::optional` guards |
| M2 | `deviceCnt_` computation missing | Fixed — Added with `std::unordered_set` |
| M3 | `findAvailableSendQp()` round-robin state missing | Fixed — Added `nextSendPhysicalQpIdx_` |
| M4 | `ibv_poll_cq()` called directly | Fixed — Uses `cq.pollCq()` |
| M5 | Unsignaled operation logic gap | Fixed — Rejected with EINVAL in multi-QP mode |
| M6 | `IBV_WR_RECV` does not exist | Fixed — Removed opcode from `IbvVirtualRecvWr` |
| m1 | Naming changes from production | Fixed — Intentional renames documented |
| m2 | `PhysicalWrStatus` type referenced but never defined | Fixed — Unified struct in `IbvVirtualWr.h` |
| m3 | Missing accessor methods | Fixed — All accessors added |
| m4 | Missing move assignment operator | Fixed — Added for both classes |
| m5 | `send_flags` always `IBV_SEND_SIGNALED` in fragmented path | Fixed — Documented as intentional |
| m6 | `IbvVirtualCq` single-CQ constructor missing | Fixed — Added |
| m7 | `IbvVirtualWc` may be missing fields | Fixed — Verified |
| m8 | Coordinator removal types not cleaned up | Fixed — Coordinator removed |
| m9 | `IbvPd::createVirtualQp()` signature change | Fixed — Single-CQ signature |
| D1 | SPRAY recv `immData` not captured from physical WC | Not a gap — CtranIb never reads `immData` from multi-QP completions |
| D2 | `IBV_WR_SEND` routing in multi-QP mode | Fixed — Opcode-based CQE routing in `pollCq()` |
| D3 | DQPLB recv `notifyCount` loop bug | Fixed — `reportRecvCompletions()` inside loop |
| D4 | Notify WR uses wrong remote address | Fixed — Uses `pending->remoteAddr` |
| D5 | `pollCq()` single-pass vs two-phase drain | Fixed — Phase 1 drain + Phase 2 return |
| D6 | `buildVirtualWc()` recv opcode bug | Fixed — `wcOpcode` captured from physicalWc |
| D7 | `dqplbSeqTracker` member name missing trailing underscore | Fixed |
| D8 | `maxMsgSize_` type mismatch (`int` vs `uint32_t`) | Fixed in design doc, **NOT fixed in implementation** (see T5) |
| D9 | VirtualCq move assignment dangling pointers | Fixed — Pointer update loop |
| D10 | `IbvQp.h` not in Section 4.1 | Fixed |
| D11 | SEND recv-side fragmentation not documented | Fixed — Limitation comment added |
| D12 | `buildPhysicalSendWr()` dangling SGE pointer | Fixed — Output-parameter pattern |
| D13 | `badWr` pointer type mismatch | Fixed — Stack-allocated struct |
| D14 | DQPLB recv path never sets `wcOpcode` | Fixed — `frontWr->wcOpcode = physicalWc.opcode` at line 1073 |
| D15 | Constructor `maxMsgSize` type mismatch (declaration vs definition) | Fixed in design doc, **NOT fixed in implementation** (see T5) |
| D16 | Default argument repeated on `dispatchPendingSends` definition | Fixed |
| D17 | `dispatchPendingSends()` return value discarded | Fixed — Error check added at line 1029-1032 |
| D18 | `flushPendingRecvNotifies()` return value discarded | Fixed — Error check added at line 995-998 |

### Implementation Issues (Fixed)

| ID | Issue | Resolution |
|----|-------|------------|
| I1 | Missing member declarations | Fixed — `maxMsgCntPerQp_`, `maxMsgSize_`, `loadBalancingScheme_` declared |
| I2 | Missing error logging in `updateWrState()` | Fixed — `XLOGF(ERR)` at line 837-841 |
| I4 | `IbvVirtualCqWrapper` uses undefined type | Fixed — Convenience layer not implemented |
| I5 | `DqplbSeqTracker::getSendImm(bool)` implementation | Fixed — Implemented |
| I6 | `IbvVirtualCq` move assignment pointer update | Fixed — D9 |
| I10 | `qpNumToIdx_` key type changed to `uint32_t` | Fixed — Intentional |
| I12 | Batch-32 CQ polling vs poll-1 | Fixed — Documented |
| I16 | `wcOpcode` not initialized in `ActiveVirtualWr` | Fixed — Default `IBV_WC_SEND` |
| I17 | `postNotifyForWr()` sets `sg_list=nullptr` with `num_sge=0` | Open in impl — line 755-756 still sets `nullptr`. Some drivers may dereference. |
| I18 | `IbvVirtualWr.h` not in Section 4.1 | Fixed — File updated with new types |
| I19 | `MemoryRegionKeys` migration | Fixed — In `IbvCommon.h` |
| I20 | `notifyQp_->` dereferences without XCHECK | Partially fixed — XCHECKs at lines 426, 501, 524, 687, 751, 780. All structural access points covered. |
| I21 | `IbvVirtualCq` move constructor | Fixed — Implemented in IbvVirtualCq.cc |
| I22 | `kMaxOutstandingNotifies=256` undocumented | Fixed — Removed. Notify QP backpressure uses `maxMsgCntPerQp_` (same limit as data QPs), consistent with CtranIb's `MAX_SEND_WR` usage. |
| I24 | Removed `enqueSendCq()`/`enqueRecvCq()` | Fixed — Replaced by `processCompletion()` |
