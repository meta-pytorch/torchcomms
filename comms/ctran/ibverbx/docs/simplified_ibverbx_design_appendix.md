---

## Appendix A: Decision Matrix

| Approach | PostSend | Completion | Complexity | Compatibility |
|----------|----------|------------|------------|---------------|
| Keep current design | 238 ns | 179 ns | Low | Full |
| Simplified (VirtualQp only) | 168 ns | 118 ns | Medium | Breaking |
| **Simplified + VirtualCqWrapper** ✓ | **168 ns** | **118-123 ns** | **Medium** | **Full** |
| Aggressive (remove fragmentation) | 115 ns | 90 ns | Low | Breaking |

**Recommendation**: Simplified design with optional `IbvVirtualCqWrapper` for user convenience.

### Key Design Decisions

1. **VirtualQp owns all completion tracking** - Eliminates Coordinator and VirtualCq overhead
2. **Registration with isMultiQp flag** - VirtualQp indicates single vs multi-QP at registration time
3. **Two-path pollCq()** - Fast path (pass-through) for single-QP VirtualQps, aggregation for multi-QP
4. **All QPs are VirtualQps** - Every QP must be registered with a VirtualQp
5. **Generic WrTracker<T> template** - Reusable three-structure design for WR tracking (multi-QP only):
   - Separate `sendTracker_` and `recvTracker_` using unified `WrTracker<ActiveVirtualWr>` type
   - Each tracker contains: `activeVirtualWrs_` (F14FastMap), `pendingQue_` (deque), `outstandingQue_` (deque)
   - Reduces code duplication and ensures consistent behavior
6. **Internal unique IDs** - User's `wrId` may have duplicates (ibverbs allows this). We generate unique `internalWrId` for tracking (map key), store user's `wrId` for reporting.
7. **Unified ActiveVirtualWr struct** - Single type for both send and recv, essential fields stored; derivable state computed on-demand:
   - Added: `needsNotify` and `notifyPosted` booleans for clean SPRAY notify tracking
   - Added: `isSendOp()` helper to distinguish send vs recv operations
   - `remainingMsgCnt` counts ALL expected CQEs (data + notify if applicable)
   - Removed: `internalWrId` (use map key), `expectedMsgCnt` (unused), `sendFlags` (unused)
   - Kept: `userWrId`, `remainingMsgCnt`, `aggregatedStatus`, `localAddr`, `length`, `remoteAddr`, `opcode`, `immData`, `deviceKeys`, `offset`
8. **In-order completion guarantee** - Virtual completions reported in posting order for both send and recv, even when physical completions arrive out-of-order. WRs are added to both `pendingQue_` and `outstandingQue_` at add() time.
9. **Unified SPRAY notify tracking** - Simple boolean-based approach replaces complex state machine:
   - `remainingMsgCnt` = ALL expected CQEs (data fragments + notify if applicable)
   - For SPRAY send: `remainingMsgCnt = numFragments + 1`, `needsNotify = true`
   - For non-SPRAY: `remainingMsgCnt = numFragments`, `needsNotify = false`
   - `notifyPosted` tracks whether notify has been posted (prevents double-posting)
   - `isComplete()` is simply `return remainingMsgCnt == 0;`
   - **Key invariant**: WR stays in `outstandingQue_` until `isComplete()` to ensure in-order reporting
10. **Notify backpressure** - `notifyQp_->physicalSendQueStatus_.size()` tracks outstanding notifies; blocked notifies set `notifyPosted = true` and are queued in `pendingNotifyQue_`
11. **No expensive queue-moves** - Queues hold 8-byte internal WR IDs only, not full WR objects
12. **processCompletion() returns vector** - May return multiple results when in-order draining unblocks several WRs
13. **Single-QP fast path** - Single-QP (`!isMultiQp_`) is pure passthrough: direct post to physical QP 0, VirtualCq returns physical WC directly to user, no tracking needed (~5 ns overhead)
14. **Opcode routing** - `IBV_WR_SEND` uses single QP (no load balancing/fragmentation), RDMA operations (`IBV_WR_RDMA_WRITE`, `IBV_WR_RDMA_WRITE_WITH_IMM`, `IBV_WR_RDMA_READ`) use load balancing across multiple QPs:
   - Assumption: User will not mix SEND and RDMA opcodes in the same VirtualQp
   - Runtime assertion detects opcode mixing and returns error
   - `postSend()` routes to `postSendSingleQp()` for SEND, `dispatchPendingSends()` for RDMA
15. **DQPLB recv tracking** - User postRecv() calls are tracked in `recvTracker_` for both SPRAY and DQPLB modes:
   - `postRecv()` adds WR to recvTracker_ with `remainingMsgCnt = 1`
   - DQPLB uses `DqplbSeqTracker.processReceivedImm()` to get `notifyCount` for in-order completions
   - `notifyCount` determines how many front WRs in recvTracker_ to complete
   - `replenishDqplbRecv(qpIdx)` maintains pre-posted recv pool after consuming one
   - This unified approach respects ordering even if SPRAY and DQPLB were mixed (though currently not supported)
16. **2x2 matrix completion dispatch** - `processCompletion()` uses a simple 2x2 matrix based on QP type × direction:
   ```
                       Send                         Recv
              ┌─────────────────────────┬─────────────────────────┐
   NotifyQp   │ processNotifyQpSend-    │ processNotifyQpRecv-    │
              │   Completion()          │   Completion()          │
              ├─────────────────────────┼─────────────────────────┤
   DataQp     │ processDataQpSend-      │ processDataQpRecv-      │
              │   Completion()          │   Completion()          │
              └─────────────────────────┴─────────────────────────┘
   ```
   - Each processor is self-contained (~30-40 lines) with consistent structure: Pop → Update → Report → Schedule
   - All functions share "process...Completion" naming → clear calling relationship
   - Processor names describe the source (QP type + direction), not the mode
   - `processDataQpRecvCompletion()` only occurs in DQPLB mode (SPRAY recv uses notifyQp)

---

## Appendix B: Scatter-Gather Support (Optional Extension)

This appendix describes how to extend the simplified design to support scatter-gather (multiple non-contiguous local buffers). This is an **optional extension** for users who need to send data from multiple scattered memory locations.

### B.1 Extended Data Structures

```cpp
// Extended IbvVirtualSendWr with scatter-gather support
struct IbvVirtualSendWr {
  uint64_t wrId;

  // Scatter-gather list for local buffers
  struct LocalBuffer {
    void* addr;
    uint32_t length;
  };
  std::vector<LocalBuffer> localBuffers;  // Multiple scattered buffers

  uint64_t remoteAddr;        // Single contiguous remote destination
  ibv_wr_opcode opcode;
  int sendFlags;
  uint32_t immData;
  std::vector<MemoryRegionKeys> deviceKeys;

  // Convenience: total length across all buffers
  uint32_t totalLength() const {
    uint32_t total = 0;
    for (const auto& buf : localBuffers) {
      total += buf.length;
    }
    return total;
  }
};

// Extended PendingVirtualWr with scatter-gather state (minimal fields only)
struct PendingVirtualWr {
  // Identity
  uint64_t userWrId;              // User's original wrId (for reporting)

  // Completion tracking
  int remainingMsgCnt;            // Decremented on each completion; 0 = complete
  ibv_wc_status aggregatedStatus; // First error wins

  // Scatter-gather list (replaces single localAddr/length)
  std::vector<IbvVirtualSendWr::LocalBuffer> localBuffers;
  uint64_t remoteAddr;
  uint32_t totalLength;           // Sum of all buffer lengths
  ibv_wr_opcode opcode;
  uint32_t immData;
  std::vector<MemoryRegionKeys> deviceKeys;

  // Scatter-gather fragmentation state
  uint32_t currentBufferIdx;      // Current buffer index in localBuffers
  uint32_t currentBufferOffset;   // Offset within current buffer
  uint32_t totalBytesSent;        // Total bytes sent so far

  // Derived state (not stored):
  // - allFragmentsSent: derive from (totalBytesSent >= totalLength)
  // - allFragmentsComplete: derive from (remainingMsgCnt == 0)
  // - needsNotifyImm: derive from (opcode == WRITE_WITH_IMM && SPRAY mode)
};
```

### B.2 Scatter-Gather Fragmentation Logic

```cpp
// Helper: Get next fragment from scatter-gather list
// Returns {addr, length} for the next fragment, handling buffer boundaries
inline std::pair<void*, uint32_t> IbvVirtualQp::getNextFragment(
    PendingVirtualWr& pending,
    uint32_t maxFragmentSize) {

  auto& currentBuf = pending.localBuffers[pending.currentBufferIdx];
  uint32_t remainingInBuffer = currentBuf.length - pending.currentBufferOffset;

  // Fragment is limited by: maxFragmentSize OR remaining bytes in current buffer
  uint32_t fragmentLength = std::min(maxFragmentSize, remainingInBuffer);
  void* fragmentAddr = static_cast<char*>(currentBuf.addr) + pending.currentBufferOffset;

  return {fragmentAddr, fragmentLength};
}

// Helper: Advance scatter-gather state after sending a fragment
inline void IbvVirtualQp::advanceScatterGatherState(
    PendingVirtualWr& pending,
    uint32_t bytesSent) {

  pending.totalBytesSent += bytesSent;
  pending.currentBufferOffset += bytesSent;

  // Check if we've exhausted the current buffer
  auto& currentBuf = pending.localBuffers[pending.currentBufferIdx];
  if (pending.currentBufferOffset >= currentBuf.length) {
    pending.currentBufferIdx++;
    pending.currentBufferOffset = 0;
  }
}
```

### B.3 Scatter-Gather Fragmentation Example

```
User's Scatter-Gather Input:
  localBuffers[0]: addr=A, length=100 bytes
  localBuffers[1]: addr=B, length=200 bytes
  localBuffers[2]: addr=C, length=150 bytes
  Total: 450 bytes

maxMsgSize = 150 bytes
Number of fragments = ceil(450 / 150) = 3 (naive calculation)

Fragment 1:
  - 100 bytes from buffer[0] (A+0 to A+100)
  - Buffer boundary reached! Fragment truncated to 100 bytes
  - Actual: 100 bytes from A

Fragment 2:
  - 150 bytes from buffer[1] (B+0 to B+150)
  - Full fragment sent

Fragment 3:
  - 50 bytes from buffer[1] (B+150 to B+200)
  - Buffer boundary reached! Fragment truncated to 50 bytes
  - Actual: 50 bytes from B+150

Fragment 4:
  - 150 bytes from buffer[2] (C+0 to C+150)
  - Full fragment sent

Note: Due to buffer boundaries, actual fragment count = 4 (not 3)
Each fragment is sent from a single contiguous buffer region.

State Tracking:
  currentBufferIdx:    0 → 1 → 1 → 2 → done
  currentBufferOffset: 0 → 0 → 150 → 0 → 150
  totalBytesSent:      0 → 100 → 250 → 300 → 450
```

### B.4 Performance Impact

| Buffers | totalLength() | localBuffers copy | Net Impact |
|---------|---------------|-------------------|------------|
| 1 | ~1 ns | ~2 ns | +3 ns vs single-buffer design |
| 2-4 | ~3 ns | ~4 ns | +5-7 ns |
| 8+ | ~5 ns | ~8 ns | +10-15 ns |

### B.5 Usage Example

```cpp
IbvVirtualSendWr wr{};
wr.wrId = wrId;

// Scatter-gather: data spread across multiple buffers
wr.localBuffers.push_back({buf0, len0});
wr.localBuffers.push_back({buf1, len1});
wr.localBuffers.push_back({buf2, len2});

wr.remoteAddr = remoteBuf;
wr.opcode = IBV_WR_RDMA_WRITE;
wr.sendFlags = IBV_SEND_SIGNALED;
wr.deviceKeys = {{mrs[0]->lkey, remoteRkeys[0]}, {mrs[1]->lkey, remoteRkeys[1]}};

vqp.postSend(wr);
```

### B.6 Risks and Considerations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Buffer boundary fragmentation | More fragments than expected | Recalculate expectedMsgCnt dynamically |
| Large scatter-gather lists | Performance degrades | Recommend ≤8 buffers for optimal performance |
| Memory region alignment | All buffers must share same MR | Document requirement |
