# Simplified Ibverbx Design: Essential VirtualCq with Registration-Based Routing

## Executive Summary

This document proposes a simplified Ibverbx architecture that **eliminates the Coordinator layer** while keeping `IbvVirtualCq` as an **essential component** (not a wrapper). The key design principles:

1. **User Interaction Model:**
   - `IbvVirtualQp::postSend()` - Users post work requests
   - `IbvVirtualCq::pollCq()` - Users poll completions (returns `IbvVirtualWc`)

2. **Registration-Based Routing:**
   - VirtualQp **registers** its physical QPs with VirtualCq
   - VirtualCq maintains mapping: `physical QP number → IbvVirtualQp*`

3. **Two Processing Paths in pollCq() based on isMultiQp flag:**
   - **Multi-QP VirtualQp** (isMultiQp=true): Route to VirtualQp → VirtualQp aggregates fragments and returns virtual CQE
   - **Single-QP VirtualQp** (isMultiQp=false): Pass through directly to user (fast path)

### Design Philosophy

| Component | Role |
|-----------|------|
| **IbvVirtualQp** | WR fragmentation, completion tracking, state management |
| **IbvVirtualCq** | CQ polling, routing registered CQEs to VirtualQp, pass-through for unregistered |

**Related Documents:**
- [Simplified Ibverbx Workflows](./simplified_ibverbx_workflows.md) - Detailed workflow diagrams
- [Simplified Ibverbx Performance Analysis](./simplified_ibverbx_performance_analysis.md) - Performance breakdown and analysis

---

## 1. Architecture Overview

### 1.1 Simplified Architecture (Essential VirtualCq)

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Code                               │
│  • postSend() → IbvVirtualQp                                     │
│  • pollCq() → IbvVirtualCq                                       │
└──────────┬─────────────────────────────────┬────────────────────┘
           │ postSend()                      │ pollCq()
           ▼                                 ▼
┌─────────────────────────────────┐  ┌─────────────────────────────┐
│      IbvVirtualQp               │  │      IbvVirtualCq           │
│  • Fragments large messages     │  │  • Polls physical CQ        │
│  • Load balancing (SPRAY/DQPLB) │  │  • Routes based on isMultiQp│
│  • Tracks pending virtual WRs   │  │  • Returns IbvVirtualWc      │
│  • Registers with VirtualCq     │◄─┤                             │
│  • processCompletion(wc)        │  │  Routing logic:             │
│    (multi-QP only)              │  │  • Single-QP: pass-through  │
│                                 │  │  • Multi-QP: aggregate      │
└──────────┬──────────────────────┘  └──────────┬──────────────────┘
           │                                    │
           │ Registration at construction       │ Poll loop
           │ physicalQpNum → {this, isMultiQp}  │
           └────────────────────────────────────┘
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
              • isMultiQp=false → Pass through directly (fast path)
              • isMultiQp=true  → VirtualQp.processCompletion()
```

**Key Differences from Current Design:**
1. **No Coordinator** - VirtualCq directly routes to registered VirtualQp
2. **Registration with isMultiQp flag** - VirtualQp indicates single vs multi-QP at registration
3. **Two-path pollCq()** - Fast path (pass-through) for single-QP, aggregation for multi-QP
4. **All QPs are VirtualQps** - Every QP is managed by a VirtualQp
5. **Simplified CQ model** - Typically 1 physical CQ (or 2 for multi-NIC), shared by all QPs

---

## 2. API Changes

### 2.1 New Data Structures

```cpp
// New: Custom send work request (replaces ibv_send_wr)
// Contains only the fields actually used by Ibverbx
struct IbvVirtualSendWr {
  uint64_t wrId;              // User's work request ID

  // Local buffer
  void* localAddr;            // Local buffer address
  uint32_t length;            // Buffer length

  // Remote buffer (for RDMA ops)
  uint64_t remoteAddr;        // Remote address

  // Operation
  ibv_wr_opcode opcode;       // IBV_WR_RDMA_WRITE, IBV_WR_RDMA_WRITE_WITH_IMM, etc.
  int sendFlags;              // IBV_SEND_SIGNALED, etc.
  uint32_t immData;           // Immediate data (for WRITE_WITH_IMM)

  // Per-device memory keys: maps deviceId -> {lkey, rkey}.
  // Mandatory field: 1 entry for single-NIC, N entries for multi-NIC.
  // Callers populate using deviceIds from IbvQp::getDeviceId().
  folly::F14FastMap<int32_t, MemoryRegionKeys> deviceKeys;
};

// Note: For scatter-gather support, see Appendix B.

// New: Custom recv work request (replaces ibv_recv_wr)
// Contains only the fields actually used by Ibverbx
struct IbvVirtualRecvWr {
  uint64_t wrId;              // User's work request ID

  // Local buffer (can be nullptr/0 for zero-length recv)
  void* localAddr;            // Local buffer address
  uint32_t length;            // Buffer length (0 = notification recv, >0 = data recv)

  // Per-device memory keys: maps deviceId -> {lkey, rkey}.
  // Mandatory field: 1 entry for single-NIC, N entries for multi-NIC.
  folly::F14FastMap<int32_t, MemoryRegionKeys> deviceKeys;
};

// New: Custom work completion (replaces ibv_wc at the VirtualQP boundary)
// Contains only the fields that callers actually consume.
// Symmetric with IbvVirtualSendWr/IbvVirtualRecvWr on the post side.
//
// For single-QP VirtualQp (isMultiQp=false):
//   Constructed from the physical ibv_wc by extracting the relevant fields.
//
// For multi-QP VirtualQp (isMultiQp=true):
//   Constructed by buildVirtualWc() from the completed ActiveVirtualWr.
//   - wrId: User's original work request ID
//   - status: Aggregated status (first error wins across all fragments)
//   - opcode: The operation type (mapped from WR opcode to WC opcode)
//   - byteLen: Total bytes of the user's original request (wr.length)
//   - qpNum: Virtual QP number (NOT a physical QP number)
//   - immData: User's original immediate data
struct IbvVirtualWc {
  uint64_t wrId{0};                        // User's original work request ID
  ibv_wc_status status{IBV_WC_SUCCESS};    // Aggregated status (first error wins)
  ibv_wc_opcode opcode{IBV_WC_SEND};       // Completion opcode (mapped from WR opcode)
  uint32_t qpNum{0};                       // Virtual QP number
  uint32_t immData{0};                     // Immediate data (if applicable)
  uint32_t byteLen{0};                     // Total byte length of the operation
};

```

```cpp
// Internal: Unified Active WR tracking (in VirtualQp)
// Full state for fragmentation, notify, and completion aggregation
// Used for both send and recv operations
struct ActiveVirtualWr {
  // Identity
  uint64_t userWrId;              // User's original wrId (for completion reporting)

  // Completion tracking - counts ALL expected CQEs (data + notify if applicable)
  // For SPRAY send: remainingMsgCnt = numFragments + 1 (includes notify CQE)
  // For SPRAY recv: remainingMsgCnt = 1 (notify only, data arrives via one-sided RDMA)
  // For DQPLB send: remainingMsgCnt = numFragments (no notify, seq# embedded in each fragment's IMM)
  // For DQPLB recv: remainingMsgCnt = 1 (logical completion, decremented by DqplbSeqTracker)
  int remainingMsgCnt;            // Decremented on each CQE; 0 = complete
  ibv_wc_status aggregatedStatus; // First error wins
  ibv_wc_opcode wcOpcode;         // Physical WC opcode (captured from last physicalWc.opcode)

  // Cached from IbvVirtualSendWr/IbvVirtualRecvWr (needed for fragmentation)
  void* localAddr;
  uint32_t length;
  uint64_t remoteAddr;            // Send only (0 for recv)
  ibv_wr_opcode opcode;           // The operation type
  uint32_t immData;               // Send only (0 for recv)
  folly::F14FastMap<int32_t, MemoryRegionKeys> deviceKeys;

  // Fragmentation progress (used by both send and recv)
  uint32_t offset;                // Current offset; allFragmentsSent = (offset >= length)

  // SPRAY notify tracking (send only, false for recv)
  bool needsNotify{false};        // True if this WR requires a notify (SPRAY mode)
  bool notifyPosted{false};       // True after notify has been posted to notifyQp

  // Helper: check if WR is fully complete
  // Simple: just check if all expected CQEs have been received
  bool isComplete() const {
    return remainingMsgCnt == 0;
  }

  // Helper: check if this is a send operation
  bool isSendOp() const {
    return opcode == IBV_WR_SEND ||
           opcode == IBV_WR_RDMA_WRITE ||
           opcode == IBV_WR_RDMA_WRITE_WITH_IMM ||
           opcode == IBV_WR_RDMA_READ;
  }

  // Derived state (computed, not stored):
  // - internalWrId: use map key directly
  // - allFragmentsSent: derive from (offset >= length)
  // - readyToPostNotify: derive from (needsNotify && !notifyPosted && remainingMsgCnt == 1)
};

// ============================================================
// Generic WR Tracker
// ============================================================
//
// Encapsulates the three-structure design for WR tracking.
// With unified ActiveVirtualWr, a single tracker handles both send and recv.
//
template <typename ActiveVirtualWrT>
struct WrTracker {
  // ----- Core three-structure design -----

  // All active (not yet completed) WRs
  // Key = internalWrId (always unique), Value = active WR state
  folly::F14FastMap<uint64_t, ActiveVirtualWrT> activeVirtualWrs_;

  // Pending queue: WRs not yet fully posted to physical QPs
  // Front = next WR to process. Popped when fully posted.
  std::deque<uint64_t> pendingQue_;

  // Outstanding queue: WRs posted, awaiting CQE (subset of active)
  // Front = next WR to report. Popped when complete.
  std::deque<uint64_t> outstandingQue_;

  // ----- ID generator -----
  // Monotonically increasing counter. uint64_t overflow is not a practical concern:
  // at 10 billion WRs/sec, it takes ~57 years to wrap around.
  uint64_t nextInternalVirtualWrId_{0};

  // ----- Core operations -----

  // Add new WR to tracker, returns internal ID
  // WR is added to pendingQue_ AND outstandingQue_ for in-order completion tracking
  uint64_t add(ActiveVirtualWrT&& wr) {
    uint64_t id = nextInternalVirtualWrId_++;
    activeVirtualWrs_.emplace(id, std::move(wr));
    pendingQue_.push_back(id);
    outstandingQue_.push_back(id);
    return id;
  }

  // O(1) lookup by internal ID
  ActiveVirtualWrT* find(uint64_t internalId) {
    auto it = activeVirtualWrs_.find(internalId);
    return it != activeVirtualWrs_.end() ? &it->second : nullptr;
  }

  const ActiveVirtualWrT* find(uint64_t internalId) const {
    auto it = activeVirtualWrs_.find(internalId);
    return it != activeVirtualWrs_.end() ? &it->second : nullptr;
  }

  // Remove completed WR from tracker
  // Caller should pop_front() from outstandingQue_ separately
  void remove(uint64_t internalId) {
    activeVirtualWrs_.erase(internalId);
  }

  // ----- Queue accessors -----

  bool hasPending() const { return !pendingQue_.empty(); }
  uint64_t frontPending() const { return pendingQue_.front(); }
  void popPending() { pendingQue_.pop_front(); }

  bool hasOutstanding() const { return !outstandingQue_.empty(); }
  uint64_t frontOutstanding() const { return outstandingQue_.front(); }
  void popOutstanding() { outstandingQue_.pop_front(); }

  // ----- Metrics -----

  size_t activeCount() const { return activeVirtualWrs_.size(); }
  size_t pendingCount() const { return pendingQue_.size(); }
  size_t outstandingCount() const { return outstandingQue_.size(); }
};
```

```cpp
// Physical WR tracking — correlates physical completions to virtual WR IDs.
// Lives in IbvQp (each physical QP maintains send and recv status deques).
//
// Production has two separate but structurally identical structs
// (PhysicalSendWrStatus, PhysicalRecvWrStatus). The design unifies them
// into a single type since the fields and semantics are the same —
// popPhysicalQueueStatus() operates on either deque generically.
struct PhysicalWrStatus {
  PhysicalWrStatus(uint64_t physicalWrId, uint64_t virtualWrId)
      : physicalWrId(physicalWrId), virtualWrId(virtualWrId) {}
  uint64_t physicalWrId{0};  // Physical WR ID (from ibv_wc.wr_id)
  uint64_t virtualWrId{0};   // Internal virtual WR ID (maps to ActiveVirtualWr)
};

// Each IbvQp holds:
//   std::deque<PhysicalWrStatus> physicalSendQueStatus_;
//   std::deque<PhysicalWrStatus> physicalRecvQueStatus_;
//
// These deques maintain FIFO ordering of posted WRs per physical QP.
// On completion, popPhysicalQueueStatus() pops the front entry and
// returns the virtualWrId for lookup in WrTracker::activeVirtualWrs_.
// The deque size also provides backpressure (limits outstanding WRs per QP).
```

### 2.2 IbvVirtualCq Class (Essential Component)

The VirtualCq maintains a **registration table** mapping physical QP numbers to VirtualQp pointers. This enables routing completions to the appropriate handler.

**Key Optimization**: During registration, VirtualQp indicates whether it uses a single physical QP or multiple physical QPs (load balancing). This enables fast-path handling:
- **Single-QP VirtualQp**: No fragmentation/aggregation needed → return CQE directly to user
- **Multi-QP VirtualQp**: Load balancing with fragmentation → route to VirtualQp for aggregation

```cpp
class IbvVirtualCq {
 public:
  IbvVirtualCq(IbvCq&& cq, int maxCqe);     // Single-CQ convenience constructor
  IbvVirtualCq(std::vector<IbvCq>&& cqs, int maxCqe);
  ~IbvVirtualCq();

  // disable copy, allow move
  IbvVirtualCq(const IbvVirtualCq&) = delete;
  IbvVirtualCq& operator=(const IbvVirtualCq&) = delete;
  IbvVirtualCq(IbvVirtualCq&& other) noexcept;
  IbvVirtualCq& operator=(IbvVirtualCq&& other) noexcept;

  // =========================================================
  // CORE API: Poll completions
  // =========================================================
  //
  // Returns a vector of IbvVirtualWc completions.
  // - Single-QP VirtualQp: constructs IbvVirtualWc from physical ibv_wc
  // - Multi-QP VirtualQp: routes to VirtualQp for aggregation, returns constructed IbvVirtualWc
  //
  folly::Expected<std::vector<IbvVirtualWc>, Error> pollCq(int numEntries);

  // =========================================================
  // REGISTRATION API: Called by VirtualQp at construction
  // =========================================================
  //
  // Registers a physical QP as belonging to a VirtualQp.
  // The `isMultiQp` flag indicates whether this VirtualQp uses
  // multiple physical QPs (load balancing) or just one.
  //
  // After registration:
  // - isMultiQp=false: CQEs passed through directly to user
  // - isMultiQp=true: CQEs routed to VirtualQp for processing
  //
  void registerPhysicalQp(
      uint32_t physicalQpNum,
      int32_t deviceId,
      IbvVirtualQp* vqp,
      bool isMultiQp,
      uint32_t virtualQpNum);
  void unregisterPhysicalQp(uint32_t physicalQpNum, int32_t deviceId);

  // Accessors
  std::vector<IbvCq>& getPhysicalCqsRef() { return physicalCqs_; }
  uint32_t getVirtualCqNum() const { return virtualCqNum_; }

 private:
  friend class IbvPd;
  friend class IbvVirtualQp;

  // VirtualCq unique ID (used by accessors and for debugging)
  uint32_t virtualCqNum_{0};
  inline static std::atomic<uint32_t> nextVirtualCqNum_{0};

  std::vector<IbvCq> physicalCqs_;
  int maxCqe_{0};

  // Completed virtual CQEs buffered between pollCq() calls.
  // Phase 1 of pollCq() drains all physical CQEs and appends completed
  // virtual WCs here. Phase 2 returns up to numEntries from the front.
  // Using deque for efficient front-pop in Phase 2.
  std::deque<IbvVirtualWc> completedVirtualWcs_;

  // =========================================================
  // REGISTRATION TABLE: Routes CQEs to VirtualQp
  // =========================================================
  //
  // Key: QpId (deviceId + qpNum)
  // Value: RegisteredQpInfo containing VirtualQp* and isMultiQp flag
  //
  struct QpId {
    int32_t deviceId;
    uint32_t qpNum;

    bool operator==(const QpId& other) const {
      return deviceId == other.deviceId && qpNum == other.qpNum;
    }
  };

  struct QpIdHash {
    std::size_t operator()(const QpId& id) const {
      return std::hash<int32_t>{}(id.deviceId) ^
             (std::hash<uint32_t>{}(id.qpNum) << 1);
    }
  };

  // Registration info for each physical QP
  struct RegisteredQpInfo {
    IbvVirtualQp* vqp;    // Non-owning pointer to VirtualQp
    bool isMultiQp;       // true if VirtualQp has >1 physical QPs
    uint32_t virtualQpNum; // Virtual QP number (for IbvVirtualWc.qpNum in passthrough)
  };

  // Registration table: QpId → RegisteredQpInfo
  folly::F14FastMap<QpId, RegisteredQpInfo, QpIdHash> registeredQps_;

  // Helper: Find registered QP info
  inline const RegisteredQpInfo* findRegisteredQpInfo(
      uint32_t qpNum, int32_t deviceId) const;
};
```

**Registration Flow:**

```
VirtualQp Construction
         │
         │ Determine isMultiQp = (physicalQps_.size() > 1)
         │
         ▼
┌─────────────────────────────────────────┐
│ For each physical QP in physicalQps_:   │
│   virtualCq->registerPhysicalQp(        │
│       qp.qpNum, qp.deviceId, this,      │
│       isMultiQp, virtualQpNum_)         │
│                                         │
│ Also register notifyQp_ (if multi-QP)   │
└─────────────────────────────────────────┘
         │
         ▼
VirtualCq.registeredQps_[{deviceId, qpNum}] = {this, isMultiQp}
         │
         ▼
pollCq() behavior:
  - isMultiQp=false: Return CQE directly to user (fast path)
  - isMultiQp=true: Route CQE to VirtualQp.processCompletion()
```

**Polling Behavior Summary:**

| VirtualQp Type | Physical QPs | Registration | pollCq() Handling |
|----------------|--------------|--------------|-------------------|
| Single-QP | 1 | isMultiQp=false | Direct pass-through |
| Multi-QP | >1 | isMultiQp=true | Route to VirtualQp |

### 2.3 Modified IbvVirtualQp Class

**IbvVirtualQpBusinessCard** — Reused from production unchanged. Carries physical QP numbers for connection setup exchange:

```cpp
// Reused from production (IbvVirtualQp.h) — no changes needed.
struct IbvVirtualQpBusinessCard {
  explicit IbvVirtualQpBusinessCard(
      std::vector<uint32_t> qpNums,
      uint32_t notifyQpNum = 0);
  IbvVirtualQpBusinessCard() = default;
  ~IbvVirtualQpBusinessCard() = default;

  // Default copy/move
  IbvVirtualQpBusinessCard(const IbvVirtualQpBusinessCard&) = default;
  IbvVirtualQpBusinessCard& operator=(const IbvVirtualQpBusinessCard&) = default;
  IbvVirtualQpBusinessCard(IbvVirtualQpBusinessCard&&) = default;
  IbvVirtualQpBusinessCard& operator=(IbvVirtualQpBusinessCard&&) = default;

  // Serialization (folly::dynamic + JSON)
  folly::dynamic toDynamic() const;
  static folly::Expected<IbvVirtualQpBusinessCard, Error> fromDynamic(
      const folly::dynamic& obj);
  std::string serialize() const;
  static folly::Expected<IbvVirtualQpBusinessCard, Error> deserialize(
      const std::string& jsonStr);

  // Ordered: ith QP connects to ith remote QP
  std::vector<uint32_t> qpNums_;
  uint32_t notifyQpNum_{0};  // 0 when no notifyQp (e.g., DQPLB mode)
};
```

```cpp
class IbvVirtualQp {
 public:
  // CONSTRUCTOR: Takes VirtualCq pointer for registration
  //
  // Design assumptions:
  // - Single CQ: sendCq and recvCq are the same VirtualCq instance.
  //   This matches current VirtualQp usage patterns.
  // - notifyQp is optional: Only needed for multi-QP VirtualQps using
  //   SPRAY mode with RDMA_WRITE_WITH_IMM. Pass std::nullopt otherwise.
  //
  IbvVirtualQp(
      std::vector<IbvQp>&& qps,
      IbvVirtualCq* virtualCq,          // Single CQ for both send/recv
      int maxMsgCntPerQp = kIbMaxMsgCntPerQp,
      uint32_t maxMsgSize = kIbMaxMsgSizeByte,
      LoadBalancingScheme loadBalancingScheme = LoadBalancingScheme::SPRAY,
      std::optional<IbvQp>&& notifyQp = std::nullopt);  // Optional: only for SPRAY mode

  ~IbvVirtualQp();  // Unregisters from VirtualCq

  // NEW: Post send request with custom IbvVirtualSendWr (simpler API)
  folly::Expected<folly::Unit, Error> postSend(const IbvVirtualSendWr& wr);

  // NEW: Post recv request with custom IbvVirtualRecvWr (simpler API)
  folly::Expected<folly::Unit, Error> postRecv(const IbvVirtualRecvWr& wr);

  // =========================================================
  // COMPLETION PROCESSING: Called by VirtualCq (multi-QP only)
  // =========================================================
  //
  // Processes a physical completion and returns virtual CQE(s).
  // Only called for multi-QP VirtualQps (load balancing mode).
  // Single-QP VirtualQps have their CQEs passed through directly.
  //
  // Returns:
  //   - Empty vector: No virtual completion ready yet (intermediate fragment)
  //   - Non-empty vector: One or more virtual completions ready
  //
  folly::Expected<std::vector<IbvVirtualWc>, Error> processCompletion(
      const ibv_wc& physicalWc,
      int32_t deviceId = 0);

  // Batch processing for maximum performance
  folly::Expected<std::vector<IbvVirtualWc>, Error> processCompletions(
      const ibv_wc* physicalWcs,
      int count,
      int32_t deviceId = 0);

  // =========================================================
  // CONNECTION SETUP (reused from production, unchanged)
  // =========================================================
  //
  // These methods handle QP state transitions (INIT→RTR→RTS) and
  // business card exchange for connection setup. They are orthogonal
  // to the data-path redesign and are reused from production with
  // only minor std::optional guards for notifyQp_.
  //

  // Modify all physical QPs (and notifyQp_ if present) with the given attributes.
  // If businessCard is provided, sets dest_qp_num per-QP from the remote side's card.
  folly::Expected<folly::Unit, Error> modifyVirtualQp(
      ibv_qp_attr* attr,
      int attrMask,
      const IbvVirtualQpBusinessCard& businessCard =
          IbvVirtualQpBusinessCard());

  // Extract physical QP numbers into a BusinessCard for exchange with remote side.
  IbvVirtualQpBusinessCard getVirtualQpBusinessCard() const;

  // Accessors
  bool isMultiQp() const { return isMultiQp_; }
  size_t getTotalQps() const { return physicalQps_.size(); }
  const std::vector<IbvQp>& getQpsRef() const { return physicalQps_; }
  std::vector<IbvQp>& getQpsRef() { return physicalQps_; }
  // Returns notifyQp reference. Caller must verify notifyQp_.has_value() first.
  const IbvQp& getNotifyQpRef() const { return *notifyQp_; }
  IbvQp& getNotifyQpRef() { return *notifyQp_; }
  uint32_t getVirtualQpNum() const { return virtualQpNum_; }
  LoadBalancingScheme getLoadBalancingScheme() const { return loadBalancingScheme_; }

 private:
#ifdef IBVERBX_TEST_FRIENDS
  IBVERBX_TEST_FRIENDS
#endif

  // Physical QPs
  std::vector<IbvQp> physicalQps_;
  std::optional<IbvQp> notifyQp_;               // Optional: only for SPRAY mode
  std::unordered_map<uint32_t, int> qpNumToIdx_;
  DqplbSeqTracker dqplbSeqTracker_;  // NOTE: getSendImm() signature changes from int to bool

  // VirtualCq reference (single CQ for both send/recv)
  IbvVirtualCq* virtualCq_;

  // Flag indicating if this VirtualQp uses multiple physical QPs
  // - isMultiQp_=false: Single QP, no fragmentation/aggregation
  // - isMultiQp_=true: Multiple QPs, load balancing with fragmentation
  bool isMultiQp_{false};
  int deviceCnt_{1};  // Number of unique devices across all physical QPs

  // ============================================================
  // WR Tracking: Separate Trackers with Unified Type
  // ============================================================
  //
  // Send and recv use separate trackers for independent ordering.
  // Both use the same ActiveVirtualWr type to eliminate code duplication.
  // See Section 2.1 for WrTracker and ActiveVirtualWr definitions.

  // ----- SEND TRACKING -----
  WrTracker<ActiveVirtualWr> sendTracker_;
  //
  // Access patterns:
  // - sendTracker_.add(wr)            → add new WR (adds to pendingQue_ AND outstandingQue_)
  // - sendTracker_.find(id)           → O(1) lookup by internal ID
  // - sendTracker_.hasPending()       → check if WRs need posting
  // - sendTracker_.frontPending()     → get next WR to post
  // - sendTracker_.popPending()       → remove WR from pending after all fragments sent
  // - sendTracker_.hasOutstanding()   → check if WRs awaiting CQE
  // - sendTracker_.frontOutstanding() → get next WR to report (in-order)
  // - sendTracker_.popOutstanding()   → remove WR from outstanding after reporting
  // - sendTracker_.remove(id)         → remove completed WR from activeVirtualWrs_

  // ----- RECV TRACKING -----
  WrTracker<ActiveVirtualWr> recvTracker_;
  //
  // Same access patterns as sendTracker_.
  // Separate tracker ensures recv completions aren't blocked by pending sends.

  // ----- SPRAY NOTIFY TRACKING -----
  //
  // SEND: Backpressure queue for WRs ready to send notify but blocked
  std::deque<uint64_t> pendingNotifyQue_;
  //
  // RECV: Backpressure queue for recv WRs waiting to post to notifyQp
  std::deque<uint64_t> pendingRecvNotifyQue_;
  //
  // In-flight tracking uses notifyQp_->physicalSendQueStatus_ (send) and
  // notifyQp_->physicalRecvQueStatus_ (recv).
  // Backpressure check: queue.size() >= maxMsgCntPerQp_
  // (same limit as data QPs — derived from QP send queue depth)

  // ----- PHYSICAL WR ID GENERATOR -----
  uint64_t nextPhysicalWrId_{0};   // Unique ID for each physical WR fragment

  // ----- VIRTUAL QP NUMBER -----
  uint32_t virtualQpNum_{0};       // Unique ID for this VirtualQp instance
  inline static std::atomic<uint32_t> nextVirtualQpNum_{0};  // Global counter

  // ----- ROUND-ROBIN STATE -----
  int nextSendPhysicalQpIdx_{0};  // Round-robin state for send QP selection

  // ----- DQPLB STATE -----
  bool dqplbReceiverInitialized_{false};

  // ============================================================
  // SEND HELPER FUNCTIONS
  // ============================================================

  // Single-QP fast path (pure passthrough, no tracking)
  inline folly::Expected<folly::Unit, Error> postSendSingleQp(const IbvVirtualSendWr& wr);

  // Multi-QP path helpers
  inline folly::Expected<folly::Unit, Error> dispatchPendingSends(int freedQpIdx = -1);
  inline void buildPhysicalSendWr(
      const ActiveVirtualWr& pending, int32_t deviceId, uint32_t fragLen,
      ibv_send_wr* sendWr, ibv_sge* sendSge);
  inline folly::Expected<std::vector<IbvVirtualWc>&, Error> reportSendCompletions(std::vector<IbvVirtualWc>& results);
  inline folly::Expected<folly::Unit, Error> flushPendingSendNotifies();  // Process pendingNotifyQue_ when backpressure clears
  inline folly::Expected<folly::Unit, Error> postNotifyForWr(uint64_t internalWrId);
  inline int findAvailableSendQp();
  inline bool hasQpCapacity(int qpIdx) const;  // Check if specific QP has available slot

  // ============================================================
  // RECEIVE HELPER FUNCTIONS
  // ============================================================

  // Single-QP fast path (pure passthrough, no tracking)
  inline folly::Expected<folly::Unit, Error> postRecvSingleQp(const IbvVirtualRecvWr& wr);

  // SPRAY mode helpers
  inline folly::Expected<folly::Unit, Error> postRecvToNotifyQp(uint64_t internalWrId);
  inline folly::Expected<folly::Unit, Error> flushPendingRecvNotifies();  // Process pendingRecvNotifyQue_ when backpressure clears

  // DQPLB mode helpers
  inline folly::Expected<folly::Unit, Error> initializeDqplbReceiver();  // Pre-post recvs for DQPLB
  inline folly::Expected<folly::Unit, Error> replenishDqplbRecv(int qpIdx);  // Replenish recv on specific QP

  // Completion helper
  inline folly::Expected<std::vector<IbvVirtualWc>&, Error> reportRecvCompletions(std::vector<IbvVirtualWc>& results);

  // ============================================================
  // COMPLETION PROCESSING HANDLERS (2x2 matrix: QP type × direction)
  // ============================================================
  //
  //                     Send                         Recv
  //            ┌─────────────────────────┬─────────────────────────┐
  // NotifyQp   │ processNotifyQpSend-    │ processNotifyQpRecv-    │
  //            │   Completion()          │   Completion()          │
  //            ├─────────────────────────┼─────────────────────────┤
  // DataQp     │ processDataQpSend-      │ processDataQpRecv-      │
  //            │   Completion()          │   Completion()          │
  //            └─────────────────────────┴─────────────────────────┘
  //
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processNotifyQpSendCompletion(
      const ibv_wc& physicalWc, std::vector<IbvVirtualWc>& results);
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processNotifyQpRecvCompletion(
      const ibv_wc& physicalWc, std::vector<IbvVirtualWc>& results);
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processDataQpSendCompletion(
      const ibv_wc& physicalWc, int qpIdx, std::vector<IbvVirtualWc>& results);
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> processDataQpRecvCompletion(
      const ibv_wc& physicalWc, int qpIdx, std::vector<IbvVirtualWc>& results);

  // Common helpers for completion handlers
  inline bool isSendOpcode(ibv_wc_opcode opcode) const;
  inline folly::Expected<folly::Unit, Error> updateWrState(
      WrTracker<ActiveVirtualWr>& tracker, uint64_t internalWrId,
      ibv_wc_status status, ibv_wc_opcode wcOpcode);
  inline IbvVirtualWc buildVirtualWc(const ActiveVirtualWr& wr) const;

  // Registration helper (called from constructor/destructor)
  void registerWithVirtualCq();
  void unregisterFromVirtualCq();
};
```

### 2.4 IbvVirtualQp Design Considerations

#### 2.4.1 Lifetime and Thread Safety

**Lifetime Dependency:**
```
VirtualCq must outlive VirtualQp
┌──────────────┐
│  VirtualCq   │  ← Created first
│              │
│  registeredQps_:
│    qp100 → VirtualQp_A
│    qp101 → VirtualQp_A
└──────────────┘
       ▲
       │ references
       │
┌──────────────┐
│ VirtualQp_A  │  ← Created second, destroyed first
│  virtualCq_ ─┼──→ VirtualCq
└──────────────┘
```

**Thread Safety:**
- VirtualCq and associated VirtualQps should be used from the same thread
- If multi-threaded access is needed, external synchronization is required
- `registeredQps_` access during pollCq() must not race with registration/unregistration

#### 2.4.2 Move Semantics

When VirtualQp is moved, the registration must be updated:

```cpp
IbvVirtualQp::IbvVirtualQp(IbvVirtualQp&& other) noexcept
    : physicalQps_(std::move(other.physicalQps_)),
      notifyQp_(std::move(other.notifyQp_)),
      virtualCq_(other.virtualCq_),
      // ... other members ...
{
  other.virtualCq_ = nullptr;  // Prevent double-unregister

  // Re-register with new 'this' pointer
  registerWithVirtualCq();
}

IbvVirtualQp& IbvVirtualQp::operator=(IbvVirtualQp&& other) noexcept {
  if (this != &other) {
    // Unregister current QPs from old VirtualCq
    unregisterFromVirtualCq();

    // Move all members
    physicalQps_ = std::move(other.physicalQps_);
    notifyQp_ = std::move(other.notifyQp_);
    qpNumToIdx_ = std::move(other.qpNumToIdx_);
    dqplbSeqTracker_ = std::move(other.dqplbSeqTracker_);
    virtualCq_ = other.virtualCq_;
    isMultiQp_ = other.isMultiQp_;
    deviceCnt_ = other.deviceCnt_;
    sendTracker_ = std::move(other.sendTracker_);
    recvTracker_ = std::move(other.recvTracker_);
    pendingNotifyQue_ = std::move(other.pendingNotifyQue_);
    pendingRecvNotifyQue_ = std::move(other.pendingRecvNotifyQue_);
    nextPhysicalWrId_ = other.nextPhysicalWrId_;
    virtualQpNum_ = other.virtualQpNum_;
    nextSendPhysicalQpIdx_ = other.nextSendPhysicalQpIdx_;
    maxMsgCntPerQp_ = other.maxMsgCntPerQp_;
    maxMsgSize_ = other.maxMsgSize_;
    loadBalancingScheme_ = other.loadBalancingScheme_;
    dqplbReceiverInitialized_ = other.dqplbReceiverInitialized_;

    other.virtualCq_ = nullptr;  // Prevent double-unregister

    // Re-register with new 'this' pointer
    registerWithVirtualCq();
  }
  return *this;
}
```

VirtualCq move assignment follows the same pattern:

```cpp
IbvVirtualCq& IbvVirtualCq::operator=(IbvVirtualCq&& other) noexcept {
  if (this != &other) {
    physicalCqs_ = std::move(other.physicalCqs_);
    maxCqe_ = other.maxCqe_;
    virtualCqNum_ = other.virtualCqNum_;
    completedVirtualWcs_ = std::move(other.completedVirtualWcs_);
    registeredQps_ = std::move(other.registeredQps_);

    // Update all registered VirtualQps' virtualCq_ pointer to new 'this'
    // (VirtualQps hold raw pointers to their VirtualCq)
    for (auto& [key, info] : registeredQps_) {
      if (info.vqp != nullptr) {
        info.vqp->virtualCq_ = this;
      }
    }
  }
  return *this;
}
```

#### 2.4.3 Connection Setup (modifyVirtualQp / BusinessCard)

Connection setup functions are **reused from production** with minor `std::optional` guards for `notifyQp_`. They are orthogonal to the data-path redesign (fragmentation, completion aggregation, Coordinator removal).

**`IbvVirtualQpBusinessCard`** — Reused unchanged. Self-contained struct with serialization, no dependency on Coordinator or data-path logic.

**`getVirtualQpBusinessCard()`** — Adapted for `std::optional<IbvQp> notifyQp_`:

```cpp
IbvVirtualQpBusinessCard IbvVirtualQp::getVirtualQpBusinessCard() const {
  std::vector<uint32_t> qpNums;
  qpNums.reserve(physicalQps_.size());
  for (auto& qp : physicalQps_) {
    qpNums.push_back(qp.qp()->qp_num);
  }
  // notifyQpNum is 0 when no notifyQp (e.g., DQPLB mode)
  uint32_t notifyQpNum = notifyQp_.has_value() ? notifyQp_->qp()->qp_num : 0;
  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}
```

**`modifyVirtualQp()`** — Adapted for `std::optional<IbvQp> notifyQp_`:

```cpp
folly::Expected<folly::Unit, Error> IbvVirtualQp::modifyVirtualQp(
    ibv_qp_attr* attr,
    int attrMask,
    const IbvVirtualQpBusinessCard& businessCard) {
  if (!businessCard.qpNums_.empty()) {
    if (businessCard.qpNums_.size() != physicalQps_.size()) {
      return folly::makeUnexpected(Error(
          EINVAL, "BusinessCard QP count doesn't match physical QP count"));
    }
    for (auto i = 0; i < physicalQps_.size(); i++) {
      attr->dest_qp_num = businessCard.qpNums_.at(i);
      auto maybeModifyQp = physicalQps_.at(i).modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    // Only modify notifyQp if it exists (SPRAY mode)
    if (notifyQp_.has_value()) {
      attr->dest_qp_num = businessCard.notifyQpNum_;
      auto maybeModifyQp = notifyQp_->modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
  } else {
    for (auto& qp : physicalQps_) {
      auto maybeModifyQp = qp.modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    if (notifyQp_.has_value()) {
      auto maybeModifyQp = notifyQp_->modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
  }
  return folly::unit;
}
```

**Changes from production:**
- `notifyQp_` access guarded by `notifyQp_.has_value()` (production accesses unconditionally since `notifyQp_` is always an `IbvQp`, not `std::optional<IbvQp>`)
- `notifyQp_->` instead of `notifyQp_.` (dereference through `std::optional`)
- `BusinessCard::notifyQpNum_` defaults to 0 when no notifyQp exists

#### 2.4.4 Shared VirtualCq

Multiple VirtualQps can share the same VirtualCq:

```
VirtualCq (shared)
  └── registeredQps_
        ├── {device0, qp100} → VirtualQp_A   ─┐
        ├── {device0, qp101} → VirtualQp_A   ─┤ VirtualQp_A's physical QPs
        ├── {device0, qp102} → VirtualQp_A   ─┘
        ├── {device1, qp200} → VirtualQp_B   ─┐
        ├── {device1, qp201} → VirtualQp_B   ─┤ VirtualQp_B's physical QPs
        └── {device1, qp202} → VirtualQp_B   ─┘

        (unregistered QPs pass through directly)
```

The simplified design uses a generic `WrTracker<T>` template (see Section 2.1) with a unified `ActiveVirtualWr` type:

**WrTracker Design:**

| Component | Type | Purpose |
|-----------|------|---------|
| `activeVirtualWrs_` | `F14FastMap<uint64_t, T>` | All active (not completed) WRs, O(1) lookup |
| `pendingQue_` | `deque<uint64_t>` | WRs not yet fully posted to physical QPs |
| `outstandingQue_` | `deque<uint64_t>` | WRs posted, awaiting CQE |
| `nextInternalVirtualWrId_` | `uint64_t` | Generates unique internal IDs |

**VirtualQp Members:**

| Member | Type | Purpose |
|--------|------|---------|
| `sendTracker_` | `WrTracker<ActiveVirtualWr>` | Send WR tracking |
| `recvTracker_` | `WrTracker<ActiveVirtualWr>` | Recv WR tracking |
| `notifyQp_->physicalSendQueStatus_` | `deque<PhysicalWrStatus>` | Notify CQE correlation + backpressure |

**Why unified ActiveVirtualWr type with separate trackers?**

1. **Unified type**: Single `ActiveVirtualWr` eliminates code duplication for WR state updates
2. **Separate trackers**: Send and recv have independent ordering - recv completions aren't blocked by pending sends
3. **Recv fragmentation support**: Recv operations (SEND/RECV mode) may also need fragmentation for large messages
4. **Memory trade-off**: ~56 extra bytes per recv WR (acceptable for most workloads)

**SPRAY Notify Ordering (Head-of-Line Blocking):**

SPRAY mode with `RDMA_WRITE_WITH_IMM` requires a separate notify message after all data fragments complete. The design provides a strong ordering guarantee:

> **When receiver gets WR_B's notify, ALL prior WRs (including non-notify WRs) are guaranteed complete.**

This is achieved via head-of-line blocking in `reportSendCompletions()`:
- Notify can only be posted when WR is at **front** of `sendTracker_.outstandingQue_`
- This ensures all prior WRs have completed before posting notify
- `notifyQp_->physicalSendQueStatus_.size()` provides backpressure (limits outstanding notifies)

**Why this design?**

1. **O(1) completion lookup**: Physical completions arrive out-of-order across NICs. F14FastMap gives O(1) lookup by internalWrId.

2. **Internal unique IDs**: User's `wrId` may have duplicates (ibverbs allows this). We generate unique `internalWrId` for tracking, store user's `wrId` for reporting. This ensures correct behavior regardless of user input.

3. **WR scheduling**: Large WRs may not fully post (QP capacity). `sendTracker_.pendingQue_` / `recvTracker_.pendingQue_` tracks which WRs need more posting, in order.

4. **In-order completion**: Users expect completions in posting order within each category (send/recv). Separate trackers ensure this without cross-blocking.

5. **Lightweight queues**: Deques hold only 8-byte internal WR IDs, not full WR objects. No expensive queue-moves.

6. **O(1) notify ordering**: Head-of-line blocking means we only check the front of `sendTracker_.outstandingQue_`, not scan the entire queue.

### 2.5 Usage Example

**Simplified Usage (Essential VirtualCq with Registration):**
```cpp
// Setup - VirtualQp registers with VirtualCq at construction
// Single CQ for both send and recv operations
IbvVirtualCq virtualCq(std::move(cqs), maxCqe);

// Create VirtualQp with new simplified API
// - Single CQ (no separate sendCq/recvCq)
// - notifyQp is optional (only needed for SPRAY mode with WRITE_WITH_IMM)
std::vector<IbvQp> qps = {...};  // Physical QPs for load balancing
IbvVirtualQp vqp(
    std::move(qps),
    &virtualCq,                        // Single CQ for send/recv
    kIbMaxMsgCntPerQp,                 // maxMsgCntPerQp (default)
    kIbMaxMsgSizeByte,                 // maxMsgSize (default)
    LoadBalancingScheme::SPRAY,        // load balancing scheme
    std::move(notifyQp));              // Optional: only for SPRAY mode
// ↑ VirtualQp automatically registers its physical QPs with virtualCq
//   Registration includes isMultiQp flag based on qps.size()

// Prepare IbvVirtualSendWr (simple, clean API)
IbvVirtualSendWr sendWr{};
sendWr.wrId = wrId;
sendWr.localAddr = localBuf;
sendWr.length = len;
sendWr.remoteAddr = remoteBuf;
sendWr.opcode = IBV_WR_RDMA_WRITE;
sendWr.sendFlags = IBV_SEND_SIGNALED;

// Device keys bundled in the struct (keyed by deviceId)
sendWr.deviceKeys[devices[0].getDeviceId()] = {mrs[0]->lkey, remoteRkeys[0]};
sendWr.deviceKeys[devices[1].getDeviceId()] = {mrs[1]->lkey, remoteRkeys[1]};

// Send (simplified - one parameter)
vqp.postSend(sendWr);

// Prepare IbvVirtualRecvWr (for RDMA WRITE_WITH_IMM notification)
IbvVirtualRecvWr recvWr{};
recvWr.wrId = recvWrId;
recvWr.localAddr = nullptr;  // Zero-length recv for WRITE_WITH_IMM
recvWr.length = 0;
// recvWr.deviceKeys empty for single-NIC, or populate for multi-NIC

// Post recv
vqp.postRecv(recvWr);

// Poll - returns IbvVirtualWc completions
auto wcs = virtualCq.pollCq(32);
for (auto& wc : *wcs) {
  handleCompletion(wc.wrId, wc.status);  // Note: wrId not wr_id
}
```

---

## 3. Implementation Details

### 3.1 Modified postSend() Implementation

The `postSend()` function validates user input, routes based on opcode, and initiates transmission.

**Opcode Routing:**

| Opcode | Path | Description |
|--------|------|-------------|
| `IBV_WR_SEND` | Single QP | No load balancing, no fragmentation. Uses `postSendSingleQp()`. |
| `IBV_WR_RDMA_WRITE` | Load balancing | Fragments across multiple QPs for bandwidth. |
| `IBV_WR_RDMA_WRITE_WITH_IMM` | Load balancing | Fragments + SPRAY notify for receiver notification. |
| `IBV_WR_RDMA_READ` | Load balancing | Fragments across multiple QPs. |
| Others | Error | Unsupported opcodes return error. |

**Design Decision - Why SEND uses Single QP:**
- `IBV_WR_SEND` requires matching recv buffers on the receiver side
- Message ordering is critical for SEND - fragments could arrive out-of-order with load balancing
- RDMA operations write directly to known remote addresses and are order-independent
- Assumption: Opcodes will not be mixed (no SEND followed by RDMA_WRITE) - enforced with runtime check

**Implementation:**

```cpp
// Helper: Single-QP fast path (pure passthrough)
// Used for: !isMultiQp_ OR IBV_WR_SEND in multi-QP mode
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSendSingleQp(
    const IbvVirtualSendWr& wr) {
  // Minimal validation
  if (wr.length == 0) {
    return folly::makeUnexpected(Error(EINVAL,
        "[Ibverbx]IbvVirtualQp::postSendSingleQp, length cannot be zero"));
  }

  // Build physical WR directly from virtual WR
  ibv_send_wr sendWr{};
  ibv_sge sendSge{};

  sendSge.addr = reinterpret_cast<uint64_t>(wr.localAddr);
  sendSge.length = wr.length;
  int32_t deviceId = physicalQps_[0].getDeviceId();
  sendSge.lkey = wr.deviceKeys.at(deviceId).lkey;

  sendWr.wr_id = wr.wrId;  // Use user's wrId directly (no internal ID)
  sendWr.sg_list = &sendSge;
  sendWr.num_sge = 1;
  sendWr.opcode = wr.opcode;
  sendWr.send_flags = wr.sendFlags;
  sendWr.wr.rdma.remote_addr = wr.remoteAddr;
  sendWr.wr.rdma.rkey = wr.deviceKeys.at(deviceId).rkey;
  if (wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
    sendWr.imm_data = wr.immData;
  }

  // Post directly to physical QP 0
  ibv_send_wr badWr{};
  auto maybePost = physicalQps_[0].postSend(&sendWr, &badWr);
  if (maybePost.hasError()) {
    return folly::makeUnexpected(maybePost.error());
  }

  // No tracking needed - VirtualCq returns physical WC directly to user
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postSend(
    const IbvVirtualSendWr& wr) {

  // Fast path: Single physical QP (pure passthrough)
  if (!isMultiQp_) {
    return postSendSingleQp(wr);
  }

  // ============================================================
  // MULTI-QP PATH
  // ============================================================

  // Parameter validation
  if (wr.length == 0) {
    return folly::makeUnexpected(Error(EINVAL,
        "[Ibverbx]IbvVirtualQp::postSend, length cannot be zero"));
  }

  // Multi-QP mode requires signaled operations for in-order delivery.
  // Unsignaled WRs cannot be tracked through outstandingQue_, which would
  // break the ordering guarantee: when WR_C completes, all prior WRs
  // (including unsignaled WR_B) must be done on the remote side.
  // Single-QP passthrough is unaffected (ibverbs guarantees within-QP ordering).
  if (!(wr.sendFlags & IBV_SEND_SIGNALED) &&
      wr.opcode != IBV_WR_RDMA_WRITE_WITH_IMM) {
    return folly::makeUnexpected(Error(EINVAL,
        "[Ibverbx]IbvVirtualQp::postSend, unsignaled operations not supported in multi-QP mode"));
  }

  // Opcode routing
  switch (wr.opcode) {
    case IBV_WR_SEND:
      // SEND uses single QP (QP[0]) - no load balancing, no fragmentation.
      // CQEs for SEND are returned as passthrough in pollCq() (opcode-based
      // routing), so they never enter processCompletion() / sendTracker_.
      return postSendSingleQp(wr);

    case IBV_WR_RDMA_WRITE:
    case IBV_WR_RDMA_WRITE_WITH_IMM:
    case IBV_WR_RDMA_READ:
      break;  // RDMA operations use load balancing

    default:
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx]IbvVirtualQp::postSend, unsupported opcode: {}", static_cast<int>(wr.opcode))));
  }

  // ============================================================
  // RDMA LOAD BALANCING PATH
  // ============================================================

  // Calculate fragment count and determine notify requirement
  bool needsNotify = (wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
                      loadBalancingScheme_ == LoadBalancingScheme::SPRAY);
  int expectedMsgCnt = (wr.length + maxMsgSize_ - 1) / maxMsgSize_;

  // Add WR to tracker for completion aggregation
  // (unsignaled multi-QP ops are rejected above, so all WRs here need tracking)
  //
  // Note on immData: In multi-QP mode, IbvVirtualQp uses immData internally
  // for control signaling only — DQPLB embeds sequence numbers, SPRAY sends
  // a zero-length RDMA_WRITE_WITH_IMM notify with the user's immData.
  // Callers (e.g., CtranIb) do NOT read immData from multi-QP IbvVirtualWc
  // completions; they consume notifications via opcode detection
  // (IBV_WC_RECV_RDMA_WITH_IMM). For single-QP passthrough, immData is
  // copied directly from the physical ibv_wc in pollCq().
  uint64_t internalId = sendTracker_.add(
      ActiveVirtualWr{
          .userWrId = wr.wrId,
          // SPRAY: +1 for notify CQE; non-SPRAY: data fragments only
          .remainingMsgCnt = needsNotify ? expectedMsgCnt + 1 : expectedMsgCnt,
          .aggregatedStatus = IBV_WC_SUCCESS,
          .localAddr = wr.localAddr,
          .length = wr.length,
          .remoteAddr = wr.remoteAddr,
          .opcode = wr.opcode,
          .immData = wr.immData,
          .deviceKeys = wr.deviceKeys,
          .offset = 0,
          .needsNotify = needsNotify,
          .notifyPosted = false
      });

  // Dispatch fragments to physical QPs
  if (dispatchPendingSends().hasError()) {
    return folly::makeUnexpected(Error(errno));
  }

  return folly::unit;
}

// Single QP receive (passthrough to physical QP 0)
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecvSingleQp(
    const IbvVirtualRecvWr& wr) {

  ibv_recv_wr recvWr{};
  ibv_sge sge{};

  recvWr.wr_id = wr.wrId;
  recvWr.next = nullptr;

  if (wr.length > 0) {
    sge.addr = reinterpret_cast<uint64_t>(wr.localAddr);
    sge.length = wr.length;
    int32_t deviceId = physicalQps_[0].getDeviceId();
    sge.lkey = wr.deviceKeys.at(deviceId).lkey;
    recvWr.sg_list = &sge;
    recvWr.num_sge = 1;
  } else {
    recvWr.sg_list = nullptr;
    recvWr.num_sge = 0;
  }

  ibv_recv_wr badWr{};
  auto maybePost = physicalQps_[0].postRecv(&recvWr, &badWr);
  if (maybePost.hasError()) {
    return folly::makeUnexpected(maybePost.error());
  }

  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecv(
    const IbvVirtualRecvWr& wr) {

  // Fast path: Single physical QP (pure passthrough)
  if (!isMultiQp_) {
    return postRecvSingleQp(wr);
  }

  // ============================================================
  // MULTI-QP PATH
  // ============================================================

  // Route based on length: 0 = notification recv, >0 = data recv
  // (Matches production pattern of using num_sge == 0 vs num_sge == 1)
  if (wr.length > 0) {
    // Data recv (IBV_WR_SEND matching) — uses single QP, no load balancing.
    // LIMITATION: Neither send-side nor recv-side SEND fragmentation is
    // supported in the simplified design. Production fragments both sides.
    // SEND messages must fit within a single physical QP's capacity.
    // This matches current CtranIb usage where SEND is only used for
    // small control messages. See I7 in ibverbx_gaps_2.md.
    return postRecvSingleQp(wr);
  }

  // ============================================================
  // ZERO-LENGTH RECV PATH (SPRAY/DQPLB notification)
  // ============================================================
  //
  // For RDMA_WRITE_WITH_IMM, data arrives via one-sided RDMA_WRITE (no recv needed).
  // Receiver only needs to post zero-length recv on notifyQp to catch IMM notification.

  // Add WR to tracker for completion tracking
  // Only essential fields needed for recv: userWrId, remainingMsgCnt, aggregatedStatus
  uint64_t internalId = recvTracker_.add(
      ActiveVirtualWr{
          .userWrId = wr.wrId,
          .remainingMsgCnt = 1,
          .aggregatedStatus = IBV_WC_SUCCESS,
          .localAddr = nullptr,
          .length = 0,
          .remoteAddr = 0,
          .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,  // Notification recv always corresponds to WRITE_WITH_IMM
          .immData = 0,
          .deviceKeys = {},
          .offset = 0,
          .needsNotify = false,   // Recv doesn't need to post notify (sender does)
          .notifyPosted = false
      });

  // DQPLB mode: Initialize receiver with pre-posted recvs on first call
  if (loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
    if (!dqplbReceiverInitialized_) {
      if (initializeDqplbReceiver().hasError()) {
        return folly::makeUnexpected(Error(errno,
            "[Ibverbx]IbvVirtualQp::postRecv, DQPLB receiver initialization failed"));
      }
    }
    // DQPLB: Recvs are pre-posted on all QPs, no further action needed
    return folly::unit;
  }

  // SPRAY mode: Post zero-length recv to notifyQp to catch IMM notification
  // Check backpressure on notifyQp first
  if (notifyQp_->physicalRecvQueStatus_.size() >= maxMsgCntPerQp_) {
    // Queue for later when backpressure clears
    pendingRecvNotifyQue_.push_back(internalId);
    return folly::unit;
  }

  // Post recv to notifyQp
  if (postRecvToNotifyQp(internalId).hasError()) {
    return folly::makeUnexpected(Error(errno,
        "[Ibverbx]IbvVirtualQp::postRecv, failed to post recv to notifyQp"));
  }

  return folly::unit;
}

// Initialize DQPLB receiver by pre-posting zero-length recvs on all physical QPs
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::initializeDqplbReceiver() {
  ibv_recv_wr recvWr{};
  ibv_recv_wr badWr{};

  recvWr.next = nullptr;
  recvWr.sg_list = nullptr;
  recvWr.num_sge = 0;  // Zero-length recv for IMM-only notifications

  // Pre-post recvs on all physical QPs
  for (int i = 0; i < maxMsgCntPerQp_; i++) {
    for (size_t j = 0; j < physicalQps_.size(); j++) {
      recvWr.wr_id = nextPhysicalWrId_++;

      auto maybeRecv = physicalQps_[j].postRecv(&recvWr, &badWr);
      if (maybeRecv.hasError()) {
        return folly::makeUnexpected(maybeRecv.error());
      }

      // Track for completion correlation (virtualWrId = -1 for pre-posted)
      physicalQps_[j].physicalRecvQueStatus_.emplace_back(recvWr.wr_id, -1);
    }
  }

  dqplbReceiverInitialized_ = true;
  return folly::unit;
}

// Replenish a single DQPLB recv on the specified QP after consuming one
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::replenishDqplbRecv(int qpIdx) {
  if (qpIdx < 0 || qpIdx >= static_cast<int>(physicalQps_.size())) {
    return folly::makeUnexpected(Error(EINVAL,
        fmt::format("[Ibverbx]IbvVirtualQp::replenishDqplbRecv, invalid qpIdx={}", qpIdx)));
  }

  ibv_recv_wr recvWr{};
  ibv_recv_wr badWr{};

  recvWr.wr_id = nextPhysicalWrId_++;
  recvWr.next = nullptr;
  recvWr.sg_list = nullptr;
  recvWr.num_sge = 0;  // Zero-length recv for IMM-only notifications

  auto maybeRecv = physicalQps_[qpIdx].postRecv(&recvWr, &badWr);
  if (maybeRecv.hasError()) {
    return folly::makeUnexpected(maybeRecv.error());
  }

  // Track for completion correlation (virtualWrId = -1 for pre-posted)
  physicalQps_[qpIdx].physicalRecvQueStatus_.emplace_back(recvWr.wr_id, -1);

  return folly::unit;
}

// Post zero-length recv to notifyQp to catch IMM notification (SPRAY mode)
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postRecvToNotifyQp(
    uint64_t internalWrId) {

  ibv_recv_wr recvWr{};
  ibv_recv_wr badWr{};

  recvWr.wr_id = nextPhysicalWrId_++;
  recvWr.next = nullptr;
  recvWr.sg_list = nullptr;
  recvWr.num_sge = 0;  // Zero-length - data arrives via RDMA_WRITE

  auto maybeRecv = notifyQp_->postRecv(&recvWr, &badWr);
  if (maybeRecv.hasError()) {
    return folly::makeUnexpected(maybeRecv.error());
  }

  // Track for completion correlation
  notifyQp_->physicalRecvQueStatus_.emplace_back(recvWr.wr_id, internalWrId);

  return folly::unit;
}

// Flush pending recv notifications when notifyQp backpressure clears
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::flushPendingRecvNotifies() {
  while (!pendingRecvNotifyQue_.empty()) {
    if (notifyQp_->physicalRecvQueStatus_.size() >= maxMsgCntPerQp_) {
      break;  // Still under backpressure
    }

    uint64_t frontId = pendingRecvNotifyQue_.front();

    if (postRecvToNotifyQp(frontId).hasError()) {
      break;  // Posting failed - will retry on next CQE
    }

    pendingRecvNotifyQue_.pop_front();
  }

  return folly::unit;
}
```

### 3.2 Fragmentation Logic (dispatchPendingSends)

The `dispatchPendingSends()` function sends fragments from the front of `sendTracker_.pendingQue_` to available physical QPs. It's called from `postSend()` (for RDMA operations) and `processCompletion()`.

**Note:** This function only handles RDMA operations (`IBV_WR_RDMA_WRITE`, `IBV_WR_RDMA_WRITE_WITH_IMM`, `IBV_WR_RDMA_READ`). `IBV_WR_SEND` operations are routed directly to `postSendSingleQp()` and never enter the tracker.

See [Simplified Ibverbx Workflows](./simplified_ibverbx_workflows.md#12-fragmentation-logic-mappendingsendtophysicalqp) for the detailed processing flow diagram.

**Implementation:**

```cpp
// freedQpIdx: Hint for which QP just freed a slot (from CQE).
//             Pass -1 if no hint available (e.g., initial post).
// Note: Only handles RDMA operations (WRITE/READ). SEND uses postSendSingleQp().
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::dispatchPendingSends(
    int freedQpIdx) {

  // Process WRs from send schedule queue (in posting order)
  // Uses tracker accessors for cleaner code
  while (sendTracker_.hasPending()) {
    uint64_t internalId = sendTracker_.frontPending();

    auto* pending = sendTracker_.find(internalId);
    if (pending == nullptr) {
      // Invariant violation: WR in pendingQue_ but not in activeVirtualWrs_
      // This should never happen given the design - fail fast to expose the bug
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx]IbvVirtualQp::dispatchPendingSends, WR {} in pendingQue_ but not found in activeVirtualWrs_", internalId)));
    }

    // Try to send fragments while QP slots are available
    while (pending->offset < pending->length) {
      // Find available QP - use hint first if provided
      int qpIdx;
      if (freedQpIdx >= 0 && hasQpCapacity(freedQpIdx)) {
        qpIdx = freedQpIdx;
        freedQpIdx = -1;  // Consume the hint (use only once)
      } else {
        qpIdx = findAvailableSendQp();
      }

      if (qpIdx == -1) {
        // All QPs full - will continue when slot frees up
        return folly::unit;
      }

      int32_t deviceId = physicalQps_.at(qpIdx).getDeviceId();

      // Calculate fragment size
      uint32_t fragLen = std::min(maxMsgSize_, pending->length - pending->offset);

      // Build and post physical send WR
      ibv_send_wr sendWr{};
      ibv_sge sendSge{};
      buildPhysicalSendWr(*pending, deviceId, fragLen, &sendWr, &sendSge);

      // Post to physical QP
      ibv_send_wr badWr{};
      auto maybePost = physicalQps_.at(qpIdx).postSend(&sendWr, &badWr);
      if (maybePost.hasError()) {
        return folly::makeUnexpected(maybePost.error());
      }

      // Track physical WR for completion correlation
      // Maps: physicalWrId → internalWrId (use loop variable, not stored field)
      physicalQps_.at(qpIdx).physicalSendQueStatus_.emplace_back(
          sendWr.wr_id, internalId);

      // Advance offset
      pending->offset += fragLen;
    }

    // All fragments sent for this WR (offset >= length now)
    sendTracker_.popPending();
    // Note: SPRAY notify is handled in reportOrderedCompletions()
    // when this WR reaches the front of sendTracker_.outstandingQue_
  }

  return folly::unit;
}

// Helper: Build a physical ibv_send_wr for one fragment
// Populates caller-owned sendWr and sendSge. sendWr->sg_list points to sendSge.
inline void IbvVirtualQp::buildPhysicalSendWr(
    const ActiveVirtualWr& pending,
    int32_t deviceId,
    uint32_t fragLen,
    ibv_send_wr* sendWr,
    ibv_sge* sendSge) {

  sendSge->addr = reinterpret_cast<uint64_t>(
      static_cast<char*>(pending.localAddr) + pending.offset);
  sendSge->length = fragLen;
  sendSge->lkey = pending.deviceKeys.at(deviceId).lkey;

  sendWr->wr_id = nextPhysicalWrId_++;
  sendWr->sg_list = sendSge;
  sendWr->num_sge = 1;
  // Always signaled: every fragment needs a CQE for completion tracking.
  // Unsignaled operations are rejected in multi-QP postSend() (see M5),
  // so all WRs reaching this point are guaranteed signaled.
  // Note: IBV_SEND_INLINE is not preserved here. It could be added as an
  // optimization for small fragments in the future.
  sendWr->send_flags = IBV_SEND_SIGNALED;

  // Set remote address (advances with offset)
  sendWr->wr.rdma.remote_addr = pending.remoteAddr + pending.offset;
  sendWr->wr.rdma.rkey = pending.deviceKeys.at(deviceId).rkey;

  // Set opcode (SPRAY uses RDMA_WRITE, DQPLB uses RDMA_WRITE_WITH_IMM)
  if (pending.opcode == IBV_WR_RDMA_WRITE_WITH_IMM &&
      loadBalancingScheme_ == LoadBalancingScheme::SPRAY) {
    sendWr->opcode = IBV_WR_RDMA_WRITE;
  } else {
    sendWr->opcode = pending.opcode;
    if (loadBalancingScheme_ == LoadBalancingScheme::DQPLB) {
      // Derive last-fragment from offset + fragLen vs total length.
      // Note: pending.offset has NOT been advanced yet at this point —
      // it is advanced in dispatchPendingSends() AFTER buildPhysicalSendWr returns.
      bool isLastFragment = (pending.offset + fragLen >= pending.length);
      sendWr->imm_data = dqplbSeqTracker_.getSendImm(isLastFragment);
    }
  }
}

// Helper: Check if a specific QP has capacity for another WR
inline bool IbvVirtualQp::hasQpCapacity(int qpIdx) const {
  if (maxMsgCntPerQp_ == -1) {
    return true;  // No limit
  }
  return physicalQps_.at(qpIdx).physicalSendQueStatus_.size() <
         static_cast<size_t>(maxMsgCntPerQp_);
}

// Helper: Find next available send QP using round-robin
inline int IbvVirtualQp::findAvailableSendQp() {
  // maxMsgCntPerQp_ == -1 means no limit — just round-robin
  if (maxMsgCntPerQp_ == -1) {
    auto availableQpIdx = nextSendPhysicalQpIdx_;
    nextSendPhysicalQpIdx_ = (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
    return availableQpIdx;
  }

  // Try each QP starting from current round-robin position
  for (int i = 0; i < physicalQps_.size(); i++) {
    if (hasQpCapacity(nextSendPhysicalQpIdx_)) {
      auto availableQpIdx = nextSendPhysicalQpIdx_;
      nextSendPhysicalQpIdx_ =
          (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
      return availableQpIdx;
    }
    nextSendPhysicalQpIdx_ = (nextSendPhysicalQpIdx_ + 1) % physicalQps_.size();
  }
  return -1;  // All QPs full
}
```

### 3.3 Send Completion Reporting (reportSendCompletions)

The `reportSendCompletions()` function handles send completion reporting AND SPRAY notify posting. It uses **head-of-line blocking** to ensure ordering guarantees.

**Ordering Guarantee**: When receiver gets WR_B's notify, ALL prior WRs (including non-notify WRs) are guaranteed complete. This is achieved by only posting notify when the WR is at the front of `sendTracker_.outstandingQue_`.

**Unified Completion Tracking**: `remainingMsgCnt` counts ALL expected CQEs (data fragments + notify if applicable). When `remainingMsgCnt == 0`, the WR is complete.

```cpp
// Returns reference to results on success, enabling chaining: return reportSendCompletions(results);
// IMPORTANT: The returned reference is to the caller's results vector. The caller must ensure
// results remains valid for the lifetime of the returned Expected (typically immediate return).
inline folly::Expected<std::vector<IbvVirtualWc>&, Error> IbvVirtualQp::reportSendCompletions(std::vector<IbvVirtualWc>& results) {
  while (sendTracker_.hasOutstanding()) {
    uint64_t frontId = sendTracker_.frontOutstanding();
    auto* frontWr = sendTracker_.find(frontId);
    if (frontWr == nullptr) {
      // Invariant violation: WR in outstandingQue_ but not in activeVirtualWrs_
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx]IbvVirtualQp::reportSendCompletions, WR {} in outstandingQue_ but not found in activeVirtualWrs_", frontId)));
    }

    // Check if WR needs notify and notify hasn't been posted yet
    // For SPRAY sends: remainingMsgCnt == 1 means all data done, only notify CQE remains
    if (frontWr->needsNotify && !frontWr->notifyPosted) {
      if (frontWr->remainingMsgCnt > 1) {
        break;  // Data fragments not complete - head-of-line blocking
      }

      // All data done (remainingMsgCnt == 1) - post notify now
      if (notifyQp_->physicalSendQueStatus_.size() >= maxMsgCntPerQp_) {
        // Backpressure: queue for later
        pendingNotifyQue_.push_back(frontId);
        frontWr->notifyPosted = true;  // Mark as "posted" (queued counts as posted)
        break;  // Wait for notify CQE
      }

      auto maybePost = postNotifyForWr(frontId);
      if (maybePost.hasError()) {
        return folly::makeUnexpected(maybePost.error());
      }
      frontWr->notifyPosted = true;
      break;  // Wait for notify CQE
    }

    // Check if WR is complete (all CQEs received including notify if applicable)
    if (frontWr->remainingMsgCnt > 0) {
      break;  // Not complete - head-of-line blocking
    }

    // Complete - report to user
    results.push_back(buildVirtualWc(*frontWr));
    sendTracker_.remove(frontId);
    sendTracker_.popOutstanding();
  }

  return results;  // Return reference to caller's vector
}

// Recv completion reporting - same logic, simpler because recv doesn't post notifies
// For SPRAY recv: remainingMsgCnt = 1 (notify only), decremented when IMM CQE arrives
// Returns reference to results on success, enabling chaining: return reportRecvCompletions(results);
// IMPORTANT: The returned reference is to the caller's results vector. The caller must ensure
// results remains valid for the lifetime of the returned Expected (typically immediate return).
inline folly::Expected<std::vector<IbvVirtualWc>&, Error> IbvVirtualQp::reportRecvCompletions(
    std::vector<IbvVirtualWc>& results) {

  while (recvTracker_.hasOutstanding()) {
    uint64_t frontId = recvTracker_.frontOutstanding();
    auto* frontWr = recvTracker_.find(frontId);
    if (frontWr == nullptr) {
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx]IbvVirtualQp::reportRecvCompletions, WR {} in outstandingQue_ but not found in activeVirtualWrs_", frontId)));
    }

    // Simple: check if all expected CQEs received
    if (frontWr->remainingMsgCnt > 0) {
      break;  // Not complete - head-of-line blocking
    }

    // Complete - report to user
    results.push_back(buildVirtualWc(*frontWr));
    recvTracker_.remove(frontId);
    recvTracker_.popOutstanding();
  }

  return results;  // Return reference to caller's vector
}

inline folly::Expected<folly::Unit, Error> IbvVirtualQp::postNotifyForWr(
    uint64_t internalWrId) {
  auto* pending = sendTracker_.find(internalWrId);
  if (pending == nullptr) {
    return folly::makeUnexpected(Error(EINVAL,
        fmt::format("[Ibverbx]IbvVirtualQp::postNotifyForWr, WR {} not found", internalWrId)));
  }

  // Build zero-length WRITE_WITH_IMM for notify
  ibv_send_wr sendWr{};
  sendWr.wr_id = nextPhysicalWrId_++;
  sendWr.sg_list = nullptr;
  sendWr.num_sge = 0;  // Zero-length - no data payload
  sendWr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  sendWr.send_flags = IBV_SEND_SIGNALED;
  sendWr.wr.rdma.remote_addr = pending->remoteAddr;  // Base address (not +offset, which points past the buffer)
  int32_t notifyDeviceId = notifyQp_->getDeviceId();
  sendWr.wr.rdma.rkey = pending->deviceKeys.at(notifyDeviceId).rkey;
  sendWr.imm_data = pending->immData;

  // Post to dedicated notify QP
  ibv_send_wr badWr{};
  auto maybePost = notifyQp_->postSend(&sendWr, &badWr);
  if (maybePost.hasError()) {
    return folly::makeUnexpected(maybePost.error());
  }

  // Track for completion correlation (same pattern as data QPs)
  notifyQp_->physicalSendQueStatus_.emplace_back(sendWr.wr_id, internalWrId);

  return folly::unit;
}

// Process pendingNotifyQue_ when notify CQE frees up capacity
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::flushPendingSendNotifies() {
  while (!pendingNotifyQue_.empty()) {
    // Check backpressure before attempting to post
    if (notifyQp_->physicalSendQueStatus_.size() >= kMaxOutstandingNotifies) {
      break;  // Still under backpressure
    }

    uint64_t frontId = pendingNotifyQue_.front();
    auto* frontWr = sendTracker_.find(frontId);
    if (frontWr == nullptr) {
      // Invariant violation: WR in pendingNotifyQue_ but not in activeVirtualWrs_
      // This should never happen given the design - fail fast to expose the bug
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx]IbvVirtualQp::flushPendingNotifies, WR {} in pendingNotifyQue_ but not found in activeVirtualWrs_", frontId)));
    }

    // Post notify
    // Note: notifyPosted was already set true when queued.
    // WR stays in outstandingQue_ - will be reported when remainingMsgCnt == 0.
    if (postNotifyForWr(frontId).hasError()) {
      break;  // Posting failed - will retry on next CQE
    }
    pendingNotifyQue_.pop_front();
  }

  return folly::unit;
}
```


### 3.4 New processCompletion() Implementation

The `processCompletion()` function now supports **in-order completion reporting**. Even though physical completions may arrive out-of-order (across multiple NICs), virtual completions are reported to users strictly in posting order.

**Key design changes**:
1. Returns `std::vector<IbvVirtualWc>` - constructed completions for multi-QP aggregation
2. VirtualCq calls this for registered QP completions
3. Uses internal IDs for tracking, but reports user's original `wrId` in the returned `IbvVirtualWc.wrId`
4. Uses `sendTracker_` for sends, `recvTracker_` for recvs - unified `ActiveVirtualWr` type eliminates code duplication

**Key Characteristics:**

| Aspect | Description |
|--------|-------------|
| **Return type** | `std::vector<IbvVirtualWc>` - may return 0, 1, or many completions |
| **Internal ID tracking** | Uses `internalWrId` for all lookups, reports `userWrId` in result |
| **Tracker selection** | `isSend` flag selects `sendTracker_` vs `recvTracker_` |
| **Unified type** | Both trackers use `ActiveVirtualWr` - same state update code |
| **In-order guarantee** | Only reports front of outstanding queue when complete (separate for send/recv) |
| **Batched reporting** | If WR_B completes before WR_A, WR_B waits. When WR_A completes, both are returned together |
| **Error aggregation** | First error wins - subsequent fragments don't overwrite the error status |
| **SPRAY notify** | Posted via `reportSendCompletions()` when WR is at front and data complete; if blocked, queued in `pendingNotifyQue_` |
| **Notify backpressure** | `notifyQp_->physicalSendQueStatus_.size()` limits outstanding notifies; blocked notifies go to `pendingNotifyQue_` |
| **Pending notify drain** | `flushPendingSendNotifies()` called on notify CQE to process backpressure queue |
| **Slot recycling** | Calls `dispatchPendingSends()` for send; calls `flushPendingRecvNotifies()` for recv (SPRAY mode) |

**Helper Function - Pop and validate physical queue status:**

```cpp
// Helper: Pop front of physical queue status and return internal WR ID
// Returns error if queue is empty or WR ID doesn't match
inline folly::Expected<uint64_t, Error> popPhysicalQueueStatus(
    std::deque<PhysicalWrStatus>& queStatus,
    uint64_t expectedPhysicalWrId,
    const char* queueName) {
  if (queStatus.empty()) {
    return folly::makeUnexpected(Error(EINVAL,
        fmt::format("[Ibverbx]IbvVirtualQp::popPhysicalQueueStatus, no pending WR in {}", queueName)));
  }

  auto& frontStatus = queStatus.front();
  if (frontStatus.physicalWrId != expectedPhysicalWrId) {
    return folly::makeUnexpected(Error(EINVAL,
        fmt::format("[Ibverbx]IbvVirtualQp::popPhysicalQueueStatus, {} WR ID mismatch: expected {}, got {}",
            queueName, frontStatus.physicalWrId, expectedPhysicalWrId)));
  }

  uint64_t internalWrId = frontStatus.virtualWrId;
  queStatus.pop_front();
  return internalWrId;
}
```

**Implementation:**

The `processCompletion()` function uses a **2x2 matrix dispatch** based on QP type (NotifyQp vs DataQp) and direction (Send vs Recv). This structure makes the code easy to follow and each handler self-contained.

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

**Main dispatch function:**

```cpp
// Helper: Check if opcode is a send operation
inline bool IbvVirtualQp::isSendOpcode(ibv_wc_opcode opcode) const {
  return opcode == IBV_WC_SEND ||
         opcode == IBV_WC_RDMA_WRITE ||
         opcode == IBV_WC_RDMA_READ;
}

// Helper: Common WR state update (decrement count + aggregate error)
inline folly::Expected<folly::Unit, Error> IbvVirtualQp::updateWrState(
    WrTracker<ActiveVirtualWr>& tracker,
    uint64_t internalWrId,
    ibv_wc_status status,
    ibv_wc_opcode wcOpcode) {

  auto* wr = tracker.find(internalWrId);
  if (wr == nullptr) {
    return folly::makeUnexpected(Error(EINVAL,
        fmt::format("[Ibverbx] WR {} not found in tracker", internalWrId)));
  }

  wr->remainingMsgCnt--;
  wr->wcOpcode = wcOpcode;  // Capture physical WC opcode for buildVirtualWc()

  // First error wins
  if (wr->aggregatedStatus == IBV_WC_SUCCESS && status != IBV_WC_SUCCESS) {
    wr->aggregatedStatus = status;
  }

  return folly::unit;
}

// Helper: Construct an IbvVirtualWc from a completed ActiveVirtualWr
// Maps WR-level fields into the IbvVirtualWc format expected by callers.
// See IbvVirtualWc struct definition for field semantics.
inline IbvVirtualWc IbvVirtualQp::buildVirtualWc(const ActiveVirtualWr& wr) const {
  IbvVirtualWc wc;

  wc.wrId = wr.userWrId;
  wc.status = wr.aggregatedStatus;
  wc.byteLen = wr.length;
  wc.qpNum = virtualQpNum_;
  wc.immData = wr.immData;

  // Use the physical WC opcode captured by updateWrState().
  // The hardware provides the correct WC opcode directly:
  //   Send-side RDMA_WRITE_WITH_IMM → IBV_WC_RDMA_WRITE
  //   Recv-side RDMA_WRITE_WITH_IMM → IBV_WC_RECV_RDMA_WITH_IMM
  // No WR→WC opcode mapping needed.
  wc.opcode = wr.wcOpcode;

  return wc;
}

inline folly::Expected<std::vector<IbvVirtualWc>, Error> IbvVirtualQp::processCompletion(
    const ibv_wc& physicalWc,
    int32_t deviceId) {

  std::vector<IbvVirtualWc> results;

  // Step 1: Identify QP source
  bool isNotifyQp = notifyQp_.has_value() &&
                    (physicalWc.qp_num == notifyQp_->getQpNum());

  // Step 2: Dispatch based on 2x2 matrix (QP type × direction)
  if (isNotifyQp) {
    bool isSend = isSendOpcode(physicalWc.opcode);
    return isSend ? processNotifyQpSendCompletion(physicalWc, results)
                  : processNotifyQpRecvCompletion(physicalWc, results);
  } else {
    // Data QP: find which physical QP
    auto qpIdxIt = qpNumToIdx_.find(physicalWc.qp_num);
    if (qpIdxIt == qpNumToIdx_.end()) {
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx] unknown physical QP number: {}", physicalWc.qp_num)));
    }
    int qpIdx = qpIdxIt->second;

    bool isSend = isSendOpcode(physicalWc.opcode);
    return isSend ? processDataQpSendCompletion(physicalWc, qpIdx, results)
                  : processDataQpRecvCompletion(physicalWc, qpIdx, results);
  }
}
```

**Processor: NotifyQp Send (SPRAY sender's notify completion)**

```cpp
inline folly::Expected<std::vector<IbvVirtualWc>, Error> IbvVirtualQp::processNotifyQpSendCompletion(
    const ibv_wc& physicalWc,
    std::vector<IbvVirtualWc>& results) {

  // [1] Pop physical queue status
  auto popResult = popPhysicalQueueStatus(
      notifyQp_->physicalSendQueStatus_,
      physicalWc.wr_id,
      "notifyQpSend");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  uint64_t internalWrId = popResult.value();

  // [2] Update WR state (decrement count + aggregate error)
  auto updateResult = updateWrState(sendTracker_, internalWrId, physicalWc.status, physicalWc.opcode);
  if (updateResult.hasError()) {
    return folly::makeUnexpected(updateResult.error());
  }

  // [3] Report completed WRs in order
  auto reportResult = reportSendCompletions(results);
  if (reportResult.hasError()) {
    return folly::makeUnexpected(reportResult.error());
  }

  // [4] Backpressure cleared - try to drain pendingNotifyQue_
  auto flushResult = flushPendingSendNotifies();
  if (flushResult.hasError()) {
    return folly::makeUnexpected(flushResult.error());
  }

  return results;
}
```

**Processor: NotifyQp Recv (SPRAY receiver's notify arrival)**

```cpp
inline folly::Expected<std::vector<IbvVirtualWc>, Error> IbvVirtualQp::processNotifyQpRecvCompletion(
    const ibv_wc& physicalWc,
    std::vector<IbvVirtualWc>& results) {

  // [1] Pop physical queue status
  auto popResult = popPhysicalQueueStatus(
      notifyQp_->physicalRecvQueStatus_,
      physicalWc.wr_id,
      "notifyQpRecv");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  uint64_t internalWrId = popResult.value();

  // [2] Update WR state (decrement count + aggregate error)
  auto updateResult = updateWrState(recvTracker_, internalWrId, physicalWc.status, physicalWc.opcode);
  if (updateResult.hasError()) {
    return folly::makeUnexpected(updateResult.error());
  }

  // [3] Report completed WRs in order
  auto reportResult = reportRecvCompletions(results);
  if (reportResult.hasError()) {
    return folly::makeUnexpected(reportResult.error());
  }

  // [4] Flush pending recv notifications when notifyQp slot freed
  auto flushResult = flushPendingRecvNotifies();
  if (flushResult.hasError()) {
    return folly::makeUnexpected(flushResult.error());
  }

  return results;
}
```

**Processor: DataQp Send (Data fragment completion)**

```cpp
inline folly::Expected<std::vector<IbvVirtualWc>, Error> IbvVirtualQp::processDataQpSendCompletion(
    const ibv_wc& physicalWc,
    int qpIdx,
    std::vector<IbvVirtualWc>& results) {

  auto& physicalQp = physicalQps_.at(qpIdx);

  // [1] Pop physical queue status
  auto popResult = popPhysicalQueueStatus(
      physicalQp.physicalSendQueStatus_,
      physicalWc.wr_id,
      "dataQpSend");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  uint64_t internalWrId = popResult.value();

  // [2] Update WR state (decrement count + aggregate error)
  auto updateResult = updateWrState(sendTracker_, internalWrId, physicalWc.status, physicalWc.opcode);
  if (updateResult.hasError()) {
    return folly::makeUnexpected(updateResult.error());
  }

  // [3] Report completed WRs in order
  auto reportResult = reportSendCompletions(results);
  if (reportResult.hasError()) {
    return folly::makeUnexpected(reportResult.error());
  }

  // [4] Schedule more sends with freed QP hint
  auto dispatchResult = dispatchPendingSends(qpIdx);
  if (dispatchResult.hasError()) {
    return folly::makeUnexpected(dispatchResult.error());
  }

  return results;
}
```

**Processor: DataQp Recv (DQPLB recv with sequence number)**

```cpp
// NOTE: In current design, DataQp recv completions only occur in DQPLB mode.
// SPRAY mode receivers use notifyQp for recv completions (processNotifyQpRecvCompletion).
// DQPLB uses pre-posted zero-length recvs on data QPs to catch IMM with seq#.
inline folly::Expected<std::vector<IbvVirtualWc>, Error> IbvVirtualQp::processDataQpRecvCompletion(
    const ibv_wc& physicalWc,
    int qpIdx,
    std::vector<IbvVirtualWc>& results) {

  auto& physicalQp = physicalQps_.at(qpIdx);

  // [1] Pop physical queue status (internalWrId is -1 for pre-posted recvs)
  auto popResult = popPhysicalQueueStatus(
      physicalQp.physicalRecvQueStatus_,
      physicalWc.wr_id,
      "dataQpRecv");
  if (popResult.hasError()) {
    return folly::makeUnexpected(popResult.error());
  }
  // Note: internalWrId is -1 for DQPLB pre-posted recvs - we use DqplbSeqTracker instead

  // [2] Process IMM data to get notifyCount (in-order user-level completions)
  // Key insight: notifyCount maps 1:1 to front entries in recvTracker_.outstandingQue_
  int notifyCount = dqplbSeqTracker_.processReceivedImm(physicalWc.imm_data);

  // [3] Decrement remainingMsgCnt for the front N WRs and report completions
  // Each notifyCount unit corresponds to one user-level WR completing.
  // We must call reportRecvCompletions() inside the loop so that completed
  // WRs are popped from outstandingQue_ before the next iteration calls
  // frontOutstanding(). Without this, the same front WR would be decremented
  // notifyCount times instead of decrementing the front N WRs once each.
  for (int i = 0; i < notifyCount; i++) {
    if (!recvTracker_.hasOutstanding()) {
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx] DQPLB notifyCount={} exceeds outstanding recvs", notifyCount)));
    }

    uint64_t frontId = recvTracker_.frontOutstanding();
    auto* frontWr = recvTracker_.find(frontId);
    if (frontWr == nullptr) {
      return folly::makeUnexpected(Error(EINVAL,
          fmt::format("[Ibverbx] DQPLB WR {} not found in recvTracker_", frontId)));
    }

    frontWr->remainingMsgCnt--;
    frontWr->wcOpcode = physicalWc.opcode;

    // Aggregate physical WC error status into the virtual WR
    if (frontWr->aggregatedStatus == IBV_WC_SUCCESS &&
        physicalWc.status != IBV_WC_SUCCESS) {
      frontWr->aggregatedStatus = physicalWc.status;
    }

    // Report completed WRs in order — pops WRs with remainingMsgCnt <= 0
    // from the front of outstandingQue_ so next iteration advances
    auto reportResult = reportRecvCompletions(results);
    if (reportResult.hasError()) {
      return folly::makeUnexpected(reportResult.error());
    }
  }

  // [5] Replenish pre-posted recv on the QP that completed
  // This maintains the DQPLB recv WR pool for continuous operation
  auto replenishResult = replenishDqplbRecv(qpIdx);
  if (replenishResult.hasError()) {
    return folly::makeUnexpected(replenishResult.error());
  }

  return results;
}
```

**In-Order Completion Guarantee:**

```
Example: User posts WR_A, WR_B, WR_C in order
Physical completions arrive: WR_B fragment, WR_A fragment, WR_C fragment, WR_A last fragment

State after each completion:

1. WR_B fragment completes:
   sendTracker_.outstandingQue_ = [A, B, C]
   WR_A.allFragmentsComplete = false (front, blocks reporting)
   WR_B.allFragmentsComplete = false
   → No results reported

2. WR_A fragment completes:
   WR_A.remainingMsgCnt--
   WR_A.allFragmentsComplete = false (still has fragments)
   → No results reported

3. WR_C fragment completes:
   WR_C.remainingMsgCnt = 0
   WR_C.allFragmentsComplete = true
   But: WR_A is still at front, not complete
   → No results reported (in-order guarantee!)

4. WR_A last fragment completes:
   WR_A.remainingMsgCnt = 0
   WR_A.allFragmentsComplete = true
   Now drain sendTracker_.outstandingQue_:
     - Pop WR_A → report to user ✓
     - Check WR_B: not complete → stop
   → Returns [WR_A]

5. (Later) WR_B last fragment completes:
   WR_B.allFragmentsComplete = true
   Drain sendTracker_.outstandingQue_:
     - Pop WR_B → report to user ✓
     - Pop WR_C → report to user ✓ (was already complete)
   → Returns [WR_B, WR_C]

Final order reported to user: WR_A, WR_B, WR_C ✓
```

### 3.5 Batch Processing Implementation

```cpp
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualQp::processCompletions(
    const ibv_wc* physicalWcs,
    int count,
    int32_t deviceId) {

  std::vector<IbvVirtualWc> allResults;
  allResults.reserve(count);  // May return more due to in-order batching

  for (int i = 0; i < count; i++) {
    auto result = processCompletion(physicalWcs[i], deviceId);
    if (result.hasError()) {
      return folly::makeUnexpected(result.error());
    }
    // Append all completed WRs (may be multiple due to in-order draining)
    for (auto& r : *result) {
      allResults.push_back(std::move(r));
    }
  }

  return allResults;
}
```

### 3.6 IbvVirtualCq pollCq() Implementation

The `pollCq()` function is the **core routing logic** of VirtualCq. It polls physical CQs and handles completions based on whether the VirtualQp uses single or multiple physical QPs:

- **Single-QP VirtualQp** (isMultiQp=false): No fragmentation/aggregation needed → return CQE directly to user (fast path)
- **Multi-QP VirtualQp** (isMultiQp=true): Load balancing with fragmentation → route to VirtualQp for aggregation

**Key Characteristics:**

| Aspect | Description |
|--------|-------------|
| **Registration lookup** | O(1) via F14FastMap `registeredQps_` |
| **Single-QP (isMultiQp=false)** | Direct pass-through, ~5 ns overhead |
| **Multi-QP (isMultiQp=true)** | Routes to VirtualQp::processCompletion() for aggregation |
| **Return type** | `std::vector<IbvVirtualWc>` - virtual CQEs |

**Implementation:**

```cpp
inline folly::Expected<std::vector<IbvVirtualWc>, Error> IbvVirtualCq::pollCq(
    int numEntries) {

  // ============================================================
  // PHASE 1: Drain ALL physical CQEs from ALL physical CQs
  // ============================================================
  // We must drain all CQEs before returning results to prevent CQ overflow.
  // Production polls one CQE at a time per CQ; we batch-poll for efficiency.
  // CQEs are processed immediately (updating VirtualQp state via
  // processCompletion()), which may trigger notify sends or replenishment.
  //
  // Note: This phase may produce more virtual completions than numEntries.
  // We collect all of them and return only numEntries worth in Phase 2.
  // The excess completions are already recorded in VirtualQp trackers and
  // will be returned on subsequent pollCq() calls.

  for (size_t cqIdx = 0; cqIdx < physicalCqs_.size(); cqIdx++) {
    auto& cq = physicalCqs_.at(cqIdx);
    int32_t deviceId = cq.getDeviceId();

    // Drain this CQ until empty
    while (true) {
      auto maybeWcs = cq.pollCq(32);
      if (maybeWcs.hasError()) {
        return folly::makeUnexpected(maybeWcs.error());
      }
      auto& physicalWcs = *maybeWcs;
      if (physicalWcs.empty()) {
        break;  // CQ drained
      }

      // Process each physical completion
      for (size_t i = 0; i < physicalWcs.size(); i++) {
        const ibv_wc& physicalWc = physicalWcs[i];

        // Lookup registration info for this QP
        const RegisteredQpInfo* info = findRegisteredQpInfo(physicalWc.qp_num, deviceId);

        if (info == nullptr) {
          return folly::makeUnexpected(Error(EINVAL,
              fmt::format("[Ibverbx]IbvVirtualCq::pollCq, unregistered QP number: {}", physicalWc.qp_num)));
        }

        if (!info->isMultiQp) {
          // =============================================
          // SINGLE-QP FAST PATH: Construct IbvVirtualWc from physical ibv_wc
          // No fragmentation/aggregation needed
          // =============================================
          IbvVirtualWc vwc;
          vwc.wrId = physicalWc.wr_id;
          vwc.status = physicalWc.status;
          vwc.opcode = physicalWc.opcode;
          vwc.qpNum = info->virtualQpNum;
          vwc.immData = physicalWc.imm_data;
          vwc.byteLen = physicalWc.byte_len;
          completedVirtualWcs_.push_back(vwc);
        } else if (physicalWc.opcode == IBV_WC_SEND ||
                   physicalWc.opcode == IBV_WC_RECV) {
          // =============================================
          // MULTI-QP PASSTHROUGH: SEND/RECV bypass processCompletion()
          // These operations are posted via postSendSingleQp() /
          // postRecvSingleQp() to QP[0] without fragmentation or tracking.
          // Their CQEs are returned directly — no aggregation needed.
          // =============================================
          IbvVirtualWc vwc;
          vwc.wrId = physicalWc.wr_id;
          vwc.status = physicalWc.status;
          vwc.opcode = physicalWc.opcode;
          vwc.qpNum = info->virtualQpNum;
          vwc.immData = physicalWc.imm_data;
          vwc.byteLen = physicalWc.byte_len;
          completedVirtualWcs_.push_back(vwc);
        } else {
          // =============================================
          // MULTI-QP AGGREGATION PATH: RDMA_WRITE / RDMA_READ /
          // RECV_RDMA_WITH_IMM — route to VirtualQp for fragment aggregation
          // =============================================
          auto maybeVirtualWcs = info->vqp->processCompletion(physicalWc, deviceId);

          if (maybeVirtualWcs.hasError()) {
            return folly::makeUnexpected(maybeVirtualWcs.error());
          }

          // VirtualQp may return:
          // - Empty vector: intermediate fragment, no virtual CQE yet
          // - Non-empty vector: one or more virtual CQEs ready
          for (auto& virtualWc : *maybeVirtualWcs) {
            completedVirtualWcs_.push_back(std::move(virtualWc));
          }
        }
      }
    }
  }

  // ============================================================
  // PHASE 2: Return up to numEntries completed virtual CQEs
  // ============================================================
  // completedVirtualWcs_ accumulates across pollCq() calls.
  // We return the oldest entries first (FIFO order).

  std::vector<IbvVirtualWc> results;
  int count = std::min(static_cast<int>(completedVirtualWcs_.size()), numEntries);
  results.reserve(count);
  for (int i = 0; i < count; i++) {
    results.push_back(std::move(completedVirtualWcs_.front()));
    completedVirtualWcs_.pop_front();
  }

  return results;
}

inline const IbvVirtualCq::RegisteredQpInfo* IbvVirtualCq::findRegisteredQpInfo(
    uint32_t qpNum, int32_t deviceId) const {

  QpId key{.deviceId = deviceId, .qpNum = qpNum};
  auto it = registeredQps_.find(key);
  if (it == registeredQps_.end()) {
    return nullptr;
  }
  return &it->second;
}
```

### 3.7 VirtualQp Registration Implementation

VirtualQp registers its physical QPs with VirtualCq at construction and unregisters at destruction. The `isMultiQp` flag is determined by the number of physical QPs.

```cpp
IbvVirtualQp::IbvVirtualQp(
    std::vector<IbvQp>&& qps,
    IbvVirtualCq* virtualCq,
    int maxMsgCntPerQp,
    uint32_t maxMsgSize,
    LoadBalancingScheme loadBalancingScheme,
    std::optional<IbvQp>&& notifyQp)
    : physicalQps_(std::move(qps)),
      virtualCq_(virtualCq),
      maxMsgCntPerQp_(maxMsgCntPerQp),
      maxMsgSize_(maxMsgSize),
      loadBalancingScheme_(loadBalancingScheme),
      notifyQp_(std::move(notifyQp)) {

  virtualQpNum_ = nextVirtualQpNum_.fetch_add(1);

  // Determine if this is a multi-QP VirtualQp (load balancing)
  isMultiQp_ = (physicalQps_.size() > 1);

  // Build qpNum → index mapping
  for (int i = 0; i < physicalQps_.size(); i++) {
    qpNumToIdx_[physicalQps_.at(i).qp()->qp_num] = i;
  }

  // Calculate number of unique devices across physical QPs (and notifyQp if present)
  std::unordered_set<uint32_t> uniqueDevices;
  for (const auto& qp : physicalQps_) {
    uniqueDevices.insert(qp.getDeviceId());
  }
  if (notifyQp_.has_value()) {
    uniqueDevices.insert(notifyQp_->getDeviceId());
  }
  deviceCnt_ = uniqueDevices.size();

  // Register with VirtualCq
  registerWithVirtualCq();
}

IbvVirtualQp::~IbvVirtualQp() {
  unregisterFromVirtualCq();
}

void IbvVirtualQp::registerWithVirtualCq() {
  if (!virtualCq_) {
    return;
  }

  // Register all physical QPs
  for (const auto& qp : physicalQps_) {
    virtualCq_->registerPhysicalQp(
        qp.qp()->qp_num, qp.getDeviceId(), this, isMultiQp_, virtualQpNum_);
  }

  // Register notifyQp if present (only for multi-QP mode with SPRAY)
  if (notifyQp_.has_value()) {
    virtualCq_->registerPhysicalQp(
        notifyQp_->qp()->qp_num, notifyQp_->getDeviceId(), this, isMultiQp_, virtualQpNum_);
  }
}

void IbvVirtualQp::unregisterFromVirtualCq() {
  if (!virtualCq_) {
    return;
  }

  for (const auto& qp : physicalQps_) {
    virtualCq_->unregisterPhysicalQp(qp.qp()->qp_num, qp.getDeviceId());
  }

  if (notifyQp_.has_value()) {
    virtualCq_->unregisterPhysicalQp(
        notifyQp_->qp()->qp_num, notifyQp_->getDeviceId());
  }
}

// VirtualCq registration methods
void IbvVirtualCq::registerPhysicalQp(
    uint32_t physicalQpNum, int32_t deviceId, IbvVirtualQp* vqp, bool isMultiQp,
    uint32_t virtualQpNum) {

  QpId key{.deviceId = deviceId, .qpNum = physicalQpNum};
  registeredQps_[key] = RegisteredQpInfo{
      .vqp = vqp, .isMultiQp = isMultiQp, .virtualQpNum = virtualQpNum};
}

void IbvVirtualCq::unregisterPhysicalQp(
    uint32_t physicalQpNum, int32_t deviceId) {

  QpId key{.deviceId = deviceId, .qpNum = physicalQpNum};
  registeredQps_.erase(key);
}

// VirtualCq constructors

IbvVirtualCq::IbvVirtualCq(IbvCq&& cq, int maxCqe) : maxCqe_(maxCqe) {
  physicalCqs_.push_back(std::move(cq));
  virtualCqNum_ = nextVirtualCqNum_.fetch_add(1);
}

IbvVirtualCq::IbvVirtualCq(std::vector<IbvCq>&& cqs, int maxCqe)
    : physicalCqs_(std::move(cqs)), maxCqe_(maxCqe) {
  virtualCqNum_ = nextVirtualCqNum_.fetch_add(1);
}
```

---

## 4. Files to Modify

### 4.1 Core Changes

| File | Changes | Impact |
|------|---------|--------|
| `IbvVirtualQp.h` | Add `processCompletion()` returning `IbvVirtualWc`, add four-structure WR tracking, add VirtualCq registration, remove Coordinator calls, keep `IbvVirtualQpBusinessCard` struct unchanged | Major |
| `IbvVirtualQp.cc` | Update constructor with VirtualCq registration, move semantics, destructor unregistration, keep `modifyVirtualQp()`/`getVirtualQpBusinessCard()` with `std::optional` guards for `notifyQp_`, keep `IbvVirtualQpBusinessCard` serialization unchanged | Medium |
| `IbvVirtualCq.h` | Add registration table (`registeredQps_`), update `pollCq()` with two-phase drain (drain all CQEs then return up to numEntries from `completedVirtualWcs_`), add opcode-based routing for multi-QP SEND/RECV passthrough, add `registerPhysicalQp()`/`unregisterPhysicalQp()`, add single-CQ constructor `IbvVirtualCq(IbvCq&&, int)`, add `getVirtualCqNum()` accessor, add `virtualCqNum_` and `completedVirtualWcs_` members, add move assignment operator | Major |
| `IbvVirtualCq.cc` | Implement registration methods, single-CQ constructor, move assignment, pollCq routing | Medium |
| `IbvCommon.h` | Remove Coordinator-only types: `RequestType` enum, `VirtualQpRequest`, `VirtualQpResponse`, `VirtualCqRequest` structs. Keep: `Error`, `LoadBalancingScheme`, `MemoryRegionKeys`, constants | Minor |
| `IbvPd.h/cc` | Update `createVirtualQp()` signature: change `IbvVirtualCq* sendCq, IbvVirtualCq* recvCq` to single `IbvVirtualCq* virtualCq` (all current callers pass the same CQ for both). Implementation uses `virtualCq->getPhysicalCqsRef().at(0).cq()` for both `initAttr->send_cq` and `initAttr->recv_cq` | Minor |
| `DqplbSeqTracker.h` | Change `getSendImm(int remainingMsgCnt)` to `getSendImm(bool isLastFragment)` | Minor |
| `IbvQp.h` | Replace nested `PhysicalSendWrStatus`/`PhysicalRecvWrStatus` with unified `PhysicalWrStatus` (defined in design Section 2.1). Rename members `physicalSendWrStatus_`/`physicalRecvWrStatus_` to `physicalSendQueStatus_`/`physicalRecvQueStatus_` | Minor |

### 4.2 Removable Files

| File | Status | Notes |
|------|--------|-------|
| `Coordinator.h` | **Remove** | Registration moves to VirtualCq |
| `Coordinator.cc` | **Remove** | No routing needed |

### 4.3 Key Design Differences vs Current

| Aspect | Current Design | Simplified Design |
|--------|----------------|-------------------|
| **Routing** | Coordinator singleton with 5 maps | VirtualCq's `registeredQps_` (1 map) |
| **VirtualCq Role** | State tracking + aggregation | Routing + pass-through |
| **VirtualQp Role** | Send WRs only | Send WRs + completion tracking |
| **Registration** | Coordinator at construction | VirtualCq at construction |
| **Completion Return** | `VirtualWc` struct | `IbvVirtualWc` (custom minimal type) |
| **Mixed QPs** | Not supported | Supported (pass-through) |
| `Coordinator.cc` | Can remove | |

### 4.4 New Dependencies

```cpp
// IbvVirtualQp.h
#include <folly/container/F14Map.h>  // For faster hash maps
```

---

## 5. Migration Guide

### 5.1 For Existing Users

**Step 1: Update VirtualQp Creation**
```cpp
// OLD
IbvVirtualCq sendCq = pd.createVirtualCq(...);
IbvVirtualCq recvCq = pd.createVirtualCq(...);
IbvVirtualQp vqp = pd.createVirtualQp(qps, notifyQp, &sendCq, &recvCq);

// NEW
IbvVirtualQp vqp = pd.createVirtualQp(qps, notifyQp);
// Create physical CQs separately if needed for polling
```

**Step 2: Update Completion Handling**
```cpp
// OLD
while (running) {
  auto virtualWcs = sendCq.pollCq(32);
  for (auto& vwc : virtualWcs) {
    handleCompletion(vwc.wc.wr_id, vwc.wc.status);
  }
}

// NEW
while (running) {
  ibv_wc wcs[32];
  int nwcs = ibv_poll_cq(physicalSendCq, 32, wcs);

  for (int i = 0; i < nwcs; i++) {
    auto results = vqp.processCompletion(wcs[i]);
    // processCompletion may return multiple IbvVirtualWc due to in-order reporting
    for (auto& result : *results) {
      handleCompletion(result.wrId, result.status);
    }
  }
}
```

### 5.2 Lightweight VirtualCq Wrapper (Optional Convenience Layer)

For users who don't want to manage physical CQs directly, we provide an **optional lightweight wrapper** that simplifies completion polling. This wrapper is fundamentally different from the original VirtualCq:

| Aspect | Original VirtualCq | Lightweight Wrapper |
|--------|-------------------|---------------------|
| **State** | VirtualWc queues, hash maps | Only physical CQs |
| **Logic** | Completion tracking, aggregation | Delegates to VirtualQp |
| **Overhead** | 42+ ns per completion | 5-10 ns per completion |
| **Purpose** | Core abstraction | Convenience layer |

#### 5.2.1 Wrapper Implementation

```cpp
class IbvVirtualCqWrapper {
 public:
  // Constructor: takes ownership of physical CQs
  IbvVirtualCqWrapper(std::vector<IbvCq> physicalCqs, IbvVirtualQp* vqp)
      : physicalCqs_(std::move(physicalCqs))
      , virtualQp_(vqp) {}

  // Basic polling - returns completed virtual WRs
  std::vector<CompletionResult> pollCq(int maxCompletions = 32) {
    std::vector<CompletionResult> results;
    results.reserve(maxCompletions);

    for (auto& cq : physicalCqs_) {
      if (results.size() >= static_cast<size_t>(maxCompletions)) break;

      ibv_wc wcs[32];
      int remaining = maxCompletions - results.size();
      int nwcs = cq.pollCq(std::min(32, remaining), wcs);

      if (nwcs < 0) {
        continue;  // Error handling - skip this CQ
      }

      for (int i = 0; i < nwcs; i++) {
        auto maybeResults = virtualQp_->processCompletion(wcs[i], cq.getDeviceId());
        if (maybeResults.hasValue()) {
          // processCompletion returns a vector (in-order batch)
          for (auto& r : *maybeResults) {
            results.push_back(std::move(r));
          }
        }
      }
    }
    return results;
  }

  // Zero-allocation callback variant for performance-sensitive code
  template <typename Callback>
  int pollCq(int maxCompletions, Callback&& onComplete) {
    int totalProcessed = 0;

    for (auto& cq : physicalCqs_) {
      if (totalProcessed >= maxCompletions) break;

      ibv_wc wcs[32];
      int remaining = maxCompletions - totalProcessed;
      int nwcs = cq.pollCq(std::min(32, remaining), wcs);

      if (nwcs < 0) continue;

      for (int i = 0; i < nwcs; i++) {
        auto maybeResults = virtualQp_->processCompletion(wcs[i], cq.getDeviceId());
        if (maybeResults.hasValue()) {
          for (auto& r : *maybeResults) {
            onComplete(r);
            totalProcessed++;
          }
        }
      }
    }
    return totalProcessed;
  }

  // Poll specific CQ (for priority handling)
  std::vector<CompletionResult> pollCq(size_t cqIndex, int maxCompletions) {
    if (cqIndex >= physicalCqs_.size()) {
      return {};
    }

    std::vector<CompletionResult> results;
    results.reserve(maxCompletions);

    auto& cq = physicalCqs_[cqIndex];
    ibv_wc wcs[32];
    int nwcs = cq.pollCq(std::min(32, maxCompletions), wcs);

    for (int i = 0; i < nwcs; i++) {
      auto maybeResults = virtualQp_->processCompletion(wcs[i], cq.getDeviceId());
      if (maybeResults.hasValue()) {
        for (auto& r : *maybeResults) {
          results.push_back(std::move(r));
        }
      }
    }
    return results;
  }

  // Non-blocking poll with immediate return
  // Returns first completed WR, or nullopt if none
  // Note: May not return all ready completions - use pollCq() for batch draining
  std::optional<CompletionResult> tryPollOne() {
    for (auto& cq : physicalCqs_) {
      ibv_wc wc;
      int nwcs = cq.pollCq(1, &wc);
      if (nwcs > 0) {
        auto maybeResults = virtualQp_->processCompletion(wc, cq.getDeviceId());
        if (maybeResults.hasValue() && !maybeResults->empty()) {
          // Return first result; remaining in-order completions
          // will be returned on subsequent polls
          return maybeResults->front();
        }
      }
    }
    return std::nullopt;
  }

  // Advanced access for power users
  std::vector<IbvCq>& getPhysicalCqs() { return physicalCqs_; }
  const std::vector<IbvCq>& getPhysicalCqs() const { return physicalCqs_; }
  IbvVirtualQp* getVirtualQp() { return virtualQp_; }
  size_t numCqs() const { return physicalCqs_.size(); }

 private:
  std::vector<IbvCq> physicalCqs_;
  IbvVirtualQp* virtualQp_;  // Non-owning; VirtualQp must outlive wrapper
};
```

#### 5.2.2 Factory Integration

```cpp
class IbvPd {
 public:
  // Original simplified factory (for advanced users)
  IbvVirtualQp createVirtualQp(std::vector<IbvQp>&& qps, IbvQp&& notifyQp);

  // Convenience factory that creates both QP and wrapper (for typical users)
  std::pair<IbvVirtualQp, IbvVirtualCqWrapper> createVirtualQpWithCq(
      std::vector<IbvQp>&& qps,
      IbvQp&& notifyQp,
      int sendCqSize = 128,
      int recvCqSize = 128) {

    // Create physical CQs
    std::vector<IbvCq> sendCqs;
    for (int i = 0; i < numDevices_; i++) {
      sendCqs.push_back(createCq(i, sendCqSize));
    }

    // Create VirtualQp
    IbvVirtualQp vqp = createVirtualQp(std::move(qps), std::move(notifyQp));

    // Create wrapper with non-owning reference to VirtualQp
    IbvVirtualCqWrapper wrapper(std::move(sendCqs), &vqp);

    return {std::move(vqp), std::move(wrapper)};
  }
};
```

#### 5.2.3 Usage Examples

**Simple usage (recommended for 80% of users):**
```cpp
// Factory creates everything
auto [vqp, sendCq] = pd.createVirtualQpWithCq(qps, notifyQp, 128, 128);

// Simple polling loop - no device ID management needed
while (running) {
  for (auto& result : sendCq.pollCq(32)) {
    if (result.isComplete) {
      onOperationComplete(result.virtualWrId, result.status);
    }
  }
}
```

**Zero-allocation callback usage (for latency-sensitive code):**
```cpp
auto [vqp, sendCq] = pd.createVirtualQpWithCq(qps, notifyQp);

// Callback-based polling - no vector allocation
while (running) {
  sendCq.pollCq(32, [](const CompletionResult& result) {
    if (result.isComplete) {
      handleCompletion(result.virtualWrId, result.status);
    }
  });
}
```

**Priority polling (intermediate users):**
```cpp
auto [vqp, sendCq] = pd.createVirtualQpWithCq(qps, notifyQp);

// Poll high-priority CQ (index 0) first
while (running) {
  // High-priority completions
  for (auto& result : sendCq.pollCq(/*cqIndex=*/0, 32)) {
    handleHighPriority(result);
  }
  // Normal completions from remaining CQs
  for (size_t i = 1; i < sendCq.numCqs(); i++) {
    for (auto& result : sendCq.pollCq(i, 32)) {
      handleNormal(result);
    }
  }
}
```

**Advanced usage (bypass wrapper entirely):**
```cpp
// Create VirtualQp without wrapper
IbvVirtualQp vqp = pd.createVirtualQp(qps, notifyQp);

// User manages CQs with full control
std::vector<IbvCq> myCqs = createCqsWithCustomSettings();

// Direct polling with custom strategies
for (auto& cq : myCqs) {
  ibv_wc wcs[32];
  int nwcs = ibv_poll_cq(cq.getCq(), 32, wcs);
  for (int i = 0; i < nwcs; i++) {
    auto results = vqp.processCompletion(wcs[i], cq.getDeviceId());
    // Handle all returned IbvVirtualWc completions (may be multiple due to in-order draining)
    for (auto& result : *results) {
      handleCompletion(result.wrId, result.status);
    }
  }
}
```

#### 5.2.4 Wrapper Performance Characteristics

| API Variant | Overhead per poll | Best for |
|-------------|-------------------|----------|
| `pollCq(32)` → vector | 5-8 ns + allocation | Simple use, moderate throughput |
| `pollCq(32, callback)` | 5-8 ns | Low-latency, high-throughput |
| `pollCq(cqIndex, 32)` | 3-5 ns | Priority-based polling |
| `tryPollOne()` | 2-3 ns | Event-driven polling |
| Direct `processCompletion()` | 0 ns (baseline) | Maximum performance |

**When to use the wrapper:**
- You have a single VirtualQp per connection
- You poll completions in a straightforward event loop
- You don't need fine-grained control over which physical CQ to poll first
- You want to avoid device ID correlation boilerplate

**When to bypass the wrapper:**
- You need custom polling strategies (priority-based, adaptive)
- You're integrating with an existing event loop that already polls CQs
- Every nanosecond of latency matters
- You need per-device error handling

---

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Multi-NIC complexity | User must manage multiple CQs | Use `IbvVirtualCqWrapper` or helper `getSendCqs()`/`getRecvCqs()` |
| Wrong CQE routing | Undefined behavior if user passes wrong CQE | Validate qp_num in processCompletion() |
| Thread safety | Race conditions if concurrent access | Document: user responsible for sync |
| Memory ordering | Out-of-order completions possible | Use F14FastMap, not deque, for O(1) lookup |
