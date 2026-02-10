// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <deque>
#include <unordered_map>
#include <vector>

#include <folly/container/F14Map.h>
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// Memory region keys for multi-NIC support
struct MemoryRegionKeys {
  uint32_t lkey{0};
  uint32_t rkey{0};
};

// ============================================================
// IbvVirtualSendWr/IbvVirtualRecvWr (input to VirtualQp::postSend/Recv)
// ============================================================

// Custom send work request (replaces ibv_send_wr in IbvVirtualQp::postSend
// call)
struct IbvVirtualSendWr {
  uint64_t wrId{0}; // User's work request ID

  // Local buffer
  void* localAddr{nullptr}; // Local buffer address
  uint32_t length{0}; // Buffer length

  // Remote buffer (for RDMA ops)
  uint64_t remoteAddr{0}; // Remote address

  // Operation
  ibv_wr_opcode opcode{IBV_WR_RDMA_WRITE}; // Operation type
  int sendFlags{0}; // IBV_SEND_SIGNALED, etc.
  uint32_t immData{0}; // Immediate data (for WRITE_WITH_IMM)

  // Per-device memory keys: maps deviceId -> {lkey, rkey}.
  // Mandatory field: 1 entry for single-NIC, N entries for multi-NIC.
  folly::F14FastMap<int32_t, MemoryRegionKeys> deviceKeys;
};

// Custom recv work request (replaces ibv_recv_wr in IbvVirtualQp::postRecv
// call)
struct IbvVirtualRecvWr {
  uint64_t wrId{0}; // User's work request ID

  // Local buffer (can be nullptr/0 for zero-length recv)
  void* localAddr{nullptr}; // Local buffer address
  uint32_t length{0}; // Buffer length (0 = notification recv)

  // Per-device memory keys: maps deviceId -> {lkey, rkey}.
  // Mandatory field: 1 entry for single-NIC, N entries for multi-NIC.
  folly::F14FastMap<int32_t, MemoryRegionKeys> deviceKeys;
};

// ============================================================
// Virtual Work Completion (output of VirtualCq::pollCq)
// ============================================================

// Custom completion entry returned by IbvVirtualCq::pollCq().
// Replaces raw ibv_wc at the VirtualQp/VirtualCq boundary.
struct IbvVirtualWc {
  uint64_t wrId{0}; // User's original work request ID
  ibv_wc_status status{IBV_WC_SUCCESS}; // Completion status
  ibv_wc_opcode opcode{IBV_WC_SEND}; // Operation type
  uint32_t qpNum{0}; // Virtual QP number
  uint32_t immData{0}; // Immediate data (for WRITE_WITH_IMM)
  uint32_t byteLen{0}; // Total byte length of the completed WR
};

// ============================================================
// Internal: Active WR tracking (used by VirtualQp)
// ============================================================

// Full state for fragmentation, notify, and completion aggregation.
// Used for both send and recv operations.
struct ActiveVirtualWr {
  // Identity
  uint64_t userWrId{0}; // User's original wrId (for completion reporting)

  // Completion tracking
  int remainingMsgCnt{0}; // Decremented on each CQE; 0 = complete
  ibv_wc_status aggregatedStatus{IBV_WC_SUCCESS}; // First error wins
  ibv_wc_opcode wcOpcode{IBV_WC_SEND}; // Physical WC opcode (captured from
                                       // physicalWc.opcode by updateWrState)

  // Cached from IbvVirtualSendWr/IbvVirtualRecvWr (needed for fragmentation)
  void* localAddr{nullptr};
  uint32_t length{0};
  uint64_t remoteAddr{0}; // Send only (0 for recv)
  ibv_wr_opcode opcode{IBV_WR_RDMA_WRITE}; // The operation type
  uint32_t immData{0}; // Send only (0 for recv)
  folly::F14FastMap<int32_t, MemoryRegionKeys> deviceKeys;

  // Fragmentation progress
  uint32_t offset{0}; // Current offset; allFragmentsSent = (offset >= length)

  // SPRAY notify tracking (send only, false for recv)
  bool needsNotify{false}; // True if this WR requires a notify (SPRAY mode)
  bool notifyPosted{false}; // True after notify has been posted to notifyQp

  // Helper: check if WR is fully complete
  bool isComplete() const {
    return remainingMsgCnt == 0;
  }

  // Helper: check if this is a send operation
  bool isSendOp() const {
    return opcode == IBV_WR_SEND || opcode == IBV_WR_RDMA_WRITE ||
        opcode == IBV_WR_RDMA_WRITE_WITH_IMM || opcode == IBV_WR_RDMA_READ;
  }
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
  // All active (not yet completed) WRs
  // Key = internalWrId (always unique), Value = active WR state
  folly::F14FastMap<uint64_t, ActiveVirtualWrT> activeVirtualWrs_;

  // Pending queue: WRs not yet fully posted to physical QPs
  std::deque<uint64_t> pendingQue_;

  // Outstanding queue: WRs posted, awaiting CQE (subset of active)
  std::deque<uint64_t> outstandingQue_;

  // ID generator
  uint64_t nextInternalVirtualWrId_{0};

  // Add new WR to tracker, returns internal ID
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
  void remove(uint64_t internalId) {
    activeVirtualWrs_.erase(internalId);
  }

  // Queue accessors
  bool hasPending() const {
    return !pendingQue_.empty();
  }
  uint64_t frontPending() const {
    return pendingQue_.front();
  }
  void popPending() {
    pendingQue_.pop_front();
  }

  bool hasOutstanding() const {
    return !outstandingQue_.empty();
  }
  uint64_t frontOutstanding() const {
    return outstandingQue_.front();
  }
  void popOutstanding() {
    outstandingQue_.pop_front();
  }

  // Metrics
  size_t activeCount() const {
    return activeVirtualWrs_.size();
  }
  size_t pendingCount() const {
    return pendingQue_.size();
  }
  size_t outstandingCount() const {
    return outstandingQue_.size();
  }
};

// ============================================================
// Legacy types (kept for backward compatibility during migration)
// ============================================================

struct VirtualSendWr {
  VirtualSendWr(
      const ibv_send_wr& wr,
      int expectedMsgCnt,
      int remainingMsgCnt,
      bool sendExtraNotifyImm,
      const std::unordered_map<int32_t, MemoryRegionKeys>& deviceIdToKeys = {})
      : expectedMsgCnt(expectedMsgCnt),
        remainingMsgCnt(remainingMsgCnt),
        sendExtraNotifyImm(sendExtraNotifyImm),
        deviceIdToKeys(deviceIdToKeys) {
    // Make an explicit copy of the ibv_send_wr structure
    this->wr = wr;

    // Deep copy the scatter-gather list
    if (wr.sg_list != nullptr && wr.num_sge > 0) {
      sgList.resize(wr.num_sge);
      std::copy(wr.sg_list, wr.sg_list + wr.num_sge, sgList.begin());
      // Update the copied work request to point to our own scatter-gather list
      this->wr.sg_list = sgList.data();
    } else {
      // Handle case where there's no scatter-gather list
      this->wr.sg_list = nullptr;
      this->wr.num_sge = 0;
    }
  }
  VirtualSendWr() = default;
  ~VirtualSendWr() = default;

  ibv_send_wr wr{}; // Copy of the work request being posted by the user
  std::vector<ibv_sge> sgList; // Copy of the scatter-gather list
  int expectedMsgCnt{0}; // Expected message count resulting from splitting a
                         // large user message into multiple parts
  int remainingMsgCnt{0}; // Number of message segments left to transmit after
                          // splitting a large user messaget
  int offset{
      0}; // Address offset to be used for the next message send operation
  bool sendExtraNotifyImm{false}; // Whether to send an extra notify IMM message
  std::unordered_map<int32_t, MemoryRegionKeys>
      deviceIdToKeys; // Map from deviceId to device-specific keys (lkey, rkey)
};

struct VirtualRecvWr {
  inline VirtualRecvWr(
      const ibv_recv_wr& wr,
      int expectedMsgCnt,
      int remainingMsgCnt)
      : expectedMsgCnt(expectedMsgCnt), remainingMsgCnt(remainingMsgCnt) {
    // Make an explicit copy of the ibv_recv_wr structure
    this->wr = wr;

    // Deep copy the scatter-gather list
    if (wr.sg_list != nullptr && wr.num_sge > 0) {
      sgList.resize(wr.num_sge);
      std::copy(wr.sg_list, wr.sg_list + wr.num_sge, sgList.begin());
      // Update the copied work request to point to our own scatter-gather list
      this->wr.sg_list = sgList.data();
    } else {
      // Handle case where there's no scatter-gather list
      this->wr.sg_list = nullptr;
      this->wr.num_sge = 0;
    }
  };
  VirtualRecvWr() = default;
  ~VirtualRecvWr() = default;

  ibv_recv_wr wr{}; // Copy of the work request being posted by the user
  std::vector<ibv_sge> sgList; // Copy of the scatter-gather list
  int expectedMsgCnt{0}; // Expected message count resulting from splitting a
                         // large user message into multiple parts
  int remainingMsgCnt{0}; // Number of message segments left to transmit after
                          // splitting a large user messaget
  int offset{
      0}; // Address offset to be used for the next message send operation
};

} // namespace ibverbx
