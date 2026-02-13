// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include <folly/container/F14Map.h>
#include <folly/logging/xlog.h>
#include <deque>
#include <vector>

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvCq.h"
#include "comms/ctran/ibverbx/IbvVirtualQp.h"
#include "comms/ctran/ibverbx/IbvVirtualWr.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// Returns true if this CQE comes from a multi-QP VirtualQp and has an RDMA
// opcode that participates in load-balanced fragmentation across physical QPs.
inline bool isUsingMultiQpLoadBalancing(bool isMultiQp, ibv_wc_opcode opcode) {
  return isMultiQp &&
      (opcode == IBV_WC_RDMA_WRITE || opcode == IBV_WC_RDMA_READ ||
       opcode == IBV_WC_RECV_RDMA_WITH_IMM);
}

// Ibv Virtual Completion Queue (CQ): Provides a virtual CQ abstraction for the
// user. When the user calls IbvVirtualQp::postSend() or
// IbvVirtualQp::postRecv(), they can track the completion of messages posted on
// the Virtual QP through this virtual CQ.
class IbvVirtualCq {
 public:
  IbvVirtualCq(IbvCq&& cq, int maxCqe);
  IbvVirtualCq(std::vector<IbvCq>&& cqs, int maxCqe);
  ~IbvVirtualCq();

  // disable copy constructor
  IbvVirtualCq(const IbvVirtualCq&) = delete;
  IbvVirtualCq& operator=(const IbvVirtualCq&) = delete;

  // move constructor
  IbvVirtualCq(IbvVirtualCq&& other) noexcept;
  IbvVirtualCq& operator=(IbvVirtualCq&& other) noexcept;

  // Drain all physical CQEs and return as IbvVirtualWc. In Single-QP
  // (isMultiQp=false) or Multi-QP Send/Recv cases, pass CQE through directly as
  // IbvVirtualWc; in Multi-QP RDMA cases, route to
  // VirtualQp::processCompletion.
  inline folly::Expected<std::vector<IbvVirtualWc>, Error> pollCq();

  // Registration API (called by VirtualQp constructor/destructor)
  void registerPhysicalQp(
      uint32_t physicalQpNum,
      int32_t deviceId,
      IbvVirtualQp* vqp,
      bool isMultiQp,
      uint32_t virtualQpNum);
  void unregisterPhysicalQp(uint32_t physicalQpNum, int32_t deviceId);

  std::vector<IbvCq>& getPhysicalCqsRef();
  uint32_t getVirtualCqNum() const;

 private:
  friend class IbvPd;
  friend class IbvVirtualQp;

#ifdef IBVERBX_TEST_FRIENDS
  IBVERBX_TEST_FRIENDS
#endif

  inline static std::atomic<uint32_t> nextVirtualCqNum_{
      0}; // Static counter for assigning unique virtual CQ numbers
  uint32_t virtualCqNum_{
      0}; // The unique virtual CQ number assigned to instance of IbvVirtualCq

  std::vector<IbvCq> physicalCqs_;
  int maxCqe_{0};

  // Registration info for each physical QP (used by pollCq)
  struct RegisteredQpInfo {
    IbvVirtualQp* vqp{nullptr}; // Non-owning pointer to VirtualQp
    bool isMultiQp{false}; // true if VirtualQp has >1 physical QPs
    uint32_t virtualQpNum{0}; // Virtual QP number (for passthrough)
  };

  // Registration table: QpId → RegisteredQpInfo
  folly::F14FastMap<QpId, RegisteredQpInfo, QpIdHash> registeredQps_;

  // Helper: Find registered QP info by physical QP num and device ID
  inline const RegisteredQpInfo* findRegisteredQpInfo(
      uint32_t qpNum,
      int32_t deviceId) const;
};

// IbvVirtualCq inline functions

// pollCq: Drain all physical CQEs and route them.
// TODO: Accept a numEntries parameter like original pollCq() and return at most
// that many completions, instead of draining all physical CQEs unconditionally.
inline folly::Expected<std::vector<IbvVirtualWc>, Error>
IbvVirtualCq::pollCq() {
  std::vector<IbvVirtualWc> results;

  for (size_t cqIdx = 0; cqIdx < physicalCqs_.size(); cqIdx++) {
    auto& cq = physicalCqs_.at(cqIdx);
    int32_t deviceId = cq.getDeviceId();

    // Drain this CQ until empty
    while (true) {
      auto maybeWcs = cq.pollCq(kPollCqBatchSize);
      if (maybeWcs.hasError()) {
        return folly::makeUnexpected(maybeWcs.error());
      }
      auto& physicalWcs = *maybeWcs;
      if (physicalWcs.empty()) {
        break; // CQ drained
      }

      // Process each physical completion
      for (size_t i = 0; i < physicalWcs.size(); i++) {
        const ibv_wc& physicalWc = physicalWcs[i];

        // Lookup registration info for this QP
        const RegisteredQpInfo* info =
            findRegisteredQpInfo(physicalWc.qp_num, deviceId);

        CHECK(info != nullptr) << fmt::format(
            "[Ibverbx]IbvVirtualCq::pollCq, unregistered QP: qpNum={}, deviceId={}",
            physicalWc.qp_num,
            deviceId);

        if (isUsingMultiQpLoadBalancing(info->isMultiQp, physicalWc.opcode)) {
          // Multi-QP RDMA: route to VirtualQp for fragment reassembly
          auto maybeVirtualWcs =
              info->vqp->processCompletion(physicalWc, deviceId);

          if (maybeVirtualWcs.hasError()) {
            return folly::makeUnexpected(maybeVirtualWcs.error());
          }

          for (auto& virtualWc : *maybeVirtualWcs) {
            results.push_back(std::move(virtualWc));
          }
        } else {
          // Passthrough: single-QP, or non-RDMA opcodes (SEND, RECV, atomics,
          // etc.) that don't need fragment aggregation
          IbvVirtualWc vwc;
          vwc.wrId = physicalWc.wr_id;
          vwc.status = physicalWc.status;
          vwc.opcode = physicalWc.opcode;
          vwc.qpNum = info->virtualQpNum;
          vwc.immData = physicalWc.imm_data;
          vwc.byteLen = physicalWc.byte_len;
          results.push_back(vwc);
        }
      }
    }
  }

  return results;
}

// Helper: Find registered QP info
inline const IbvVirtualCq::RegisteredQpInfo* IbvVirtualCq::findRegisteredQpInfo(
    uint32_t qpNum,
    int32_t deviceId) const {
  QpId key{.deviceId = deviceId, .qpNum = qpNum};
  auto it = registeredQps_.find(key);
  if (it == registeredQps_.end()) {
    return nullptr;
  }
  return &it->second;
}

} // namespace ibverbx
