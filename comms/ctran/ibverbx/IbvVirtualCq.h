// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include <folly/logging/xlog.h>
#include <deque>
#include <unordered_map>
#include <vector>

#include "comms/ctran/ibverbx/Coordinator.h"
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvCq.h"
#include "comms/ctran/ibverbx/IbvVirtualQp.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

struct VirtualWc {
  VirtualWc() = default;
  ~VirtualWc() = default;

  struct ibv_wc wc{};
  int expectedMsgCnt{0};
  int remainingMsgCnt{0};
  bool sendExtraNotifyImm{
      false}; // Whether to expect an extra notify IMM
              // message to be sent for the current virtualWc
};

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

  inline folly::Expected<std::vector<ibv_wc>, Error> pollCq(int numEntries);

  std::vector<IbvCq>& getPhysicalCqsRef();
  uint32_t getVirtualCqNum() const;

  void enqueSendCq(VirtualWc virtualWc);
  void enqueRecvCq(VirtualWc virtualWc);

  inline void processRequest(VirtualCqRequest&& request);

 private:
  friend class IbvPd;
  friend class IbvVirtualQp;

  inline static std::atomic<uint32_t> nextVirtualCqNum_{
      0}; // Static counter for assigning unique virtual CQ numbers
  uint32_t virtualCqNum_{
      0}; // The unique virtual CQ number assigned to instance of IbvVirtualCq

  std::vector<IbvCq> physicalCqs_;
  int maxCqe_{0};
  std::deque<VirtualWc> pendingSendVirtualWcQue_;
  std::deque<VirtualWc> pendingRecvVirtualWcQue_;
  inline void updateVirtualWcFromPhysicalWc(
      const ibv_wc& physicalWc,
      VirtualWc* virtualWc);
  std::unordered_map<uint64_t, VirtualWc*> virtualWrIdToVirtualWc_;

  // Helper function for IbvVirtualCq::pollCq.
  // Continuously polls the underlying physical Completion Queue (CQ) in a loop,
  // retrieving all available Completion Queue Entries (CQEs) until none remain.
  // For each physical CQE polled, the corresponding virtual CQE entries in the
  // virtual CQ are also updated. This function ensures that all ready physical
  // CQEs are polled, processed, and reflected in the virtual CQ state.
  inline folly::Expected<folly::Unit, Error> loopPollPhysicalCqUntilEmpty();

  // Helper function for IbvVirtualCq::pollCq.
  // Continuously polls the underlying virtual Completion Queues (CQs) in a
  // loop. The function collects up to numEntries virtual Completion Queue
  // Entries (CQEs), or stops early if there are no more virtual CQEs available
  // to poll. Returns a vector containing the polled virtual CQEs.
  inline std::vector<ibv_wc> loopPollVirtualCqUntil(int numEntries);
};

inline void Coordinator::submitRequestToVirtualCq(VirtualCqRequest&& request) {
  if (request.type == RequestType::SEND) {
    auto virtualCq = getVirtualSendCq(request.virtualQpNum);
    virtualCq->processRequest(std::move(request));
  } else {
    auto virtualCq = getVirtualRecvCq(request.virtualQpNum);
    virtualCq->processRequest(std::move(request));
  }
}

// IbvVirtualCq inline functions
inline folly::Expected<std::vector<ibv_wc>, Error> IbvVirtualCq::pollCq(
    int numEntries) {
  auto maybeLoopPollPhysicalCq = loopPollPhysicalCqUntilEmpty();
  if (maybeLoopPollPhysicalCq.hasError()) {
    return folly::makeUnexpected(maybeLoopPollPhysicalCq.error());
  }

  return loopPollVirtualCqUntil(numEntries);
}

inline folly::Expected<folly::Unit, Error>
IbvVirtualCq::loopPollPhysicalCqUntilEmpty() {
  // Poll from physical CQ one by one and process immediately
  int cqIdx = 0;
  while (true) {
    // Poll one completion at a time
    auto maybePhysicalWcsVector = physicalCqs_.at(cqIdx).pollCq(1);
    if (maybePhysicalWcsVector.hasError()) {
      return folly::makeUnexpected(maybePhysicalWcsVector.error());
    }

    // If there are no available completions, increment cqIdx and continue
    // looping. Repeat this process until cqIdx reaches physicalCqs_.size().
    if (maybePhysicalWcsVector->empty()) {
      cqIdx += 1;
      // If cqIdx equals physicalCqs_size, we have already polled CQEs from all
      // physical CQs. No more CQEs are available to poll at this time, so break
      // the loop.
      if (cqIdx == physicalCqs_.size()) {
        break;
      }
      continue;
    }

    // Process the single completion immediately
    const auto& physicalWc = maybePhysicalWcsVector->front();

    if (physicalWc.opcode == IBV_WC_RECV ||
        physicalWc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      VirtualQpRequest request = {
          .type = RequestType::RECV,
          .wrId = physicalWc.wr_id,
          .physicalQpNum = physicalWc.qp_num,
          .deviceId = physicalCqs_.at(cqIdx).getDeviceId()};
      if (physicalWc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        request.immData = physicalWc.imm_data;
      }
      auto coordinator = Coordinator::getCoordinator();
      CHECK(coordinator) << "Coordinator should not be nullptr during pollCq!";
      auto response = coordinator->submitRequestToVirtualQp(std::move(request));
      if (response.hasError()) {
        return folly::makeUnexpected(response.error());
      }

      if (response->useDqplb) {
        int processedCount = 0;
        for (int i = 0; i < pendingRecvVirtualWcQue_.size() &&
             processedCount < response->notifyCount;
             i++) {
          if (pendingRecvVirtualWcQue_.at(i).remainingMsgCnt != 0) {
            pendingRecvVirtualWcQue_.at(i).remainingMsgCnt = 0;
            processedCount++;
          }
        }
      } else {
        auto virtualWc = virtualWrIdToVirtualWc_.at(response->virtualWrId);
        virtualWc->remainingMsgCnt--;
        updateVirtualWcFromPhysicalWc(physicalWc, virtualWc);
      }
    } else {
      // Except for the above two conditions, all other conditions indicate a
      // send message, and we should poll from send queue
      VirtualQpRequest request = {
          .type = RequestType::SEND,
          .wrId = physicalWc.wr_id,
          .physicalQpNum = physicalWc.qp_num,
          .deviceId = physicalCqs_.at(cqIdx).getDeviceId()};
      auto coordinator = Coordinator::getCoordinator();
      CHECK(coordinator) << "Coordinator should not be nullptr during pollCq!";
      auto response = coordinator->submitRequestToVirtualQp(std::move(request));
      if (response.hasError()) {
        return folly::makeUnexpected(response.error());
      }

      auto virtualWc = virtualWrIdToVirtualWc_.at(response->virtualWrId);
      virtualWc->remainingMsgCnt--;
      updateVirtualWcFromPhysicalWc(physicalWc, virtualWc);
      if (virtualWc->remainingMsgCnt == 1 && virtualWc->sendExtraNotifyImm) {
        VirtualQpRequest request = {
            .type = RequestType::SEND_NOTIFY,
            .wrId = response->virtualWrId,
            .physicalQpNum = physicalWc.qp_num,
            .deviceId = physicalCqs_.at(cqIdx).getDeviceId()};

        auto coordinator = Coordinator::getCoordinator();
        CHECK(coordinator)
            << "Coordinator should not be nullptr during pollCq!";
        auto response =
            coordinator->submitRequestToVirtualQp(std::move(request));
        if (response.hasError()) {
          return folly::makeUnexpected(response.error());
        }
      }
    }
  }

  return folly::unit;
}

inline std::vector<ibv_wc> IbvVirtualCq::loopPollVirtualCqUntil(
    int numEntries) {
  std::vector<ibv_wc> wcs;
  wcs.reserve(numEntries);
  bool virtualSendCqPollComplete = false;
  bool virtualRecvCqPollComplete = false;
  while (wcs.size() < static_cast<size_t>(numEntries) &&
         (!virtualSendCqPollComplete || !virtualRecvCqPollComplete)) {
    if (!virtualSendCqPollComplete) {
      if (pendingSendVirtualWcQue_.empty() ||
          pendingSendVirtualWcQue_.front().remainingMsgCnt > 0) {
        virtualSendCqPollComplete = true;
      } else {
        auto vSendCqHead = pendingSendVirtualWcQue_.front();
        virtualWrIdToVirtualWc_.erase(vSendCqHead.wc.wr_id);
        wcs.push_back(std::move(vSendCqHead.wc));
        pendingSendVirtualWcQue_.pop_front();
      }
    }

    if (!virtualRecvCqPollComplete) {
      if (pendingRecvVirtualWcQue_.empty() ||
          pendingRecvVirtualWcQue_.front().remainingMsgCnt > 0) {
        virtualRecvCqPollComplete = true;
      } else {
        auto vRecvCqHead = pendingRecvVirtualWcQue_.front();
        virtualWrIdToVirtualWc_.erase(vRecvCqHead.wc.wr_id);
        wcs.push_back(std::move(vRecvCqHead.wc));
        pendingRecvVirtualWcQue_.pop_front();
      }
    }
  }

  return wcs;
}

inline void IbvVirtualCq::updateVirtualWcFromPhysicalWc(
    const ibv_wc& physicalWc,
    VirtualWc* virtualWc) {
  // Updates the vWc status field based on the statuses of all pWc instances.
  // If all physicalWc statuses indicate success, returns success.
  // If any of the physicalWc statuses indicate an error, return the first
  // encountered error code.
  // Additionally, log all error statuses for debug purposes.
  if (physicalWc.status != IBV_WC_SUCCESS) {
    if (virtualWc->wc.status == IBV_WC_SUCCESS) {
      virtualWc->wc.status = physicalWc.status;
    }

    // Log the error
    XLOGF(
        ERR,
        "Physical WC error: status={}, vendor_err={}, qp_num={}, wr_id={}",
        physicalWc.status,
        physicalWc.vendor_err,
        physicalWc.qp_num,
        physicalWc.wr_id);
  }

  // Update the OP code in virtualWc. Note that for the same user message, the
  // opcode must remain consistent, because all sub-messages within that user
  // message will be postSend using the same opcode.
  virtualWc->wc.opcode = physicalWc.opcode;

  // Update the vendor error in virtualWc. For now, assume that all pWc
  // instances will report the same vendor_error across all sub-messages
  // within a single user message.
  virtualWc->wc.vendor_err = physicalWc.vendor_err;

  virtualWc->wc.src_qp = physicalWc.src_qp;
  virtualWc->wc.byte_len += physicalWc.byte_len;
  virtualWc->wc.imm_data = physicalWc.imm_data;
  virtualWc->wc.wc_flags = physicalWc.wc_flags;
  virtualWc->wc.pkey_index = physicalWc.pkey_index;
  virtualWc->wc.slid = physicalWc.slid;
  virtualWc->wc.sl = physicalWc.sl;
  virtualWc->wc.dlid_path_bits = physicalWc.dlid_path_bits;
}

inline void IbvVirtualCq::processRequest(VirtualCqRequest&& request) {
  VirtualWc* virtualWcPtr = nullptr;
  uint64_t wrId;
  if (request.type == RequestType::SEND) {
    wrId = request.sendWr->wr_id;
    if (request.sendWr->send_flags & IBV_SEND_SIGNALED ||
        request.sendWr->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
      VirtualWc virtualWc{};
      virtualWc.wc.wr_id = request.sendWr->wr_id;
      virtualWc.wc.qp_num = request.virtualQpNum;
      virtualWc.wc.status = IBV_WC_SUCCESS;
      virtualWc.wc.byte_len = 0;
      virtualWc.expectedMsgCnt = request.expectedMsgCnt;
      virtualWc.remainingMsgCnt = request.expectedMsgCnt;
      virtualWc.sendExtraNotifyImm = request.sendExtraNotifyImm;
      pendingSendVirtualWcQue_.push_back(std::move(virtualWc));
      virtualWcPtr = &pendingSendVirtualWcQue_.back();
    }
  } else {
    wrId = request.recvWr->wr_id;
    VirtualWc virtualWc{};
    virtualWc.wc.wr_id = request.recvWr->wr_id;
    virtualWc.wc.qp_num = request.virtualQpNum;
    virtualWc.wc.status = IBV_WC_SUCCESS;
    virtualWc.wc.byte_len = 0;
    virtualWc.expectedMsgCnt = request.expectedMsgCnt;
    virtualWc.remainingMsgCnt = request.expectedMsgCnt;
    pendingRecvVirtualWcQue_.push_back(std::move(virtualWc));
    virtualWcPtr = &pendingRecvVirtualWcQue_.back();
  }
  virtualWrIdToVirtualWc_[wrId] = virtualWcPtr;
}

} // namespace ibverbx
