// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <unordered_map>
#include <vector>
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

struct MemoryRegionKeys {
  uint32_t lkey{0};
  uint32_t rkey{0};
};

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
