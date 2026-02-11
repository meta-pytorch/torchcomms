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

// Per-buffer device-specific keys for scatter-gather operations
struct ScatterGatherBufferKeys {
  // Map from deviceId to device-specific keys for a single buffer
  std::unordered_map<int32_t, MemoryRegionKeys> deviceIdToKeys;
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

      // Initialize per-buffer lengths from the scatter-gather entries
      for (int i = 0; i < wr.num_sge; ++i) {
        sbufLens.push_back(wr.sg_list[i].length);
      }
    } else {
      // Handle case where there's no scatter-gather list
      this->wr.sg_list = nullptr;
      this->wr.num_sge = 0;
    }
  }

  // Constructor for scatter-gather operations with per-buffer keys
  VirtualSendWr(
      const ibv_send_wr& wr,
      int expectedMsgCnt,
      int remainingMsgCnt,
      bool sendExtraNotifyImm,
      const std::vector<ScatterGatherBufferKeys>& perBufferKeys)
      : expectedMsgCnt(expectedMsgCnt),
        remainingMsgCnt(remainingMsgCnt),
        sendExtraNotifyImm(sendExtraNotifyImm),
        perBufferDeviceKeys(perBufferKeys) {
    // Make an explicit copy of the ibv_send_wr structure
    this->wr = wr;

    // Deep copy the scatter-gather list
    if (wr.sg_list != nullptr && wr.num_sge > 0) {
      sgList.resize(wr.num_sge);
      std::copy(wr.sg_list, wr.sg_list + wr.num_sge, sgList.begin());
      // Update the copied work request to point to our own scatter-gather list
      this->wr.sg_list = sgList.data();

      // Initialize per-buffer lengths from the scatter-gather entries
      for (int i = 0; i < wr.num_sge; ++i) {
        sbufLens.push_back(wr.sg_list[i].length);
      }
    } else {
      // Handle case where there's no scatter-gather list
      this->wr.sg_list = nullptr;
      this->wr.num_sge = 0;
    }
  }

  VirtualSendWr() = default;
  ~VirtualSendWr() = default;

  // Helper function to check if scatter-gather mode is enabled
  bool isScatterGatherEnabled() const {
    return sgList.size() > 1;
  }

  // Helper function to get the current source buffer and remaining length
  // based on the current offset for scatter-gather operations.
  // Returns {buffer_index, offset_in_buffer, remaining_in_buffer}
  std::tuple<int, uint64_t, uint64_t> getCurrentBufferPosition() const {
    if (sgList.empty()) {
      return {-1, 0, 0};
    }

    uint64_t cumulativeOffset = 0;
    for (size_t i = 0; i < sgList.size(); ++i) {
      uint64_t bufLen = sbufLens.empty() ? sgList[i].length : sbufLens[i];
      if (static_cast<uint64_t>(offset) < cumulativeOffset + bufLen) {
        // The offset falls within this buffer
        uint64_t bufferOffset = offset - cumulativeOffset;
        uint64_t remainingInBuffer = bufLen - bufferOffset;
        return {static_cast<int>(i), bufferOffset, remainingInBuffer};
      }
      cumulativeOffset += bufLen;
    }

    // Offset is beyond all buffers
    return {-1, 0, 0};
  }

  // Helper function to calculate total length across all scatter-gather buffers
  uint64_t getTotalLength() const {
    uint64_t totalLen = 0;
    for (size_t i = 0; i < sgList.size(); ++i) {
      totalLen += sbufLens.empty() ? sgList[i].length : sbufLens[i];
    }
    return totalLen;
  }

  ibv_send_wr wr{}; // Copy of the work request being posted by the user
  std::vector<ibv_sge> sgList; // Copy of the scatter-gather list
  std::vector<uint64_t>
      sbufLens; // Length of each buffer in scatter-gather list
  int expectedMsgCnt{0}; // Expected message count resulting from splitting a
                         // large user message into multiple parts
  int remainingMsgCnt{0}; // Number of message segments left to transmit after
                          // splitting a large user messaget
  int offset{
      0}; // Address offset to be used for the next message send operation
  bool sendExtraNotifyImm{false}; // Whether to send an extra notify IMM message
  std::unordered_map<int32_t, MemoryRegionKeys>
      deviceIdToKeys; // Map from deviceId to device-specific keys (lkey, rkey)
                      // - used for single-buffer operations

  // Per-buffer device keys for scatter-gather operations
  // perBufferDeviceKeys[buf_idx].deviceIdToKeys[deviceId] = {lkey, rkey}
  std::vector<ScatterGatherBufferKeys> perBufferDeviceKeys;
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
