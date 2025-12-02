// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <unordered_map>
#include "comms/ctran/ibverbx/IbvCommon.h"

namespace ibverbx {

class DqplbSeqTracker {
 public:
  DqplbSeqTracker() = default;
  ~DqplbSeqTracker() = default;

  // Explicitly default move constructor and move assignment operator
  DqplbSeqTracker(DqplbSeqTracker&&) = default;
  DqplbSeqTracker& operator=(DqplbSeqTracker&&) = default;

  // This helper function calculates sender IMM message in DQPLB mode.
  inline uint32_t getSendImm(int remainingMsgCnt);
  // This helper function processes received IMM message and update
  // receivedSeqNums_ map and receiveNext_ field.
  inline int processReceivedImm(uint32_t receivedImm);

 private:
  int sendNext_{0};
  int receiveNext_{0};
  std::unordered_map<uint32_t, bool> receivedSeqNums_;
};

// DqplbSeqTracker inline functions
inline uint32_t DqplbSeqTracker::getSendImm(int remainingMsgCnt) {
  uint32_t immData = sendNext_;
  sendNext_ = (sendNext_ + 1) % kSeqNumMask;
  if (remainingMsgCnt == 1) {
    immData |= (1 << kNotifyBit);
  }
  return immData;
}

inline int DqplbSeqTracker::processReceivedImm(uint32_t immData) {
  int notifyCount = 0;
  receivedSeqNums_[immData & kSeqNumMask] = immData & (1U << kNotifyBit);
  auto it = receivedSeqNums_.find(receiveNext_);

  while (it != receivedSeqNums_.end()) {
    if (it->second) {
      notifyCount++;
    }
    receivedSeqNums_.erase(it);
    receiveNext_ = (receiveNext_ + 1) % kSeqNumMask;
    it = receivedSeqNums_.find(receiveNext_);
  }
  return notifyCount;
}

} // namespace ibverbx
