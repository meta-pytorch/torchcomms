// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

#include "comms/ctran/algos/IPersistPlan.h"

namespace ctran::allgather::ctsrd {

// Flow control mode for streamed recursive doubling AllGather.
//
// kRecvOnly: receiver-only flow control. Each rank pre-sends its own chunk
//   to all step peers upfront and forwards received chunks to all future
//   step peers immediately. Maximizes send/recv overlap at the cost of
//   more concurrent flows (up to log2(nRanks) peers simultaneously).
//
// kFull: sender and receiver flow control. Each rank forwards received
//   chunks to the next step peer immediately, but stages forwards for
//   further steps until those steps begin. Limits concurrent flows to
//   at most 2 peers (current step + next step).
enum class FcMode {
  kRecvOnly,
  kFull,
};

class Plan {
 public:
  const std::vector<int>& chunks(int step) const {
    return steps_.at(step);
  }

  int chunk(int step, int k) const {
    return steps_.at(step).at(k);
  }

  int nSteps() const {
    return static_cast<int>(steps_.size());
  }

  FcMode fcMode() const {
    return fcMode_;
  }

  int peer(int step) const {
    return peers_.at(step);
  }

  int lastChunk(int step) const {
    return steps_.at(step).back();
  }

  bool isLastChunk(int step, int chunkOffset) const {
    return steps_.at(step).back() == chunkOffset;
  }

 private:
  Plan(std::vector<std::vector<int>> steps, std::vector<int> peers, FcMode fc)
      : steps_(std::move(steps)), peers_(std::move(peers)), fcMode_(fc) {}

  std::vector<std::vector<int>> steps_;
  std::vector<int> peers_;
  FcMode fcMode_;

  friend Plan createRecvPlan(int myRank, int nRanks, FcMode fcMode);
  friend Plan createSendPlan(int myRank, int nRanks, FcMode fcMode);
};

Plan createRecvPlan(int myRank, int nRanks, FcMode fcMode);
Plan createSendPlan(int myRank, int nRanks, FcMode fcMode);

class PersistPlan : public ctran::algos::IPersistPlan {
 public:
  const Plan& recvPlan() const {
    return recvPlan_;
  }

  const Plan& sendPlan() const {
    return sendPlan_;
  }

 private:
  PersistPlan(Plan recvPlan, Plan sendPlan)
      : recvPlan_(std::move(recvPlan)), sendPlan_(std::move(sendPlan)) {}

  Plan recvPlan_;
  Plan sendPlan_;

  friend PersistPlan createPersistPlan(int myRank, int nRanks, FcMode fcMode);
};

PersistPlan createPersistPlan(int myRank, int nRanks, FcMode fcMode);

} // namespace ctran::allgather::ctsrd
