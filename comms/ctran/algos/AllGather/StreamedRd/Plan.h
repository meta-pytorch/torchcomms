// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <vector>

#include "comms/ctran/algos/IPersistPlan.h"

namespace ctran::allgather::ctsrd {

// Plan for streamed recursive-doubling AllGather, parameterized by the
// "immediate-forward depth" fwdPeers:
//   fwdPeers = 0          : ctrd-equivalent (no streaming of received
//                           chunks; they're staged and flushed at the start
//                           of each step).
//   fwdPeers = 1          : forward each received chunk to the next step's
//                           peer immediately; defer further forwards.
//   fwdPeers >= nSteps    : recvOnly-equivalent (forward each received
//                           chunk to all future step peers immediately).
//   Intermediate values are valid and produce a continuum between the two
//   extremes for the *received-chunk* forwarding policy.
//
// Independent of fwdPeers, the sender's OWN chunk is always pre-enqueued at
// position 0 of every step's send-plan (own has no network dependency, so
// we get it on the wire as early as possible — this matches legacy
// pre-consolidation behavior).
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

  int fwdPeers() const {
    return fwdPeers_;
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
  Plan(
      std::vector<std::vector<int>> steps,
      std::vector<int> peers,
      int fwdPeers)
      : steps_(std::move(steps)),
        peers_(std::move(peers)),
        fwdPeers_(fwdPeers) {}

  std::vector<std::vector<int>> steps_;
  std::vector<int> peers_;
  int fwdPeers_;

  friend Plan createRecvPlan(int myRank, int nRanks, int fwdPeers);
  friend Plan createSendPlan(int myRank, int nRanks, int fwdPeers);
};

Plan createRecvPlan(int myRank, int nRanks, int fwdPeers);
Plan createSendPlan(int myRank, int nRanks, int fwdPeers);

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

  friend PersistPlan createPersistPlan(int myRank, int nRanks, int fwdPeers);
};

PersistPlan createPersistPlan(int myRank, int nRanks, int fwdPeers);

} // namespace ctran::allgather::ctsrd
