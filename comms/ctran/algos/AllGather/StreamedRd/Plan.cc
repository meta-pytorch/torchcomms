// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"

#include <utility>

#include "comms/ctran/utils/Checks.h"

namespace ctran::allgather::ctsrd {

namespace {

int log2i(int n) {
  int r = 0;
  while ((1 << r) < n) {
    r++;
  }
  return r;
}

int computePeer(int rank, int step, int nRanks) {
  const int dist = nRanks / (2 << step);
  const int pos = (rank / dist) % 2;
  return pos == 0 ? rank + dist : rank - dist;
}

// Compute chunk order for a single sub-tree using iterative stack.
// kRecvOnly: children in ascending step order (0, 1, ..., step-1)
std::vector<int> computeChunksRecvOnly(int senderRank, int step, int nRanks) {
  std::vector<int> chunks;
  chunks.reserve(1 << step);

  std::vector<std::pair<int, int>> stack;
  stack.push_back({senderRank, step});

  while (!stack.empty()) {
    const auto [rank, depth] = stack.back();
    stack.pop_back();
    chunks.push_back(rank);
    for (int child = depth - 1; child >= 0; child--) {
      const int peer = computePeer(rank, child, nRanks);
      stack.push_back({peer, child});
    }
  }

  return chunks;
}

// kFull: most recent step first, then staged (step-1, 0, 1, ..., step-2)
std::vector<int> computeChunksFull(int senderRank, int step, int nRanks) {
  std::vector<int> chunks;
  chunks.reserve(1 << step);

  std::vector<std::pair<int, int>> stack;
  stack.push_back({senderRank, step});

  while (!stack.empty()) {
    const auto [rank, depth] = stack.back();
    stack.pop_back();
    chunks.push_back(rank);
    if (depth == 0) {
      continue;
    }
    // Push staged children (step 0..depth-2) in reverse for stack ordering
    for (int child = depth - 2; child >= 0; child--) {
      const int peer = computePeer(rank, child, nRanks);
      stack.push_back({peer, child});
    }
    // Push most recent child (step depth-1) last so it's popped first
    const int newest = computePeer(rank, depth - 1, nRanks);
    stack.push_back({newest, depth - 1});
  }

  return chunks;
}

// Build all steps incrementally for recvOnly mode.
// step[i] = step[i-1] + subTree(peer_{i-1}, i-1)
std::vector<std::vector<int>>
buildAllStepsRecvOnly(int senderRank, int nSteps, int nRanks) {
  std::vector<std::vector<int>> steps(nSteps);
  steps[0] = {senderRank};
  for (int i = 1; i < nSteps; i++) {
    const int peer = computePeer(senderRank, i - 1, nRanks);
    auto newChunks = computeChunksRecvOnly(peer, i - 1, nRanks);
    steps[i].reserve(1 << i);
    steps[i] = steps[i - 1];
    steps[i].insert(steps[i].end(), newChunks.begin(), newChunks.end());
  }
  return steps;
}

// Build all steps incrementally for full mode.
// step[i] = [sender] + subTree(peer_{i-1}, i-1) + stagedTail
std::vector<std::vector<int>>
buildAllStepsFull(int senderRank, int nSteps, int nRanks) {
  std::vector<std::vector<int>> steps(nSteps);
  steps[0] = {senderRank};
  std::vector<int> stagedTail;
  for (int i = 1; i < nSteps; i++) {
    const int peer = computePeer(senderRank, i - 1, nRanks);
    auto newChunks = computeChunksFull(peer, i - 1, nRanks);
    steps[i].reserve(1 << i);
    steps[i].push_back(senderRank);
    steps[i].insert(steps[i].end(), newChunks.begin(), newChunks.end());
    steps[i].insert(steps[i].end(), stagedTail.begin(), stagedTail.end());
    stagedTail.insert(stagedTail.end(), newChunks.begin(), newChunks.end());
  }
  return steps;
}

std::vector<int> computePeers(int myRank, int nSteps, int nRanks) {
  std::vector<int> peers(nSteps);
  for (int i = 0; i < nSteps; i++) {
    peers[i] = computePeer(myRank, i, nRanks);
  }
  return peers;
}

inline void validateArgs(int myRank, int nRanks) {
  FB_CHECKABORT(nRanks > 0, "nRanks must be positive, got {}", nRanks);
  FB_CHECKABORT(
      (nRanks & (nRanks - 1)) == 0,
      "nRanks must be a power of 2, got {}",
      nRanks);
  FB_CHECKABORT(
      myRank >= 0 && myRank < nRanks,
      "myRank {} out of range [0, {})",
      myRank,
      nRanks);
}

} // namespace

Plan createSendPlan(int myRank, int nRanks, FcMode fcMode) {
  validateArgs(myRank, nRanks);
  const int nSteps = log2i(nRanks);
  if (nSteps == 0) {
    return Plan({}, {}, fcMode);
  }
  auto peers = computePeers(myRank, nSteps, nRanks);
  if (fcMode == FcMode::kRecvOnly) {
    return Plan(
        buildAllStepsRecvOnly(myRank, nSteps, nRanks),
        std::move(peers),
        fcMode);
  }
  return Plan(
      buildAllStepsFull(myRank, nSteps, nRanks), std::move(peers), fcMode);
}

Plan createRecvPlan(int myRank, int nRanks, FcMode fcMode) {
  validateArgs(myRank, nRanks);
  const int nSteps = log2i(nRanks);
  if (nSteps == 0) {
    return Plan({}, {}, fcMode);
  }
  auto peers = computePeers(myRank, nSteps, nRanks);
  std::vector<std::vector<int>> steps(nSteps);
  for (int s = 0; s < nSteps; s++) {
    if (fcMode == FcMode::kRecvOnly) {
      steps[s] = computeChunksRecvOnly(peers[s], s, nRanks);
    } else {
      steps[s] = computeChunksFull(peers[s], s, nRanks);
    }
  }
  return Plan(std::move(steps), std::move(peers), fcMode);
}

PersistPlan createPersistPlan(int myRank, int nRanks, FcMode fcMode) {
  return PersistPlan(
      createRecvPlan(myRank, nRanks, fcMode),
      createSendPlan(myRank, nRanks, fcMode));
}

} // namespace ctran::allgather::ctsrd
