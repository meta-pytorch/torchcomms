// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/AllGather/StreamedRd/Plan.h"

#include <algorithm>
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

// Build the per-step ordered chunk sequence that sender `senderRank` produces
// at each outer step under the given fwdPeers value.
//
// Own (root) is ALWAYS pre-enqueued first at every step regardless of
// fwdPeers — it has no network dependency, so we get it on the wire as
// early as possible.
//
// For a sender `root` at depth `depth`, fwdPeers controls only the ordering
// of *received* chunks that root accumulated from its earlier-step peers:
//   Phase 1: chunks received at outer steps [depth-fwdPeers, depth-1] are
//     forwarded immediately on receipt (they appear first after own).
//   Phase 2: chunks received at outer steps [0, depth-fwdPeers-1] are
//     deferred and flushed at the start of outer step `depth` (they appear
//     after Phase 1 chunks).
// At fwdPeers >= depth, Phase 1 covers all of [0, depth-1] and Phase 2 is
// empty; at fwdPeers == 0, Phase 1 is empty and Phase 2 covers all.
std::vector<int>
chunksForSubtree(int root, int depth, int fwdPeers, int nRanks) {
  std::vector<int> result;
  result.reserve(1 << depth);

  // Own always leads every step's send-plan.
  result.push_back(root);

  const int phase1Start = std::max(0, depth - fwdPeers);

  // Phase 1: immediate forwards from receives at steps [phase1Start, depth-1].
  for (int i = phase1Start; i < depth; i++) {
    const int child = computePeer(root, i, nRanks);
    auto sub = chunksForSubtree(child, i, fwdPeers, nRanks);
    result.insert(result.end(), sub.begin(), sub.end());
  }

  // Phase 2: deferred flush — chunks received at steps [0, phase1Start - 1].
  for (int i = 0; i < phase1Start; i++) {
    const int child = computePeer(root, i, nRanks);
    auto sub = chunksForSubtree(child, i, fwdPeers, nRanks);
    result.insert(result.end(), sub.begin(), sub.end());
  }

  return result;
}

// Build all steps for sender `senderRank`. step[i] gives the ordered list of
// chunks the sender enqueues for the peer at step i.
std::vector<std::vector<int>>
buildAllSteps(int senderRank, int nSteps, int nRanks, int fwdPeers) {
  std::vector<std::vector<int>> steps(nSteps);
  for (int i = 0; i < nSteps; i++) {
    steps[i] = chunksForSubtree(senderRank, i, fwdPeers, nRanks);
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

inline void validateArgs(int myRank, int nRanks, int fwdPeers) {
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
  FB_CHECKABORT(fwdPeers >= 0, "fwdPeers must be >= 0, got {}", fwdPeers);
}

} // namespace

Plan createSendPlan(int myRank, int nRanks, int fwdPeers) {
  validateArgs(myRank, nRanks, fwdPeers);
  const int nSteps = log2i(nRanks);
  if (nSteps == 0) {
    return Plan({}, {}, fwdPeers);
  }
  auto peers = computePeers(myRank, nSteps, nRanks);
  return Plan(
      buildAllSteps(myRank, nSteps, nRanks, fwdPeers),
      std::move(peers),
      fwdPeers);
}

Plan createRecvPlan(int myRank, int nRanks, int fwdPeers) {
  validateArgs(myRank, nRanks, fwdPeers);
  const int nSteps = log2i(nRanks);
  if (nSteps == 0) {
    return Plan({}, {}, fwdPeers);
  }
  auto peers = computePeers(myRank, nSteps, nRanks);
  std::vector<std::vector<int>> steps(nSteps);
  for (int s = 0; s < nSteps; s++) {
    steps[s] = chunksForSubtree(peers[s], s, fwdPeers, nRanks);
  }
  return Plan(std::move(steps), std::move(peers), fwdPeers);
}

PersistPlan createPersistPlan(int myRank, int nRanks, int fwdPeers) {
  return PersistPlan(
      createRecvPlan(myRank, nRanks, fwdPeers),
      createSendPlan(myRank, nRanks, fwdPeers));
}

} // namespace ctran::allgather::ctsrd
