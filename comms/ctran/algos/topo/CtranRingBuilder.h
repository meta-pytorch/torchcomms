// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

namespace ctran::algos::topo {

// ring type used in kernel and GPE side
struct CtranRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;
  int* userRanks;
  int index; // This rank's index in the ring
  int nRanks; // Number of ranks in the ring
};

/**
 * Initialize ring topology for distributed communication.
 *
 * Creates multiple ring configurations where each ring represents a different
 * communication pattern. The rings are designed to provide load balancing
 * across nodes while maintaining efficient intra-node and inter-node
 * communication.
 *
 * @param nNodes       Number of nodes in the cluster
 * @param nLocalRanks  Number of ranks (processes) per node
 * @param nRanks       Total number of ranks across all nodes
 * @return             2D vector where each row represents a ring configuration
 *                     and each column represents a rank's position in that ring
 */
static std::vector<std::vector<int>>
getMultiFlatRing(int nNodes, int nLocalRanks, int nRanks) {
  std::vector<std::vector<int>> ring(nLocalRanks, std::vector<int>(nRanks));

  // set up the first ring
  for (int n = 0; n < nNodes; n++) {
    // always start with 0
    ring[0][n * nLocalRanks] = n * nLocalRanks;
    for (int i = 1; i < nLocalRanks; i++) {
      ring[0][n * nLocalRanks + i] = n * nLocalRanks + i;
    }
  }
  for (int lr = 1; lr < nLocalRanks; lr++) {
    ring[lr] = ring[lr - 1];

    // load balance the inter-node ring
    for (int n = 0; n < nNodes - 1; n++) {
      int pos = nLocalRanks - lr;
      std::swap(
          ring[lr][pos + n * nLocalRanks],
          ring[lr][pos + (n + 1) * nLocalRanks]);
    }
  }

  return ring;
}

/**
 * Build multiple ring topologies and compute prev/next relationships for each
 * rank.
 *
 * This function creates multiple ring configurations based on the base rings
 * from ringInit(), then for each rank in each ring, determines its predecessor
 * and successor ranks. This is essential for ring-based collective
 * communication algorithms like AllReduce.
 *
 * @param rank         Current rank (unused in current implementation)
 * @param nLocalRanks  Number of ranks per node
 * @param nRanks       Total number of ranks
 * @param nNodes       Number of nodes
 * @param nRings       Number of ring configurations to create
 * @param ringPrev     Output: predecessor rank for each ring and rank
 * [ring][rank]
 * @param ringNext     Output: successor rank for each ring and rank
 * [ring][rank]
 * @param rings        Output: ring configurations [ring][position] = rank
 */
static void buildMultiFlatRing(
    int rank,
    int nLocalRanks,
    int nRanks,
    int nNodes,
    int nRings,
    std::vector<std::vector<int>>& ringPrev,
    std::vector<std::vector<int>>& ringNext,
    std::vector<std::vector<int>>& rings) {
  std::vector<std::vector<int>> baseRings =
      getMultiFlatRing(nNodes, nLocalRanks, nRanks);

  // Copy base rings to output rings
  for (int i = 0; i < std::min((int)baseRings.size(), nRings); i++) {
    rings[i] = baseRings[i];
  }
  for (int i = std::min((int)baseRings.size(), nRings); i < nRings; i++) {
    rings[i] = baseRings[i % baseRings.size()];
  }

  // For each rank, find its position in the global ring and set prev/next
  for (int r = 0; r < nRanks; r++) {
    for (int c = 0; c < nRings; c++) {
      // Find rank's position in the ring
      int pos = -1;
      for (int i = 0; i < nRanks; i++) {
        if (rings[c][i] == r) {
          pos = i;
          break;
        }
      }

      if (pos != -1) {
        int prevPos = (pos - 1 + nRanks) % nRanks;
        int nextPos = (pos + 1) % nRanks;

        ringPrev[c][r] = rings[c][prevPos];
        ringNext[c][r] = rings[c][nextPos];
      }
    }
  }
}

} // namespace ctran::algos::topo
