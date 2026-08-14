// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "comms/prims/transport/MultiPeerIbTransport.h"

namespace comms::prims {
namespace {

// The Data-Direct config knob (NCCL_IB_DATA_DIRECT 0/1/2, tunneled into
// MultipeerIbTransportConfig::enableDataDirect) must reach registerBuffer's
// per-NIC registration decision: registerBuffer() takes the Data-Direct
// (BAR1) registration path exactly when dataDirectActiveForNic() holds. These
// pure checks pin that config -> registration tunnel without needing a NIC.
// enableDataDirect is the single shared comms::prims::DataDirectMode, also used
// by NIC discovery.

// Default config requests Data-Direct (Only = NCCL's default of 1).
TEST(MultiPeerIbTransportConfigTest, DataDirectDefaultsToOnly) {
  MultipeerIbTransportConfig config;
  EXPECT_EQ(config.enableDataDirect, DataDirectMode::Only);
}

// Any non-Disabled mode (Only or Both) + a DD-capable NIC -> registerBuffer
// uses the Data-Direct path.
TEST(MultiPeerIbTransportConfigTest, DataDirectActiveOnCapableNic) {
  MultipeerIbTransportConfig config;
  config.enableDataDirect = DataDirectMode::Only;
  EXPECT_TRUE(dataDirectActiveForNic(config, /*nicIsDataDirect=*/true));
  config.enableDataDirect = DataDirectMode::Both;
  EXPECT_TRUE(dataDirectActiveForNic(config, /*nicIsDataDirect=*/true));
}

// Non-Disabled but a non-DD NIC -> no Data-Direct path; registerBuffer falls
// back to the regular DMA-BUF / reg_mr path (e.g. H100).
TEST(MultiPeerIbTransportConfigTest, DataDirectInactiveOnNonCapableNic) {
  MultipeerIbTransportConfig config;
  config.enableDataDirect = DataDirectMode::Only;
  EXPECT_FALSE(dataDirectActiveForNic(config, /*nicIsDataDirect=*/false));
}

// Disabled -> never use Data-Direct, even on a DD-capable NIC.
TEST(MultiPeerIbTransportConfigTest, DataDirectDisabledNeverActivates) {
  MultipeerIbTransportConfig config;
  config.enableDataDirect = DataDirectMode::Disabled;
  EXPECT_FALSE(dataDirectActiveForNic(config, /*nicIsDataDirect=*/true));
  EXPECT_FALSE(dataDirectActiveForNic(config, /*nicIsDataDirect=*/false));
}

// The key automatic behavior: with a default-constructed config (no caller
// opt-in), registerBuffer must AUTOMATICALLY select the Data-Direct path on a
// DD-capable NIC and only on a DD-capable NIC. dataDirectActiveForNic() is the
// exact predicate registerBuffer gates the DD registration path on, so this
// asserts the auto-select decision end to end for the default configuration.
TEST(
    MultiPeerIbTransportConfigTest,
    RegisterBufferAutoSelectsDataDirectByDefault) {
  MultipeerIbTransportConfig defaultConfig; // no explicit enableDataDirect

  // DD-capable NIC: auto-selected, no configuration needed.
  EXPECT_TRUE(dataDirectActiveForNic(defaultConfig, /*nicIsDataDirect=*/true));
  // Non-DD NIC: not selected (transparent fallback to the regular path).
  EXPECT_FALSE(
      dataDirectActiveForNic(defaultConfig, /*nicIsDataDirect=*/false));
}

// The PCIe Relaxed Ordering knob (NCCL_IB_PCI_RELAXED_ORDERING, tunneled into
// enablePciRelaxedOrdering) reaches registerBuffer's access-flag decision via
// relaxedOrderingActiveForNic(): the IBV_ACCESS_RELAXED_ORDERING flag is set
// exactly when this holds. Crucially, it is also gated on NIC capability
// (probed during openNics), so on a NIC whose driver rejects the flag both
// Auto and Enabled fall back to strict ordering instead of failing
// registration. These pure checks pin that gating without needing a NIC.

// Default config requests Relaxed Ordering (Auto), matching NCCL's default.
TEST(MultiPeerIbTransportConfigTest, RelaxedOrderingDefaultsToAuto) {
  MultipeerIbTransportConfig config;
  EXPECT_EQ(
      config.enablePciRelaxedOrdering,
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Auto);
}

// Auto + RO-capable NIC -> registerBuffer sets the Relaxed Ordering flag.
TEST(MultiPeerIbTransportConfigTest, RelaxedOrderingAutoActiveOnCapableNic) {
  MultipeerIbTransportConfig config;
  config.enablePciRelaxedOrdering =
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Auto;
  EXPECT_TRUE(
      relaxedOrderingActiveForNic(config, /*nicRelaxedOrderingCapable=*/true));
}

// Auto but the NIC's driver rejects the flag -> fall back to strict ordering
// (no throw). This is the case the review flagged.
TEST(
    MultiPeerIbTransportConfigTest,
    RelaxedOrderingAutoFallsBackOnIncapableNic) {
  MultipeerIbTransportConfig config;
  config.enablePciRelaxedOrdering =
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Auto;
  EXPECT_FALSE(
      relaxedOrderingActiveForNic(config, /*nicRelaxedOrderingCapable=*/false));
}

// Even an explicit Enabled request falls back when the NIC can't do RO, so
// transport setup never breaks on an unsupporting driver (a warning is logged).
TEST(
    MultiPeerIbTransportConfigTest,
    RelaxedOrderingEnabledFallsBackOnIncapableNic) {
  MultipeerIbTransportConfig config;
  config.enablePciRelaxedOrdering =
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Enabled;
  EXPECT_TRUE(
      relaxedOrderingActiveForNic(config, /*nicRelaxedOrderingCapable=*/true));
  EXPECT_FALSE(
      relaxedOrderingActiveForNic(config, /*nicRelaxedOrderingCapable=*/false));
}

// Disabled -> never set the flag, even on a capable NIC.
TEST(MultiPeerIbTransportConfigTest, RelaxedOrderingDisabledNeverActive) {
  MultipeerIbTransportConfig config;
  config.enablePciRelaxedOrdering =
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Disabled;
  EXPECT_FALSE(
      relaxedOrderingActiveForNic(config, /*nicRelaxedOrderingCapable=*/true));
  EXPECT_FALSE(
      relaxedOrderingActiveForNic(config, /*nicRelaxedOrderingCapable=*/false));
}

TEST(MultiPeerIbTransportConfigTest, ReliableDoorbellAutoUsesNicCapability) {
  const MultipeerIbTransportConfig config;
  EXPECT_FALSE(config.enableReliableDoorbell.has_value());
  EXPECT_TRUE(reliableDoorbellNeedsCapabilityQuery(config));
  EXPECT_TRUE(reliableDoorbellActiveForNic(
      config, /*nicReliableDoorbellCapable=*/true));
  EXPECT_FALSE(reliableDoorbellActiveForNic(
      config, /*nicReliableDoorbellCapable=*/false));
}

TEST(MultiPeerIbTransportConfigTest, ReliableDoorbellEnableRequiresCapability) {
  MultipeerIbTransportConfig config;
  config.enableReliableDoorbell = true;
  EXPECT_TRUE(reliableDoorbellNeedsCapabilityQuery(config));
  EXPECT_TRUE(reliableDoorbellActiveForNic(
      config, /*nicReliableDoorbellCapable=*/true));
  EXPECT_THROW(
      reliableDoorbellActiveForNic(
          config, /*nicReliableDoorbellCapable=*/false),
      std::invalid_argument);
}

TEST(MultiPeerIbTransportConfigTest, ReliableDoorbellDisableForcesValidDbr) {
  MultipeerIbTransportConfig config;
  config.enableReliableDoorbell = false;
  EXPECT_FALSE(reliableDoorbellNeedsCapabilityQuery(config));
  EXPECT_FALSE(reliableDoorbellActiveForNic(
      config, /*nicReliableDoorbellCapable=*/true));
  EXPECT_FALSE(reliableDoorbellActiveForNic(
      config, /*nicReliableDoorbellCapable=*/false));
}

TEST(MultiPeerIbTransportConfigTest, PeerMaterializationDefaultsOnDemand) {
  const MultipeerIbTransportConfig config;
  EXPECT_TRUE(config.ibLazyConnect);
}

// connectPeers() walks each rank's pending peers in peerMaterializationKey
// order and materializes them one at a time against a peer that must be doing
// the same. The checks below cover the two properties that buys: the schedule
// never stalls on a symmetric request graph, and a ring pairs up in a constant
// number of rounds rather than one per rank.

using PeerKey = int (*)(int myRank, int peerRank);

// Rendezvous rounds needed to materialize every edge of `adjacency` when each
// rank walks its peers in `key` order. Mirrors the connectPeers() loop for a
// single connect round from a clean state: an edge advances only once both ends
// have selected it, so a round is one doMaterializePeer() of wall clock.
// Returns -1 if the schedule stalls. Does not model the peerMaterialized_ skip
// across repeated calls, nor the failure/cleanup path.
// `key == nullptr` orders peers with the production `sortPendingPeers`, so a
// change to the shipped schedule moves these results. Passing a key overrides
// that, which is only used to model the pre-change rank ordering.
int rendezvousRounds(
    std::vector<std::vector<int>> adjacency,
    PeerKey key = nullptr) {
  const int nRanks = static_cast<int>(adjacency.size());
  int remainingEdges = 0;
  for (int rank = 0; rank < nRanks; ++rank) {
    auto& peers = adjacency[rank];
    if (key == nullptr) {
      sortPendingPeers(rank, peers);
    } else {
      std::sort(peers.begin(), peers.end(), [rank, key](int lhs, int rhs) {
        return key(rank, lhs) < key(rank, rhs);
      });
    }
    remainingEdges += static_cast<int>(peers.size());
  }
  remainingEdges /= 2;

  std::vector<std::size_t> head(nRanks, 0);
  int rounds = 0;
  while (remainingEdges > 0) {
    std::vector<std::pair<int, int>> paired;
    for (int a = 0; a < nRanks; ++a) {
      if (head[a] >= adjacency[a].size()) {
        continue;
      }
      const int b = adjacency[a][head[a]];
      if (b > a && head[b] < adjacency[b].size() &&
          adjacency[b][head[b]] == a) {
        paired.emplace_back(a, b);
      }
    }
    if (paired.empty()) {
      return -1;
    }
    for (const auto& [a, b] : paired) {
      ++head[a];
      ++head[b];
    }
    remainingEdges -= static_cast<int>(paired.size());
    ++rounds;
  }
  return rounds;
}

// queuePeerForMaterialization() rejects self-peers and dedups, so the models
// below must too -- a 2-rank ring has one edge, not two.
std::vector<std::vector<int>> fromEdges(
    int nRanks,
    const std::vector<std::pair<int, int>>& edges) {
  std::vector<std::set<int>> unique(nRanks);
  for (const auto& [a, b] : edges) {
    if (a != b) {
      unique[a].insert(b);
      unique[b].insert(a);
    }
  }
  std::vector<std::vector<int>> adjacency(nRanks);
  for (int rank = 0; rank < nRanks; ++rank) {
    adjacency[rank].assign(unique[rank].begin(), unique[rank].end());
  }
  return adjacency;
}

std::vector<std::vector<int>> ringAdjacency(int nRanks) {
  std::vector<std::pair<int, int>> edges;
  edges.reserve(nRanks);
  for (int rank = 0; rank < nRanks; ++rank) {
    edges.emplace_back(rank, (rank + 1) % nRanks);
  }
  return fromEdges(nRanks, edges);
}

std::vector<std::vector<int>> cliqueAdjacency(int nRanks) {
  std::vector<std::pair<int, int>> edges;
  edges.reserve(static_cast<std::size_t>(nRanks) * (nRanks - 1) / 2);
  for (int a = 0; a < nRanks; ++a) {
    for (int b = a + 1; b < nRanks; ++b) {
      edges.emplace_back(a, b);
    }
  }
  return fromEdges(nRanks, edges);
}

std::vector<std::vector<int>> binaryTreeAdjacency(int nRanks) {
  std::vector<std::pair<int, int>> edges;
  edges.reserve(nRanks);
  for (int rank = 1; rank < nRanks; ++rank) {
    edges.emplace_back(rank, (rank - 1) / 2);
  }
  return fromEdges(nRanks, edges);
}

std::vector<std::vector<int>>
randomSymmetricAdjacency(int nRanks, int degree, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0, nRanks - 1);
  std::vector<std::pair<int, int>> edges;
  edges.reserve(static_cast<std::size_t>(nRanks) * degree);
  for (int rank = 0; rank < nRanks; ++rank) {
    for (int i = 0; i < degree; ++i) {
      edges.emplace_back(rank, pick(rng));
    }
  }
  return fromEdges(nRanks, edges);
}

// The regression this ordering exists for. A ring is the shape every ring
// collective requests, and rank order pairs it off one edge at a time.
int rankOrderKey(int /*myRank*/, int peerRank) {
  return peerRank;
}

TEST(MultiPeerIbTransportConfigTest, RankOrderSerializesRing) {
  for (const int nRanks : {8, 64, 256, 644}) {
    EXPECT_EQ(rendezvousRounds(ringAdjacency(nRanks), rankOrderKey), nRanks - 1)
        << "nRanks=" << nRanks;
  }
}

// Under peerMaterializationKey the same ring costs a constant instead: two
// rounds when the ring can be 2-edge-coloured, three when it cannot.
TEST(MultiPeerIbTransportConfigTest, RingPairsUpInConstantRounds) {
  for (const int nRanks : {8, 64, 256, 644, 9, 65, 645}) {
    const int expected = (nRanks % 2 == 0) ? 2 : 3;
    EXPECT_EQ(rendezvousRounds(ringAdjacency(nRanks)), expected)
        << "nRanks=" << nRanks;
  }
}

// The invariant the ordering has to preserve: given the symmetric request graph
// materializePeers requires, every edge eventually pairs up. Covers the other
// shapes comms requests -- cliques from the direct algorithms, trees from
// AllReduceFusedTree -- plus irregular graphs the fixed topologies do not
// reach.
TEST(MultiPeerIbTransportConfigTest, SymmetricRequestGraphsNeverStall) {
  for (const int nRanks : {2, 3, 8, 33, 64}) {
    EXPECT_GT(rendezvousRounds(ringAdjacency(nRanks)), 0)
        << "ring nRanks=" << nRanks;
    EXPECT_GT(rendezvousRounds(cliqueAdjacency(nRanks)), 0)
        << "clique nRanks=" << nRanks;
    EXPECT_GT(rendezvousRounds(binaryTreeAdjacency(nRanks)), 0)
        << "tree nRanks=" << nRanks;
  }
  for (uint32_t seed = 0; seed < 200; ++seed) {
    EXPECT_GT(
        rendezvousRounds(
            randomSymmetricAdjacency(/*nRanks=*/32, /*degree=*/3, seed)),
        0)
        << "random seed=" << seed;
  }
}

} // namespace
} // namespace comms::prims
