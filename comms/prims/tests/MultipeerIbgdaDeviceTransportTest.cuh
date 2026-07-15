// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <vector>

namespace comms::prims::tests {

// Wrapper function to launch test kernel (defined in .cu, called from .cc)
// Tests the indexToRank mapping logic on device
void runTestRankMappingKernel(
    int myRank,
    int nRanks,
    int* d_results,
    int* d_expected,
    int numTestCases,
    bool* d_success);

// Which NIC and physical QP slot a group's put was mapped to.
struct V4LaneResult {
  int nic_id;
  int qp_slot_per_nic;
};

// Runs select_put_lane() for every (block, group) over a
// P2pIbgdaTransportDevice built with fake NIC resources, returning the chosen
// lane per group_id (outResults[group_id]). Hardware-free; exercises the V4
// block-owned-QP + group-staggered-NIC mapping. Uses qpDirectionCount =
// kIbDirections.
void runV4LaneMappingTest(
    int numNics,
    int numBlocks,
    int groupsPerBlock,
    int qpsPerConn,
    std::vector<V4LaneResult>& outResults);

} // namespace comms::prims::tests
