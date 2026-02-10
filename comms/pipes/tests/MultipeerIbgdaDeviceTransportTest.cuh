// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::tests {

// Wrapper function to launch test kernel (defined in .cu, called from .cc)
// Tests the indexToRank mapping logic on device
void runTestRankMappingKernel(
    int myRank,
    int nRanks,
    int* d_results,
    int* d_expected,
    int numTestCases,
    bool* d_success);

} // namespace comms::pipes::tests
