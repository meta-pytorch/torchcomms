// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/prims/transport/nvl/MultimemNvlTransport.h"

namespace comms::prims::tests {

namespace {

template <typename Fn>
void expectRuntimeErrorContains(Fn&& fn, const std::string& expected) {
  try {
    std::forward<Fn>(fn)();
    FAIL() << "expected std::runtime_error containing " << expected;
  } catch (const std::runtime_error& ex) {
    EXPECT_NE(std::string(ex.what()).find(expected), std::string::npos)
        << ex.what();
  }
}

} // namespace

// validateRankMap runs before any GPU access in the constructor; these cases
// must reject bad topologies on CPU-only hosts.

TEST(MultimemNvlTransportValidationTest, AcceptsValidRankMap) {
  EXPECT_NO_THROW(MultimemNvlTransport::validateRankMap(2, {0, 1, 2, 3}));
}

TEST(MultimemNvlTransportValidationTest, RejectsEmptyRankMap) {
  expectRuntimeErrorContains(
      [] { MultimemNvlTransport::validateRankMap(0, {}); },
      "nvlRankToCommRank must be non-empty");
}

TEST(MultimemNvlTransportValidationTest, RejectsNegativeRankInMap) {
  expectRuntimeErrorContains(
      [] { MultimemNvlTransport::validateRankMap(0, {0, -1}); },
      "contains a negative rank");
}

TEST(MultimemNvlTransportValidationTest, RejectsDuplicateRankInMap) {
  expectRuntimeErrorContains(
      [] { MultimemNvlTransport::validateRankMap(0, {0, 1, 0}); },
      "contains duplicate ranks");
}

TEST(MultimemNvlTransportValidationTest, RejectsMissingCommRank) {
  expectRuntimeErrorContains(
      [] { MultimemNvlTransport::validateRankMap(7, {0, 1, 2}); },
      "commRank must appear in nvlRankToCommRank");
}

TEST(MultimemNvlTransportValidationTest, IsEligibleRequiresMoreThanTwoRanks) {
  // The cudaDevice check needs a valid device; pass -1 to force the
  // isMultimemSupported branch to short-circuit to false. The point of this
  // test is the nRanks gate.
  EXPECT_FALSE(MultimemNvlTransport::isEligible(2, -1));
  EXPECT_FALSE(MultimemNvlTransport::isEligible(1, -1));
}

// Compat constructor: the identity-map path prechecks nvlRank bounds before
// delegating so misuse surfaces as a targeted message instead of the generic
// "commRank must appear in nvlRankToCommRank" from validateRankMap. The
// precheck runs in the delegating-ctor argument list (before cudaGetDevice),
// so it is exercisable on CPU-only hosts with a null bootstrap.

TEST(MultimemNvlTransportValidationTest, CompatCtorRejectsNegativeNvlRank) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 1024;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*nvlRank=*/-1, /*nvlRanks=*/4, /*bootstrap=*/nullptr, config);
      },
      "nvlRank must be in [0, nvlRanks)");
}

TEST(MultimemNvlTransportValidationTest, CompatCtorRejectsOutOfRangeNvlRank) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 1024;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*nvlRank=*/4, /*nvlRanks=*/4, /*bootstrap=*/nullptr, config);
      },
      "nvlRank must be in [0, nvlRanks)");
}

// Config-validation guards. All three run in the primary ctor body BEFORE
// cudaGetDevice, so they are exercisable on CPU-only hosts with a null
// bootstrap: none of the code paths past these throws is reached.

TEST(MultimemNvlTransportValidationTest, PrimaryCtorRejectsZeroDataBufferSize) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 0; // triggers the guard
  config.userSignalCount = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "dataBufferSize must be non-zero");
}

TEST(MultimemNvlTransportValidationTest, PrimaryCtorRejectsZeroSignalCount) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 1024;
  config.userSignalCount = 0;
  config.internalSignalCount = 0;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "at least one signal slot is required");
}

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsSignalCountOverflow) {
  // Sum of two uint32_t maxes overflows the int32 clamp used downstream.
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 1024;
  config.userSignalCount = std::numeric_limits<uint32_t>::max();
  config.internalSignalCount = std::numeric_limits<uint32_t>::max();
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "signalCount too large");
}

} // namespace comms::prims::tests
