// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
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

MultimemNvlTransportConfig makeConfig(
    std::size_t dataBufferSize,
    uint32_t userSignalCount = 1,
    std::size_t pipelineDepth = 0,
    std::size_t maxGroups = 0) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = dataBufferSize;
  config.userSignalCount = userSignalCount;
  config.pipelineDepth = pipelineDepth;
  config.maxGroups = maxGroups;
  return config;
}

} // namespace

TEST(MultimemNvlTransportConfigTest, DerivesUserOnlyConfiguration) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(makeConfig(256), 4), 0);
}

TEST(MultimemNvlTransportConfigTest, DerivesStagingOnlyConfiguration) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(256, 0, 2, 2), 4),
      48);
}

TEST(MultimemNvlTransportConfigTest, DerivesMixedConfiguration) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(256, 7, 2, 2), 4),
      48);
}

TEST(MultimemNvlTransportConfigTest, RejectsPartialStagingGeometry) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(256, 1, 2, 0), 4),
      std::nullopt);
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(256, 1, 0, 2), 4),
      std::nullopt);
}

TEST(MultimemNvlTransportConfigTest, RejectsInvalidRankCount) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(256, 1, 1, 1), 0),
      std::nullopt);
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(std::numeric_limits<std::size_t>::max(), 1, 1, 1),
          std::numeric_limits<int>::max()),
      std::nullopt);
}

TEST(MultimemNvlTransportConfigTest, RejectsInsufficientDataCapacity) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(255, 1, 2, 2), 4),
      std::nullopt);
}

TEST(MultimemNvlTransportConfigTest, RejectsInternalSignalOverflow) {
  constexpr std::size_t kRanks = 4;
  constexpr std::size_t kSignalsPerLane =
      multimem_staging_signals_per_lane(static_cast<uint32_t>(kRanks));
  const std::size_t pipelineDepth =
      std::numeric_limits<int>::max() / kSignalsPerLane + 1;
  const std::size_t requiredDataBytes = pipelineDepth * kRanks * 16;
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(requiredDataBytes, 1, pipelineDepth, 1), kRanks),
      std::nullopt);
}

TEST(MultimemNvlTransportConfigTest, RejectsTotalSignalOverflow) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(64, std::numeric_limits<int>::max(), 1, 1), 4),
      std::nullopt);
}

TEST(MultimemNvlTransportConfigTest, RejectsUserSignalCountOutsideSignedRange) {
  EXPECT_EQ(
      detail::checked_multimem_internal_signal_count(
          makeConfig(64, std::numeric_limits<uint32_t>::max(), 0, 0), 4),
      std::nullopt);
}

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

// These config-validation guards run in the primary ctor body before
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

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsDefaultZeroSignalCount) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 1024;
  EXPECT_EQ(config.userSignalCount, 0);
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
    PrimaryCtorRejectsTotalSignalCountOverflow) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = 64;
  config.userSignalCount = std::numeric_limits<int>::max();
  config.pipelineDepth = 1;
  config.maxGroups = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "invalid staging geometry or capacity");
}

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsDataBufferAlignmentOverflow) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize = std::numeric_limits<std::size_t>::max();
  config.userSignalCount = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "dataBufferSize alignment overflows");
}

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsCombinedAllocationSizeOverflow) {
  MultimemNvlTransportConfig config{};
  config.dataBufferSize =
      std::numeric_limits<std::size_t>::max() - (alignof(SignalState) - 1);
  config.userSignalCount = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "combined allocation size overflows");
}

} // namespace comms::prims::tests
