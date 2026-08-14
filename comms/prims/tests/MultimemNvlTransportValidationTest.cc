// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
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
    std::size_t perChannelSize,
    uint32_t userSignalCount = 0,
    std::size_t pipelineDepth = 0,
    std::size_t maxChannels = 1,
    std::size_t maxBlocks = 0) {
  return make_multimem_nvl_transport_config({
      .perChannelSize = perChannelSize,
      .pipelineDepth = pipelineDepth,
      .maxChannels = maxChannels,
      .maxBlocks = maxBlocks,
      .userSignalCount = userSignalCount,
  });
}

} // namespace

TEST(MultimemNvlTransportConfigTest, DerivesUserOnlyConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(256, 1), 4);
  EXPECT_TRUE(validation);
  EXPECT_EQ(validation.internalSignalCount, 0);
}

TEST(
    MultimemNvlTransportConfigTest,
    DerivesDataAndSignalsFromChannelCapacityAndIndependentBlockLimit) {
  const auto config = make_multimem_nvl_transport_config({
      .perChannelSize = 1024,
      .pipelineDepth = 2,
      .maxChannels = 8,
      .maxBlocks = 3,
      .userSignalCount = 1,
  });

  const auto validation = validate_multimem_nvl_transport_config(config, 4);

  ASSERT_TRUE(validation);
  EXPECT_EQ(validation.dataBufferSize, 8192);
  EXPECT_EQ(validation.internalSignalCount, 160);
}

TEST(MultimemNvlTransportConfigTest, DefinesSignalGeometryPerChannel) {
  EXPECT_EQ(detail::kMultimemSignalsPerPeer, 3);
  EXPECT_EQ(detail::kMultimemSignalsPerLane, 4);
  EXPECT_EQ(multimem_staging_signals_per_channel(4, 2), 20);
  constexpr uint64_t kMax = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(multimem_staging_signals_per_channel(kMax, kMax), 7 * kMax);
}

TEST(MultimemNvlTransportConfigTest, RejectsDataBufferMultiplicationOverflow) {
  const auto config = make_multimem_nvl_transport_config({
      .perChannelSize = std::numeric_limits<std::size_t>::max(),
      .pipelineDepth = 0,
      .maxChannels = 2,
      .maxBlocks = 0,
      .userSignalCount = 1,
  });

  EXPECT_EQ(
      validate_multimem_nvl_transport_config(config, 4).errorMessage,
      "per-channel size times maximum channels overflows");
}

TEST(MultimemNvlTransportConfigTest, PreservesDefaultUserSignalCount) {
  EXPECT_EQ(MultimemNvlTransportConfig{}.userSignalCount, 0);
}

TEST(MultimemNvlTransportConfigTest, RejectsLegacyPositionalConstruction) {
  EXPECT_FALSE((std::is_constructible_v<
                MultimemNvlTransportConfig,
                std::size_t,
                uint32_t,
                std::size_t,
                std::size_t>));
}

TEST(MultimemNvlTransportDeviceTest, PreservesLegacyPositionalRankFields) {
  const MultimemNvlTransportDevice device{
      nullptr, nullptr, {}, {}, {}, {}, 0, 2, 4};
  EXPECT_EQ(device.nvlRank, 2);
  EXPECT_EQ(device.nvlRanks, 4);
}

TEST(MultimemNvlTransportConfigTest, DerivesStagingOnlyConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(128, 0, 2, 2, 2), 4);
  EXPECT_TRUE(validation);
  EXPECT_EQ(validation.internalSignalCount, 40);
}

TEST(MultimemNvlTransportConfigTest, DerivesMixedConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(128, 7, 2, 2, 2), 4);
  EXPECT_TRUE(validation);
  EXPECT_EQ(validation.internalSignalCount, 40);
}

TEST(MultimemNvlTransportConfigTest, RejectsPartialStagingGeometry) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(256, 1, 2, 1, 0), 4)
          .errorMessage,
      "pipeline depth and maximum blocks must both be zero or non-zero");
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(128, 1, 0, 2, 2), 4)
          .errorMessage,
      "pipeline depth and maximum blocks must both be zero or non-zero");
}

TEST(MultimemNvlTransportConfigTest, RejectsInvalidRankCount) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(256, 1, 1, 1, 1), 0)
          .errorMessage,
      "NVL rank count must be positive");
}

TEST(MultimemNvlTransportConfigTest, RejectsMissingPerChannelSize) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(0, 1), 4).errorMessage,
      "per-channel size must be non-zero");
}

TEST(MultimemNvlTransportConfigTest, RejectsMissingMaxChannels) {
  const auto config = make_multimem_nvl_transport_config({
      .perChannelSize = 256,
      .pipelineDepth = 0,
      .maxChannels = 0,
      .maxBlocks = 0,
      .userSignalCount = 1,
  });
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(config, 4).errorMessage,
      "maximum channels must be non-zero");
}

TEST(MultimemNvlTransportConfigTest, AcceptsSignalFreeConfiguration) {
  const auto validation =
      validate_multimem_nvl_transport_config(makeConfig(256, 0), 4);

  ASSERT_TRUE(validation);
  EXPECT_EQ(validation.dataBufferSize, 256);
  EXPECT_EQ(validation.signalsPerChannel, 0);
  EXPECT_EQ(validation.internalSignalCount, 0);
  EXPECT_EQ(validation.signalRegionOffset, 256);
  EXPECT_EQ(validation.backingAllocationSize, 256);
}

TEST(MultimemNvlTransportConfigTest, RejectsMaxBlocksAboveMaxChannels) {
  const auto config = make_multimem_nvl_transport_config({
      .perChannelSize = 64,
      .pipelineDepth = 1,
      .maxChannels = 2,
      .maxBlocks = 3,
      .userSignalCount = 1,
  });

  EXPECT_EQ(
      validate_multimem_nvl_transport_config(config, 4).errorMessage,
      "maximum blocks must not exceed maximum channels");
}

TEST(MultimemNvlTransportConfigTest, RejectsInsufficientDataCapacity) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(64, 1, 2, 2, 2), 4)
          .errorMessage,
      "data buffer is too small for the staging geometry");
}

TEST(MultimemNvlTransportConfigTest, RejectsMisalignedPerChannelSize) {
  const auto config = make_multimem_nvl_transport_config({
      .perChannelSize = 48,
      .pipelineDepth = 2,
      .maxChannels = 4,
      .maxBlocks = 1,
      .userSignalCount = 1,
  });
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(config, 4).errorMessage,
      "per-channel size must be divisible by pipeline depth times 16");
}

TEST(MultimemNvlTransportConfigTest, RejectsInternalSignalOverflow) {
  constexpr std::size_t kRanks = 4;
  constexpr std::size_t kSignalsPerPeer = detail::kMultimemSignalsPerPeer;
  constexpr std::size_t kSignalsPerLane = detail::kMultimemSignalsPerLane;
  const std::size_t pipelineDepth =
      (std::numeric_limits<int>::max() - kRanks * kSignalsPerPeer) /
          kSignalsPerLane +
      1;
  const std::size_t requiredDataBytes = pipelineDepth * kRanks * 16;
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(requiredDataBytes, 1, pipelineDepth, 1, 1), kRanks)
          .errorMessage,
      "signal count exceeds INT_MAX");
}

TEST(MultimemNvlTransportConfigTest, RejectsPipelineDepthOutsideDeviceRange) {
  constexpr std::size_t kOutsideDeviceRange =
      static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(
              /*perChannelSize=*/16,
              /*userSignalCount=*/1,
              /*pipelineDepth=*/kOutsideDeviceRange,
              /*maxChannels=*/1,
              /*maxBlocks=*/1),
          4)
          .errorMessage,
      "transport geometry exceeds UINT32_MAX");
}

TEST(MultimemNvlTransportConfigTest, RejectsMaxChannelsOutsideDeviceRange) {
  constexpr std::size_t kOutsideDeviceRange =
      static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(
              /*perChannelSize=*/16,
              /*userSignalCount=*/1,
              /*pipelineDepth=*/1,
              /*maxChannels=*/kOutsideDeviceRange,
              /*maxBlocks=*/1),
          4)
          .errorMessage,
      "transport geometry exceeds UINT32_MAX");
}

TEST(MultimemNvlTransportConfigTest, RejectsTotalSignalOverflow) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(64, std::numeric_limits<int>::max(), 1, 1, 1), 4)
          .errorMessage,
      "signal count exceeds INT_MAX");
}

TEST(MultimemNvlTransportConfigTest, RejectsDataAlignmentOverflow) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(std::numeric_limits<std::size_t>::max(), 1), 4)
          .errorMessage,
      "combined data and signal allocation size overflows");
}

TEST(MultimemNvlTransportConfigTest, RejectsCombinedAllocationOverflow) {
  constexpr auto kAlignment = detail::kMultimemSignalAlignment;
  const auto dataBufferSize =
      std::numeric_limits<std::size_t>::max() - (kAlignment - 1);
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(makeConfig(dataBufferSize, 1), 4)
          .errorMessage,
      "combined data and signal allocation size overflows");
}

TEST(MultimemNvlTransportConfigTest, RejectsUserSignalCountOutsideSignedRange) {
  EXPECT_EQ(
      validate_multimem_nvl_transport_config(
          makeConfig(64, std::numeric_limits<uint32_t>::max()), 4)
          .errorMessage,
      "signal count exceeds INT_MAX");
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
  config.perChannelSize = 1024;
  config.maxChannels = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*nvlRank=*/-1, /*nvlRanks=*/4, /*bootstrap=*/nullptr, config);
      },
      "nvlRank must be in [0, nvlRanks)");
}

TEST(MultimemNvlTransportValidationTest, CompatCtorRejectsOutOfRangeNvlRank) {
  MultimemNvlTransportConfig config{};
  config.perChannelSize = 1024;
  config.maxChannels = 1;
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

TEST(MultimemNvlTransportValidationTest, PrimaryCtorRejectsZeroPerChannelSize) {
  MultimemNvlTransportConfig config{};
  config.perChannelSize = 0;
  config.maxChannels = 1;
  config.userSignalCount = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "per-channel size must be non-zero");
}

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsTotalSignalCountOverflow) {
  MultimemNvlTransportConfig config{};
  config.perChannelSize = 64;
  config.userSignalCount = std::numeric_limits<int>::max();
  config.pipelineDepth = 1;
  config.maxChannels = 1;
  config.maxBlocks = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "signal count exceeds INT_MAX");
}

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsDataBufferAlignmentOverflow) {
  MultimemNvlTransportConfig config{};
  config.perChannelSize = std::numeric_limits<std::size_t>::max();
  config.maxChannels = 1;
  config.userSignalCount = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "combined data and signal allocation size overflows");
}

TEST(
    MultimemNvlTransportValidationTest,
    PrimaryCtorRejectsCombinedAllocationSizeOverflow) {
  MultimemNvlTransportConfig config{};
  config.perChannelSize =
      std::numeric_limits<std::size_t>::max() - (alignof(SignalState) - 1);
  config.maxChannels = 1;
  config.userSignalCount = 1;
  expectRuntimeErrorContains(
      [&] {
        MultimemNvlTransport(
            /*bootstrap=*/nullptr,
            /*commRank=*/0,
            /*nvlRankToCommRank=*/std::vector<int>{0, 1, 2, 3},
            config);
      },
      "combined data and signal allocation size overflows");
}

} // namespace comms::prims::tests
