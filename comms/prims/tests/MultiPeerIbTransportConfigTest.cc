// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <array>
#include <cerrno>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/transport/MultiPeerIbTransport.h"
#include "comms/prims/transport/MultiPeerTransport.h"

using ::testing::_;
using ::testing::StrictMock;

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

TEST(MultiPeerIbTransportConfigTest, PeerMaterializationDefaultsOnDemand) {
  const MultipeerIbTransportConfig config;
  EXPECT_TRUE(config.ibLazyConnect);
}

TEST(MultiPeerIbTransportConfigTest, LazyChannelsDefaultOff) {
  const MultipeerIbTransportConfig config;
  EXPECT_FALSE(config.lazyChannels);
}

TEST(MultiPeerTransportInitTest, MatchingRecordsSucceed) {
  const detail::ChannelProtocolRecord record{
      .mode = detail::PrimsChannelMode::kLazyPrefix,
      .channelCapacity = 64,
  };
  const std::array records{record, record};
  EXPECT_NO_THROW(detail::validateChannelProtocolRecords(records));
}

TEST(MultiPeerTransportInitTest, MismatchedChannelModesFail) {
  detail::ChannelProtocolRecord eager;
  auto lazy = eager;
  lazy.mode = detail::PrimsChannelMode::kLazyPrefix;
  const std::array records{eager, lazy};
  EXPECT_THROW(
      detail::validateChannelProtocolRecords(records), std::runtime_error);
}

TEST(MultiPeerTransportInitTest, MismatchedChannelCapacitiesFail) {
  detail::ChannelProtocolRecord smaller;
  smaller.channelCapacity = 4;
  auto larger = smaller;
  larger.channelCapacity = 8;
  const std::array records{smaller, larger};
  EXPECT_THROW(
      detail::validateChannelProtocolRecords(records), std::runtime_error);
}

TEST(MultiPeerTransportInitTest, SymmetricRoutesSucceed) {
  using Route = detail::PrimsTransportRoute;
  const std::array routes{
      Route::kSelf,
      Route::kIbgda,
      Route::kIbgda,
      Route::kSelf,
  };
  EXPECT_NO_THROW(detail::validatePrimsTransportRoutes(routes, 2));
}

TEST(MultiPeerTransportInitTest, AsymmetricRoutesFail) {
  using Route = detail::PrimsTransportRoute;
  const std::array routes{
      Route::kSelf,
      Route::kNvl,
      Route::kIbgda,
      Route::kSelf,
  };
  EXPECT_THROW(
      detail::validatePrimsTransportRoutes(routes, 2), std::runtime_error);
}

TEST(MultiPeerTransportInitTest, AllGatherFailureFailsInitialization) {
  StrictMock<meta::comms::testing::MockBootstrap> bootstrap;
  EXPECT_CALL(
      bootstrap,
      allGather(
          _, static_cast<int>(sizeof(detail::ChannelProtocolRecord)), 0, 2))
      .WillOnce(
          [](void*, int, int, int) { return folly::makeSemiFuture(EIO); });

  EXPECT_THROW(
      detail::exchangeAndValidateChannelProtocol(
          bootstrap, 0, 2, detail::ChannelProtocolRecord{}),
      std::runtime_error);
}

class TestLazyChannelTransport
    : public MultiPeerIbTransport<TestLazyChannelTransport> {
 public:
  struct MaterializedRange {
    int peerRank;
    uint32_t beginChannel;
    uint32_t endChannel;

    bool operator==(const MaterializedRange&) const = default;
  };

  static constexpr bool supportsLazyChannelPrefixGrowth() {
    return true;
  }

  static constexpr PeerChannelBackend peerChannelBackend() {
    return PeerChannelBackend::kIbgda;
  }

  explicit TestLazyChannelTransport(bool lazyChannels)
      : MultiPeerIbTransport<TestLazyChannelTransport>(
            /*myRank=*/0,
            /*nRanks=*/2,
            std::make_shared<
                ::testing::NiceMock<meta::comms::testing::MockBootstrap>>(),
            makeConfig(lazyChannels)) {
    // This fake isolates local target and watermark behavior.
    channelRangeProtocolEnabled_ = false;
  }

  void materializePeerChannelRange(
      int peerRank,
      uint32_t beginChannel,
      uint32_t endChannel) {
    observedWatermarks.push_back(
        materializedChannels_[rankToPeerIndex(peerRank)]);
    materializedRanges.push_back({peerRank, beginChannel, endChannel});
    if (failNextMaterialization) {
      throw std::runtime_error("injected materialization failure");
    }
  }

  uint32_t rawMaterializedChannelCount(int peerRank) const {
    return materializedChannels_[rankToPeerIndex(peerRank)];
  }

  int configuredQpsPerConnection() const {
    return config_.qpsPerConnection;
  }

  std::vector<MaterializedRange> materializedRanges;
  std::vector<uint32_t> observedWatermarks;
  bool failNextMaterialization{false};

 private:
  static MultipeerIbTransportConfig makeConfig(bool lazyChannels) {
    MultipeerIbTransportConfig config;
    config.gpuNicMap[0] = {"test_nic"};
    config.maxGroups = 8;
    config.qpsPerBlockPerNic = 2;
    config.lazyChannels = lazyChannels;
    return config;
  }
};

TEST(MultiPeerIbTransportConfigTest, LegacyQpGeometryIsNormalized) {
  const TestLazyChannelTransport transport(/*lazyChannels=*/false);
  EXPECT_EQ(8, transport.channelCapacity());
  EXPECT_EQ(2, transport.configuredQpsPerConnection());
}

TEST(MultiPeerIbTransportConfigTest, LazyChannelsMaterializeMissingPrefix) {
  TestLazyChannelTransport transport(/*lazyChannels=*/true);

  transport.queuePeerForMaterialization(/*peerRank=*/1, /*targetChannels=*/1);
  transport.connectPeers();
  transport.queuePeerForMaterialization(/*peerRank=*/1, /*targetChannels=*/3);
  transport.connectPeers();
  transport.queuePeerForMaterialization(/*peerRank=*/1, /*targetChannels=*/2);
  transport.connectPeers();

  const std::vector<TestLazyChannelTransport::MaterializedRange> expected{
      {1, 0, 1},
      {1, 1, 4},
  };
  EXPECT_EQ(expected, transport.materializedRanges);
  EXPECT_EQ((std::vector<uint32_t>{0, 1}), transport.observedWatermarks);
  EXPECT_EQ(4, transport.materializedChannelCount(/*peerRank=*/1));
}

TEST(MultiPeerIbTransportConfigTest, EagerChannelsMaterializeCapacity) {
  TestLazyChannelTransport transport(/*lazyChannels=*/false);

  transport.queuePeerForMaterialization(/*peerRank=*/1, /*targetChannels=*/1);
  transport.connectPeers();

  const std::vector<TestLazyChannelTransport::MaterializedRange> expected{
      {1, 0, 8},
  };
  EXPECT_EQ(expected, transport.materializedRanges);
  EXPECT_EQ(8, transport.materializedChannelCount(/*peerRank=*/1));
}

TEST(
    MultiPeerIbTransportConfigTest,
    FailedGrowthDoesNotPublishAndPoisonsTransport) {
  TestLazyChannelTransport transport(/*lazyChannels=*/true);
  transport.queuePeerForMaterialization(/*peerRank=*/1, /*targetChannels=*/1);
  transport.connectPeers();
  transport.failNextMaterialization = true;

  transport.queuePeerForMaterialization(/*peerRank=*/1, /*targetChannels=*/4);
  EXPECT_THROW(transport.connectPeers(), std::runtime_error);

  EXPECT_EQ(1, transport.rawMaterializedChannelCount(/*peerRank=*/1));
  EXPECT_THROW(
      transport.materializedChannelCount(/*peerRank=*/1), std::runtime_error);
  EXPECT_THROW(
      transport.queuePeerForMaterialization(
          /*peerRank=*/1, /*targetChannels=*/2),
      std::runtime_error);
}

} // namespace
} // namespace comms::prims
