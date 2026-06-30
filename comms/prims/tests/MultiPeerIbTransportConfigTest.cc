// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

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

} // namespace
} // namespace comms::prims
