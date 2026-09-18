// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Integration test for MultiTransportFactory::selectNics.
/// Requires real GPUs, NVML, and ibverbs — not for CI without GPU hardware.

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <string>

#include <gtest/gtest.h>

namespace uniflow {

// Named MultiTransportFactoryTest to match the friend declaration in
// MultiTransport.h, granting access to private members.
// All private member accesses must go through fixture helper methods since
// TEST_F creates derived classes that do not inherit friend access.
class MultiTransportFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    topo_ = &sharedTopology();
    if (!topo_->available()) {
      GTEST_SKIP() << "Topology not available";
    }
  }

  // Create a lightweight factory for calling selectNics without constructing
  // real transport backends. Uses the private vector<factory> constructor.
  std::unique_ptr<MultiTransportFactory> makeTestFactory(
      int deviceId,
      NicFilter nicFilter = NicFilter()) {
    auto factory =
        std::unique_ptr<MultiTransportFactory>(new MultiTransportFactory({}));
    factory->deviceId_ = deviceId;
    factory->options_.nicFilter = std::move(nicFilter);
    return factory;
  }

  std::vector<std::string> callSelectNics(MultiTransportFactory& factory) {
    return factory.selectNics();
  }

  size_t factoryCount(const MultiTransportFactory& factory) {
    return factory.factories_.size();
  }

  TransportType factoryTransportType(
      const MultiTransportFactory& factory,
      size_t idx) {
    return factory.factories_[idx]->transportType();
  }

  Status isPlatformSupported(
      std::string_view platform = "",
      const NicFilter& nicFilter = NicFilter()) {
    size_t nicCount = topo_->nicCount();
    if (nicCount == 0) {
      return Err(ErrCode::ResourceExhausted, "No NICs available");
    }

    size_t matchCount = 0;
    for (size_t i = 0; i < nicCount; ++i) {
      if (topo_->filterNic(static_cast<int>(i), nicFilter)) {
        ++matchCount;
      }
    }

    if (matchCount == 0) {
      return Err(
          ErrCode::ResourceExhausted, "No NICs matching filter available");
    }

    if (!platform.empty()) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return Err(ErrCode::DriverError, "cudaGetDeviceProperties failed");
      }
      std::string gpuName(prop.name);
      if (gpuName.find(platform) == std::string::npos) {
        std::string errMsg =
            "Not " + std::string(platform) + " (got " + gpuName + ")";
        return Err(ErrCode::ResourceExhausted, errMsg);
      }
    }

    return Ok();
  }

  Topology* topo_{nullptr};
};

// --- selectNics tests ---

TEST_F(MultiTransportFactoryTest, SelectNicsGpuH100) {
  NicFilter filter(
      "mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1");
  auto st = isPlatformSupported("NVIDIA H100", filter);
  if (!st) {
    GTEST_SKIP() << st.error().message();
  }

  if (topo_->gpuCount() != 8) {
    GTEST_SKIP() << "GPU count incorrect";
  }

  const std::vector<std::vector<std::string>> expectedNics{
      {"mlx5_0"},
      {"mlx5_3"},
      {"mlx5_4"},
      {"mlx5_5"},
      {"mlx5_6"},
      {"mlx5_9"},
      {"mlx5_10"},
      {"mlx5_11"},
  };

  for (size_t i = 0; i < topo_->gpuCount(); ++i) {
    auto factory = makeTestFactory(static_cast<int>(i), filter);
    auto nics = callSelectNics(*factory);
    EXPECT_EQ(nics, expectedNics[i]);
  }
}

TEST_F(MultiTransportFactoryTest, SelectNicsGpuGB200) {
  NicFilter filter("mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1");
  auto st = isPlatformSupported("NVIDIA GB200", filter);
  if (!st) {
    GTEST_SKIP() << st.error().message();
  }
  if (topo_->gpuCount() != 2) {
    GTEST_SKIP() << "GPU count incorrect";
  }

  const std::vector<std::vector<std::string>> expectedNics{
      {"mlx5_0", "mlx5_1"},
      {"mlx5_3", "mlx5_4"},
  };

  for (size_t i = 0; i < topo_->gpuCount(); ++i) {
    auto factory = makeTestFactory(static_cast<int>(i), filter);
    auto nics = callSelectNics(*factory);
    EXPECT_EQ(nics, expectedNics[i]);
  }
}

// --- Constructor integration tests ---
#ifdef UNIFLOW_ENABLE_TCP_TRANSPORT
TEST_F(MultiTransportFactoryTest, ConstructorCpuCreatesTcpAndOptionalRdma) {
  const auto rdmaSupported = isPlatformSupported("").hasValue();

  // TCP is opt-in.
  MultiTransportFactoryOptions opts;
  opts.enableTcp = true;
  MultiTransportFactory factory(-1, opts);
  ASSERT_GE(factoryCount(factory), 1u);
  EXPECT_EQ(
      factoryTransportType(factory, factoryCount(factory) - 1),
      TransportType::TCP);
  if (rdmaSupported) {
    EXPECT_EQ(factoryTransportType(factory, 0), TransportType::RDMA);
    EXPECT_EQ(factoryCount(factory), 2u);
  }
}

// TCP is skipped when only a loopback bind address is available, so a loopback-
// bound TCP transport cannot break cross-host connections.
TEST_F(MultiTransportFactoryTest, ConstructorOmitsTcpOnLoopbackBindHost) {
  MultiTransportFactoryOptions opts;
  opts.tcpBindHost = "127.0.0.1";
  MultiTransportFactory factory(-1, opts);
  for (size_t i = 0; i < factoryCount(factory); ++i) {
    EXPECT_NE(factoryTransportType(factory, i), TransportType::TCP);
  }
}

// TCP auto-registers (without enableTcp) when a routable bind address resolves.
TEST_F(MultiTransportFactoryTest, ConstructorRegistersTcpWhenRoutable) {
  MultiTransportFactoryOptions opts;
  opts.tcpBindHost =
      "2401:db00:eef0:1120:3520:0:0:1"; // routable (non-loopback)
  MultiTransportFactory factory(-1, opts);
  bool hasTcp = false;
  for (size_t i = 0; i < factoryCount(factory); ++i) {
    if (factoryTransportType(factory, i) == TransportType::TCP) {
      hasTcp = true;
    }
  }
  EXPECT_TRUE(hasTcp);
}

TEST_F(MultiTransportFactoryTest, ConstructorGpuCreatesNvlinkAndOptionalRdma) {
  if (topo_->gpuCount() == 0) {
    GTEST_SKIP() << "No GPUs available";
  }

  MultiTransportFactoryOptions opts;
  opts.enableTcp = true;
  MultiTransportFactory factory(0, opts);
  ASSERT_GE(factoryCount(factory), 1u);
  EXPECT_EQ(
      factoryTransportType(factory, factoryCount(factory) - 1),
      TransportType::TCP);
#if defined(__HIP_PLATFORM_AMD__)
  // Registration order is [interconnect tier?, RDMA?, TCP]; each earlier tier
  // is present only if its hardware is. On AMD the interconnect tier is
  // P2P/XGMI (advertised as TransportType::NVLink), registered only on all-XGMI
  // nodes; RDMA is registered only when a NIC is present. TCP is last (asserted
  // above). Walk the earlier tiers by index, gating each on availability -- a
  // node may have P2P but no RDMA NIC (or vice versa), so don't assume a fixed
  // RDMA index.
  const bool nvlinkSupported =
      !MultiTransportFactory::supported(TransportType::NVLink).hasError();
  const bool rdmaSupported = isPlatformSupported("").hasValue();
  size_t idx = 0;
  if (nvlinkSupported) {
    ASSERT_GT(factoryCount(factory), idx);
    EXPECT_EQ(factoryTransportType(factory, idx++), TransportType::NVLink);
  }
  if (rdmaSupported) {
    ASSERT_GT(factoryCount(factory), idx);
    EXPECT_EQ(factoryTransportType(factory, idx++), TransportType::RDMA);
  }
#else
  if (factoryCount(factory) > 1) {
    EXPECT_EQ(factoryTransportType(factory, 0), TransportType::NVLink);
  }
  if (factoryCount(factory) > 2) {
    EXPECT_EQ(factoryTransportType(factory, 1), TransportType::RDMA);
  }
#endif
}

TEST_F(MultiTransportFactoryTest, ConstructorRejectsInvalidDeviceId) {
  int gpuCount = static_cast<int>(topo_->gpuCount());
  EXPECT_THROW(MultiTransportFactory factory(gpuCount), std::runtime_error);
  EXPECT_THROW(MultiTransportFactory factory(-2), std::runtime_error);
}

TEST_F(
    MultiTransportFactoryTest,
    ConstructorRejectsTcpRequestWithoutBindConfig) {
  // Requesting TCP via preferred/intra/interNodeTransport requires enableTcp
  // AND a bind host; otherwise TCP can't register and selection would silently
  // fall back. Fail fast at construction.
  MultiTransportFactoryOptions preferredNoConfig;
  preferredNoConfig.preferredTransport = TransportType::TCP;
  EXPECT_THROW(
      MultiTransportFactory factory(-1, preferredNoConfig),
      std::invalid_argument);

  MultiTransportFactoryOptions interEnabledNoBind;
  interEnabledNoBind.interNodeTransport = TransportType::TCP;
  interEnabledNoBind.enableTcp = true; // bind host still empty
  EXPECT_THROW(
      MultiTransportFactory factory(-1, interEnabledNoBind),
      std::invalid_argument);

  // Fully configured TCP request (enableTcp + routable bind) does not throw.
  MultiTransportFactoryOptions configured;
  configured.preferredTransport = TransportType::TCP;
  configured.enableTcp = true;
  configured.tcpBindHost = "::1";
  EXPECT_NO_THROW(MultiTransportFactory factory(-1, configured));
}
#else
TEST_F(MultiTransportFactoryTest, TcpCapabilityIsNotImplemented) {
  auto status = MultiTransportFactory::supported(TransportType::TCP);
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::NotImplemented);
}

TEST_F(MultiTransportFactoryTest, ConstructorNeverRegistersTcp) {
  MultiTransportFactoryOptions opts;
  opts.enableTcp = true;
  opts.tcpBindHost = "127.0.0.1";
  opts.preferredTransport = TransportType::TCP;
  MultiTransportFactory factory(-1, opts);
  for (size_t i = 0; i < factoryCount(factory); ++i) {
    EXPECT_NE(factoryTransportType(factory, i), TransportType::TCP);
  }
}
#endif

} // namespace uniflow
