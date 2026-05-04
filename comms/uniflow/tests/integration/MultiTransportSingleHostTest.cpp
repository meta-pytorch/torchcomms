// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/// Integration test for MultiTransportFactory::selectNics.
/// Requires real GPUs, NVML, and ibverbs — not for CI without GPU hardware.

#include "comms/uniflow/MultiTransport.h"

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
    topo_ = &Topology::get();
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
    factory->nicFilter_ = std::move(nicFilter);
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
      cudaGetDeviceProperties(&prop, 0);
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
TEST_F(MultiTransportFactoryTest, ConstructorCpuCreatesRdmaFactory) {
  auto st = isPlatformSupported("");
  if (!st) {
    GTEST_SKIP() << st.error().message();
  }

  MultiTransportFactory factory(-1);
  // CPU mode: no NVLink, only RDMA (if NICs available).
  EXPECT_EQ(factoryCount(factory), 1u);
  EXPECT_EQ(factoryTransportType(factory, 0), TransportType::RDMA);
}

TEST_F(MultiTransportFactoryTest, ConstructorGpuCreatesNvlinkAndRdma) {
  if (topo_->gpuCount() == 0) {
    GTEST_SKIP() << "No GPUs available";
  }

  MultiTransportFactory factory(0);
  // GPU mode: NVLink factory first, then RDMA if PIX NICs exist.
  ASSERT_GE(factoryCount(factory), 1u);
  EXPECT_EQ(factoryTransportType(factory, 0), TransportType::NVLink);

  if (factoryCount(factory) > 1) {
    EXPECT_EQ(factoryTransportType(factory, 1), TransportType::RDMA);
  }
}

TEST_F(MultiTransportFactoryTest, ConstructorRejectsInvalidDeviceId) {
  int gpuCount = static_cast<int>(topo_->gpuCount());
  EXPECT_THROW(MultiTransportFactory factory(gpuCount), std::runtime_error);
  EXPECT_THROW(MultiTransportFactory factory(-2), std::runtime_error);
}

} // namespace uniflow
