// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/DeviceAdapterAccess.h"

#include <future>
#include <latch>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include "comms/uniflow/drivers/HostDeviceAdapter.h"
#include "comms/uniflow/drivers/topology/GpuTopologyCache.h"

using namespace uniflow;

namespace {

class TestDiscovery final : public TopologyDiscoveryBackend {
 public:
  explicit TestDiscovery(std::string name, Status status = Ok())
      : name_(std::move(name)), status_(std::move(status)) {}

  Status discover(Topology& topology) override {
    CHECK_EXPR(status_);
    topology.clear();
    for (int deviceId = 0; deviceId < 2; ++deviceId) {
      const int id = topology.addNode({
          .type = NodeType::GPU,
          .name = deviceId == 0 ? name_ : name_ + ":1",
          .data = TopoNode::GpuData{.cudaDeviceId = deviceId},
      });
      topology.registerGpuNode(deviceId, id);
    }
    topology.setStatus(Ok());
    return Ok();
  }

 private:
  std::string name_;
  Status status_;
};

PlatformRegistration registration(std::string name) {
  return {
      .name = name,
      .createDeviceAdapter = []() -> Result<std::shared_ptr<DeviceAdapter>> {
        return createHostDeviceAdapter();
      },
      .createDiscoveryBackend = [name = std::move(name)]()
          -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
        return std::make_unique<TestDiscovery>(name);
      },
  };
}

} // namespace

TEST(PlatformResourcesTest, HostStorageDoesNotResolveOrClaimGpu) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", "invalid-environment");
  DeviceAdapterAccess access(std::nullopt, platforms, selector);

  auto allocation = access.hostAdapter()->pinnedHostAlloc(4096);
  ASSERT_TRUE(allocation);
  ASSERT_NE(allocation.value(), nullptr);
  auto hostAddress =
      access.hostAdapter()->hostGetDevicePointer(allocation.value());
  ASSERT_TRUE(hostAddress);
  EXPECT_EQ(hostAddress.value(), allocation.value());
  EXPECT_FALSE(
      access.hostAdapter()->hostGetDevicePointer(0, allocation.value()));
  EXPECT_TRUE(access.hostAdapter()->pinnedHostFree(allocation.value()));
  EXPECT_FALSE(access.hostAdapter()->pinnedHostAlloc(0));
  EXPECT_FALSE(selector.selectedBackend());
  EXPECT_TRUE(platforms.registeredPlatforms().empty());
}

TEST(PlatformResourcesTest, MissingGpuPlatformDoesNotClaimDefault) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  DeviceAdapterAccess access("amd", platforms, selector);

  auto adapter = access.gpuAdapter(0);

  ASSERT_FALSE(adapter);
  EXPECT_EQ(adapter.error().code(), ErrCode::NotImplemented);
  EXPECT_NE(adapter.error().message().find("amd"), std::string::npos);
  EXPECT_FALSE(selector.selectedBackend());
  EXPECT_FALSE(access.gpuAdapter(-1));
}

TEST(PlatformResourcesTest, OrdinaryRemotePointersNeedNoGpuFactory) {
  for (const auto* vendor : {"nvidia", "amd"}) {
    PlatformRegistry platforms;
    GpuBackendSelector selector("nvidia", std::nullopt);
    DeviceAdapterAccess access(vendor, platforms, selector);
    int storage{0};

    auto decoder = access.devicePointerResolver();

    ASSERT_TRUE(decoder);
    EXPECT_EQ(decoder.value()(&storage), reinterpret_cast<uint64_t>(&storage));
    EXPECT_FALSE(selector.selectedBackend());
    EXPECT_TRUE(platforms.registeredPlatforms().empty());
  }
}

TEST(PlatformResourcesTest, MtiaRemotePointersRequireRegisteredPurePolicy) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  DeviceAdapterAccess access("mtia", platforms, selector);
  auto missing = access.devicePointerResolver();
  ASSERT_FALSE(missing);
  EXPECT_NE(
      missing.error().message().find("metadata policy"), std::string::npos);
  int factoryCalls{0};
  int decoderCalls{0};
  auto entry = registration("mtia");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    ++factoryCalls;
    return createHostDeviceAdapter();
  };
  ASSERT_TRUE(platforms.registerPlatform(entry));
  EXPECT_FALSE(access.devicePointerResolver());

  PlatformRegistry complete;
  entry.resolveDevicePointer = [&](const void*) noexcept {
    ++decoderCalls;
    return uint64_t{123};
  };
  ASSERT_TRUE(complete.registerPlatform(std::move(entry)));
  DeviceAdapterAccess decoded("mtia", complete, selector);
  auto decoder = decoded.devicePointerResolver();
  ASSERT_TRUE(decoder);
  EXPECT_EQ(decoder.value()(nullptr), 123u);
  EXPECT_EQ(decoderCalls, 1);
  EXPECT_EQ(factoryCalls, 0);
  EXPECT_FALSE(selector.selectedBackend());
}

TEST(
    PlatformResourcesTest,
    InjectedAdapterAndDecoderStayAliveWithoutSelection) {
  auto adapter = createHostDeviceAdapter();
  std::weak_ptr<DeviceAdapter> weak = adapter;
  DevicePointerResolver decoder;
  {
    DeviceAdapterAccess access(adapter);
    adapter.reset();
    EXPECT_FALSE(weak.expired());
    auto selected = access.gpuAdapter(3);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected.value(), access.hostAdapter());
    auto result = access.devicePointerResolver();
    ASSERT_TRUE(result);
    decoder = std::move(result).value();
  }
  EXPECT_FALSE(weak.expired());
  int storage{0};
  EXPECT_EQ(decoder(&storage), reinterpret_cast<uint64_t>(&storage));
  decoder = {};
  EXPECT_TRUE(weak.expired());
  EXPECT_THROW(
      DeviceAdapterAccess(std::shared_ptr<DeviceAdapter>{}),
      std::invalid_argument);
}

TEST(PlatformResourcesTest, GpuAdapterClaimsOnceAndCachesAcrossDeviceIds) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  int calls{0};
  auto entry = registration("amd");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    ++calls;
    EXPECT_TRUE(platforms.contains("amd"));
    return createHostDeviceAdapter();
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));
  DeviceAdapterAccess access("amd", platforms, selector);
  EXPECT_FALSE(selector.selectedBackend());

  auto first = access.gpuAdapter(0);
  auto second = access.gpuAdapter(1);

  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.value(), second.value());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(selector.selectedBackend(), "amd");
  DeviceAdapterAccess conflicting("nvidia", platforms, selector);
  EXPECT_FALSE(conflicting.gpuAdapter(0));
}

TEST(PlatformResourcesTest, GpuAdapterCreationErrorIsVisibleAndRetryable) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  int calls{0};
  auto entry = registration("nvidia");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    if (++calls == 1) {
      return Err(ErrCode::ResourceExhausted, "adapter allocation failed");
    }
    return createHostDeviceAdapter();
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));
  DeviceAdapterAccess access(std::nullopt, platforms, selector);

  auto failed = access.gpuAdapter(0);
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code(), ErrCode::ResourceExhausted);
  EXPECT_EQ(failed.error().message(), "adapter allocation failed");
  EXPECT_EQ(selector.selectedBackend(), "nvidia");
  EXPECT_TRUE(access.gpuAdapter(0));
  EXPECT_EQ(calls, 2);
}

TEST(
    PlatformResourcesTest,
    RecursiveAdapterCreationReportsErrorWithoutDeadlock) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  DeviceAdapterAccess access(std::nullopt, platforms, selector);
  auto entry = registration("nvidia");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    return access.gpuAdapter(0);
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));

  auto result = access.gpuAdapter(0);

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
  EXPECT_NE(result.error().message().find("recursive"), std::string::npos);
}

TEST(PlatformResourcesTest, GpuTopologyUsesSelectedVendorAndCachesDiscovery) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  ASSERT_TRUE(platforms.registerPlatform(registration("amd")));
  GpuTopologyCache cache(platforms, selector);
  EXPECT_FALSE(selector.selectedBackend());

  auto explicitVendor = cache.get("amd");
  auto implicitVendor = cache.get();

  ASSERT_TRUE(explicitVendor);
  ASSERT_TRUE(implicitVendor);
  EXPECT_EQ(explicitVendor.value(), implicitVendor.value());
  EXPECT_EQ(explicitVendor.value()->getGpuNode(0).name, "amd");
  EXPECT_EQ(selector.selectedBackend(), "amd");
  EXPECT_FALSE(cache.get("nvidia"));
}

TEST(PlatformResourcesTest, ConcurrentGpuDiscoveryInvokesOneUnlockedFactory) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  GpuTopologyCache cache(platforms, selector);
  std::latch entered(1);
  std::latch release(1);
  int calls{0};
  auto entry = registration("nvidia");
  entry.createDiscoveryBackend =
      [&]() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    ++calls;
    EXPECT_TRUE(platforms.contains("nvidia"));
    entered.count_down();
    release.wait();
    return std::make_unique<TestDiscovery>("nvidia");
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));
  auto first = std::async(std::launch::async, [&] { return cache.get(); });
  entered.wait();
  auto second = std::async(std::launch::async, [&] { return cache.get(); });
  release.count_down();
  auto a = first.get();
  auto b = second.get();

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_EQ(a.value(), b.value());
  EXPECT_EQ(calls, 1);
}

TEST(PlatformResourcesTest, DiscoveryErrorIsVisibleAndRetryable) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  GpuTopologyCache cache(platforms, selector);
  int calls{0};
  auto entry = registration("nvidia");
  entry.createDiscoveryBackend =
      [&]() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    if (++calls == 1) {
      return Err(ErrCode::DriverError, "discovery unavailable");
    }
    return std::make_unique<TestDiscovery>("nvidia");
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));

  auto failed = cache.get();
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().message(), "discovery unavailable");
  EXPECT_TRUE(cache.get());
  EXPECT_EQ(calls, 2);
}

TEST(PlatformResourcesTest, BackendDiscoveryFailureCachesPartialTopology) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  int calls{0};
  auto entry = registration("mtia");
  entry.createDiscoveryBackend =
      [&]() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    ++calls;
    return std::make_unique<TestDiscovery>(
        "mtia", Err(ErrCode::DriverError, "MTIA topology unavailable"));
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));
  GpuTopologyCache cache(platforms, selector);

  auto result = cache.get("mtia");

  ASSERT_TRUE(result);
  EXPECT_EQ(result.value()->gpuCount(), 0u);
  auto cached = cache.get("mtia");
  ASSERT_TRUE(cached);
  EXPECT_EQ(cached.value(), result.value());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(selector.selectedBackend(), "mtia");
}

TEST(
    PlatformResourcesTest,
    InvalidGpuOrdinalPreventsAdapterAndStagingCreation) {
  PlatformRegistry platforms;
  GpuBackendSelector selector("nvidia", std::nullopt);
  int adapterCalls{0};
  auto entry = registration("nvidia");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    ++adapterCalls;
    return createHostDeviceAdapter();
  };
  ASSERT_TRUE(platforms.registerPlatform(std::move(entry)));
  DeviceAdapterAccess access(std::nullopt, platforms, selector);

  auto invalid = access.gpuAdapter(2);

  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code(), ErrCode::InvalidArgument);
  EXPECT_NE(invalid.error().message().find("numDevices=2"), std::string::npos);
  EXPECT_EQ(adapterCalls, 0);
  EXPECT_EQ(selector.selectedBackend(), "nvidia");
  auto valid = access.gpuAdapter(1);
  ASSERT_TRUE(valid);
  EXPECT_EQ(adapterCalls, 1);
}
