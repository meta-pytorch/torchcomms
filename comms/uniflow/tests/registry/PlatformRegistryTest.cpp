// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/PlatformRegistry.h"

#include <utility>

#include <gtest/gtest.h>

using namespace uniflow;

namespace {

class TestAdapter : public DeviceAdapter {
 public:
  Result<void*> pinnedHostAlloc(size_t) override {
    return ErrCode::NotImplemented;
  }

  Status pinnedHostFree(void*) override {
    return ErrCode::NotImplemented;
  }

  Result<void*> hostGetDevicePointer(void*) override {
    return ErrCode::NotImplemented;
  }

  Result<bool> isDmaBuffSupported(int) override {
    return false;
  }

  Result<DmaBuff> exportDmaBuff(int, void*, size_t, DmaBufMapping) override {
    return ErrCode::NotImplemented;
  }

  Status closeDmaBuff(DmaBuff&) override {
    return Ok();
  }
};

class TestDiscovery : public TopologyDiscoveryBackend {
 public:
  Status discover(Topology&) override {
    return Ok();
  }
};

PlatformRegistration registration(std::string name) {
  return {
      .name = std::move(name),
      .createDeviceAdapter = []() -> Result<std::shared_ptr<DeviceAdapter>> {
        return std::make_shared<TestAdapter>();
      },
      .createDiscoveryBackend =
          []() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
        return std::make_unique<TestDiscovery>();
      },
  };
}

UNIFLOW_REGISTER_PLATFORM(kStaticTestPlatform, registration("registry_test"));

} // namespace

TEST(PlatformRegistryTest, RegistrationDoesNotInvokeFactories) {
  PlatformRegistry registry;
  auto adapter = std::make_shared<TestAdapter>();
  int adapterCalls{0};
  int discoveryCalls{0};
  auto entry = registration("nvidia");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    ++adapterCalls;
    return adapter;
  };
  entry.createDiscoveryBackend =
      [&]() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    ++discoveryCalls;
    return std::make_unique<TestDiscovery>();
  };

  ASSERT_TRUE(registry.registerPlatform(std::move(entry)));

  EXPECT_TRUE(registry.contains("nvidia"));
  EXPECT_EQ(registry.registeredPlatforms(), std::vector<std::string>{"nvidia"});
  EXPECT_EQ(adapterCalls, 0);
  EXPECT_EQ(discoveryCalls, 0);
  auto selectedAdapter = registry.createDeviceAdapter("nvidia");
  ASSERT_TRUE(selectedAdapter);
  EXPECT_EQ(selectedAdapter.value(), adapter);
  EXPECT_EQ(adapterCalls, 1);
  EXPECT_EQ(discoveryCalls, 0);
  auto selectedDiscovery = registry.createDiscoveryBackend("nvidia");
  ASSERT_TRUE(selectedDiscovery);
  EXPECT_NE(selectedDiscovery.value(), nullptr);
  EXPECT_EQ(discoveryCalls, 1);
}

TEST(PlatformRegistryTest, RejectsDuplicateAndIncompleteRegistrations) {
  PlatformRegistry registry;
  EXPECT_FALSE(registry.registerPlatform(registration("")));
  auto incomplete = registration("nvidia");
  incomplete.createDiscoveryBackend = {};
  EXPECT_FALSE(registry.registerPlatform(std::move(incomplete)));
  EXPECT_TRUE(registry.registeredPlatforms().empty());

  ASSERT_TRUE(registry.registerPlatform(registration("nvidia")));
  auto duplicate = registry.registerPlatform(registration("nvidia"));
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code(), ErrCode::InvalidArgument);
  EXPECT_NE(duplicate.error().message().find("duplicate"), std::string::npos);
  EXPECT_EQ(registry.registeredPlatforms(), std::vector<std::string>{"nvidia"});
}

TEST(PlatformRegistryTest, ReportsMissingAndPreservesCreationErrors) {
  PlatformRegistry registry;
  auto missing = registry.createDeviceAdapter("amd");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code(), ErrCode::NotImplemented);
  EXPECT_NE(missing.error().message().find("amd"), std::string::npos);

  auto entry = registration("amd");
  entry.createDeviceAdapter = []() -> Result<std::shared_ptr<DeviceAdapter>> {
    return Err(ErrCode::DriverError, "adapter failed");
  };
  entry.createDiscoveryBackend =
      []() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    return Err(ErrCode::ResourceExhausted, "discovery failed");
  };
  ASSERT_TRUE(registry.registerPlatform(std::move(entry)));
  auto adapter = registry.createDeviceAdapter("amd");
  auto discovery = registry.createDiscoveryBackend("amd");
  ASSERT_FALSE(adapter);
  EXPECT_EQ(adapter.error().code(), ErrCode::DriverError);
  EXPECT_EQ(adapter.error().message(), "adapter failed");
  ASSERT_FALSE(discovery);
  EXPECT_EQ(discovery.error().code(), ErrCode::ResourceExhausted);
  EXPECT_EQ(discovery.error().message(), "discovery failed");
}

TEST(PlatformRegistryTest, RejectsSuccessfulNullFactories) {
  PlatformRegistry registry;
  auto entry = registration("amd");
  entry.createDeviceAdapter = []() -> Result<std::shared_ptr<DeviceAdapter>> {
    return nullptr;
  };
  entry.createDiscoveryBackend =
      []() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    return nullptr;
  };
  ASSERT_TRUE(registry.registerPlatform(std::move(entry)));

  auto adapter = registry.createDeviceAdapter("amd");
  auto discovery = registry.createDiscoveryBackend("amd");

  ASSERT_FALSE(adapter);
  EXPECT_EQ(adapter.error().code(), ErrCode::DriverError);
  ASSERT_FALSE(discovery);
  EXPECT_EQ(discovery.error().code(), ErrCode::DriverError);
}

TEST(PlatformRegistryTest, FactoriesCanReenterTheRegistry) {
  PlatformRegistry registry;
  Status added = ErrCode::NotImplemented;
  std::vector<std::string> observed;
  auto entry = registration("nvidia");
  entry.createDeviceAdapter = [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
    added = registry.registerPlatform(registration("amd"));
    return std::make_shared<TestAdapter>();
  };
  entry.createDiscoveryBackend =
      [&]() -> Result<std::unique_ptr<TopologyDiscoveryBackend>> {
    observed = registry.registeredPlatforms();
    return std::make_unique<TestDiscovery>();
  };
  ASSERT_TRUE(registry.registerPlatform(std::move(entry)));

  auto adapter = registry.createDeviceAdapter("nvidia");
  auto discovery = registry.createDiscoveryBackend("nvidia");

  ASSERT_TRUE(adapter);
  ASSERT_TRUE(added) << added.error().toString();
  ASSERT_TRUE(discovery);
  EXPECT_EQ(observed, (std::vector<std::string>{"amd", "nvidia"}));
}

TEST(PlatformRegistryTest, VersionFailuresDoNotRegisterOrCreate) {
  PlatformRegistry registry;
  int calls{0};
  for (bool packageMismatch : {false, true}) {
    auto entry = registration("amd");
    if (packageMismatch) {
      entry.version.packageVersion += ".different";
    } else {
      ++entry.version.abiVersion;
    }
    entry.createDeviceAdapter =
        [&]() -> Result<std::shared_ptr<DeviceAdapter>> {
      ++calls;
      return std::make_shared<TestAdapter>();
    };

    auto result = registry.registerPlatform(std::move(entry));

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
  }
  EXPECT_TRUE(registry.registeredPlatforms().empty());
  EXPECT_EQ(calls, 0);
}

TEST(PlatformRegistryTest, StaticRegistrationDoesNotPopulateInstances) {
  EXPECT_TRUE(platformRegistry().contains("registry_test"));
  auto adapter = platformRegistry().createDeviceAdapter("registry_test");
  ASSERT_TRUE(adapter);
  PlatformRegistry isolated;
  EXPECT_TRUE(isolated.registeredPlatforms().empty());
}
