// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/TransportProviderRegistry.h"
#include "comms/uniflow/transport/Topology.h"

#include <array>
#include <barrier>
#include <functional>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

using namespace uniflow;

namespace {

class TestFactory : public TransportFactory {
 public:
  TestFactory(TransportType type, uint8_t marker)
      : TransportFactory(type), marker_(marker) {}

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment&) override {
    return ErrCode::NotImplemented;
  }

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t,
      std::span<const uint8_t>) override {
    return ErrCode::NotImplemented;
  }

  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t>) override {
    return ErrCode::NotImplemented;
  }

  std::vector<uint8_t> getTopology() override {
    return {marker_};
  }

  Status canConnect(std::span<const uint8_t>) override {
    return Ok();
  }

 private:
  uint8_t marker_;
};

class TestProvider : public TransportProvider {
 public:
  TestProvider(TransportType type, uint8_t marker) : type_(type) {
    create = [type, marker](const TransportProviderContext&)
        -> Result<std::shared_ptr<TransportFactory>> {
      return std::make_shared<TestFactory>(type, marker);
    };
  }

  TransportType transportType() const noexcept override {
    return type_;
  }

  Status supported() const override {
    ++supportCalls;
    return support();
  }

  Result<std::shared_ptr<TransportFactory>> createFactory(
      const TransportProviderContext& context) const override {
    ++createCalls;
    return create(context);
  }

  std::function<Status()> support{[] { return Ok(); }};
  std::function<Result<std::shared_ptr<TransportFactory>>(
      const TransportProviderContext&)>
      create;
  mutable int supportCalls{0};
  mutable int createCalls{0};

 private:
  TransportType type_;
};

TransportProviderRegistration registration(
    std::string name,
    TransportType type,
    std::shared_ptr<const TransportProvider> provider,
    std::string platform = {}) {
  return {
      .name = std::move(name),
      .type = type,
      .platform = std::move(platform),
      .provider = std::move(provider),
  };
}

UNIFLOW_REGISTER_TRANSPORT_PROVIDER(
    kStaticTestProvider,
    registration(
        "registry_static_mock",
        TransportType::Mock,
        std::make_shared<TestProvider>(TransportType::Mock, 0xA1)));

} // namespace

TEST(TransportProviderRegistryTest, RegistrationDoesNotProbeOrCreate) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);

  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, provider)));

  EXPECT_EQ(registry.registeredTypes(), std::vector{TransportType::RDMA});
  EXPECT_EQ(provider->supportCalls, 0);
  EXPECT_EQ(provider->createCalls, 0);
}

TEST(TransportProviderRegistryTest, CreatesFactoriesInWireOrder) {
  TransportProviderRegistry registry;
  auto rdma = std::make_shared<TestProvider>(TransportType::RDMA, 0x20);
  ASSERT_TRUE(registry.registerProvider(registration(
      "tcp",
      TransportType::TCP,
      std::make_shared<TestProvider>(TransportType::TCP, 0x33))));
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, rdma)));
  ASSERT_TRUE(registry.registerProvider(registration(
      "peer-slot",
      TransportType::NVLink,
      std::make_shared<TestProvider>(TransportType::NVLink, 0x44))));

  auto result = registry.createFactories({.deviceId = 0});

  ASSERT_TRUE(result) << result.error().toString();
  std::vector<TransportType> types;
  std::vector<std::vector<uint8_t>> topology;
  for (const auto& factory : result.value()) {
    types.push_back(factory->transportType());
    topology.push_back(factory->getTopology());
  }
  EXPECT_EQ(
      types,
      (std::vector{
          TransportType::NVLink, TransportType::RDMA, TransportType::TCP}));
  EXPECT_EQ(
      topology, (std::vector<std::vector<uint8_t>>{{0x44}, {0x20}, {0x33}}));
  EXPECT_EQ(rdma->supportCalls, 1);
  EXPECT_EQ(rdma->createCalls, 1);
}

TEST(TransportProviderRegistryTest, SkipsUnsupportedAndInapplicableProviders) {
  TransportProviderRegistry registry;
  auto rdma = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  auto inapplicable = std::make_shared<TestProvider>(TransportType::TCP, 0x22);
  inapplicable->create = [](const TransportProviderContext&)
      -> Result<std::shared_ptr<TransportFactory>> { return nullptr; };
  auto unsupported =
      std::make_shared<TestProvider>(TransportType::NVLink, 0x33);
  unsupported->support = [] { return Status(ErrCode::NotImplemented); };
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, rdma)));
  ASSERT_TRUE(registry.registerProvider(
      registration("tcp", TransportType::TCP, inapplicable)));
  ASSERT_TRUE(registry.registerProvider(
      registration("peer", TransportType::NVLink, unsupported)));

  auto result = registry.createFactories({});

  ASSERT_TRUE(result) << result.error().toString();
  ASSERT_EQ(result.value().size(), 1);
  EXPECT_EQ(result.value().front()->getTopology(), std::vector<uint8_t>{0x11});
  EXPECT_EQ(unsupported->createCalls, 0);
  EXPECT_EQ(inapplicable->createCalls, 1);
}

TEST(TransportProviderRegistryTest, FiltersOtherPlatformsBeforeProbing) {
  TransportProviderRegistry registry;
  auto amd = std::make_shared<TestProvider>(TransportType::NVLink, 0x11);
  auto nvidia = std::make_shared<TestProvider>(TransportType::NVLink, 0x22);
  ASSERT_TRUE(registry.registerProvider(
      registration("amd-peer", TransportType::NVLink, amd, "amd")));
  ASSERT_TRUE(registry.registerProvider(
      registration("nvidia-peer", TransportType::NVLink, nvidia, "nvidia")));

  auto result = registry.createFactories({}, "amd");

  ASSERT_TRUE(result) << result.error().toString();
  ASSERT_EQ(result.value().size(), 1);
  EXPECT_EQ(result.value().front()->getTopology(), std::vector<uint8_t>{0x11});
  EXPECT_EQ(nvidia->supportCalls, 0);
  EXPECT_EQ(nvidia->createCalls, 0);
  EXPECT_TRUE(registry.registeredTypes().empty());
}

TEST(TransportProviderRegistryTest, RejectsDuplicateNames) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, provider)));

  auto duplicate = registry.registerProvider(
      registration("rdma", TransportType::TCP, provider));

  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code(), ErrCode::InvalidArgument);
  EXPECT_NE(duplicate.error().message().find("duplicate"), std::string::npos);
  EXPECT_EQ(registry.registeredTypes(), std::vector{TransportType::RDMA});
}

TEST(TransportProviderRegistryTest, RejectsOverlappingPlatformScopes) {
  for (const auto& [firstPlatform, secondPlatform] : std::array{
           std::pair{"", ""},
           std::pair{"amd", "amd"},
           std::pair{"", "amd"},
           std::pair{"amd", ""}}) {
    SCOPED_TRACE(std::string(firstPlatform) + "/" + secondPlatform);
    TransportProviderRegistry registry;
    auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
    ASSERT_TRUE(registry.registerProvider(registration(
        "first-rdma", TransportType::RDMA, provider, firstPlatform)));

    auto overlap = registry.registerProvider(registration(
        "second-rdma", TransportType::RDMA, provider, secondPlatform));

    ASSERT_FALSE(overlap);
    EXPECT_EQ(overlap.error().code(), ErrCode::InvalidArgument);
    EXPECT_NE(overlap.error().message().find("overlapping"), std::string::npos);
    EXPECT_EQ(provider->supportCalls, 0);
    EXPECT_EQ(provider->createCalls, 0);
  }
}

TEST(TransportProviderRegistryTest, PreservesCreationErrors) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  provider->create = [](const TransportProviderContext&)
      -> Result<std::shared_ptr<TransportFactory>> {
    return Err(ErrCode::DriverError, "creation failed");
  };
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, provider)));

  auto result = registry.createFactories({});

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
  EXPECT_EQ(result.error().message(), "creation failed");
}

TEST(TransportProviderRegistryTest, PreservesCapabilityErrors) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  const Err error(ErrCode::DriverError, "capability probe failed");
  provider->support = [error] { return Status(error); };
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, provider)));

  auto result = registry.createFactories({});

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code(), error.code());
  EXPECT_EQ(result.error().message(), error.message());
  EXPECT_EQ(provider->createCalls, 0);
}

TEST(TransportProviderRegistryTest, RejectsInvalidAndMismatchedIds) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  EXPECT_FALSE(registry.registerProvider(
      registration("", TransportType::RDMA, provider)));
  EXPECT_FALSE(registry.registerProvider(
      registration("null", TransportType::RDMA, nullptr)));
  EXPECT_FALSE(registry.registerProvider(
      registration("invalid", TransportType::NumTransportType, provider)));
  EXPECT_TRUE(registry.registeredTypes().empty());

  ASSERT_TRUE(registry.registerProvider(
      registration("wrong-provider", TransportType::TCP, provider)));
  auto wrongProvider = registry.createFactories({});
  ASSERT_FALSE(wrongProvider);
  EXPECT_EQ(wrongProvider.error().code(), ErrCode::InvalidArgument);

  TransportProviderRegistry second;
  provider->create = [](const TransportProviderContext&)
      -> Result<std::shared_ptr<TransportFactory>> {
    return std::make_shared<TestFactory>(TransportType::TCP, 0x22);
  };
  ASSERT_TRUE(second.registerProvider(
      registration("wrong-factory", TransportType::RDMA, provider)));
  auto wrongFactory = second.createFactories({});
  ASSERT_FALSE(wrongFactory);
  EXPECT_EQ(wrongFactory.error().code(), ErrCode::InvalidArgument);
}

TEST(TransportProviderRegistryTest, MissingAndUnsupportedAreReported) {
  TransportProviderRegistry registry;
  auto missing = registry.supported(TransportType::TCP);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code(), ErrCode::NotImplemented);
  EXPECT_NE(missing.error().message().find("not linked"), std::string::npos);

  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  provider->support = [] {
    return Status(Err(ErrCode::ResourceExhausted, "no active NIC"));
  };
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, provider)));
  auto unsupported = registry.supported(TransportType::RDMA);
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().message(), "no active NIC");
  auto factories = registry.createFactories({});
  ASSERT_TRUE(factories);
  EXPECT_TRUE(factories.value().empty());
}

TEST(TransportProviderRegistryTest, ReentrantCreationUsesTheNextSnapshot) {
  TransportProviderRegistry registry;
  Topology topology;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  Status added = ErrCode::NotImplemented;
  bool registered{false};
  const Topology* receivedTopology{nullptr};
  int receivedDevice{-1};
  provider->create = [&](const TransportProviderContext& context)
      -> Result<std::shared_ptr<TransportFactory>> {
    receivedTopology = context.topology;
    receivedDevice = context.deviceId;
    if (!registered) {
      added = registry.registerProvider(registration(
          "tcp",
          TransportType::TCP,
          std::make_shared<TestProvider>(TransportType::TCP, 0x22)));
      registered = true;
    }
    return std::make_shared<TestFactory>(TransportType::RDMA, 0x11);
  };
  ASSERT_TRUE(registry.registerProvider(
      registration("rdma", TransportType::RDMA, provider)));

  auto first = registry.createFactories({.deviceId = 3, .topology = &topology});
  auto second =
      registry.createFactories({.deviceId = 3, .topology = &topology});

  ASSERT_TRUE(added) << added.error().toString();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.value().size(), 1);
  EXPECT_EQ(second.value().size(), 2);
  EXPECT_EQ(receivedDevice, 3);
  EXPECT_EQ(receivedTopology, &topology);
}

TEST(TransportProviderRegistryTest, ConcurrentOverlapHasOneWinner) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  std::array<Status, 2> results;
  std::barrier<> start(3);
  auto add = [&](size_t index) {
    start.arrive_and_wait();
    results[index] = registry.registerProvider(registration(
        "rdma-" + std::to_string(index), TransportType::RDMA, provider));
  };
  std::thread first(add, 0);
  std::thread second(add, 1);
  start.arrive_and_wait();
  first.join();
  second.join();

  EXPECT_NE(results[0].hasValue(), results[1].hasValue());
  EXPECT_EQ(registry.registeredTypes(), std::vector{TransportType::RDMA});
  EXPECT_EQ(provider->supportCalls, 0);
  EXPECT_EQ(provider->createCalls, 0);
}

TEST(TransportProviderRegistryTest, VersionFailuresDoNotRegisterOrProbe) {
  TransportProviderRegistry registry;
  auto provider = std::make_shared<TestProvider>(TransportType::RDMA, 0x11);
  for (bool packageMismatch : {false, true}) {
    auto entry = registration("rdma", TransportType::RDMA, provider);
    if (packageMismatch) {
      entry.version.packageVersion += ".different";
    } else {
      ++entry.version.abiVersion;
    }

    auto result = registry.registerProvider(std::move(entry));

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
  }
  EXPECT_TRUE(registry.registeredTypes().empty());
  EXPECT_EQ(provider->supportCalls, 0);
  EXPECT_EQ(provider->createCalls, 0);
}

TEST(
    TransportProviderRegistryTest,
    StaticRegistrationDoesNotPopulateInstances) {
  auto result = transportProviderRegistry().createFactories({});
  ASSERT_TRUE(result);
  bool found{false};
  for (const auto& factory : result.value()) {
    if (factory->transportType() == TransportType::Mock) {
      EXPECT_EQ(factory->getTopology(), std::vector<uint8_t>{0xA1});
      found = true;
    }
  }
  EXPECT_TRUE(found);
  TransportProviderRegistry isolated;
  EXPECT_TRUE(isolated.registeredTypes().empty());
}
