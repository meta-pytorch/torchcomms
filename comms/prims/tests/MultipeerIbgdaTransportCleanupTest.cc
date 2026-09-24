// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/transport/MultiPeerIbTransportInternal.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportInternal.h"

namespace comms::prims::detail {
namespace {

struct FakeNicResources {
  std::vector<IbgdaQpSlotResources> qpSlots;
};

using Event = std::pair<std::string, const void*>;
using StrictMockBootstrap =
    ::testing::StrictMock<meta::comms::testing::MockBootstrap>;

[[noreturn]] void throwLocalValidationFailure() {
  throw std::runtime_error("local validation failure");
}

class PeerExchangeHarness final : private MultiPeerIbTransportBase {
 public:
  PeerExchangeHarness(
      int myRank,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap)
      : MultiPeerIbTransportBase(
            myRank,
            /*nRanks=*/2,
            std::move(bootstrap),
            makeConfig()) {}

  int exchange(
      int peerRank,
      int localPayload,
      const std::function<void()>& beforeSend) {
    return exchangeWithPeer(peerRank, localPayload, /*tag=*/0, beforeSend);
  }

 private:
  static MultipeerIbTransportConfig makeConfig() {
    MultipeerIbTransportConfig config;
    config.gpuNicMap[0] = {"test_nic"};
    return config;
  }
};

class MaterializationFailureHarness final
    : public MultiPeerIbTransport<MaterializationFailureHarness> {
 public:
  explicit MaterializationFailureHarness(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap)
      : MultiPeerIbTransport(
            /*myRank=*/0,
            /*nRanks=*/2,
            std::move(bootstrap),
            makeConfig()) {
    peerMaterialized_.resize(1, false);
  }

  [[noreturn]] void doMaterializePeer(int) {
    rkeysPossiblyExposed_ = true;
    throw std::runtime_error("original materialization failure");
  }

  void cleanupPeerOnFailure(int) {
    cleanupCalled_ = true;
    if (rkeysPossiblyExposed_) {
      requiresProcessLifetimeQuarantine_ = true;
      return;
    }
  }

  bool requiresProcessLifetimeQuarantine() const {
    return requiresProcessLifetimeQuarantine_;
  }

  bool cleanupCalled() const {
    return cleanupCalled_;
  }

 private:
  static MultipeerIbTransportConfig makeConfig() {
    MultipeerIbTransportConfig config;
    config.gpuNicMap[0] = {"test_nic"};
    return config;
  }

  bool rkeysPossiblyExposed_{false};
  bool requiresProcessLifetimeQuarantine_{false};
  bool cleanupCalled_{false};
};

TEST(
    MultipeerIbgdaTransportCleanupTest,
    TransitionsEveryQpBeforeFirstBufferRelease) {
  doca_gpu_verbs_qp_group_hl groupedQp{};
  doca_gpu_verbs_qp_hl groupedLoopbackQp{};
  doca_gpu_verbs_qp_hl standaloneMainQp{};
  doca_gpu_verbs_qp_group_hl secondGroupedQp{};
  doca_gpu_verbs_qp_hl secondGroupedLoopbackQp{};
  std::vector<FakeNicResources> nics(2);
  nics[0].qpSlots = {
      {.group = &groupedQp, .loopback = &groupedLoopbackQp},
      {.standaloneMain = &standaloneMainQp},
  };
  nics[1].qpSlots = {
      {.group = &secondGroupedQp, .loopback = &secondGroupedLoopbackQp},
      {},
  };

  std::vector<Event> events;
  quiesceQpsThenReleaseBuffers(
      true,
      nics,
      [&](doca_gpu_verbs_qp_group_hl* qp) {
        events.emplace_back("group", qp);
        return DOCA_SUCCESS;
      },
      [&](doca_gpu_verbs_qp_hl* qp) {
        events.emplace_back("single", qp);
        return DOCA_SUCCESS;
      },
      [&]() { events.emplace_back("release_gpu", nullptr); },
      [&]() { events.emplace_back("release_send_recv", nullptr); });

  EXPECT_EQ(
      events,
      (std::vector<Event>{
          {"group", &groupedQp},
          {"single", &groupedLoopbackQp},
          {"single", &standaloneMainQp},
          {"group", &secondGroupedQp},
          {"single", &secondGroupedLoopbackQp},
          {"release_gpu", nullptr},
          {"release_send_recv", nullptr},
      }));
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    TransitionsFailedPeerBeforeReleasingItsResources) {
  doca_gpu_verbs_qp_hl otherPeerQp0{};
  doca_gpu_verbs_qp_hl otherPeerQp1{};
  doca_gpu_verbs_qp_group_hl failedPeerGroup0{};
  doca_gpu_verbs_qp_hl failedPeerLoopback0{};
  doca_gpu_verbs_qp_hl failedPeerStandalone0{};
  doca_gpu_verbs_qp_hl failedPeerStandalone1{};
  doca_gpu_verbs_qp_group_hl failedPeerGroup1{};
  doca_gpu_verbs_qp_hl failedPeerLoopback1{};
  std::vector<FakeNicResources> nics(2);
  nics[0].qpSlots = {
      {.standaloneMain = &otherPeerQp0},
      {},
      {.group = &failedPeerGroup0, .loopback = &failedPeerLoopback0},
      {.standaloneMain = &failedPeerStandalone0},
  };
  nics[1].qpSlots = {
      {},
      {.standaloneMain = &otherPeerQp1},
      {.standaloneMain = &failedPeerStandalone1},
      {.group = &failedPeerGroup1, .loopback = &failedPeerLoopback1},
  };

  std::vector<Event> events;
  quiescePeerQpsThenReleaseResources(
      true,
      nics,
      /*peerIndex=*/1,
      /*slotsPerPeer=*/2,
      [&](doca_gpu_verbs_qp_group_hl* qp) {
        events.emplace_back("group", qp);
        return DOCA_SUCCESS;
      },
      [&](doca_gpu_verbs_qp_hl* qp) {
        events.emplace_back("single", qp);
        return DOCA_SUCCESS;
      },
      [&]() { events.emplace_back("release_peer", nullptr); });

  EXPECT_EQ(
      events,
      (std::vector<Event>{
          {"group", &failedPeerGroup0},
          {"single", &failedPeerLoopback0},
          {"single", &failedPeerStandalone0},
          {"single", &failedPeerStandalone1},
          {"group", &failedPeerGroup1},
          {"single", &failedPeerLoopback1},
          {"release_peer", nullptr},
      }));
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    LowerRankReceiveFailureKeepsRkeysLocal) {
  auto bootstrap = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*bootstrap, duplicate()).WillOnce([] { return nullptr; });
  EXPECT_CALL(*bootstrap, recv(::testing::_, sizeof(int), 1, ::testing::_))
      .WillOnce([] { return folly::makeSemiFuture(-1); });
  EXPECT_CALL(*bootstrap, send(::testing::_, ::testing::_, 1, ::testing::_))
      .Times(0);
  PeerExchangeHarness transport(/*myRank=*/0, bootstrap);
  PeerRkeyExposureState exposureState = PeerRkeyExposureState::kLocalOnly;

  EXPECT_THROW(
      exchangePeerBufferPayloadWithExposureTracking(
          exposureState,
          [&](const auto& beforeSend) {
            return transport.exchange(/*peerRank=*/1, 7, beforeSend);
          }),
      std::runtime_error);
  EXPECT_EQ(exposureState, PeerRkeyExposureState::kLocalOnly);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    LowerRankSendFailureMarksRkeysPossiblyExposed) {
  auto bootstrap = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*bootstrap, duplicate()).WillOnce([] { return nullptr; });
  EXPECT_CALL(*bootstrap, recv(::testing::_, sizeof(int), 1, ::testing::_))
      .WillOnce([](void* payload, int, int, int) {
        *static_cast<int*>(payload) = 11;
        return folly::makeSemiFuture(0);
      });
  PeerRkeyExposureState exposureState = PeerRkeyExposureState::kLocalOnly;
  EXPECT_CALL(*bootstrap, send(::testing::_, sizeof(int), 1, ::testing::_))
      .WillOnce([&](void*, int, int, int) {
        EXPECT_EQ(exposureState, PeerRkeyExposureState::kPossiblyExposed);
        return folly::makeSemiFuture(-1);
      });
  PeerExchangeHarness transport(/*myRank=*/0, bootstrap);

  EXPECT_THROW(
      exchangePeerBufferPayloadWithExposureTracking(
          exposureState,
          [&](const auto& beforeSend) {
            return transport.exchange(/*peerRank=*/1, 7, beforeSend);
          }),
      std::runtime_error);
  EXPECT_EQ(exposureState, PeerRkeyExposureState::kPossiblyExposed);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    HigherRankReceiveFailureLeavesRkeysPossiblyExposed) {
  auto bootstrap = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*bootstrap, duplicate()).WillOnce([] { return nullptr; });
  PeerRkeyExposureState exposureState = PeerRkeyExposureState::kLocalOnly;
  EXPECT_CALL(*bootstrap, send(::testing::_, sizeof(int), 0, ::testing::_))
      .WillOnce([&](void*, int, int, int) {
        EXPECT_EQ(exposureState, PeerRkeyExposureState::kPossiblyExposed);
        return folly::makeSemiFuture(0);
      });
  EXPECT_CALL(*bootstrap, recv(::testing::_, sizeof(int), 0, ::testing::_))
      .WillOnce([] { return folly::makeSemiFuture(-1); });
  PeerExchangeHarness transport(/*myRank=*/1, bootstrap);

  EXPECT_THROW(
      exchangePeerBufferPayloadWithExposureTracking(
          exposureState,
          [&](const auto& beforeSend) {
            return transport.exchange(/*peerRank=*/0, 7, beforeSend);
          }),
      std::runtime_error);
  EXPECT_EQ(exposureState, PeerRkeyExposureState::kPossiblyExposed);
}

TEST(MultipeerIbgdaTransportCleanupTest, FindsFirstPossiblyExposedPeer) {
  const std::vector<PeerRkeyExposureState> exposureStates = {
      PeerRkeyExposureState::kPossiblyExposed,
      PeerRkeyExposureState::kLocalOnly,
      PeerRkeyExposureState::kPossiblyExposed,
  };

  EXPECT_EQ(findPossiblyExposedPeer(exposureStates), 0);
}

TEST(MultipeerIbgdaTransportCleanupTest, NoPossiblyExposedPeerReturnsNullopt) {
  const std::vector<PeerRkeyExposureState> exposureStates(
      2, PeerRkeyExposureState::kLocalOnly);

  EXPECT_EQ(findPossiblyExposedPeer(exposureStates), std::nullopt);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    ProcessLifetimeQuarantineDetachesOnlyWhenOwnerIsDestroyed) {
  bool destroyed = false;
  struct DestructionProbe {
    explicit DestructionProbe(bool& destroyed) : destroyed(destroyed) {}
    ~DestructionProbe() {
      destroyed = true;
    }
    bool& destroyed;
  };
  auto owner = std::make_unique<DestructionProbe>(destroyed);
  auto* quarantined = releaseTransportForProcessLifetimeIfQuarantined(
      owner, /*processLifetimeQuarantineRequired=*/true);

  EXPECT_EQ(owner, nullptr);
  EXPECT_FALSE(destroyed);

  delete quarantined;
  EXPECT_TRUE(destroyed);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    RkeyExchangeFailureAfterExposureQuarantinesAndPreservesOriginalError) {
  bool rkeysPossiblyExposed = false;
  bool quarantined = false;
  std::string quarantineContext;

  try {
    runWithProcessLifetimeQuarantineOnFailureAfterRkeyExposure(
        rkeysPossiblyExposed,
        [&]() {
          rkeysPossiblyExposed = true;
          throw std::runtime_error("original window exchange failure");
        },
        [&](std::string_view context) {
          quarantined = true;
          quarantineContext = context;
        });
    FAIL() << "expected original window exchange failure";
  } catch (const std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "original window exchange failure");
  }

  EXPECT_TRUE(quarantined);
  EXPECT_EQ(quarantineContext, "original window exchange failure");
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    RkeyExchangeFailureBeforeExposureDoesNotQuarantine) {
  bool rkeysPossiblyExposed = false;
  bool quarantined = false;

  EXPECT_THROW(
      runWithProcessLifetimeQuarantineOnFailureAfterRkeyExposure(
          rkeysPossiblyExposed,
          throwLocalValidationFailure,
          [&](std::string_view) { quarantined = true; }),
      std::runtime_error);
  EXPECT_FALSE(quarantined);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    LaterWindowFailureQuarantinesPreviouslyExposedCallerBuffer) {
  bool ibgdaRkeysPossiblyExposed = false;
  bool callerBufferPossiblyExposed = true;
  bool ibgdaTransportQuarantined = false;
  bool callerBufferQuarantined = false;

  EXPECT_THROW(
      runWithProcessLifetimeQuarantineOnFailureAfterWindowExposure(
          ibgdaRkeysPossiblyExposed,
          callerBufferPossiblyExposed,
          throwLocalValidationFailure,
          [&](std::string_view) { ibgdaTransportQuarantined = true; },
          [&](std::string_view) { callerBufferQuarantined = true; }),
      std::runtime_error);
  EXPECT_FALSE(ibgdaTransportQuarantined);
  EXPECT_TRUE(callerBufferQuarantined);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    WindowFailureQuarantinesBothExposedResourceClasses) {
  bool ibgdaRkeysPossiblyExposed = true;
  bool callerBufferPossiblyExposed = true;
  std::vector<std::string> quarantined;

  try {
    runWithProcessLifetimeQuarantineOnFailureAfterWindowExposure(
        ibgdaRkeysPossiblyExposed,
        callerBufferPossiblyExposed,
        throwLocalValidationFailure,
        [&](std::string_view context) {
          EXPECT_EQ(context, "local validation failure");
          quarantined.emplace_back("ibgda");
        },
        [&](std::string_view context) {
          EXPECT_EQ(context, "local validation failure");
          quarantined.emplace_back("caller");
        });
    FAIL() << "expected local validation failure";
  } catch (const std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "local validation failure");
  }

  EXPECT_EQ(quarantined, (std::vector<std::string>{"ibgda", "caller"}));
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    MaterializationFailurePreservesOriginalErrorAfterOwnerQuarantine) {
  auto bootstrap = std::make_shared<StrictMockBootstrap>();
  EXPECT_CALL(*bootstrap, duplicate()).WillOnce([] { return nullptr; });
  auto owner = std::make_unique<MaterializationFailureHarness>(bootstrap);
  MaterializationFailureHarness* quarantinedTransport = nullptr;
  bool quarantined = false;
  bool poisoned = false;
  auto materialize = [&]() {
    runWithProcessLifetimeQuarantineOnFailure(
        *owner,
        [](auto& transport) { transport.materializePeer(/*peerRank=*/1); },
        [&](std::string_view context) {
          EXPECT_EQ(context, "original materialization failure");
          quarantined = true;
          poisoned = true;
        });
  };

  try {
    materialize();
    FAIL() << "expected original materialization failure";
  } catch (const std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "original materialization failure");
  }
  EXPECT_NE(owner, nullptr);
  EXPECT_TRUE(quarantined);
  EXPECT_TRUE(poisoned);
  quarantinedTransport =
      releaseTransportForProcessLifetimeIfQuarantined(owner, quarantined);
  ASSERT_NE(quarantinedTransport, nullptr);
  EXPECT_TRUE(quarantinedTransport->cleanupCalled());

  delete quarantinedTransport;
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    ProcessLifetimeQuarantineRetainsDependentResources) {
  bool released = false;
  releaseUnlessProcessLifetimeQuarantined(true, [&]() { released = true; });
  EXPECT_FALSE(released);

  releaseUnlessProcessLifetimeQuarantined(false, [&]() { released = true; });
  EXPECT_TRUE(released);
}

TEST(MultipeerIbgdaTransportCleanupTest, NormalCleanupRunsBeforeOwnerRelease) {
  bool cleanupRan = false;
  bool releasedAfterCleanup = false;
  {
    auto owner = std::shared_ptr<void>(new int(7), [&](void* ptr) {
      releasedAfterCleanup = cleanupRan;
      delete static_cast<int*>(ptr);
    });
    std::vector<std::unique_ptr<std::shared_ptr<void>>> keepAlives;
    keepAlives.push_back(
        std::make_unique<std::shared_ptr<void>>(std::move(owner)));

    EXPECT_TRUE(releaseResourcesOrRetainKeepAlives(
        /*resourceLifetimeQuarantineRequired=*/false,
        /*keepAliveLifetimeQuarantineRequired=*/false,
        keepAlives,
        [&]() noexcept {
          cleanupRan = true;
          return true;
        }));
    EXPECT_FALSE(releasedAfterCleanup);
  }
  EXPECT_TRUE(releasedAfterCleanup);
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    ExceptionUnwindAfterRkeyExposureRetainsCallerOwner) {
  struct CleanupScope {
    ~CleanupScope() {
      static_cast<void>(releaseResourcesOrRetainKeepAlives(
          quarantine, quarantine, keepAlives, [this]() noexcept {
            cleanupRan = true;
            return true;
          }));
    }

    bool& quarantine;
    bool& cleanupRan;
    std::vector<std::unique_ptr<std::shared_ptr<void>>> keepAlives;
  };

  bool rkeysPossiblyExposed = false;
  bool quarantined = false;
  bool cleanupRan = false;
  std::weak_ptr<void> weakOwner;
  std::shared_ptr<void>* retainedHolder = nullptr;

  try {
    auto owner = std::shared_ptr<void>(
        new int(7), [](void* ptr) { delete static_cast<int*>(ptr); });
    weakOwner = owner;
    CleanupScope cleanup{quarantined, cleanupRan, {}};
    cleanup.keepAlives.push_back(
        std::make_unique<std::shared_ptr<void>>(std::move(owner)));
    retainedHolder = cleanup.keepAlives.front().get();

    runWithProcessLifetimeQuarantineOnFailureAfterRkeyExposure(
        rkeysPossiblyExposed,
        [&]() {
          rkeysPossiblyExposed = true;
          throw std::runtime_error("post-exposure failure");
        },
        [&](std::string_view) { quarantined = true; });
    FAIL() << "expected post-exposure failure";
  } catch (const std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "post-exposure failure");
  }

  EXPECT_TRUE(quarantined);
  EXPECT_FALSE(cleanupRan);
  EXPECT_FALSE(weakOwner.expired());
  delete retainedHolder;
  EXPECT_TRUE(weakOwner.expired());
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    FailureBeforeRkeyExposureReleasesNormally) {
  struct CleanupScope {
    ~CleanupScope() {
      static_cast<void>(releaseResourcesOrRetainKeepAlives(
          quarantine, quarantine, keepAlives, [this]() noexcept {
            cleanupRan = true;
            return true;
          }));
    }

    bool& quarantine;
    bool& cleanupRan;
    std::vector<std::unique_ptr<std::shared_ptr<void>>> keepAlives;
  };

  bool rkeysPossiblyExposed = false;
  bool quarantined = false;
  bool cleanupRan = false;
  std::weak_ptr<void> weakOwner;

  try {
    auto owner = std::shared_ptr<void>(
        new int(7), [](void* ptr) { delete static_cast<int*>(ptr); });
    weakOwner = owner;
    CleanupScope cleanup{quarantined, cleanupRan, {}};
    cleanup.keepAlives.push_back(
        std::make_unique<std::shared_ptr<void>>(std::move(owner)));

    runWithProcessLifetimeQuarantineOnFailureAfterRkeyExposure(
        rkeysPossiblyExposed,
        throwLocalValidationFailure,
        [&](std::string_view) { quarantined = true; });
    FAIL() << "expected local validation failure";
  } catch (const std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "local validation failure");
  }

  EXPECT_FALSE(quarantined);
  EXPECT_TRUE(cleanupRan);
  EXPECT_TRUE(weakOwner.expired());
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    CallerBufferQuarantineStillReleasesUnrelatedResources) {
  bool cleanupRan = false;
  auto owner = std::shared_ptr<void>(
      new int(7), [](void* ptr) { delete static_cast<int*>(ptr); });
  std::weak_ptr<void> weakOwner = owner;
  std::vector<std::unique_ptr<std::shared_ptr<void>>> keepAlives;
  keepAlives.push_back(
      std::make_unique<std::shared_ptr<void>>(std::move(owner)));
  auto* retainedHolder = keepAlives.front().get();

  EXPECT_FALSE(releaseResourcesOrRetainKeepAlives(
      /*resourceLifetimeQuarantineRequired=*/false,
      /*keepAliveLifetimeQuarantineRequired=*/true,
      keepAlives,
      [&]() noexcept {
        cleanupRan = true;
        return true;
      }));

  EXPECT_TRUE(cleanupRan);
  EXPECT_FALSE(weakOwner.expired());
  delete retainedHolder;
  EXPECT_TRUE(weakOwner.expired());
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    DisabledQuiescingStillReleasesBuffers) {
  doca_gpu_verbs_qp_hl qp{};
  std::vector<FakeNicResources> nics(1);
  nics[0].qpSlots = {{.standaloneMain = &qp}};

  std::vector<std::string> events;
  quiesceQpsThenReleaseBuffers(
      false,
      nics,
      [&](doca_gpu_verbs_qp_group_hl*) {
        events.emplace_back("group");
        return DOCA_SUCCESS;
      },
      [&](doca_gpu_verbs_qp_hl*) {
        events.emplace_back("single");
        return DOCA_SUCCESS;
      },
      [&]() { events.emplace_back("release_gpu"); },
      [&]() { events.emplace_back("release_send_recv"); });

  EXPECT_EQ(
      events, (std::vector<std::string>{"release_gpu", "release_send_recv"}));
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    FailedQpTransitionStopsBeforeBufferRelease) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  doca_gpu_verbs_qp_hl qp{};
  std::vector<FakeNicResources> nics(1);
  nics[0].qpSlots = {{.standaloneMain = &qp}};

  EXPECT_DEATH(
      quiesceQpsThenReleaseBuffers(
          true,
          nics,
          [](doca_gpu_verbs_qp_group_hl*) { return DOCA_SUCCESS; },
          [](doca_gpu_verbs_qp_hl*) { return DOCA_ERROR_DRIVER; },
          []() { std::abort(); },
          []() { std::abort(); }),
      "QP transition failed; refusing to continue teardown.*"
      "qp_kind=standalone_main nic_index=0 qp_index=0");
}

} // namespace
} // namespace comms::prims::detail
