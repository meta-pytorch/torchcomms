// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportInternal.h"

namespace comms::prims::detail {
namespace {

struct FakeNicResources {
  std::vector<IbgdaQpSlotResources> qpSlots;
};

using Event = std::pair<std::string, const void*>;

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
