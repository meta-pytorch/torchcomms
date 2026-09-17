// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportInternal.h"

namespace comms::prims::detail {
namespace {

struct FakeQpSlotResources {
  int* group{nullptr};
  int* standaloneMain{nullptr};
  int* loopback{nullptr};
};

struct FakeNicResources {
  std::vector<FakeQpSlotResources> qpSlots;
};

TEST(
    MultipeerIbgdaTransportCleanupTest,
    TransitionsEveryQpBeforeFirstBufferRelease) {
  int groupedQp = 1;
  int groupedLoopbackQp = 2;
  int standaloneMainQp = 3;
  int secondGroupedQp = 4;
  int secondGroupedLoopbackQp = 5;
  std::vector<FakeNicResources> nics(2);
  nics[0].qpSlots = {
      {.group = &groupedQp, .loopback = &groupedLoopbackQp},
      {.standaloneMain = &standaloneMainQp},
  };
  nics[1].qpSlots = {
      {.group = &secondGroupedQp, .loopback = &secondGroupedLoopbackQp},
      {},
  };

  std::vector<std::string> events;
  quiesceQpsThenReleaseBuffers(
      true,
      nics,
      [&](int* qp) {
        events.push_back("group:" + std::to_string(*qp));
        return DOCA_SUCCESS;
      },
      [&](int* qp) {
        events.push_back("single:" + std::to_string(*qp));
        return DOCA_SUCCESS;
      },
      [&]() { events.emplace_back("release_gpu"); },
      [&]() { events.emplace_back("release_send_recv"); });

  EXPECT_EQ(
      events,
      (std::vector<std::string>{
          "group:1",
          "single:2",
          "single:3",
          "group:4",
          "single:5",
          "release_gpu",
          "release_send_recv",
      }));
}

TEST(
    MultipeerIbgdaTransportCleanupTest,
    DisabledQuiescingStillReleasesBuffers) {
  int qp = 1;
  std::vector<FakeNicResources> nics(1);
  nics[0].qpSlots = {{.standaloneMain = &qp}};

  std::vector<std::string> events;
  quiesceQpsThenReleaseBuffers(
      false,
      nics,
      [&](int*) {
        events.emplace_back("group");
        return DOCA_SUCCESS;
      },
      [&](int*) {
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
  int qp = 1;
  std::vector<FakeNicResources> nics(1);
  nics[0].qpSlots = {{.standaloneMain = &qp}};

  EXPECT_DEATH(
      quiesceQpsThenReleaseBuffers(
          true,
          nics,
          [](int*) { return DOCA_SUCCESS; },
          [](int*) { return DOCA_ERROR_DRIVER; },
          []() { std::abort(); },
          []() { std::abort(); }),
      "QP transition failed; refusing to release memory.*"
      "qp_kind=standalone_main nic_index=0 qp_index=0");
}

} // namespace
} // namespace comms::prims::detail
