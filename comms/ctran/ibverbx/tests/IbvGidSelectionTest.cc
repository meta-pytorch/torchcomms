// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/ctran/ibverbx/IbvDevice.h"
#include "comms/ctran/ibverbx/Ibverbx.h"

namespace ibverbx {
namespace {

ibv_gid ipv4Gid(uint8_t address) {
  ibv_gid gid{};
  gid.raw[10] = 0xff;
  gid.raw[11] = 0xff;
  gid.raw[15] = address;
  return gid;
}

TEST(IbvGidSelectionTest, ExplicitIndexUsesOnlyRequestedEntry) {
  int queriedIndex = -1;
  auto index = detail::selectRoceGidIndex(
      4,
      3,
      [&](int candidate) -> Expected<ibv_gid> {
        queriedIndex = candidate;
        return ipv4Gid(3);
      },
      [](int) -> std::string {
        ADD_FAILURE() << "Explicit GID index should not query GID types";
        return {};
      });
  ASSERT_FALSE(index.hasError());
  EXPECT_EQ(*index, 3);
  EXPECT_EQ(queriedIndex, 3);
}

TEST(IbvGidSelectionTest, AutoSelectsIpv4RoceV2) {
  std::array<ibv_gid, 5> gids{};
  gids[1].raw[0] = 0xfe;
  gids[1].raw[1] = 0x80; // link-local
  gids[2] = ipv4Gid(2); // RoCE v1
  gids[3].raw[0] = 0x20; // RoCE v2, IPv6
  gids[3].raw[1] = 0x01;
  gids[4] = ipv4Gid(4); // RoCE v2, IPv4
  const std::array<std::string, 5> types{
      "", "RoCE v2", "RoCE v1", "RoCE v2", "RoCE v2"};

  auto index = detail::selectRoceGidIndex(
      gids.size(),
      -1,
      [&](int candidate) -> Expected<ibv_gid> { return gids.at(candidate); },
      [&](int candidate) { return types.at(candidate); });
  ASSERT_FALSE(index.hasError());
  EXPECT_EQ(*index, 4);
}

TEST(IbvGidSelectionTest, AutoFallsBackWhenGidTypesAreUnavailable) {
  auto index = detail::selectRoceGidIndex(
      4,
      -1,
      [](int candidate) -> Expected<ibv_gid> {
        return candidate < 2 ? ibv_gid{} : ipv4Gid(candidate);
      },
      [](int) { return std::string{}; });
  ASSERT_FALSE(index.hasError());
  EXPECT_EQ(*index, 3);
}

TEST(IbvGidSelectionTest, UntypedV1V2PairsChooseTheLastV2Candidate) {
  const std::array<ibv_gid, 4> gids{
      ipv4Gid(1), ipv4Gid(1), ipv4Gid(2), ipv4Gid(2)};
  auto index = detail::selectRoceGidIndex(
      gids.size(),
      -1,
      [&](int candidate) -> Expected<ibv_gid> { return gids.at(candidate); },
      [](int) { return std::string{}; });
  ASSERT_FALSE(index.hasError());
  EXPECT_EQ(*index, 3);
}

TEST(IbvGidSelectionTest, RejectsNonEthernetPortBeforeQueryingVerbs) {
  if (ibvInit().hasError()) {
    GTEST_SKIP() << "ibverbs unavailable on this host";
  }
  std::vector<IbvDevice> devices;
  try {
    auto available = IbvDevice::ibvGetDeviceList();
    if (available.hasError()) {
      GTEST_SKIP() << "no accessible IB device on this host";
    }
    devices = std::move(*available);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "cannot open IB device on this host: " << e.what();
  }
  if (devices.empty()) {
    GTEST_SKIP() << "no accessible IB device on this host";
  }
  ibv_port_attr port{};
  port.link_layer = IBV_LINK_LAYER_INFINIBAND;
  auto result = devices.front().resolveRoceGidIndex(1, port, -1);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().errNum, EINVAL);
}

TEST(IbvGidSelectionTest, PropagatesGidTypeReadFailure) {
  auto index = detail::selectRoceGidIndex(
      1,
      -1,
      [](int) -> Expected<ibv_gid> { return ipv4Gid(1); },
      [](int) -> Expected<std::string> {
        return makeUnexpected(Error(EACCES, "GID type unreadable"));
      });
  ASSERT_TRUE(index.hasError());
  EXPECT_EQ(index.error().errNum, EACCES);
}

TEST(IbvGidSelectionTest, AutoSkipsUnmappedGidTypeAndSelectsNextGid) {
  std::vector<int> queriedTypes;
  auto index = detail::selectRoceGidIndex(
      3,
      -1,
      [](int candidate) -> Expected<ibv_gid> { return ipv4Gid(candidate + 1); },
      [&](int candidate) -> Expected<std::string> {
        queriedTypes.push_back(candidate);
        if (candidate == 1) {
          return makeUnexpected(Error(EINVAL, "GID type not mapped"));
        }
        return std::string(candidate == 2 ? "RoCE v2" : "RoCE v1");
      });
  ASSERT_FALSE(index.hasError());
  EXPECT_EQ(*index, 2);
  EXPECT_EQ(queriedTypes, (std::vector<int>{0, 1, 2}));
}

TEST(IbvGidSelectionTest, RejectsInvalidOverridesAndEmptyTable) {
  auto query = [](int) -> Expected<ibv_gid> {
    ADD_FAILURE() << "Invalid index must not query the device";
    return ibv_gid{};
  };
  auto type = [](int) { return std::string{}; };
  EXPECT_EQ(
      detail::selectRoceGidIndex(4, -2, query, type).error().errNum, EINVAL);
  EXPECT_EQ(
      detail::selectRoceGidIndex(4, 4, query, type).error().errNum, EINVAL);
  EXPECT_EQ(
      detail::selectRoceGidIndex(0, -1, query, type).error().errNum, ENOENT);
}

TEST(IbvGidSelectionTest, PropagatesExplicitQueryFailure) {
  auto index = detail::selectRoceGidIndex(
      4,
      3,
      [](int) -> Expected<ibv_gid> { return makeUnexpected(Error(EIO)); },
      [](int) { return std::string{}; });
  ASSERT_TRUE(index.hasError());
  EXPECT_EQ(index.error().errNum, EIO);
}

} // namespace
} // namespace ibverbx
