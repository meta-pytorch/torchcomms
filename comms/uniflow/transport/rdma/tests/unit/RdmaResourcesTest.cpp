// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaResources.h"

#include <cstdio>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/uniflow/drivers/ibverbs/mock/MockIbvApi.h"

using namespace uniflow;
using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

// --- dataDirectDomainMatchesGpu ---

TEST(DataDirectDomainMatchTest, MatchingDomain) {
  EXPECT_TRUE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0008:00/0008:00:00.0/infiniband/mlx5_4",
      "00000008:06:00.0"));
}

TEST(DataDirectDomainMatchTest, MismatchingDomain) {
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0009:00/0009:00:00.0/infiniband/mlx5_0",
      "00000008:06:00.0"));
}

TEST(DataDirectDomainMatchTest, DiffersOnlyByLeadingZeros) {
  /*
   * The domain is compared as a hex value, so widths/leading zeros are
   * irrelevant: GPU "8" matches data-direct "pci0008".
   */
  EXPECT_TRUE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0008:00/0008:00:00.0", "0008:06:00.0"));
}

TEST(DataDirectDomainMatchTest, NonZeroDomains) {
  EXPECT_TRUE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0018:00/0018:00:00.0", "00000018:07:00.0"));
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0018:00/0018:00:00.0", "00000019:07:00.0"));
}

TEST(DataDirectDomainMatchTest, MalformedInputsReturnFalse) {
  // Missing ':' in the bus id.
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0008:00/0008:00:00.0", "00000008"));
  // No "/pci" token in the sysfs path.
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/no-domain-here", "0008:06:00.0"));
  // Non-hex domain token must not silently parse to 0 and match a pci0000 path.
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0000:00/0000:00:00.0", "zzzz:06:00.0"));
  // A signed token (e.g. "+8") must be rejected, not accepted by strtol.
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0008:00/0008:00:00.0", "+8:06:00.0"));
  // An overflowing token (strtol ERANGE -> LONG_MAX) must be rejected, not
  // accepted as a valid value.
  EXPECT_FALSE(dataDirectDomainMatchesGpu(
      "/sys/devices/pci0008:00/0008:00:00.0", "ffffffffffffffffffff:06:00.0"));
  // Empty inputs.
  EXPECT_FALSE(dataDirectDomainMatchesGpu("", ""));
}

// --- selectDataDirectNicsForGpu ---

namespace {

/*
 * Configure a mock exposing three NICs: mlx5_0 (domain 0008), mlx5_1 (domain
 * 0009), mlx5_2 (no data-direct path). The device/context handles are opaque —
 * the code only compares these pointers, never dereferences them — so distinct
 * fake addresses suffice. devList is a function-local static so getDeviceList
 * can hand back a stable ibv_device** that outlives this setup call.
 */
void configureThreeNicMock(NiceMock<MockIbvApi>& mock) {
  static ibv_device* devList[3] = {
      reinterpret_cast<ibv_device*>(0x1001),
      reinterpret_cast<ibv_device*>(0x1002),
      reinterpret_cast<ibv_device*>(0x1003),
  };
  ibv_device* const dev0 = devList[0];
  ibv_device* const dev1 = devList[1];
  ibv_device* const dev2 = devList[2];
  ibv_context* const ctx0 = reinterpret_cast<ibv_context*>(0x2001);
  ibv_context* const ctx1 = reinterpret_cast<ibv_context*>(0x2002);
  ibv_context* const ctx2 = reinterpret_cast<ibv_context*>(0x2003);

  ON_CALL(mock, getDeviceList(_)).WillByDefault([](int* numDevices) {
    *numDevices = 3;
    return Result<ibv_device**>(devList);
  });
  ON_CALL(mock, freeDeviceList(_)).WillByDefault(Return(Ok()));
  ON_CALL(mock, closeDevice(_)).WillByDefault(Return(Ok()));

  ON_CALL(mock, getDeviceName(dev0))
      .WillByDefault(Return(Result<const char*>("mlx5_0")));
  ON_CALL(mock, getDeviceName(dev1))
      .WillByDefault(Return(Result<const char*>("mlx5_1")));
  ON_CALL(mock, getDeviceName(dev2))
      .WillByDefault(Return(Result<const char*>("mlx5_2")));

  ON_CALL(mock, openDevice(dev0))
      .WillByDefault(Return(Result<ibv_context*>(ctx0)));
  ON_CALL(mock, openDevice(dev1))
      .WillByDefault(Return(Result<ibv_context*>(ctx1)));
  ON_CALL(mock, openDevice(dev2))
      .WillByDefault(Return(Result<ibv_context*>(ctx2)));

  ON_CALL(mock, mlx5dvGetDataDirectSysfsPath(ctx0, _, _))
      .WillByDefault([](ibv_context*, char* buf, size_t len) {
        std::snprintf(buf, len, "/sys/devices/pci0008:00/0008:00:00.0");
        return Ok();
      });
  ON_CALL(mock, mlx5dvGetDataDirectSysfsPath(ctx1, _, _))
      .WillByDefault([](ibv_context*, char* buf, size_t len) {
        std::snprintf(buf, len, "/sys/devices/pci0009:00/0009:00:00.0");
        return Ok();
      });
  // mlx5_2 has no data-direct path.
  ON_CALL(mock, mlx5dvGetDataDirectSysfsPath(ctx2, _, _))
      .WillByDefault(Return(Status(ErrCode::NotImplemented)));
}

} // namespace

TEST(SelectDataDirectNicsTest, KeepsOnlyDomainMatchedNic) {
  NiceMock<MockIbvApi> mock;
  configureThreeNicMock(mock);

  auto matched = selectDataDirectNicsForGpu(
      mock, {"mlx5_0", "mlx5_1", "mlx5_2"}, "00000008:06:00.0");

  EXPECT_EQ(matched, std::vector<std::string>({"mlx5_0"}));
}

TEST(SelectDataDirectNicsTest, MatchesDifferentDomain) {
  NiceMock<MockIbvApi> mock;
  configureThreeNicMock(mock);

  auto matched = selectDataDirectNicsForGpu(
      mock, {"mlx5_0", "mlx5_1", "mlx5_2"}, "00000009:06:00.0");

  EXPECT_EQ(matched, std::vector<std::string>({"mlx5_1"}));
}

TEST(SelectDataDirectNicsTest, NoMatchReturnsEmpty) {
  NiceMock<MockIbvApi> mock;
  configureThreeNicMock(mock);

  auto matched = selectDataDirectNicsForGpu(
      mock, {"mlx5_0", "mlx5_1", "mlx5_2"}, "00000018:06:00.0");

  EXPECT_TRUE(matched.empty());
}

TEST(SelectDataDirectNicsTest, PreservesInputOrder) {
  NiceMock<MockIbvApi> mock;
  configureThreeNicMock(mock);

  /*
   * Both mlx5_0 and (hypothetically) another 0008 NIC would match; here only
   * mlx5_0 does, but the candidate order must be respected in the output.
   */
  auto matched = selectDataDirectNicsForGpu(
      mock, {"mlx5_2", "mlx5_1", "mlx5_0"}, "00000008:06:00.0");

  EXPECT_EQ(matched, std::vector<std::string>({"mlx5_0"}));
}

TEST(SelectDataDirectNicsTest, SkipsUnknownCandidate) {
  NiceMock<MockIbvApi> mock;
  configureThreeNicMock(mock);

  // "mlx5_99" is not in the device list; it is silently skipped.
  auto matched = selectDataDirectNicsForGpu(
      mock, {"mlx5_99", "mlx5_0"}, "00000008:06:00.0");

  EXPECT_EQ(matched, std::vector<std::string>({"mlx5_0"}));
}
