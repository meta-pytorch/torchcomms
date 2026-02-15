// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstring>

#include <cuda_runtime.h>

#include "comms/pipes/NvmlFabricInfo.h"

namespace comms::pipes::tests {

TEST(NvmlFabricInfoTest, DefaultConstruction) {
  NvmlFabricInfo info;

  EXPECT_FALSE(info.available);
  EXPECT_EQ(info.cliqueId, 0u);

  char zeros[NvmlFabricInfo::kUuidLen]{};
  EXPECT_EQ(std::memcmp(info.clusterUuid, zeros, NvmlFabricInfo::kUuidLen), 0);
}

TEST(NvmlFabricInfoTest, UuidLenConstant) {
  EXPECT_EQ(NvmlFabricInfo::kUuidLen, 16);
}

TEST(NvmlFabricInfoTest, QueryDoesNotCrash) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  char busId[NvmlFabricInfo::kBusIdLen];
  ASSERT_EQ(
      cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, device),
      cudaSuccess);

  NvmlFabricInfo info = NvmlFabricInfo::query(busId);

  EXPECT_TRUE(info.available == true || info.available == false);
}

TEST(NvmlFabricInfoTest, QueryResultConsistency) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  char busId[NvmlFabricInfo::kBusIdLen];
  ASSERT_EQ(
      cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, device),
      cudaSuccess);

  NvmlFabricInfo info = NvmlFabricInfo::query(busId);

  char zeros[NvmlFabricInfo::kUuidLen]{};

  if (info.available) {
    EXPECT_NE(std::memcmp(info.clusterUuid, zeros, NvmlFabricInfo::kUuidLen), 0)
        << "Fabric info is available but clusterUuid is all-zero";
  } else {
    EXPECT_EQ(
        std::memcmp(info.clusterUuid, zeros, NvmlFabricInfo::kUuidLen), 0);
    EXPECT_EQ(info.cliqueId, 0u);
  }
}

TEST(NvmlFabricInfoTest, InvalidBusIdReturnsUnavailable) {
  NvmlFabricInfo info = NvmlFabricInfo::query("0000:FF:FF.F");

  EXPECT_FALSE(info.available);
}

TEST(NvmlFabricInfoTest, QueryIsDeterministic) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  char busId[NvmlFabricInfo::kBusIdLen];
  ASSERT_EQ(
      cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, device),
      cudaSuccess);

  NvmlFabricInfo a = NvmlFabricInfo::query(busId);
  NvmlFabricInfo b = NvmlFabricInfo::query(busId);

  EXPECT_EQ(a.available, b.available);
  EXPECT_EQ(a.cliqueId, b.cliqueId);
  EXPECT_EQ(
      std::memcmp(a.clusterUuid, b.clusterUuid, NvmlFabricInfo::kUuidLen), 0);
}

TEST(NvmlFabricInfoTest, MultiGpuConsistency) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Requires >= 2 GPUs, got " << deviceCount;
  }

  std::vector<NvmlFabricInfo> infos(deviceCount);
  for (int d = 0; d < deviceCount; ++d) {
    char busId[NvmlFabricInfo::kBusIdLen];
    ASSERT_EQ(
        cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, d),
        cudaSuccess);
    infos[d] = NvmlFabricInfo::query(busId);
  }

  for (int d = 1; d < deviceCount; ++d) {
    EXPECT_EQ(infos[0].available, infos[d].available)
        << "GPU 0 and GPU " << d << " disagree on fabric availability";
  }

  if (infos[0].available) {
    for (int d = 1; d < deviceCount; ++d) {
      EXPECT_EQ(
          std::memcmp(
              infos[0].clusterUuid,
              infos[d].clusterUuid,
              NvmlFabricInfo::kUuidLen),
          0)
          << "GPU 0 and GPU " << d << " have different clusterUuid";
      EXPECT_EQ(infos[0].cliqueId, infos[d].cliqueId)
          << "GPU 0 and GPU " << d << " have different cliqueId";
    }
  }
}

} // namespace comms::pipes::tests
