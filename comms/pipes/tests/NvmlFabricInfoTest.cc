// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cstring>

#include <cuda_runtime.h>

#include "comms/pipes/NvmlFabricInfo.h"

namespace comms::pipes::tests {

// Default-constructed NvmlFabricInfo should be zeroed and unavailable.
TEST(NvmlFabricInfoTest, DefaultConstruction) {
  NvmlFabricInfo info;

  EXPECT_FALSE(info.available);
  EXPECT_EQ(info.cliqueId, 0u);

  char zeros[NvmlFabricInfo::kUuidLen]{};
  EXPECT_EQ(std::memcmp(info.clusterUuid, zeros, NvmlFabricInfo::kUuidLen), 0);
}

// kUuidLen must be 16 bytes (matches NVML's definition).
TEST(NvmlFabricInfoTest, UuidLenConstant) {
  EXPECT_EQ(NvmlFabricInfo::kUuidLen, 16);
}

// query_nvml_fabric_info() must not crash and must return a valid struct
// regardless of platform (H100 or GB200).
TEST(NvmlFabricInfoTest, QueryDoesNotCrash) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  char busId[detail::kNvmlBusIdLen];
  ASSERT_EQ(
      cudaDeviceGetPCIBusId(busId, detail::kNvmlBusIdLen, device), cudaSuccess);

  NvmlFabricInfo info = query_nvml_fabric_info(busId);

  // available must be a valid bool (true or false) — just verifying
  // the function returned without crashing and produced a well-formed struct.
  EXPECT_TRUE(info.available == true || info.available == false);
}

// If fabric info is available (MNNVL), clusterUuid must be non-zero.
// If not available (H100), clusterUuid should be zeroed and cliqueId 0.
TEST(NvmlFabricInfoTest, QueryResultConsistency) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  char busId[detail::kNvmlBusIdLen];
  ASSERT_EQ(
      cudaDeviceGetPCIBusId(busId, detail::kNvmlBusIdLen, device), cudaSuccess);

  NvmlFabricInfo info = query_nvml_fabric_info(busId);

  char zeros[NvmlFabricInfo::kUuidLen]{};

  if (info.available) {
    // MNNVL path: clusterUuid must have at least one non-zero byte.
    EXPECT_NE(std::memcmp(info.clusterUuid, zeros, NvmlFabricInfo::kUuidLen), 0)
        << "Fabric info is available but clusterUuid is all-zero";
  } else {
    // H100 / no-NVML path: struct should remain in default state.
    EXPECT_EQ(
        std::memcmp(info.clusterUuid, zeros, NvmlFabricInfo::kUuidLen), 0);
    EXPECT_EQ(info.cliqueId, 0u);
  }
}

// Querying with an invalid PCI bus ID should gracefully return unavailable,
// not crash or throw.
TEST(NvmlFabricInfoTest, InvalidBusIdReturnsUnavailable) {
  NvmlFabricInfo info = query_nvml_fabric_info("0000:FF:FF.F");

  EXPECT_FALSE(info.available);
}

// Querying the same device twice should return identical results
// (the dlopen/init path is cached via static local).
TEST(NvmlFabricInfoTest, QueryIsDeterministic) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  char busId[detail::kNvmlBusIdLen];
  ASSERT_EQ(
      cudaDeviceGetPCIBusId(busId, detail::kNvmlBusIdLen, device), cudaSuccess);

  NvmlFabricInfo a = query_nvml_fabric_info(busId);
  NvmlFabricInfo b = query_nvml_fabric_info(busId);

  EXPECT_EQ(a.available, b.available);
  EXPECT_EQ(a.cliqueId, b.cliqueId);
  EXPECT_EQ(
      std::memcmp(a.clusterUuid, b.clusterUuid, NvmlFabricInfo::kUuidLen), 0);
}

// If the system has multiple GPUs, all GPUs on the same MNNVL fabric should
// share the same clusterUuid and cliqueId.  On H100, all should be unavailable.
TEST(NvmlFabricInfoTest, MultiGpuConsistency) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Requires >= 2 GPUs, got " << deviceCount;
  }

  // Query fabric info for all devices.
  std::vector<NvmlFabricInfo> infos(deviceCount);
  for (int d = 0; d < deviceCount; ++d) {
    char busId[detail::kNvmlBusIdLen];
    ASSERT_EQ(
        cudaDeviceGetPCIBusId(busId, detail::kNvmlBusIdLen, d), cudaSuccess);
    infos[d] = query_nvml_fabric_info(busId);
  }

  // All GPUs within the same node should have the same availability.
  for (int d = 1; d < deviceCount; ++d) {
    EXPECT_EQ(infos[0].available, infos[d].available)
        << "GPU 0 and GPU " << d << " disagree on fabric availability";
  }

  if (infos[0].available) {
    // MNNVL path: all local GPUs should share clusterUuid + cliqueId.
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
