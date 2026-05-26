// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comms/pipes/MultimemNvlTransport.h"
#include "comms/pipes/tests/MultimemNvlTransportTest.cuh"
#include "comms/testinfra/DistEnvironmentBase.h"
#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::tests {

class MultimemNvlTransportUnalignedTestFixture
    : public ::testing::Test,
      public meta::comms::DistBaseTest {
 protected:
  void SetUp() override {
    distSetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    distTearDown();
  }
};

std::vector<int> identityRankMap(int size) {
  std::vector<int> rankMap(static_cast<std::size_t>(size));
  for (int rank = 0; rank < size; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

std::shared_ptr<meta::comms::IBootstrap> makeBootstrap(
    const std::string& prefix) {
  return std::shared_ptr<meta::comms::IBootstrap>(
      meta::comms::createBootstrap(prefix));
}

bool allRanksMultimemEligible(
    const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
    int rank,
    int nRanks,
    int cudaDevice) {
  std::vector<int> eligible(static_cast<std::size_t>(nRanks));
  eligible[static_cast<std::size_t>(rank)] =
      MultimemNvlTransport::isEligible(nRanks, cudaDevice) ? 1 : 0;
  auto rc =
      bootstrap->allGather(eligible.data(), sizeof(int), rank, nRanks).get();
  EXPECT_EQ(rc, 0);
  if (rc != 0) {
    return false;
  }
  for (const auto value : eligible) {
    if (value == 0) {
      return false;
    }
  }
  return true;
}

uint8_t expectedByte(int rank, std::size_t byte) {
  return static_cast<uint8_t>((rank * 17 + byte * 3 + 5) & 0xFF);
}

TEST_F(
    MultimemNvlTransportUnalignedTestFixture,
    StoreSupportsUnalignedSrcDstAndSize) {
  auto bootstrap = makeBootstrap("multimem_nvl_transport_unaligned_test");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kStoreBytes = 257;
  constexpr std::size_t kRankStride = kStoreBytes;
  constexpr std::size_t kDstSkew = 1;
  const std::size_t dataBufferSize =
      kDstSkew + kRankStride * static_cast<std::size_t>(numRanks);

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      MultimemNvlTransportConfig{
          .dataBufferSize = dataBufferSize,
          .userSignalCount = 1,
      });
  transport.exchange();
  const auto deviceTransport = transport.getDeviceTransport();

  std::vector<uint8_t> hostSrc(kStoreBytes + 1, 0xA5);
  for (std::size_t byte = 0; byte < kStoreBytes; ++byte) {
    hostSrc[byte + 1] = expectedByte(globalRank, byte);
  }
  DeviceBuffer src(hostSrc.size());
  CUDACHECK_TEST(cudaMemcpy(
      src.get(), hostSrc.data(), hostSrc.size(), cudaMemcpyHostToDevice));

  const std::size_t dstOffset =
      kDstSkew + static_cast<std::size_t>(globalRank) * kRankStride;
  test::launchMultimemRawStoreTest(
      deviceTransport,
      static_cast<const char*>(src.get()) + 1,
      dstOffset,
      kStoreBytes,
      numRanks);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<uint8_t> hostDst(dataBufferSize);
  CUDACHECK_TEST(cudaMemcpy(
      hostDst.data(),
      deviceTransport.localData,
      hostDst.size(),
      cudaMemcpyDeviceToHost));

  for (int rank = 0; rank < numRanks; ++rank) {
    const std::size_t rankOffset =
        kDstSkew + static_cast<std::size_t>(rank) * kRankStride;
    for (std::size_t byte = 0; byte < kStoreBytes; ++byte) {
      ASSERT_EQ(hostDst[rankOffset + byte], expectedByte(rank, byte))
          << "rank=" << rank << " byte=" << byte;
    }
  }
}

TEST_F(
    MultimemNvlTransportUnalignedTestFixture,
    StoreUsesAlignedBulkForAlignedSrcDstAndUnalignedSize) {
  auto bootstrap =
      makeBootstrap("multimem_nvl_transport_aligned_bulk_tail_test");
  if (!allRanksMultimemEligible(bootstrap, globalRank, numRanks, localRank)) {
    GTEST_SKIP() << "CUDA multimem/NVLS multicast is not eligible";
  }

  constexpr std::size_t kStoreBytes = 257;
  constexpr std::size_t kRankStride = 272;
  const std::size_t dataBufferSize =
      kRankStride * static_cast<std::size_t>(numRanks);

  MultimemNvlTransport transport(
      bootstrap,
      globalRank,
      identityRankMap(numRanks),
      MultimemNvlTransportConfig{
          .dataBufferSize = dataBufferSize,
          .userSignalCount = 1,
      });
  transport.exchange();
  const auto deviceTransport = transport.getDeviceTransport();

  std::vector<uint8_t> hostSrc(kStoreBytes);
  for (std::size_t byte = 0; byte < kStoreBytes; ++byte) {
    hostSrc[byte] = expectedByte(globalRank, byte);
  }
  DeviceBuffer src(hostSrc.size());
  CUDACHECK_TEST(cudaMemcpy(
      src.get(), hostSrc.data(), hostSrc.size(), cudaMemcpyHostToDevice));

  const std::size_t dstOffset =
      static_cast<std::size_t>(globalRank) * kRankStride;
  test::launchMultimemRawStoreTest(
      deviceTransport,
      static_cast<const char*>(src.get()),
      dstOffset,
      kStoreBytes,
      numRanks);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<uint8_t> hostDst(dataBufferSize);
  CUDACHECK_TEST(cudaMemcpy(
      hostDst.data(),
      deviceTransport.localData,
      hostDst.size(),
      cudaMemcpyDeviceToHost));

  for (int rank = 0; rank < numRanks; ++rank) {
    const std::size_t rankOffset = static_cast<std::size_t>(rank) * kRankStride;
    for (std::size_t byte = 0; byte < kStoreBytes; ++byte) {
      ASSERT_EQ(hostDst[rankOffset + byte], expectedByte(rank, byte))
          << "rank=" << rank << " byte=" << byte;
    }
  }
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::DistEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
