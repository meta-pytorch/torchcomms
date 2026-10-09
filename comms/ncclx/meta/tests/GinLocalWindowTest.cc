// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <vector>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "nccl.h"
#include "nccl_device.h"

#include "comms/ncclx/meta/tests/GinLocalWindowTestKernels.h"

namespace {

constexpr size_t kCount = 1024;
constexpr int kRankStride = 100000;

} // namespace

class GinLocalWindowTest : public NcclxBaseTestFixture {};

TEST_F(GinLocalWindowTest, PutFromLocalOnlyWindow) {
  const size_t bytes = kCount * sizeof(int);
  void* recvBuf = nullptr;
  void* srcBuf = nullptr;
  void* keptBuf = nullptr;
  {
    ncclx::test::NcclCommRAII comm(
        globalRank, numRanks, localRank, bootstrap_.get());
    ASSERT_NE(nullptr, comm.get());

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));
    if (!props.deviceApiSupport || props.ginType == NCCL_GIN_TYPE_NONE) {
      GTEST_SKIP() << "needs the device API and a GIN backend "
                      "(NCCL_GIN_ENABLE=1 on a host with RDMA NICs)";
    }

    ASSERT_EQ(ncclSuccess, ncclMemAlloc(&recvBuf, bytes * numRanks));
    ASSERT_EQ(ncclSuccess, ncclMemAlloc(&srcBuf, bytes));
    ASSERT_EQ(ncclSuccess, ncclMemAlloc(&keptBuf, bytes));

    ncclWindow_t recvWin;
    ASSERT_EQ(
        ncclSuccess,
        ncclCommWindowRegister(
            comm,
            recvBuf,
            bytes * numRanks,
            &recvWin,
            NCCL_WIN_COLL_SYMMETRIC));

    ncclDevComm devComm;
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.worldGinBarrierCount = 1;
    reqs.ginSignalCount = 1;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    ASSERT_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &devComm));

    // Local-only registration needs GIN up (the devcomm above) and is not
    // collective, so each rank registers on its own.
    ncclWindow_t srcWin;
    ASSERT_EQ(
        ncclSuccess,
        ncclCommWindowRegister(
            comm, srcBuf, bytes, &srcWin, NCCL_WIN_LOCAL_ONLY));
    // Only rank 0 registers this one. If the registration were collective,
    // rank 0 would block in it and the allreduce below would hang.
    ncclWindow_t keptWin = nullptr;
    if (globalRank == 0) {
      ASSERT_EQ(
          ncclSuccess,
          ncclCommWindowRegister(
              comm, keptBuf, bytes, &keptWin, NCCL_WIN_LOCAL_ONLY));
    }

    std::vector<int> src(kCount);
    for (size_t i = 0; i < kCount; i++) {
      src[i] = globalRank * kRankStride + static_cast<int>(i);
    }
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(srcBuf, src.data(), bytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(recvBuf, 0, bytes * numRanks));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(
        ncclSuccess,
        ncclAllReduce(recvBuf, recvBuf, 1, ncclInt, ncclSum, comm, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemset(recvBuf, 0, bytes * numRanks));
    ASSERT_EQ(
        cudaSuccess,
        launchPutFromLocalWindow(srcWin, recvWin, bytes, devComm, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const int prev = (globalRank + numRanks - 1) % numRanks;
    std::vector<int> expected(kCount);
    for (size_t i = 0; i < kCount; i++) {
      expected[i] = prev * kRankStride + static_cast<int>(i);
    }
    std::vector<int> got(kCount);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            got.data(),
            static_cast<char*>(recvBuf) + prev * bytes,
            bytes,
            cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, got);

    EXPECT_EQ(ncclSuccess, ncclCommWindowDeregister(comm, srcWin));
    EXPECT_EQ(ncclSuccess, ncclDevCommDestroy(comm, &devComm));
    EXPECT_EQ(ncclSuccess, ncclCommWindowDeregister(comm, recvWin));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    // keptWin (rank 0) stays registered: destroying the communicator must
    // release it.
  }
  EXPECT_EQ(ncclSuccess, ncclMemFree(keptBuf));
  EXPECT_EQ(ncclSuccess, ncclMemFree(srcBuf));
  EXPECT_EQ(ncclSuccess, ncclMemFree(recvBuf));
}

// Needs no GIN backend, so it runs wherever the 2-rank target runs.
TEST_F(
    GinLocalWindowTest,
    LocalOnlyIsRejectedWithoutGinAndWithCollectiveFlags) {
  const size_t bytes = kCount * sizeof(int);
  void* buf = nullptr;
  {
    ncclx::test::NcclCommRAII comm(
        globalRank, numRanks, localRank, bootstrap_.get());
    ASSERT_NE(nullptr, comm.get());
    ASSERT_EQ(ncclSuccess, ncclMemAlloc(&buf, bytes));

    // Non-collective, so rank 0 alone can be rejected without the other ranks.
    if (globalRank == 0) {
      ncclWindow_t win = nullptr;
      EXPECT_EQ(
          ncclInvalidUsage,
          ncclCommWindowRegister(comm, buf, bytes, &win, NCCL_WIN_LOCAL_ONLY));
      EXPECT_EQ(
          ncclInvalidArgument,
          ncclCommWindowRegister(
              comm,
              buf,
              bytes,
              &win,
              NCCL_WIN_LOCAL_ONLY | NCCL_WIN_COLL_SYMMETRIC));
    }

    // Every rank is still in step after rank 0's rejected registrations.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(
        ncclSuccess,
        ncclAllReduce(buf, buf, 1, ncclInt, ncclSum, comm, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  }
  EXPECT_EQ(ncclSuccess, ncclMemFree(buf));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
