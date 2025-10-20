// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/common/IpcMemHandler.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>
#include "comms/common/tests/TestBaselineBootstrap.h"
#include "comms/rcclx/develop/meta/lib/tests/RcclxTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::rcclx;
using namespace meta::comms;

class IpcMemHandlerTest : public RcclxBaseTestFixture {};

TEST_F(IpcMemHandlerTest, exchangeMemPtrs) {
  // must test with multiple GPUs
  ASSERT_GT(numRanks, 1);
  CUDA_CHECK(cudaSetDevice(globalRank));

  ncclComm_t comm{nullptr};
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  NCCL_CHECK(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  XLOGF(INFO, "rank {} init done; total ranks: {}", globalRank, numRanks);

  int myValue = globalRank;
  DeviceBuffer myBuf(sizeof(myValue));
  CUDA_CHECK(
      cudaMemcpy(myBuf.get(), &myValue, sizeof(myValue), cudaMemcpyDefault));

  auto bootstrap = std::make_shared<TestBaselineBootstrap>(comm);
  IpcMemHandler handler(bootstrap, globalRank, numRanks);
  handler.addSelfDeviceMemPtr(myBuf.get());

  // exchangeMemPtrs must be called before getPeerDeviceMemPtr
  EXPECT_THROW(handler.getPeerDeviceMemPtr(0), std::runtime_error);

  handler.exchangeMemPtrs();
  for (int i = 0; i < numRanks; ++i) {
    int* peerDevPtr = static_cast<int*>(handler.getPeerDeviceMemPtr(i));
    int peerVal{-1};
    CUDA_CHECK(
        cudaMemcpy(&peerVal, peerDevPtr, sizeof(peerVal), cudaMemcpyDefault));
    EXPECT_EQ(peerVal, i);
    if (i == globalRank) {
      // If peer is self, the handler should return local ptr
      EXPECT_EQ(peerDevPtr, myBuf.get());
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
