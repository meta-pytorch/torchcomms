// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Distributed correctness test for the low-latency (Proto=LL) IBGDA send/recv
// path. Rank 0 sends a known byte pattern via detail::send<..., LL>; rank 1
// receives it over the same channel and verifies every byte. 2 ranks; intended
// for 2-host x 1-GPU (nnodes=2, ppn=1) over RDMA, but also works
// 2-ranks-on-1-node. Skips gracefully when no RDMA transport is available.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <memory>

#include "comms/prims/transport/llx/tests/SendRecvSweepHarness.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::prims::tests {

class LLIbgdaSendRecvFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_F(LLIbgdaSendRecvFixture, SendRecvRoundTrip) {
  test::runSendRecvSweep(
      globalRank, numRanks, localRank, &test::launchLLSendRecv, "LL");
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
