// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/HostCollectiveEngine.h"

#include <stdexcept>

#include <gtest/gtest.h>

namespace comms::prims {
namespace {

/*
 * These cover the Copy-Engine-only constructor, which is the half of the
 * engine that needs neither a transport nor a peer. The stream-memop and RDMA
 * paths are exercised by the collectives that drive real hardware.
 *
 * Construction resolves the CUDA driver entry points, so a host without a
 * driver cannot build an engine at all -- skip rather than fail there, since
 * that is the documented behaviour and not a defect.
 */
class HostCollectiveEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (cuda_driver_lazy_init() != 0) {
      GTEST_SKIP() << "CUDA driver entry points unavailable on this host";
    }
  }
};

// The Copy-Engine-only engine has no transport, so the RDMA half must refuse
// rather than dereference a null one.
TEST_F(HostCollectiveEngineTest, CopyEngineOnlyRejectsHostWriter) {
  HostCollectiveEngine engine(cudaStreamDefault);
  EXPECT_THROW((void)engine.hostWriter(/*peerRank=*/1), std::runtime_error);
}

// setStream is how a cached engine follows the caller's per-call stream; the
// accessor has to report the retarget, since every enqueue keys off it.
TEST_F(HostCollectiveEngineTest, SetStreamRetargetsTheEngine) {
  cudaStream_t first = nullptr;
  cudaStream_t second = nullptr;
  ASSERT_EQ(cudaStreamCreate(&first), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&second), cudaSuccess);

  HostCollectiveEngine engine(first);
  EXPECT_EQ(engine.stream(), first);

  engine.setStream(second);
  EXPECT_EQ(engine.stream(), second);

  EXPECT_EQ(cudaStreamDestroy(first), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(second), cudaSuccess);
}

} // namespace
} // namespace comms::prims
