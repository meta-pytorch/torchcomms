// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/prims/tests/ProgressRecvAcquireInstantiationTest.cuh"

// This is a compile-and-link guard, not a behavioral test: the acquire/release
// recv seam is device-only template code, so the failure mode it protects
// against (an argument list that no longer matches progress_recv_ready) is a
// build break, and the build of the kernel TU is the assertion. Behavioral
// coverage of the same path lives in the multi-GPU transport tests.
TEST(ProgressRecvAcquireInstantiationTest, AllTransportProtocolPairsCompile) {
  EXPECT_TRUE(
      comms::prims::test::progress_recv_acquire_instantiations_linked());
}
