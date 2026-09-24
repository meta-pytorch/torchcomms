// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <vector>

#include "AlgoInit.h"
#include "comm.h"

namespace {

class AlgoInitTest : public ::testing::Test {
 protected:
  // Brings up nRanks real comms in this process, one per device.
  //
  // Rank count is per-test rather than fixture-wide because it is itself an
  // input to the decision under test. Four ranks exercise the rank-count
  // guard; each case must use a layout whose guard returns before DDA
  // construction (DDA engages only at NRANKS == 8, IpcGpuBarrier.cuh, and
  // same-process IPC imports are invalid).
  //
  // The device count is asserted rather than skipped: coming up short means
  // num_gpus on the BUCK target is too low, which should be red rather than a
  // green run that tested nothing.
  void initComms(int nRanks) {
    int numDevices = 0;
    ASSERT_EQ(hipGetDeviceCount(&numDevices), hipSuccess);
    ASSERT_GE(numDevices, nRanks)
        << "test needs " << nRanks << " GPUs but the host has " << numDevices
        << "; check num_gpus on the BUCK target";

    comms_.resize(nRanks);
    // A null devlist means devices 0..nRanks-1.
    ASSERT_EQ(ncclCommInitAll(comms_.data(), nRanks, nullptr), ncclSuccess);
  }

  void TearDown() override {
    for (ncclComm_t comm : comms_) {
      if (comm != nullptr) {
        ncclCommDestroy(comm);
      }
    }
  }

  std::vector<ncclComm_t> comms_;
};

// initAlgoFactory's nNodes > 1 branch is deliberately not covered here. A
// single process cannot make it the only guard that fires: either the rank
// count is not NRANKS, or every rank shares a process. Either way another
// guard would return nullptr too, so such a test would still pass with the
// multi-node check deleted. It belongs in a two-node distributed test.

// DDA engages only at NRANKS == 8 ranks; this comm has four. Rank 0 alone is
// enough: the guard reads nRanks, which is identical across the comms.
TEST_F(AlgoInitTest, RankCountOtherThanDdaNRanksDisablesDda) {
  ASSERT_NO_FATAL_FAILURE(initComms(4));

  ASSERT_EQ(comms_[0]->nNodes, 1);
  EXPECT_EQ(initAlgoFactory(comms_[0]), nullptr);
}

} // namespace
