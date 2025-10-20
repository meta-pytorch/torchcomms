// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/torchcomms/transport/RdmaTransport.h"
#include "comms/utils/cvars/nccl_cvars.h"

TEST(RdmaMemoryTest, QuerySupport) {
  setenv("NCCL_DEBUG", "WARN", 0);
  ncclCvarInit();

  constexpr int nIters = 100;
  for (auto x = 0; x < nIters; x++) {
#ifdef TEST_NO_IB_BACKEND
    EXPECT_FALSE(torch::comms::RdmaTransport::supported());
#else
    EXPECT_TRUE(torch::comms::RdmaTransport::supported());
#endif
  }
}
