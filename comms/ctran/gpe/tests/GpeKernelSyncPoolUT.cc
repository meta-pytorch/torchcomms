// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"

class GpeKernelSyncPoolTest : public ::testing::Test {
 public:
  int cudaDev;
  GpeKernelSyncPoolTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);
  }
};

TEST_F(GpeKernelSyncPoolTest, Initialize) {
  constexpr int poolSize = 1000;
  auto pool = std::make_unique<GpeKernelSyncPool>(poolSize);

  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->size(), poolSize);
  EXPECT_EQ(pool->capacity(), poolSize);
}

TEST_F(GpeKernelSyncPoolTest, PopTest) {
  constexpr int poolSize = 10;
  auto pool = std::make_unique<GpeKernelSyncPool>(poolSize);

  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->size(), poolSize);
  EXPECT_EQ(pool->capacity(), poolSize);

  for (int i = 0; i < poolSize; ++i) {
    pool->pop();
    EXPECT_EQ(pool->size(), poolSize - (i + 1));
    // Capacity is unchanged
    EXPECT_EQ(pool->capacity(), poolSize);
  }

  auto another_kernel_sync = pool->pop();
  EXPECT_EQ(another_kernel_sync, nullptr);
}

TEST_F(GpeKernelSyncPoolTest, ReclaimTest) {
  constexpr int poolSize = 10;
  constexpr int popSize = 6;
  constexpr int reclaimSize = 2;
  auto pool = std::make_unique<GpeKernelSyncPool>(poolSize);

  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->size(), poolSize);

  std::vector<ctran::algos::GpeKernelSync*> allocated_kernel_syncs;
  for (int i = 0; i < popSize; ++i) {
    auto kernel_sync = pool->pop();
    ASSERT_NE(kernel_sync, nullptr);
    allocated_kernel_syncs.push_back(kernel_sync);
  }
  EXPECT_EQ(pool->size(), poolSize - popSize);
  // Capacity is unchanged
  EXPECT_EQ(pool->capacity(), poolSize);

  for (int i = 0; i < reclaimSize; ++i) {
    allocated_kernel_syncs[i]->reset();
  }
  EXPECT_EQ(pool->size(), poolSize - popSize);
  // Capacity is unchanged
  EXPECT_EQ(pool->capacity(), poolSize);

  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize - popSize + reclaimSize);
  // Capacity is unchanged
  EXPECT_EQ(pool->capacity(), poolSize);
}

TEST_F(GpeKernelSyncPoolTest, allocGpeKernelSyncs) {
  constexpr int poolSize = 10;
  constexpr int popSize = 3;
  constexpr int niters = 10;
  int nworkers[niters] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  auto pool = std::make_unique<GpeKernelSyncPool>(poolSize);
  std::vector<ctran::algos::GpeKernelSync*> gpeKernelSyncs;
  gpeKernelSyncs.reserve(popSize);

  for (int iter = 0; iter < niters; ++iter) {
    int nw = nworkers[iter];
    // alloc from pool. The first reclaim happens on the ceil(poolSize /
    // popSize) th iteration,
    ::allocGpeKernelSyncs(
        pool.get(),
        popSize,
        /*nworkers=*/nw,
        gpeKernelSyncs);

    ASSERT_EQ(gpeKernelSyncs.size(), popSize) << "iter " << iter;

    for (int i = 0; i < popSize; i++) {
      auto* g = gpeKernelSyncs[i];

      // check null and pool status
      ASSERT_NE(g, nullptr) << "iter " << iter << " gpeKernelSync[" << i << "]";
      ASSERT_TRUE(g->inUse()) << "gpeKernelSync[" << i << "]";

      // check item state
      EXPECT_EQ(g->nworkers, nw) << "gpeKernelSync[" << i << "]";
      for (int w = 0; w < nw; w++) {
        EXPECT_EQ(g->postFlag[w], ctran::algos::GpeKernelSync::kUnset)
            << "gpeKernelSync[" << i << "].postFlag[" << w << "]";
        EXPECT_EQ(g->completeFlag[w], ctran::algos::GpeKernelSync::kUnset)
            << "gpeKernelSync[" << i << "].completeFlag[" << w << "]";
      }
    }

    // reset, to allow reclaim
    for (int i = 0; i < popSize; ++i) {
      gpeKernelSyncs[i]->reset();
    }

    gpeKernelSyncs.clear();
  }
}
