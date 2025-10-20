// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"

class KernelFlagPoolTest : public ::testing::Test {
 public:
  int cudaDev;
  KernelFlagPoolTest() = default;

 protected:
  void SetUp() override {
    cudaDev = 0;
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);
  }

  void runReclaimTest(
      std::function<void(std::vector<KernelFlagItem*>&, int)> resetCb,
      int numGroups) {
    constexpr int poolSize = 10;
    constexpr int popSize = 6;
    constexpr int reclaimSize = 2;
    auto flagPool = std::make_unique<KernelFlagPool>(poolSize);

    ASSERT_NE(flagPool, nullptr);
    EXPECT_EQ(flagPool->size(), poolSize);

    std::vector<KernelFlagItem*> allocated_flags;
    for (int i = 0; i < popSize; ++i) {
      auto flag = flagPool->pop();
      ASSERT_NE(flag, nullptr);

      flag->numGroups_ = numGroups;
      for (int j = 0; j < flag->numGroups_; ++j) {
        flag->flag_[j] = KERNEL_SCHEDULED;
      }
      allocated_flags.push_back(flag);
    }
    EXPECT_EQ(flagPool->size(), poolSize - popSize);
    // Capacity is unchanged
    EXPECT_EQ(flagPool->capacity(), poolSize);

    resetCb(allocated_flags, reclaimSize);
    EXPECT_EQ(flagPool->size(), poolSize - popSize);
    // Capacity is unchanged
    EXPECT_EQ(flagPool->capacity(), poolSize);

    flagPool->reclaim();
    EXPECT_EQ(flagPool->size(), poolSize - popSize + reclaimSize);
    // Capacity is unchanged
    EXPECT_EQ(flagPool->capacity(), poolSize);
  }
  void runReclaimTestDevice(int numGroups) {
    runReclaimTest(
        [](std::vector<KernelFlagItem*>& allocated_flags, int reclaimSize) {
          for (int i = 0; i < reclaimSize; ++i) {
            auto* allocated_flag = allocated_flags[i];
            for (int j = 0; j < allocated_flag->numGroups_; ++j) {
              allocated_flag->flag_[j] = KERNEL_UNSET;
            }
          }
        },
        numGroups);
  }
};

TEST_F(KernelFlagPoolTest, Initialize) {
  constexpr int poolSize = 1000;
  auto flagPool = std::make_unique<KernelFlagPool>(poolSize);

  ASSERT_NE(flagPool, nullptr);
  EXPECT_EQ(flagPool->size(), poolSize);
  EXPECT_EQ(flagPool->capacity(), poolSize);
}

TEST_F(KernelFlagPoolTest, PopTest) {
  constexpr int poolSize = 10;
  auto flagPool = std::make_unique<KernelFlagPool>(poolSize);

  ASSERT_NE(flagPool, nullptr);
  EXPECT_EQ(flagPool->size(), poolSize);
  EXPECT_EQ(flagPool->capacity(), poolSize);

  for (int i = 0; i < poolSize; ++i) {
    auto allocated_flag = flagPool->pop();
    EXPECT_EQ(allocated_flag->numGroups_, 1);
    EXPECT_EQ(flagPool->size(), poolSize - (i + 1));
    // Capacity is unchanged
    EXPECT_EQ(flagPool->capacity(), poolSize);
  }

  auto another_flag = flagPool->pop();
  EXPECT_EQ(another_flag, nullptr);
}

TEST_F(KernelFlagPoolTest, ReclaimTestHost) {
  runReclaimTest(
      [](std::vector<KernelFlagItem*>& allocated_flags, int reclaimSize) {
        for (int i = 0; i < reclaimSize; ++i) {
          allocated_flags[i]->reset();
        }
      },
      /*numGroups=*/1);
}

TEST_F(KernelFlagPoolTest, ReclaimTestDevice) {
  runReclaimTestDevice(/*numGroups=*/1);
}

TEST_F(KernelFlagPoolTest, ReclaimTestHostAndDevice) {
  runReclaimTest(
      [](std::vector<KernelFlagItem*>& allocated_flags, int reclaimSize) {
        for (int i = 0; i < reclaimSize; ++i) {
          auto* allocated_flag = allocated_flags[i];
          allocated_flag->reset();
          for (int j = 0; j < allocated_flag->numGroups_; ++j) {
            allocated_flag->flag_[j] = KERNEL_UNSET;
          }
        }
      },
      /*numGroups=*/1);
}

TEST_F(KernelFlagPoolTest, ReclaimTestDevice2Blocks) {
  runReclaimTestDevice(/*numGroups=*/2);
}

TEST_F(KernelFlagPoolTest, ReclaimTestDevice4Blocks) {
  runReclaimTestDevice(/*numGroups=*/4);
}

TEST_F(KernelFlagPoolTest, ReclaimTestDeviceMaxBlocks) {
  runReclaimTestDevice(/*numGroups=*/CTRAN_ALGO_MAX_THREAD_BLOCKS);
}

TEST_F(KernelFlagPoolTest, SetAndTestFlags) {
  constexpr int poolSize = 10;
  auto flagPool = std::make_unique<KernelFlagPool>(poolSize);

  ASSERT_NE(flagPool, nullptr);
  EXPECT_EQ(flagPool->size(), poolSize);
  EXPECT_EQ(flagPool->capacity(), poolSize);

  constexpr int numGroups = 3;
  auto allocated_flag = flagPool->pop();
  allocated_flag->numGroups_ = numGroups;

  allocated_flag->setFlagPerGroup(KERNEL_TERMINATE);

  for (int i = 0; i < numGroups; ++i) {
    EXPECT_EQ(allocated_flag->flag_[i], KERNEL_TERMINATE)
        << "allocated_flag[" << i << "] not properly set";
  }

  EXPECT_TRUE(allocated_flag->testFlagAllGroups(KERNEL_TERMINATE));
  EXPECT_FALSE(allocated_flag->testFlagAllGroups(KERNEL_UNSET));
}
