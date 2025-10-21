// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/Profiler.h"
#include <gtest/gtest.h>

using namespace ::testing;

namespace ctran {

class ProfilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    comm_ = new CtranComm();
    profiler_ = std::make_shared<ctran::Profiler>(comm_);
  }
  void TearDown() override {
    delete comm_;
    comm_ = nullptr;
  }

  uint64_t getOpCount() {
    return profiler_->opCount_;
  }

 protected:
  CtranComm* comm_{nullptr};
  std::shared_ptr<ctran::Profiler> profiler_{nullptr};
};

TEST_F(ProfilerTest, testInitForEachColl) {
  uint64_t opCount = 100;
  // test negative sampling weight
  profiler_->initForEachColl(opCount, -1);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(getOpCount(), opCount);

  // test zero sampling weight
  profiler_->initForEachColl(opCount, 0);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(getOpCount(), opCount);

  // test sampling weight = 1
  profiler_->initForEachColl(opCount, 1);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(getOpCount(), opCount);

  // test opCount is the multiple of sampling weight
  profiler_->initForEachColl(opCount, 20);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(getOpCount(), opCount);

  // test opCount is not the multiple of sampling weight
  ++opCount;
  profiler_->initForEachColl(opCount, 20);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(getOpCount(), opCount);
}

} // namespace ctran
