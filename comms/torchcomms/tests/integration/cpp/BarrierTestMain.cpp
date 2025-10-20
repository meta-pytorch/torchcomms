// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BarrierTest.hpp"

#include <gtest/gtest.h>

TEST_F(BarrierTest, SyncBarrier) {
  testSyncBarrier();
}

TEST_F(BarrierTest, SyncBarrierNoWork) {
  testSyncBarrierNoWork();
}

TEST_F(BarrierTest, AsyncBarrier) {
  testAsyncBarrier();
}

TEST_F(BarrierTest, AsyncBarrierEarlyReset) {
  testAsyncBarrierEarlyReset();
}

TEST_F(BarrierTest, GraphBarrier) {
  testGraphBarrier();
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
