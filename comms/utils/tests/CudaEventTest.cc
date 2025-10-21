// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/CudaRAII.h"

#include <gtest/gtest.h>

using namespace meta::comms;
TEST(CudaEventTest, ConstructorCreatesValidEvent) {
  CudaEvent event;
  ASSERT_NE(nullptr, event.get());
}

TEST(CudaEventTest, MoveConstructor) {
  CudaEvent event1;
  cudaEvent_t originalEvent = event1.get();
  ASSERT_NE(nullptr, originalEvent);

  CudaEvent event2(std::move(event1));
  // event2 should now have the original event
  ASSERT_EQ(originalEvent, event2.get());
  // event1 should now be null
  // This use after move is intentional to test the correctness of the move
  // operation.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_EQ(nullptr, event1.get());
}

TEST(CudaEventTest, MoveAssignment) {
  CudaEvent event1;
  CudaEvent event2;
  cudaEvent_t originalEvent = event1.get();
  ASSERT_NE(nullptr, originalEvent);

  event2 = std::move(event1);
  // event2 should now have the original event
  ASSERT_EQ(originalEvent, event2.get());
  // event1 should now be null
  // This use after move is intentional to test the correctness of the move
  // operation.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_EQ(nullptr, event1.get());
}

TEST(CudaEventTest, GetReturnsEvent) {
  CudaStream stream;
  CudaEvent event;
  ASSERT_NE(nullptr, event.get());

  // Verify the event is usable with CUDA API
  cudaError_t result = cudaEventRecord(event.get(), stream.get());
  ASSERT_EQ(cudaSuccess, result);

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
