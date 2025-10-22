// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BarrierTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>

std::unique_ptr<TorchCommTestWrapper> BarrierTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void BarrierTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void BarrierTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous barrier with work object
void BarrierTest::testSyncBarrier() {
  SCOPED_TRACE(::testing::Message() << "Testing sync barrier");

  // Call barrier
  auto work = torchcomm_->barrier(false);
  work->wait();

  // No explicit verification needed for barrier, just ensure it completes
}

// Test function for synchronous barrier without work object
void BarrierTest::testSyncBarrierNoWork() {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync barrier without work object");

  // Call barrier without keeping the work object
  torchcomm_->barrier(false);
}

// Test function for asynchronous barrier with wait
void BarrierTest::testAsyncBarrier() {
  SCOPED_TRACE(::testing::Message() << "Testing async barrier");

  // Call barrier
  auto work = torchcomm_->barrier(true);

  // Wait for the barrier to complete
  work->wait();
}

// Test function for asynchronous barrier with early reset
void BarrierTest::testAsyncBarrierEarlyReset() {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async barrier with early reset");

  // Call barrier
  auto work = torchcomm_->barrier(true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();
}

// CUDA Graph test function for barrier
void BarrierTest::testGraphBarrier() {
  SCOPED_TRACE(::testing::Message() << "Testing CUDA Graph barrier");

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the barrier operation in the graph
  graph.capture_begin();

  // Call barrier without keeping the work object
  torchcomm_->barrier(false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    graph.replay();

    // No explicit verification needed for barrier, just ensure it completes
  }
}
