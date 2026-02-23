// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BarrierTest.hpp"

#include <gtest/gtest.h>

// Test function for synchronous barrier with work object
template <typename Fixture>
void BarrierTest<Fixture>::testSync() {
  auto execute = [&]() {
    auto work = torchcomm_->barrier(false);
    work->wait();
  };
  run(execute);
}

// Test function for synchronous barrier without work object
template <typename Fixture>
void BarrierTest<Fixture>::testSyncNoWork() {
  auto execute = [&]() { torchcomm_->barrier(false); };
  run(execute);
}

// Test function for asynchronous barrier with wait
template <typename Fixture>
void BarrierTest<Fixture>::testAsync() {
  auto execute = [&]() {
    auto work = torchcomm_->barrier(true);
    work->wait();
  };
  run(execute);
}

// Test function for asynchronous barrier with early reset
template <typename Fixture>
void BarrierTest<Fixture>::testAsyncEarlyReset() {
  auto execute = [&]() {
    auto work = torchcomm_->barrier(true);
    work->wait();
    work.reset();
  };
  run(execute);
}

template class BarrierTest<EagerTestFixture<BarrierParams>>;
template class BarrierTest<GraphTestFixture<BarrierParams, 1>>;
template class BarrierTest<GraphTestFixture<BarrierParams, 2>>;
