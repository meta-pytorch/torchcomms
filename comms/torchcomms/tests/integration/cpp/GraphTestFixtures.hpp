// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

template <typename Params>
class TorchCommsTestBase : public ::testing::TestWithParam<Params> {
 public:
  TorchCommsTestBase()
      : TorchCommsTestBase(
            isRunningOnCPU() ? c10::DeviceType::CPU : c10::DeviceType::CUDA) {}
  explicit TorchCommsTestBase(c10::DeviceType deviceType)
      : rank_(0), num_ranks_(0), device_type_(deviceType) {}

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper() {
    return std::make_unique<TorchCommTestWrapper>();
  }

  void SetUp() override {
    wrapper_ = createWrapper();
    torchcomm_ = wrapper_->getTorchComm();
    rank_ = torchcomm_->getRank();
    num_ranks_ = torchcomm_->getSize();
  }

  void TearDown() override {
    torchcomm_.reset();
    wrapper_.reset();
  }

  // Unified test execution entry point. Fixture controls how the op runs
  // (eager vs graph capture/replay).
  //
  // Usage patterns:
  //   Standard test:    run(execute, reset, verify)
  //   InputDeleted:     run(execute, {}, {}, cleanup)
  //
  // Parameters:
  //   execute      - the collective op to run (captured into graph in graph
  //                  mode)
  //   reset        - restores input tensors to pre-op state (called before each
  //                  graph replay; ignored in eager mode)
  //   verify       - checks correctness (called after execute/replay)
  //   afterCapture - one-shot callback after graph capture completes, before
  //                  replay begins (graph mode only; e.g. destroy tensors for
  //                  InputDeleted tests)
  virtual void run(
      const std::function<void()>& execute,
      const std::function<void()>& reset = {},
      const std::function<void()>& verify = {},
      const std::function<void()>& afterCapture = {}) = 0;

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;
};

// Eager mode: execute once, verify once. No CUDA graph involvement.
template <typename Params>
class EagerTestFixture : public TorchCommsTestBase<Params> {
 protected:
  void run(
      const std::function<void()>& execute,
      const std::function<void()>& /* reset */ = {},
      const std::function<void()>& verify = {},
      const std::function<void()>& /* afterCapture */ = {}) override {
    execute();
    if (verify) {
      verify();
    }
  }
};

// Graph fixture: sets up non-default stream, captures op into NumGraphs
// separate graphs, replays each. Skips on CPU.
// NumGraphs=1 for single-graph tests, NumGraphs=2 for multi-graph tests.
template <typename Params, int NumGraphs = 1>
class GraphTestFixture : public TorchCommsTestBase<Params> {
 protected:
  void SetUp() override {
    if (isRunningOnCPU()) {
      GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
    }
    TorchCommsTestBase<Params>::SetUp();
    stream_ = at::cuda::getStreamFromPool();
    guard_.emplace(*stream_);
  }

  void TearDown() override {
    guard_.reset();
    stream_.reset();
    TorchCommsTestBase<Params>::TearDown();
  }

  void run(
      const std::function<void()>& execute,
      const std::function<void()>& reset = {},
      const std::function<void()>& verify = {},
      const std::function<void()>& afterCapture = {}) override {
    std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs;
    for (int g = 0; g < NumGraphs; ++g) {
      graphs.push_back(std::make_unique<at::cuda::CUDAGraph>());
      graphs.back()->capture_begin();
      execute();
      graphs.back()->capture_end();
    }

    if (afterCapture) {
      afterCapture();
    }

    for (int i = 0; i < kNumReplays; ++i) {
      for (int g = 0; g < NumGraphs; ++g) {
        if (reset) {
          reset();
        }
        graphs[g]->replay();
        if (verify) {
          verify();
        }
      }
    }
  }

 private:
  static constexpr int kNumReplays = 4;
  std::optional<at::cuda::CUDAStream> stream_;
  std::optional<at::cuda::CUDAStreamGuard> guard_;
};
