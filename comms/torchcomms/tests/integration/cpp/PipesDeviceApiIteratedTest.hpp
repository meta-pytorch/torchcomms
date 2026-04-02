// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Iterated functional tests for TorchComm Device API — Pipes (IBGDA+NVLink)
// backend.

#pragma once

#include <gtest/gtest.h>
#include <memory>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/cuda/MemPool.h>

#include "IteratedTestHelpers.hpp"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

class PipesDeviceApiIteratedTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  void testIteratedPut(size_t msg_bytes, torchcomms::device::CoopScope scope);
  void testIteratedSignal(torchcomms::device::CoopScope scope);
  void testIteratedBarrier(torchcomms::device::CoopScope scope);
  void testIteratedCombined(size_t msg_bytes);
  void testMultiWindow();
  void testMultiComm();
  void testWindowLifecycle();

  torchcomms::device::test::IteratedTestConfig config_;
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<c10::Allocator> allocator_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
};
