// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test - Pipes Backend (IBGDA + NVLink)
//
// This test validates device window creation using the Pipes backend.
// It exercises tensor_register() and get_device_window() from the host side.
//
// NOTE: This test requires NCCLX 2.28+ with device API headers and Pipes
// support (ENABLE_PIPES defined at compile time).
//
// Runtime prerequisites:
//   - RUN_PIPES_DEVICE_API_TEST=true (skip gate)
//   - NCCL_CTRAN_USE_PIPES=1 (initialize ctran multiPeerTransport and select
//     Pipes backend in new_window())
//   - NCCL_P2P_DISABLE=1 (route traffic through ctran/RDMA path)

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/cuda/MemPool.h>

#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

class PipesDeviceApiTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // Check if test should be skipped
  bool checkIfSkip();

  // Create a wrapper for TorchComm
  std::unique_ptr<TorchCommTestWrapper> createWrapper();

  // Test helper functions
  at::Tensor createTestTensor(int64_t count, at::ScalarType dtype);
  std::string getDtypeName(at::ScalarType dtype);

  // Test functions

  // Verify that tensor_register() + get_device_window() succeeds.
  // get_device_window() is COLLECTIVE: all ranks call
  // ctran_win->get_device_win() which performs an allGather to exchange IBGDA
  // registration info and NVLink-mapped pointers.
  void testPipesDeviceWindowCreation(int count, at::ScalarType dtype);

  // Test per-peer signal via device kernels (ring pattern: signal + wait).
  void testPerPeerSignal();

  // Test wait_signal_from: point-to-point signal wait from a specific peer.
  void testWaitSignalFrom();

  // Test device barrier: all ranks synchronize via barrier().
  void testDeviceBarrier();

  // Test local buffer registration: register_local_buffer returns valid lkey.
  void testLocalBufferRegistration(int count, at::ScalarType dtype);

  // Member variables
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<c10::Allocator> allocator_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
  at::DeviceType device_type_{at::kCUDA};
};
