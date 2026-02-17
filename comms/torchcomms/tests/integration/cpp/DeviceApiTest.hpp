// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test - NCCL GIN Backend
//
// This test validates the device-side communication primitives using NCCL GIN.
// It exercises get_device_window(), register_local_buffer(), and
// deregister_local_buffer() operations from the host side.
//
// NOTE: This test requires NCCLX 2.28+ with device API headers.

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
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

class DeviceApiTest : public ::testing::Test {
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
  void testDeviceWindowCreation(int count, at::ScalarType dtype);
  void testLocalBufferRegistration(int count, at::ScalarType dtype);
  void testDeviceWindowWithSignals(int count, at::ScalarType dtype);
  void testDeviceWindowWithCounters(int count, at::ScalarType dtype);

  // Device-initiated RMA test - uses CUDA kernel to perform put
  void testDevicePut(int count, at::ScalarType dtype);

  // GIN atomicAdd test - uses CUDA kernel to perform remote atomic
  // fetch-and-add
  void testGinAtomicAdd();

  // Per-peer signal test - validates resource buffer signal model
  // Each rank signals the next in a ring, receiver aggregates per-peer slots
  void testPerPeerSignal();

  // Wait signal from specific peer test - validates point-to-point sync
  void testWaitSignalFrom();

  // Member variables
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<c10::Allocator> allocator_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
  at::DeviceType device_type_{at::kCUDA};
};
