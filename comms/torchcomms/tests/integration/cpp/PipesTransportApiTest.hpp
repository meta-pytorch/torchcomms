// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Pipes Transport API Integration Test
//
// This test validates get_device_transport() on TorchCommNCCLX and exercises
// NVL send/recv through the returned MultiPeerDeviceHandle.
//
// NOTE: This test requires NCCLX with Pipes support (ENABLE_PIPES defined
// at compile time).
//
// Runtime prerequisites:
//   - RUN_PIPES_TRANSPORT_API_TEST=true (skip gate)
//   - NCCL_CTRAN_USE_PIPES=1 (initialize ctran multiPeerTransport)

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "TorchCommTestHelpers.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/torchcomms/TorchComm.hpp"

class PipesTransportApiTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // Check if test should be skipped
  bool checkIfSkip();

  // Create a wrapper for TorchComm
  std::unique_ptr<TorchCommTestWrapper> createWrapper();

  // Helper to get transport handle (returns false + skips if unavailable)
  bool getTransportHandle(comms::pipes::MultiPeerDeviceHandle& handle);

  // Test functions

  // Verify that get_device_transport() returns a valid MultiPeerDeviceHandle.
  void testGetDeviceTransport();

  // Test NVL send/recv: rank 0 sends nbytes to rank 1 via NVLink transport.
  void testNvlSendRecv(size_t nbytes);

  // Test NVL signal: both ranks signal each other and wait.
  void testNvlSignal();

  // Test NVL LL128 send/recv: rank 0 sends, rank 1 receives via LL128.
  void testNvlLl128SendRecv(size_t nbytes);

  // Member variables
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
};
