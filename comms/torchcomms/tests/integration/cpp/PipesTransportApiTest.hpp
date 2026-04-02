// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Stress tests for P2pNvlTransportDevice APIs.
// Tests transport-layer send/recv, signal, combined, and LL128
// operations under sustained load to catch race conditions and leaks.

#pragma once

#include <gtest/gtest.h>
#include <memory>

#include "StressTestHelpers.hpp"
#include "TorchCommTestHelpers.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/torchcomms/TorchComm.hpp"

class PipesTransportApiTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  void testStressSendRecv(size_t msg_bytes, int num_threads);
  void testStressSignal(int num_threads);
  void testStressCombined(size_t msg_bytes, int num_threads);
  void testStressLl128(size_t nbytes);

  torchcomms::device::test::StressTestConfig config_;
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  comms::pipes::MultiPeerDeviceHandle handle_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
  int peer_{-1};
};
