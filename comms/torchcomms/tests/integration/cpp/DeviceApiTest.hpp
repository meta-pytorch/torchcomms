// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Stress functional tests for TorchComm Device API — NCCLx (GIN+LSA) backend.
// Runs each device API operation many times with data verification to catch
// correctness issues under sustained load.

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/cuda/MemPool.h>

#include "StressTestHelpers.hpp"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

class DeviceApiTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // --- Category 1: Stress correctness (in-kernel loops) ---

  // Put soak: ring put with fill/verify per iteration, parameterized by scope
  void testStressPut(size_t msg_bytes, torchcomms::device::CoopScope scope);

  // Signal soak: ring signal+wait_signal_from per iteration
  void testStressSignal(torchcomms::device::CoopScope scope);

  // Barrier soak: barrier called repeatedly
  void testStressBarrier(torchcomms::device::CoopScope scope);

  // Combined: barrier -> put -> wait -> verify per iteration
  void testStressCombined(size_t msg_bytes);

  // Aggregated wait_signal + read_signal + reset: exercises aggregated
  // (non-per-peer) signal path with read and reset verification
  void testStressAggregatedSignal();

  // Half-precision put soak: ring put with at::kHalf data
  void testStressPutHalf(size_t msg_bytes, torchcomms::device::CoopScope scope);

  // --- Category 2: Concurrency (host-side orchestration) ---

  // Multiple windows on the same communicator, each doing independent puts
  void testMultiWindow();

  // Multiple communicators, each doing independent puts
  void testMultiComm();

  // --- Category 3: Resource exhaustion (host-side loops) ---

  // Repeated window create/register/operate/deregister/destroy
  void testWindowLifecycle();

  // --- Address query tests ---

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  // Verify that device-side get_multimem_address matches host-side result
  void testGetMultimemAddress();
#endif

  // Member variables
  torchcomms::device::test::StressTestConfig config_;
  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<c10::Allocator> allocator_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
};
