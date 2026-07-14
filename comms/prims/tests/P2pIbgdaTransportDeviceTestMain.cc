// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for P2pIbgdaTransportDevice
// Note: P2pIbgdaTransportDevice.cuh includes DOCA GPUNetIO headers with
// __device__ annotations that cannot be compiled by a regular C++ compiler.
// Device-side tests are implemented in P2pIbgdaTransportDeviceTest.cu and
// launched via kernel wrapper functions.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <vector>

// On AMD (`__HIP_PLATFORM_AMD__`), this maps `cuda*` runtime APIs to
// `hip*` and provides a HIP-backed `meta::comms::DeviceBuffer`. No-op
// on NVIDIA — the existing CudaRAII / cuda_runtime path stays in use.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#endif
#include "comms/prims/tests/P2pIbgdaTransportDeviceTest.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"
#ifndef __HIP_PLATFORM_AMD__
#include "comms/utils/CudaRAII.h"
#endif

using namespace meta::comms;

namespace comms::prims::tests {

// =============================================================================
// Test Fixture
// =============================================================================

class P2pIbgdaTransportDeviceTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  // Helper to run a device-side test and check result
  void runAndVerify(const std::function<void(bool*)>& runKernel) {
    DeviceBuffer successBuf(sizeof(bool));
    auto* d_success = static_cast<bool*>(successBuf.get());

    bool initSuccess = true;
    CUDACHECK_TEST(cudaMemcpy(
        d_success, &initSuccess, sizeof(bool), cudaMemcpyHostToDevice));

    runKernel(d_success);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    bool success = false;
    CUDACHECK_TEST(
        cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(success);
  }
};

// =============================================================================
// P2pIbgdaTransportDevice Device-side Tests
// These tests verify that the transport can be constructed and accessed on GPU.
// The actual P2pIbgdaTransportDevice type cannot be used here because its
// header includes DOCA GPUNetIO headers with __device__ annotations.
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, DeviceConstruction) {
  // Test that transport can be constructed on device with null QP
  runAndVerify(
      [](bool* d_success) { runTestP2pTransportConstruction(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, DefaultConstruction) {
  // Test that default-constructed transport has null values
  runAndVerify([](bool* d_success) {
    runTestP2pTransportDefaultConstruction(d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, ReadSignal) {
  // Test that read_signal returns correct values for each signal slot
  const int numSignals = 4;

  // Allocate device memory for signal buffer
  DeviceBuffer signalBuf(numSignals * sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  // Initialize signal buffer with known values: slot[i] = (i+1) * 100
  std::vector<uint64_t> h_signals(numSignals);
  for (int i = 0; i < numSignals; ++i) {
    h_signals[i] = static_cast<uint64_t>(i + 1) * 100;
  }
  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf,
      h_signals.data(),
      numSignals * sizeof(uint64_t),
      cudaMemcpyHostToDevice));

  runAndVerify([&](bool* d_success) {
    runTestP2pTransportReadSignal(d_signalBuf, numSignals, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, BufferSubBuffer) {
  // Test that sub-buffers work correctly (pointer arithmetic and key
  // preservation)
  const size_t offset = 32;
  char localSignalData[128];
  char remoteSignalData[128];

  // Create base buffers
  IbgdaLocalBuffer baseLBuf(localSignalData, NetworkLKeys{NetworkLKey(0x7777)});
  IbgdaRemoteBuffer baseRBuf(
      remoteSignalData, NetworkRKeys{NetworkRKey(0x8888)});

  // Create sub-buffers at offset
  IbgdaLocalBuffer subLBuf = baseLBuf.subBuffer(offset);
  IbgdaRemoteBuffer subRBuf = baseRBuf.subBuffer(offset);

  // Verify sub-buffer pointers
  EXPECT_EQ(subLBuf.ptr, localSignalData + offset);
  EXPECT_EQ(subRBuf.ptr, remoteSignalData + offset);

  // Verify keys are preserved
  EXPECT_EQ(subLBuf.lkey_per_device[0], baseLBuf.lkey_per_device[0]);
  EXPECT_EQ(subRBuf.rkey_per_device[0], baseRBuf.rkey_per_device[0]);
}

// =============================================================================
// wait_signal Tests
// These tests verify the spin-wait logic for GE (>=) comparison.
// Signal buffers are pre-set to values that satisfy the condition so
// wait_signal returns immediately without blocking.
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalGE_Equal) {
  // Test wait_signal with GE comparison when signal == target
  const uint64_t signalValue = 50;
  const uint64_t targetValue = 50; // Equal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalGE_Greater) {
  // Test wait_signal with GE comparison when signal > target
  const uint64_t signalValue = 100;
  const uint64_t targetValue = 50; // Less than signal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalMultipleSlots) {
  // Test wait_signal operates on correct slot in multi-signal setup
  const int numSignals = 4;

  DeviceBuffer signalBuf(numSignals * sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  // Initialize signal buffer with known values: slot[i] = (i+1) * 100
  std::vector<uint64_t> h_signals(numSignals);
  for (int i = 0; i < numSignals; ++i) {
    h_signals[i] = static_cast<uint64_t>(i + 1) * 100;
  }
  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf,
      h_signals.data(),
      numSignals * sizeof(uint64_t),
      cudaMemcpyHostToDevice));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalMultipleSlots(d_signalBuf, numSignals, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalZeroValue) {
  // Test wait_signal with zero value (edge case)
  // For uint64_t, any value >= 0 is always true, so this should pass
  // immediately
  const uint64_t targetValue = 0;

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  // Pre-set signal to 0
  CUDACHECK_TEST(cudaMemset(d_signalBuf, 0, sizeof(uint64_t)));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalMaxValue) {
  // Test wait_signal with max uint64 value (edge case)
  const uint64_t targetValue = UINT64_MAX;

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &targetValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, targetValue, d_success);
  });
}

// =============================================================================
// Group-Level API Tests
// These tests verify put_cooperative partitioning
// and broadcast logic. Actual RDMA operations require a real DOCA QP, so
// these tests focus on the GPU-side logic: data partitioning, sub-buffer
// offset calculation, and signal ticket broadcast.
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, PutCooperativePartitioning) {
  // Test that put_cooperative correctly partitions data across warp lanes
  runAndVerify(
      [](bool* d_success) { runTestPutCooperativePartitioning(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, PutSignalGroupBroadcast) {
  // Test that the cooperative put signal path broadcasts the leader's signal
  // ticket to all lanes.
  runAndVerify(
      [](bool* d_success) { runTestPutSignalGroupBroadcast(d_success); });
}

// =============================================================================
// broadcast Tests for non-warp scopes
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, Broadcast64Block) {
  // Test broadcast<uint64_t> with BLOCK scope
  runAndVerify([](bool* d_success) { runTestBroadcast64Block(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, Broadcast64Multiwarp) {
  // Test broadcast<uint64_t> with MULTIWARP scope
  runAndVerify([](bool* d_success) { runTestBroadcast64Multiwarp(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, Broadcast64DoubleSafety) {
  // Test that two consecutive broadcasts with different values don't race
  runAndVerify(
      [](bool* d_success) { runTestBroadcast64DoubleSafety(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, PutCooperativePartitioningBlock) {
  // Test put_cooperative partitioning logic with block-sized groups
  runAndVerify([](bool* d_success) {
    runTestPutCooperativePartitioningBlock(d_success);
  });
}

#ifndef __HIP_PLATFORM_AMD__
TEST_F(P2pIbgdaTransportDeviceTestFixture, TraceIbgdaEventWritesEvent) {
  cudaDeviceProp props{};
  CUDACHECK_TEST(cudaGetDeviceProperties(&props, 0));
  if (props.major < 9) {
    GTEST_SKIP() << "HRDWRingBuffer trace writes require sm_90+";
  }

  ::hrdw_ring_buffer::HRDWRingBuffer<PipesTraceEvent> traceBuffer(8);
  ASSERT_TRUE(traceBuffer.valid());

  auto deviceHandle = traceBuffer.deviceHandle();
  runTestTraceIbgdaEvent(
      PipesTraceHandle{
          .ring = reinterpret_cast<PipesTraceEntry*>(deviceHandle.ring),
          .writeIndex = deviceHandle.writeIndex,
          .mask = deviceHandle.mask,
          .shift = deviceHandle.shift});
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<PipesTraceEvent> events;
  ::hrdw_ring_buffer::HRDWRingBufferReader<PipesTraceEvent> reader(traceBuffer);
  const auto result = reader.poll(
      [&](const auto& entry, uint64_t) { events.push_back(entry.data); });

  EXPECT_EQ(result.entriesLost, 0u);
  ASSERT_EQ(result.entriesRead, 1u);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].step, 0x12345678u);
  EXPECT_EQ(events[0].detail, 0x4321u);
  EXPECT_EQ(
      events[0].type, static_cast<uint8_t>(PipesTraceEventType::kIbSendBegin));
  EXPECT_EQ(events[0].rank, 7u);
}
#endif // !__HIP_PLATFORM_AMD__

// =============================================================================
// wait_signal Timeout Tests
// Verify that the Timeout parameter on wait_signal correctly traps when the
// timeout expires and does not interfere when the signal is already satisfied.
// =============================================================================

// The timeout-trap tests rely on `cudaErrorIllegalInstruction` /
// `cudaErrorAssert` / `cudaErrorLaunchFailure` to detect a `__trap()` from
// device code. HIP does not expose `hipErrorIllegalInstruction` and AMD's
// trap surface differs from CUDA's. Skip the trap-detection block on AMD;
// the underlying `wait_signal(timeout)` behavior is exercised by the AMD
// `doca_compat_amd_smoke` build target (compile-time correctness) and can
// be added once a HIP-specific trap-detection helper exists.
#ifndef __HIP_PLATFORM_AMD__

// Test fixture for timeout trap tests that resets the device after each test
// to clear __trap() state.
class P2pIbgdaWaitSignalTimeoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    // Reset device to clear any trap state
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaDeviceReset();
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGetLastError(); // Clear any pending errors
  }

  bool isExpectedTrapError(cudaError_t err) {
    return err == cudaErrorIllegalInstruction || err == cudaErrorAssert ||
        err == cudaErrorLaunchFailure;
  }
};

TEST_F(P2pIbgdaWaitSignalTimeoutTest, WaitSignalTimeoutTraps) {
  // Set up a signal buffer with value 0, then wait for GE 999 with a
  // short timeout. The signal will never satisfy GE 999, so the timeout
  // should fire and __trap().
  // Do not use DeviceBuffer here: after the expected trap, cudaFree reports
  // the launch failure before TearDown can reset the device.
  uint64_t* d_signalBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&d_signalBuf, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMemset(d_signalBuf, 0, sizeof(uint64_t)));

  // 10ms timeout - should trigger quickly
  cudaError_t err = runTestWaitSignalTimeout(d_signalBuf, 0, 10);
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error from wait_signal timeout, got: "
      << cudaGetErrorString(err);
}

TEST_F(P2pIbgdaWaitSignalTimeoutTest, WaitSignalNoTimeoutWhenSatisfied) {
  // Set up a signal buffer with value 42, then wait for GE 42 with a
  // timeout enabled. The signal already satisfies GE 42, so wait_signal
  // should return immediately without trapping.
  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());
  const uint64_t signalValue = 42;
  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  DeviceBuffer successBuf(sizeof(bool));
  auto* d_success = static_cast<bool*>(successBuf.get());
  bool initSuccess = false;
  CUDACHECK_TEST(cudaMemcpy(
      d_success, &initSuccess, sizeof(bool), cudaMemcpyHostToDevice));

  // 1000ms timeout - kernel should complete well before this
  runTestWaitSignalNoTimeout(d_signalBuf, 0, 1000, d_success);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  bool success = false;
  CUDACHECK_TEST(
      cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(success);
}

#endif // !__HIP_PLATFORM_AMD__

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
