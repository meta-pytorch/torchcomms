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

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/tests/P2pIbgdaTransportDeviceTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

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

    bool initSuccess = false;
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
  // Test that transport can be copied to device and accessed there

  // Create mock buffers for signal data
  char localSignalData[64];
  char remoteSignalData[64];
  IbgdaLocalBuffer localBuf(localSignalData, NetworkLKey(0xAAAA));
  IbgdaRemoteBuffer remoteBuf(remoteSignalData, NetworkRKey(0xBBBB));

  runAndVerify([&](bool* d_success) {
    runTestP2pTransportConstruction(localBuf, remoteBuf, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, DefaultConstruction) {
  // Test that default-constructed transport has null values
  runAndVerify([](bool* d_success) {
    runTestP2pTransportDefaultConstruction(d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, NumSignalsDefault) {
  // Test default numSignals (should be 1)
  char localSignalData[64];
  char remoteSignalData[64];
  IbgdaLocalBuffer localBuf(localSignalData, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(remoteSignalData, NetworkRKey(0x2222));

  // When numSignals is not specified, default is 1
  DeviceBuffer successBuf(sizeof(bool));
  auto* d_success = static_cast<bool*>(successBuf.get());

  bool initSuccess = false;
  CUDACHECK_TEST(cudaMemcpy(
      d_success, &initSuccess, sizeof(bool), cudaMemcpyHostToDevice));

  // numSignals = 1 (default)
  runTestP2pTransportNumSignals(localBuf, remoteBuf, 1, d_success);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  bool success = false;
  CUDACHECK_TEST(
      cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(success) << "Default numSignals should be 1";
}

// Parameterized test for different numSignals values
class NumSignalsTestFixture : public P2pIbgdaTransportDeviceTestFixture,
                              public ::testing::WithParamInterface<int> {};

TEST_P(NumSignalsTestFixture, NumSignalsAccessor) {
  int numSignals = GetParam();

  char localSignalData[512]; // Enough for multiple signals
  char remoteSignalData[512];
  IbgdaLocalBuffer localBuf(localSignalData, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(remoteSignalData, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestP2pTransportNumSignals(localBuf, remoteBuf, numSignals, d_success);
  });
}

INSTANTIATE_TEST_SUITE_P(
    NumSignalsVariations,
    NumSignalsTestFixture,
    ::testing::Values(1, 2, 4, 8, 16, 32));

TEST_F(P2pIbgdaTransportDeviceTestFixture, SignalPointerArithmetic) {
  // Test that signal pointer arithmetic works correctly for multi-signal setup
  const int numSignals = 4;
  char localSignalData[numSignals * sizeof(uint64_t)];
  char remoteSignalData[numSignals * sizeof(uint64_t)];
  IbgdaLocalBuffer localBuf(localSignalData, NetworkLKey(0x3333));
  IbgdaRemoteBuffer remoteBuf(remoteSignalData, NetworkRKey(0x4444));

  runAndVerify([&](bool* d_success) {
    runTestP2pTransportSignalPointerArithmetic(
        localBuf, remoteBuf, numSignals, d_success);
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

  // Create buffers pointing to device memory
  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x5555));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x6666));

  runAndVerify([&](bool* d_success) {
    runTestP2pTransportReadSignal(
        d_signalBuf, localBuf, remoteBuf, numSignals, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, IbgdaWorkConstruction) {
  // Test IbgdaWork struct construction and value access
  runAndVerify([](bool* d_success) { runTestIbgdaWork(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, BufferSubBufferWithTransport) {
  // Test that sub-buffers work correctly with transport construction
  const size_t offset = 32;
  char localSignalData[128];
  char remoteSignalData[128];

  // Create base buffers
  IbgdaLocalBuffer baseLBuf(localSignalData, NetworkLKey(0x7777));
  IbgdaRemoteBuffer baseRBuf(remoteSignalData, NetworkRKey(0x8888));

  // Create sub-buffers at offset
  IbgdaLocalBuffer subLBuf = baseLBuf.subBuffer(offset);
  IbgdaRemoteBuffer subRBuf = baseRBuf.subBuffer(offset);

  // Verify sub-buffer pointers
  EXPECT_EQ(subLBuf.ptr, localSignalData + offset);
  EXPECT_EQ(subRBuf.ptr, remoteSignalData + offset);

  // Verify keys are preserved
  EXPECT_EQ(subLBuf.lkey, baseLBuf.lkey);
  EXPECT_EQ(subRBuf.rkey, baseRBuf.rkey);

  // Test transport construction with sub-buffers
  runAndVerify([&](bool* d_success) {
    runTestP2pTransportConstruction(subLBuf, subRBuf, d_success);
  });
}

// =============================================================================
// wait_signal Tests
// These tests verify the spin-wait logic for each comparison operation.
// Signal buffers are pre-set to values that satisfy the condition so
// wait_signal returns immediately without blocking.
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalEQ) {
  // Test wait_signal with EQ comparison
  const uint64_t targetValue = 42;

  // Allocate device memory for signal buffer
  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  // Pre-set signal to targetValue so EQ condition is satisfied
  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &targetValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalEQ(
        d_signalBuf, localBuf, remoteBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalNE) {
  // Test wait_signal with NE comparison
  const uint64_t signalValue = 100;
  const uint64_t targetValue = 42; // Different from signalValue

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  // Pre-set signal to signalValue (which != targetValue) so NE is satisfied
  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalNE(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalGE_Equal) {
  // Test wait_signal with GE comparison when signal == target
  const uint64_t signalValue = 50;
  const uint64_t targetValue = 50; // Equal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
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

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalGT) {
  // Test wait_signal with GT comparison
  const uint64_t signalValue = 100;
  const uint64_t targetValue = 50; // Less than signal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGT(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalLE_Equal) {
  // Test wait_signal with LE comparison when signal == target
  const uint64_t signalValue = 50;
  const uint64_t targetValue = 50; // Equal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalLE(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalLE_Less) {
  // Test wait_signal with LE comparison when signal < target
  const uint64_t signalValue = 25;
  const uint64_t targetValue = 50; // Greater than signal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalLE(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalLT) {
  // Test wait_signal with LT comparison
  const uint64_t signalValue = 25;
  const uint64_t targetValue = 50; // Greater than signal

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalLT(
        d_signalBuf, localBuf, remoteBuf, signalValue, targetValue, d_success);
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

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x3333));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x4444));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalMultipleSlots(
        d_signalBuf, localBuf, remoteBuf, numSignals, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalZeroValue) {
  // Test wait_signal with zero value (edge case)
  const uint64_t targetValue = 0;

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  // Pre-set signal to 0
  CUDACHECK_TEST(cudaMemset(d_signalBuf, 0, sizeof(uint64_t)));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalEQ(
        d_signalBuf, localBuf, remoteBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalMaxValue) {
  // Test wait_signal with max uint64 value (edge case)
  const uint64_t targetValue = UINT64_MAX;

  DeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      d_signalBuf, &targetValue, sizeof(uint64_t), cudaMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));
  IbgdaRemoteBuffer remoteBuf(d_signalBuf, NetworkRKey(0x2222));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalEQ(
        d_signalBuf, localBuf, remoteBuf, targetValue, d_success);
  });
}

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
