#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD/HIP port of comms/pipes/tests/P2pIbgdaTransportDeviceTestMain.cc
// Same test logic, adapted for HIP runtime APIs.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <vector>

#include <hip/hip_runtime.h>

#include "HipDeviceBuffer.h"
#include "P2pIbgdaTransportDeviceAmdTestKernels.h"
#include "PipesGdaShared.h"

using namespace pipes_gda;

namespace pipes_gda::tests {

// =============================================================================
// HIP helpers (matching CUDA DeviceBuffer / CUDACHECK_TEST pattern)
// =============================================================================

#define HIPCHECK_TEST(cmd)                                                    \
  do {                                                                        \
    hipError_t err = (cmd);                                                   \
    if (err != hipSuccess) {                                                  \
      FAIL() << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ \
             << ":" << __LINE__;                                              \
    }                                                                         \
  } while (0)

using ::pipes_gda::HipDeviceBuffer;

// =============================================================================
// Test Fixture
// =============================================================================

class P2pIbgdaTransportDeviceTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No HIP GPU available";
    }
    HIPCHECK_TEST(hipSetDevice(0));
  }

  void runAndVerify(const std::function<void(bool*)>& runKernel) {
    HipDeviceBuffer successBuf(sizeof(bool));
    auto* d_success = static_cast<bool*>(successBuf.get());

    bool initSuccess = true;
    HIPCHECK_TEST(hipMemcpy(
        d_success, &initSuccess, sizeof(bool), hipMemcpyHostToDevice));

    runKernel(d_success);
    HIPCHECK_TEST(hipDeviceSynchronize());

    bool success = false;
    HIPCHECK_TEST(
        hipMemcpy(&success, d_success, sizeof(bool), hipMemcpyDeviceToHost));

    EXPECT_TRUE(success);
  }
};

// =============================================================================
// Construction and accessor tests
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, DeviceConstruction) {
  runAndVerify(
      [&](bool* d_success) { runTestP2pTransportConstruction(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, DefaultConstruction) {
  runAndVerify([](bool* d_success) {
    runTestP2pTransportDefaultConstruction(d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, ReadSignal) {
  const int numSignals = 4;

  HipDeviceBuffer signalBuf(numSignals * sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  std::vector<uint64_t> h_signals(numSignals);
  for (int i = 0; i < numSignals; ++i) {
    h_signals[i] = static_cast<uint64_t>(i + 1) * 100;
  }
  HIPCHECK_TEST(hipMemcpy(
      d_signalBuf,
      h_signals.data(),
      numSignals * sizeof(uint64_t),
      hipMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x5555));

  runAndVerify([&](bool* d_success) {
    runTestP2pTransportReadSignal(d_signalBuf, localBuf, numSignals, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, IbgdaWorkConstruction) {
  runAndVerify([](bool* d_success) { runTestIbgdaWork(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, BufferSubBufferWithTransport) {
  const size_t offset = 32;
  char localSignalData[128];
  char remoteSignalData[128];

  IbgdaLocalBuffer baseLBuf(localSignalData, NetworkLKey(0x7777));
  IbgdaRemoteBuffer baseRBuf(remoteSignalData, NetworkRKey(0x8888));

  IbgdaLocalBuffer subLBuf = baseLBuf.subBuffer(offset);
  IbgdaRemoteBuffer subRBuf = baseRBuf.subBuffer(offset);

  EXPECT_EQ(subLBuf.ptr, localSignalData + offset);
  EXPECT_EQ(subRBuf.ptr, remoteSignalData + offset);
  EXPECT_EQ(subLBuf.lkey, baseLBuf.lkey);
  EXPECT_EQ(subRBuf.rkey, baseRBuf.rkey);
}

// =============================================================================
// wait_signal Tests (GE-only comparison)
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalGE_Equal) {
  const uint64_t targetValue = 50;

  HipDeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  HIPCHECK_TEST(hipMemcpy(
      d_signalBuf, &targetValue, sizeof(uint64_t), hipMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, localBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalGE_Greater) {
  const uint64_t signalValue = 100;
  const uint64_t targetValue = 50;

  HipDeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  HIPCHECK_TEST(hipMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), hipMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, localBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalMultipleSlots) {
  const int numSignals = 4;

  HipDeviceBuffer signalBuf(numSignals * sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  std::vector<uint64_t> h_signals(numSignals);
  for (int i = 0; i < numSignals; ++i) {
    h_signals[i] = static_cast<uint64_t>(i + 1) * 100;
  }
  HIPCHECK_TEST(hipMemcpy(
      d_signalBuf,
      h_signals.data(),
      numSignals * sizeof(uint64_t),
      hipMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x3333));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalMultipleSlots(
        d_signalBuf, localBuf, numSignals, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalZeroValue) {
  const uint64_t targetValue = 0;

  HipDeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  HIPCHECK_TEST(hipMemset(d_signalBuf, 0, sizeof(uint64_t)));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, localBuf, targetValue, d_success);
  });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, WaitSignalMaxValue) {
  const uint64_t targetValue = UINT64_MAX;

  HipDeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());

  HIPCHECK_TEST(hipMemcpy(
      d_signalBuf, &targetValue, sizeof(uint64_t), hipMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));

  runAndVerify([&](bool* d_success) {
    runTestWaitSignalGE(d_signalBuf, localBuf, targetValue, d_success);
  });
}

// =============================================================================
// ThreadGroup / Group-Level API Tests
// =============================================================================

TEST_F(P2pIbgdaTransportDeviceTestFixture, PutGroupPartitioning) {
  runAndVerify([](bool* d_success) { runTestPutGroupPartitioning(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, PutSignalGroupBroadcast) {
  runAndVerify(
      [](bool* d_success) { runTestPutSignalGroupBroadcast(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, Broadcast64Block) {
  runAndVerify([](bool* d_success) { runTestBroadcast64Block(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, Broadcast64Multiwarp) {
  runAndVerify([](bool* d_success) { runTestBroadcast64Multiwarp(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, Broadcast64DoubleSafety) {
  runAndVerify(
      [](bool* d_success) { runTestBroadcast64DoubleSafety(d_success); });
}

TEST_F(P2pIbgdaTransportDeviceTestFixture, PutGroupPartitioningBlock) {
  runAndVerify(
      [](bool* d_success) { runTestPutGroupPartitioningBlock(d_success); });
}

// =============================================================================
// wait_signal Timeout Tests
// =============================================================================

class P2pIbgdaWaitSignalTimeoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No HIP GPU available";
    }
    HIPCHECK_TEST(hipSetDevice(0));
  }

  void TearDown() override {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)hipDeviceReset();
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)hipSetDevice(0);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)hipGetLastError();
  }

  bool isExpectedTrapError(hipError_t err) {
    return err == hipErrorAssert || err == hipErrorLaunchFailure ||
        err == hipErrorNotReady;
  }
};

// Note: WaitSignalTimeoutTraps is intentionally omitted on AMD. On HIP,
// abort() in device code terminates the process entirely, unlike CUDA's
// __trap() which generates a recoverable cudaErrorIllegalInstruction.

TEST_F(P2pIbgdaWaitSignalTimeoutTest, WaitSignalNoTimeoutWhenSatisfied) {
  HipDeviceBuffer signalBuf(sizeof(uint64_t));
  auto* d_signalBuf = static_cast<uint64_t*>(signalBuf.get());
  const uint64_t signalValue = 42;
  HIPCHECK_TEST(hipMemcpy(
      d_signalBuf, &signalValue, sizeof(uint64_t), hipMemcpyHostToDevice));

  IbgdaLocalBuffer localBuf(d_signalBuf, NetworkLKey(0x1111));

  HipDeviceBuffer successBuf(sizeof(bool));
  auto* d_success = static_cast<bool*>(successBuf.get());
  bool initSuccess = false;
  HIPCHECK_TEST(
      hipMemcpy(d_success, &initSuccess, sizeof(bool), hipMemcpyHostToDevice));

  runTestWaitSignalNoTimeout(d_signalBuf, localBuf, 0, 1000, d_success);
  HIPCHECK_TEST(hipDeviceSynchronize());

  bool success = false;
  HIPCHECK_TEST(
      hipMemcpy(&success, d_success, sizeof(bool), hipMemcpyDeviceToHost));
  EXPECT_TRUE(success);
}

} // namespace pipes_gda::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
