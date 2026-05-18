#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// MultipeerIbgdaTransportTest — Real NIC Integration Tests
// =============================================================================
//
// AMD/HIP port of comms/pipes/tests/MultipeerIbgdaTransportTest.cc.
// Exercises P2pIbgdaTransportDevice methods over actual RDMA QPs.
//
// Requires 2 MPI ranks with AMD GPUs and RDMA NICs.
// Uses MultipeerIbgdaTransportAmd for transport setup (matching NVIDIA
// pattern).
// =============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <hip/hip_runtime.h>

#include "HipDeviceBuffer.h"
#include "MultipeerIbgdaTransportAmd.h"
#include "MultipeerIbgdaTransportAmdTestKernels.h"
#include "PipesGdaShared.h"

#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using namespace meta::comms;
using pipes_gda::HipDeviceBuffer;
using pipes_gda::MultipeerIbgdaTransportAmd;
using pipes_gda::MultipeerIbgdaTransportAmdConfig;

namespace pipes_gda::tests {

#define HIPCHECK_TEST(cmd)                                                    \
  do {                                                                        \
    hipError_t err = (cmd);                                                   \
    if (err != hipSuccess) {                                                  \
      FAIL() << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ \
             << ":" << __LINE__;                                              \
    }                                                                         \
  } while (0)

#define HIP_EXPECT(cmd)                                             \
  do {                                                              \
    hipError_t _hip_err = (cmd);                                    \
    EXPECT_EQ(_hip_err, hipSuccess) << hipGetErrorString(_hip_err); \
  } while (0)

// Maximum number of signal slots used across all tests.
static constexpr size_t kMaxSignalSlots = 16;
static constexpr size_t kSignalBufferSize = kMaxSignalSlots * sizeof(uint64_t);

// =============================================================================
// Test Fixture — uses MultipeerIbgdaTransportAmd (matching NVIDIA pattern)
// =============================================================================

class MultipeerIbgdaTransportTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    HIPCHECK_TEST(hipSetDevice(localRank));
  }

  // Helper to create transport, buffers, and exchange with peer
  void initTransport(size_t dataSize) {
    if (numRanks != 2) {
      GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }

    peerRank_ = (globalRank == 0) ? 1 : 0;

    try {
      MultipeerIbgdaTransportAmdConfig config{.hipDevice = localRank};
      auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
      transport_ = std::make_unique<MultipeerIbgdaTransportAmd>(
          globalRank, numRanks, bootstrap, config);
      transport_->exchange();

      // Allocate and register data buffer
      dataBuffer_ = std::make_unique<HipDeviceBuffer>(dataSize);
      localDataBuf_ = transport_->registerBuffer(dataBuffer_->get(), dataSize);
      auto remoteDataBufs = transport_->exchangeBuffer(localDataBuf_);
      int peerIndex = (peerRank_ < globalRank) ? peerRank_ : (peerRank_ - 1);
      remoteDataBuf_ = remoteDataBufs[peerIndex];

      // Allocate and register signal buffer
      signalBuffer_ = std::make_unique<HipDeviceBuffer>(kSignalBufferSize);
      HIPCHECK_TEST(hipMemset(signalBuffer_->get(), 0, kSignalBufferSize));
      localSignalBuf_ =
          transport_->registerBuffer(signalBuffer_->get(), kSignalBufferSize);
      auto remoteSignalBufs = transport_->exchangeBuffer(localSignalBuf_);
      remoteSignalBuf_ = remoteSignalBufs[peerIndex];

      deviceTransport_ = transport_->getP2pTransportDevice(peerRank_);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } catch (const std::exception& e) {
      GTEST_SKIP() << "AMD IBGDA transport not available: " << e.what();
    }
  }

  // --- Verification helpers ---

  void zeroDataBuffer(size_t nbytes) {
    HIPCHECK_TEST(hipMemset(dataBuffer_->get(), 0, nbytes));
    HIPCHECK_TEST(hipDeviceSynchronize());
  }

  uint64_t readSignalFromHost(int signalId = 0) {
    uint64_t val = 0;
    HIP_EXPECT(hipMemcpy(
        &val,
        static_cast<uint64_t*>(signalBuffer_->get()) + signalId,
        sizeof(uint64_t),
        hipMemcpyDeviceToHost));
    return val;
  }

  bool verifyDataFromHost(size_t nbytes, uint8_t seed) {
    bool* d_success = nullptr;
    HIP_EXPECT(hipMalloc(&d_success, sizeof(bool)));
    verifyBufferPattern(dataBuffer_->get(), nbytes, seed, d_success);
    bool h_success = false;
    HIP_EXPECT(
        hipMemcpy(&h_success, d_success, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_EXPECT(hipFree(d_success));
    return h_success;
  }

  bool* allocDeviceBool(bool initVal = false) {
    bool* d_ptr = nullptr;
    HIP_EXPECT(hipMalloc(&d_ptr, sizeof(bool)));
    HIP_EXPECT(hipMemcpy(d_ptr, &initVal, sizeof(bool), hipMemcpyHostToDevice));
    return d_ptr;
  }

  bool readDeviceBool(bool* d_ptr) {
    bool val = false;
    HIP_EXPECT(hipMemcpy(&val, d_ptr, sizeof(bool), hipMemcpyDeviceToHost));
    return val;
  }

  void freeDeviceBool(bool* d_ptr) {
    if (d_ptr)
      HIPCHECK_TEST(hipFree(d_ptr));
  }

  int peerRank_{0};
  std::unique_ptr<MultipeerIbgdaTransportAmd> transport_;
  std::unique_ptr<HipDeviceBuffer> dataBuffer_;
  std::unique_ptr<HipDeviceBuffer> signalBuffer_;
  IbgdaLocalBuffer localDataBuf_;
  IbgdaRemoteBuffer remoteDataBuf_;
  IbgdaLocalBuffer localSignalBuf_;
  IbgdaRemoteBuffer remoteSignalBuf_;
  P2pIbgdaTransportDevice* deviceTransport_{nullptr};
};

// =============================================================================
// 1. PutSignalBasic — put_signal() + wait_local(), verify data
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalBasic) {
  constexpr size_t kDataSize = 4096;
  constexpr uint8_t kSeed = 0x42;
  initTransport(kDataSize);

  if (globalRank == 0) {
    fillBufferWithPattern(dataBuffer_->get(), kDataSize, kSeed);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestPutAndSignal(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        kDataSize,
        remoteSignalBuf_,
        0,
        1, );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    zeroDataBuffer(kDataSize);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 0, 1, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_TRUE(verifyDataFromHost(kDataSize, kSeed))
        << "Data verification failed on rank 1";

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

// =============================================================================
// 2. PutSignalGroupBasic — put_signal_group_local() (wavefront group)
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalGroupBasic) {
  constexpr size_t kDataSize = 64 * 1024;
  constexpr uint8_t kSeed = 0xAB;
  initTransport(kDataSize);

  if (globalRank == 0) {
    fillBufferWithPattern(dataBuffer_->get(), kDataSize, kSeed);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestPutAndSignalGroup(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        kDataSize,
        remoteSignalBuf_,
        0,
        1, );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    zeroDataBuffer(kDataSize);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 0, 1, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_TRUE(verifyDataFromHost(kDataSize, kSeed));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

// =============================================================================
// 3. PutSignalGroupMultiWavefront — put_signal_group_global() multi-wavefront
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalGroupMultiWavefront) {
  constexpr size_t kDataSize = 256 * 1024;
  constexpr uint8_t kSeed = 0xCD;
  initTransport(kDataSize);

  if (globalRank == 0) {
    fillBufferWithPattern(dataBuffer_->get(), kDataSize, kSeed);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestPutAndSignalGroupMultiWarp(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        kDataSize,
        remoteSignalBuf_,
        0,
        1, );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    zeroDataBuffer(kDataSize);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 0, 1, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_TRUE(verifyDataFromHost(kDataSize, kSeed));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

// =============================================================================
// 4. PutSignalGroupBlock — put_signal_group_global() block-scope
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalGroupBlock) {
  constexpr size_t kDataSize = 256 * 1024;
  constexpr uint8_t kSeed = 0xEF;
  initTransport(kDataSize);

  if (globalRank == 0) {
    fillBufferWithPattern(dataBuffer_->get(), kDataSize, kSeed);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestPutAndSignalGroupBlock(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        kDataSize,
        remoteSignalBuf_,
        0,
        1, );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    zeroDataBuffer(kDataSize);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 0, 1, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_TRUE(verifyDataFromHost(kDataSize, kSeed));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

// =============================================================================
// 5. TransferSizeVariations — parameterized sizes
// =============================================================================

class TransferSizeTest
    : public MultipeerIbgdaTransportTestFixture,
      public ::testing::WithParamInterface<std::pair<size_t, const char*>> {};

TEST_P(TransferSizeTest, PutSignal) {
  auto [dataSize, label] = GetParam();
  constexpr uint8_t kSeed = 0x77;

  initTransport(dataSize);

  if (globalRank == 0) {
    fillBufferWithPattern(dataBuffer_->get(), dataSize, kSeed);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestPutAndSignal(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        dataSize,
        remoteSignalBuf_,
        0,
        1, );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    zeroDataBuffer(dataSize);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 0, 1, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_TRUE(verifyDataFromHost(dataSize, kSeed))
        << "Data verification failed for size " << label;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TransferSizeVariations,
    TransferSizeTest,
    ::testing::Values(
        std::make_pair(size_t{1024}, "1KB"),
        std::make_pair(size_t{4 * 1024}, "4KB"),
        std::make_pair(size_t{64 * 1024}, "64KB"),
        std::make_pair(size_t{256 * 1024}, "256KB"),
        std::make_pair(size_t{1024 * 1024}, "1MB"),
        std::make_pair(size_t{4 * 1024 * 1024}, "4MB"),
        std::make_pair(size_t{16 * 1024 * 1024}, "16MB")));

// =============================================================================
// 6. Bidirectional — both ranks send and receive
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, Bidirectional) {
  constexpr size_t kDataSize = 4096;
  constexpr uint8_t kSeedRank0 = 0x11;
  constexpr uint8_t kSeedRank1 = 0x22;
  // Allocate 2x: first half for send, second half for receive.
  initTransport(2 * kDataSize);

  // Send region: [0, kDataSize), Receive region: [kDataSize, 2*kDataSize)
  auto remoteRecvBuf = remoteDataBuf_.subBuffer(kDataSize);

  // Zero the receive region
  HIPCHECK_TEST(hipMemset(
      static_cast<char*>(dataBuffer_->get()) + kDataSize, 0, kDataSize));
  HIPCHECK_TEST(hipDeviceSynchronize());

  uint8_t mySeed = (globalRank == 0) ? kSeedRank0 : kSeedRank1;
  fillBufferWithPattern(dataBuffer_->get(), kDataSize, mySeed);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Both ranks: send data + signal to peer's receive region, wait for signal
  bool* d_success = allocDeviceBool();
  runTestBidirectionalPutAndWait(
      deviceTransport_,
      localDataBuf_,
      remoteRecvBuf,
      kDataSize,
      remoteSignalBuf_,
      0,
      1,
      localSignalBuf_,
      0,
      1,
      d_success);
  EXPECT_TRUE(readDeviceBool(d_success));
  freeDeviceBool(d_success);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify received data in the receive region (offset kDataSize)
  uint8_t peerSeed = (globalRank == 0) ? kSeedRank1 : kSeedRank0;
  bool* d_verify = allocDeviceBool();
  verifyBufferPattern(
      static_cast<char*>(dataBuffer_->get()) + kDataSize,
      kDataSize,
      peerSeed,
      d_verify);
  EXPECT_TRUE(readDeviceBool(d_verify))
      << "Bidirectional data verification failed on rank " << globalRank;
  freeDeviceBool(d_verify);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

// =============================================================================
// 7. StressTest — 100 iterations of put+signal+verify
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, StressTest) {
  constexpr size_t kDataSize = 4096;
  constexpr uint32_t kNumIters = 100;
  initTransport(kDataSize);

  int totalErrors = 0;

  for (uint32_t iter = 0; iter < kNumIters; ++iter) {
    const uint8_t testPattern = static_cast<uint8_t>(iter % 256);

    if (globalRank == 0) {
      fillBufferWithPattern(dataBuffer_->get(), kDataSize, testPattern);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      runTestPutAndSignal(
          deviceTransport_,
          localDataBuf_,
          remoteDataBuf_,
          kDataSize,
          remoteSignalBuf_,
          0,
          iter + 1, );

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      zeroDataBuffer(kDataSize);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      bool* d_success = allocDeviceBool();
      runTestWaitSignal(
          localSignalBuf_, 0, static_cast<uint64_t>(iter + 1), d_success);
      EXPECT_TRUE(readDeviceBool(d_success));
      freeDeviceBool(d_success);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (!verifyDataFromHost(kDataSize, testPattern)) {
        totalErrors++;
      }
    }
  }

  if (globalRank == 1) {
    EXPECT_EQ(totalErrors, 0)
        << "Stress test found " << totalErrors << " errors across " << kNumIters
        << " iterations";
  }
}

// =============================================================================
// 8. SignalOnly — signal_remote() without data
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, SignalOnly) {
  constexpr size_t kDataSize = 64;
  initTransport(kDataSize);

  if (globalRank == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestSignalOnly(deviceTransport_, remoteSignalBuf_, 0, 42, );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 0, 42, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_EQ(readSignalFromHost(0), 42u);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

// =============================================================================
// 9. ResetSignal — signal + reset cycle
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, ResetSignal) {
  constexpr size_t kDataSize = 64;
  initTransport(kDataSize);

  for (int iter = 0; iter < 3; ++iter) {
    if (globalRank == 0) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      runTestSignalOnly(deviceTransport_, remoteSignalBuf_, 0, 1, );

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      runTestResetSignal(deviceTransport_, remoteSignalBuf_, 0);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      bool* d_success = allocDeviceBool();
      runTestWaitSignal(localSignalBuf_, 0, 1, d_success);
      EXPECT_TRUE(readDeviceBool(d_success))
          << "Signal wait failed at iteration " << iter;
      freeDeviceBool(d_success);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Wait for reset to complete
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      EXPECT_EQ(readSignalFromHost(0), 0u)
          << "Signal not reset at iteration " << iter;
    }
  }
}

// =============================================================================
// 10. MultipleSignalSlots — multiple signal IDs
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, MultipleSignalSlots) {
  constexpr size_t kDataSize = 64;
  constexpr int kNumSlots = 4;
  initTransport(kDataSize);

  if (globalRank == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (int slot = 0; slot < kNumSlots; ++slot) {
      runTestSignalOnly(
          deviceTransport_,
          remoteSignalBuf_,
          slot,
          static_cast<uint64_t>(slot + 1) * 100, );
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (int slot = 0; slot < kNumSlots; ++slot) {
      uint64_t expected = static_cast<uint64_t>(slot + 1) * 100;
      bool* d_success = allocDeviceBool();
      runTestWaitSignal(localSignalBuf_, slot, expected, d_success);
      EXPECT_TRUE(readDeviceBool(d_success))
          << "Signal slot " << slot << " wait failed";
      freeDeviceBool(d_success);

      EXPECT_EQ(readSignalFromHost(slot), expected)
          << "Signal slot " << slot << " value mismatch";
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

// =============================================================================
// 11. PutSignalWaitForReady — ready/data handshake
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalWaitForReady) {
  constexpr size_t kDataSize = 4096;
  constexpr uint8_t kSeed = 0x99;
  initTransport(kDataSize);

  if (globalRank == 0) {
    fillBufferWithPattern(dataBuffer_->get(), kDataSize, kSeed);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestWaitReadyThenPutAndSignal(
        deviceTransport_,
        localDataBuf_,
        remoteDataBuf_,
        kDataSize,
        localSignalBuf_,
        0, // readySignalId
        1, // readyExpected
        remoteSignalBuf_,
        1, // dataSignalId
        1, // dataSignalVal
    );

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    zeroDataBuffer(kDataSize);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    runTestSignalOnly(deviceTransport_, remoteSignalBuf_, 0, 1, );

    bool* d_success = allocDeviceBool();
    runTestWaitSignal(localSignalBuf_, 1, 1, d_success);
    EXPECT_TRUE(readDeviceBool(d_success));
    freeDeviceBool(d_success);

    EXPECT_TRUE(verifyDataFromHost(kDataSize, kSeed));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }
}

} // namespace pipes_gda::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpiEnv = std::make_unique<meta::comms::MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpiEnv.release());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
#endif
