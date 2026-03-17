// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "comms/pipes/Transport.cuh"
#include "comms/pipes/tests/DeviceWindowTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class DeviceWindowTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

// =============================================================================
// Construction & Accessor Tests
// =============================================================================

TEST_F(DeviceWindowTestFixture, Construction) {
  const int myRank = 0;
  const int nRanks = 4;
  const int signalCount = 2;

  DeviceBuffer resultsBuffer(3 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowConstruction(myRank, nRanks, signalCount, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(3);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
  EXPECT_EQ(results_h[2], nRanks - 1)
      << "num_nvl_peers() should return " << (nRanks - 1);
}

TEST_F(DeviceWindowTestFixture, BasicAccessors) {
  const int myRank = 2;
  const int nRanks = 4;

  DeviceBuffer resultsBuffer(2 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowBasicAccessors(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 2 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
}

// =============================================================================
// NVL Signal Write + Read
// =============================================================================

TEST_F(DeviceWindowTestFixture, NvlSignalWriteRead) {
  const int myRank = 0;
  const int nRanks = 2;
  const int signalCount = 4;
  const int targetPeerRank = 1;
  const int signalId = 2;

  DeviceBuffer resultsBuffer(2 * sizeof(uint64_t));
  CUDACHECK_TEST(cudaMemset(resultsBuffer.get(), 0, 2 * sizeof(uint64_t)));
  auto results_d = static_cast<uint64_t*>(resultsBuffer.get());

  test::testDeviceWindowSignalWriteRead(
      myRank, nRanks, signalCount, targetPeerRank, signalId, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<uint64_t> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], 1u) << "read_signal_from() should see signal";
  EXPECT_EQ(results_h[1], 1u) << "read_signal() should see total";
}

// =============================================================================
// read_signal
// =============================================================================

TEST_F(DeviceWindowTestFixture, ReadSignal) {
  const int myRank = 0;
  const int nRanks = 2;
  const int signalCount = 2;

  DeviceBuffer resultsBuffer(sizeof(uint64_t));
  CUDACHECK_TEST(cudaMemset(resultsBuffer.get(), 0, sizeof(uint64_t)));
  auto results_d = static_cast<uint64_t*>(resultsBuffer.get());

  test::testDeviceWindowReadSignalGroup(myRank, nRanks, signalCount, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, results_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 3u) << "read_signal() should return aggregate value";
}

// =============================================================================
// Generic NVL Put via DeviceWindow
// =============================================================================

TEST_F(DeviceWindowTestFixture, NvlPutViaGenericApi) {
  const std::size_t nbytes = 4096;
  const std::size_t numInts = nbytes / sizeof(int);
  const int testValue = 0xCAFE;

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  std::vector<int> srcHost(numInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::testDeviceWindowNvlPut(
      0,
      2,
      reinterpret_cast<char*>(dst_d),
      reinterpret_cast<const char*>(src_d),
      nbytes);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> dstHost(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(dstHost.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  const std::vector<int> expected(numInts, testValue);
  EXPECT_EQ(dstHost, expected) << "Generic NVL put should copy all data";
}

// =============================================================================
// Offset-Based NVL Put via DeviceWindow
// =============================================================================

TEST_F(DeviceWindowTestFixture, NvlOffsetPut) {
  // Source buffer: 8KB filled with 0xCAFE
  const std::size_t srcBufSize = 8192;
  const std::size_t srcNumInts = srcBufSize / sizeof(int);
  const int testValue = 0xCAFE;

  // Window buffer: 16KB, zero-initialized
  const std::size_t windowBufSize = 16384;
  const std::size_t windowNumInts = windowBufSize / sizeof(int);

  DeviceBuffer srcBuffer(srcBufSize);
  DeviceBuffer windowBuffer(windowBufSize);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto window_d = static_cast<int*>(windowBuffer.get());

  std::vector<int> srcHost(srcNumInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), srcBufSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(window_d, 0, windowBufSize));

  // Copy 4KB from src_offset=2048 to dst_offset=4096
  const size_t dst_offset = 4096;
  const size_t src_offset = 2048;
  const std::size_t nbytes = 4096;

  test::testDeviceWindowNvlOffsetPut(
      0,
      2,
      reinterpret_cast<char*>(window_d),
      reinterpret_cast<const char*>(src_d),
      srcBufSize,
      dst_offset,
      src_offset,
      nbytes);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> windowHost(windowNumInts);
  CUDACHECK_TEST(cudaMemcpy(
      windowHost.data(), window_d, windowBufSize, cudaMemcpyDeviceToHost));

  // Expected: zeros everywhere except [dst_offset, dst_offset+nbytes)
  const std::size_t dstStartInt = dst_offset / sizeof(int);
  const std::size_t copyNumInts = nbytes / sizeof(int);
  std::vector<int> expected(windowNumInts, 0);
  std::fill(
      expected.begin() + dstStartInt,
      expected.begin() + dstStartInt + copyNumInts,
      testValue);
  EXPECT_EQ(windowHost, expected)
      << "Offset put should copy data to correct region only";
}

// =============================================================================
// Offset-Based NVL Put + Signal via DeviceWindow
// =============================================================================

TEST_F(DeviceWindowTestFixture, NvlOffsetPutSignal) {
  // Source buffer: 8KB filled with 0xBEEF
  const std::size_t srcBufSize = 8192;
  const std::size_t srcNumInts = srcBufSize / sizeof(int);
  const int testValue = 0xBEEF;

  // Window buffer: 16KB, zero-initialized
  const std::size_t windowBufSize = 16384;
  const std::size_t windowNumInts = windowBufSize / sizeof(int);

  DeviceBuffer srcBuffer(srcBufSize);
  DeviceBuffer windowBuffer(windowBufSize);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto window_d = static_cast<int*>(windowBuffer.get());

  std::vector<int> srcHost(srcNumInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), srcBufSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(window_d, 0, windowBufSize));

  // Copy 4KB from src_offset=0 to dst_offset=8192
  const size_t dst_offset = 8192;
  const size_t src_offset = 0;
  const std::size_t nbytes = 4096;
  const int signalId = 0;

  test::testDeviceWindowNvlOffsetPutSignal(
      0,
      2,
      reinterpret_cast<char*>(window_d),
      reinterpret_cast<const char*>(src_d),
      srcBufSize,
      dst_offset,
      src_offset,
      nbytes,
      signalId);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> windowHost(windowNumInts);
  CUDACHECK_TEST(cudaMemcpy(
      windowHost.data(), window_d, windowBufSize, cudaMemcpyDeviceToHost));

  // Expected: zeros everywhere except [dst_offset, dst_offset+nbytes)
  const std::size_t dstStartInt = dst_offset / sizeof(int);
  const std::size_t copyNumInts = nbytes / sizeof(int);
  std::vector<int> expected(windowNumInts, 0);
  std::fill(
      expected.begin() + dstStartInt,
      expected.begin() + dstStartInt + copyNumInts,
      testValue);
  EXPECT_EQ(windowHost, expected)
      << "Offset put_signal should copy data to correct region only";
}

// =============================================================================
// Bidirectional Offset-Based NVL Put + Signal via DeviceWindow
// =============================================================================

TEST_F(DeviceWindowTestFixture, NvlBidirectionalOffsetPutSignal) {
  // Two source buffers with different patterns
  const std::size_t srcBufSize = 8192;
  const std::size_t srcNumInts = srcBufSize / sizeof(int);
  const int testValue0 = 0xAAAA; // rank 0's data
  const int testValue1 = 0xBBBB; // rank 1's data

  // Two window buffers (one per rank), zero-initialized
  const std::size_t windowBufSize = 16384;
  const std::size_t windowNumInts = windowBufSize / sizeof(int);

  DeviceBuffer srcBuffer0(srcBufSize);
  DeviceBuffer srcBuffer1(srcBufSize);
  DeviceBuffer windowBuffer0(windowBufSize);
  DeviceBuffer windowBuffer1(windowBufSize);

  auto src0_d = static_cast<int*>(srcBuffer0.get());
  auto src1_d = static_cast<int*>(srcBuffer1.get());
  auto window0_d = static_cast<int*>(windowBuffer0.get());
  auto window1_d = static_cast<int*>(windowBuffer1.get());

  // Fill source buffers with distinct patterns
  std::vector<int> srcHost0(srcNumInts, testValue0);
  std::vector<int> srcHost1(srcNumInts, testValue1);
  CUDACHECK_TEST(
      cudaMemcpy(src0_d, srcHost0.data(), srcBufSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemcpy(src1_d, srcHost1.data(), srcBufSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(window0_d, 0, windowBufSize));
  CUDACHECK_TEST(cudaMemset(window1_d, 0, windowBufSize));

  // Both ranks put 4KB from src_offset=0 to dst_offset=4096
  const std::size_t dst_offset = 4096;
  const std::size_t src_offset = 0;
  const std::size_t nbytes = 4096;
  const int signalId = 0;

  test::testDeviceWindowNvlBidirectionalOffsetPutSignal(
      reinterpret_cast<char*>(window0_d),
      reinterpret_cast<char*>(window1_d),
      reinterpret_cast<const char*>(src0_d),
      reinterpret_cast<const char*>(src1_d),
      srcBufSize,
      dst_offset,
      src_offset,
      nbytes,
      signalId);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify rank 0's put landed in windowBuffer1
  const std::size_t dstStartInt = dst_offset / sizeof(int);
  const std::size_t copyNumInts = nbytes / sizeof(int);

  std::vector<int> window1Host(windowNumInts);
  CUDACHECK_TEST(cudaMemcpy(
      window1Host.data(), window1_d, windowBufSize, cudaMemcpyDeviceToHost));

  std::vector<int> expected1(windowNumInts, 0);
  std::fill(
      expected1.begin() + dstStartInt,
      expected1.begin() + dstStartInt + copyNumInts,
      testValue0);
  EXPECT_EQ(window1Host, expected1)
      << "Rank 0's put_signal should write 0xAAAA into rank 1's window buffer";

  // Verify rank 1's put landed in windowBuffer0
  std::vector<int> window0Host(windowNumInts);
  CUDACHECK_TEST(cudaMemcpy(
      window0Host.data(), window0_d, windowBufSize, cudaMemcpyDeviceToHost));

  std::vector<int> expected0(windowNumInts, 0);
  std::fill(
      expected0.begin() + dstStartInt,
      expected0.begin() + dstStartInt + copyNumInts,
      testValue1);
  EXPECT_EQ(window0Host, expected0)
      << "Rank 1's put_signal should write 0xBBBB into rank 0's window buffer";
}

// =============================================================================
// signal_all + read_signal Aggregate
// =============================================================================

TEST_F(DeviceWindowTestFixture, SignalAllAggregate) {
  const int myRank = 0;
  const int nRanks = 4;
  const int signalCount = 2;
  const int signalId = 1;

  DeviceBuffer resultsBuffer(sizeof(uint64_t));
  CUDACHECK_TEST(cudaMemset(resultsBuffer.get(), 0, sizeof(uint64_t)));
  auto results_d = static_cast<uint64_t*>(resultsBuffer.get());

  test::testDeviceWindowSignalAllAggregate(
      myRank, nRanks, signalCount, signalId, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, results_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // signal_all signals every peer (nRanks-1 = 3 peers) with value 1
  // Since we're writing to our own inbox simulating peers, each peer's
  // slot gets 1, so aggregate = nRanks - 1
  EXPECT_EQ(result_h, static_cast<uint64_t>(nRanks - 1))
      << "signal_all aggregate should equal number of peers";
}

// =============================================================================
// IBGDA Signal Read Tests
// =============================================================================

TEST_F(DeviceWindowTestFixture, IbgdaSignalReadFrom) {
  const int myRank = 0;
  const int nRanks = 4;
  const int signalCount = 3;
  const int sourceRank = 2;
  const int signalId = 1;
  const uint64_t seedValue = 42;

  DeviceBuffer resultsBuffer(2 * sizeof(uint64_t));
  CUDACHECK_TEST(cudaMemset(resultsBuffer.get(), 0, 2 * sizeof(uint64_t)));
  auto results_d = static_cast<uint64_t*>(resultsBuffer.get());

  test::testIbgdaSignalRead(
      myRank, nRanks, signalCount, sourceRank, signalId, seedValue, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<uint64_t> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], seedValue)
      << "read_signal_from(rank=" << sourceRank << ", signal=" << signalId
      << ") should return seeded value";
  EXPECT_EQ(results_h[1], seedValue)
      << "read_signal() should include the seeded value in aggregate";
}

// Verify that different (source_rank, signal_id) pairs are isolated —
// writing to one slot doesn't affect another.
TEST_F(DeviceWindowTestFixture, IbgdaSignalSlotIsolation) {
  const int myRank = 1;
  const int nRanks = 4;
  const int signalCount = 4;
  const uint64_t seedValue = 99;

  // Seed rank 0, signal 2
  const int sourceRank = 0;
  const int signalId = 2;

  DeviceBuffer resultsBuffer(2 * sizeof(uint64_t));
  CUDACHECK_TEST(cudaMemset(resultsBuffer.get(), 0, 2 * sizeof(uint64_t)));
  auto results_d = static_cast<uint64_t*>(resultsBuffer.get());

  test::testIbgdaSignalRead(
      myRank, nRanks, signalCount, sourceRank, signalId, seedValue, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<uint64_t> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], seedValue)
      << "read_signal_from() should see seeded value";

  // Now read a DIFFERENT slot — should still be zero
  const int otherRank = 2;
  const int otherSignal = 3;
  CUDACHECK_TEST(cudaMemset(resultsBuffer.get(), 0, 2 * sizeof(uint64_t)));

  test::testIbgdaSignalRead(
      myRank, nRanks, signalCount, otherRank, otherSignal, 0, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], 0u)
      << "read_signal_from() on unseeded slot should be zero";
}

TEST_F(DeviceWindowTestFixture, IbgdaSignalAggregateRead) {
  const int myRank = 0;
  const int nRanks = 4;
  const int signalCount = 2;
  const int signalId = 1;
  const int nPeers = nRanks - 1;

  std::vector<uint64_t> peerValues = {10, 20, 30};

  DeviceBuffer resultBuf(sizeof(uint64_t));
  CUDACHECK_TEST(cudaMemset(resultBuf.get(), 0, sizeof(uint64_t)));

  test::testIbgdaSignalAggregateRead(
      myRank,
      nRanks,
      signalCount,
      signalId,
      peerValues.data(),
      nPeers,
      static_cast<uint64_t*>(resultBuf.get()));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, resultBuf.get(), sizeof(uint64_t), cudaMemcpyDeviceToHost));

  uint64_t expectedTotal = 10 + 20 + 30;
  EXPECT_EQ(result_h, expectedTotal)
      << "read_signal() aggregate should sum all peers";
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(DeviceWindowTestFixture, SingleRank) {
  const int myRank = 0;
  const int nRanks = 1;
  const int signalCount = 1;

  DeviceBuffer resultsBuffer(3 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowConstruction(myRank, nRanks, signalCount, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(3);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank);
  EXPECT_EQ(results_h[1], nRanks);
  EXPECT_EQ(results_h[2], nRanks - 1);
}

TEST_F(DeviceWindowTestFixture, MaxRanks) {
  const int myRank = 7;
  const int nRanks = 8;

  DeviceBuffer resultsBuffer(2 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowBasicAccessors(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 2 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
}

// =============================================================================
// Self-Transport Tests
// =============================================================================

TEST_F(DeviceWindowTestFixture, GetTransportReturnsCorrectType) {
  P2pSelfTransportDevice selfTransport;
  Transport hostTransport(selfTransport);

  Transport* transport_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&transport_d, sizeof(Transport)));
  CUDACHECK_TEST(cudaMemcpy(
      transport_d, &hostTransport, sizeof(Transport), cudaMemcpyHostToDevice));

  DeviceBuffer resultsBuffer(sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0, sizeof(int)));

  test::testGetTransportType(transport_d, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, results_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Transport type should be SELF";

  CUDACHECK_TEST(cudaFree(transport_d));
}

TEST_F(DeviceWindowTestFixture, SelfTransportPutCopiesData) {
  const std::size_t nbytes = 4096;
  const std::size_t numInts = nbytes / sizeof(int);
  const int testValue = 42;

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  std::vector<int> srcHost(numInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  P2pSelfTransportDevice selfTransport;
  Transport hostTransport(selfTransport);

  Transport* transport_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&transport_d, sizeof(Transport)));
  CUDACHECK_TEST(cudaMemcpy(
      transport_d, &hostTransport, sizeof(Transport), cudaMemcpyHostToDevice));

  test::testSelfTransportPut(
      transport_d,
      reinterpret_cast<char*>(dst_d),
      reinterpret_cast<const char*>(src_d),
      nbytes,
      4,
      256);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> dstHost(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(dstHost.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  const std::vector<int> expected(numInts, testValue);
  EXPECT_EQ(dstHost, expected) << "Self-transport put should copy all data";

  CUDACHECK_TEST(cudaFree(transport_d));
}

TEST_F(DeviceWindowTestFixture, SelfTransportPutLargeTransfer) {
  const std::size_t nbytes = 1024 * 1024;
  const std::size_t numInts = nbytes / sizeof(int);
  const int testValue = 0xDEADBEEF;

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  std::vector<int> srcHost(numInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  P2pSelfTransportDevice selfTransport;
  Transport hostTransport(selfTransport);

  Transport* transport_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&transport_d, sizeof(Transport)));
  CUDACHECK_TEST(cudaMemcpy(
      transport_d, &hostTransport, sizeof(Transport), cudaMemcpyHostToDevice));

  test::testSelfTransportPut(
      transport_d,
      reinterpret_cast<char*>(dst_d),
      reinterpret_cast<const char*>(src_d),
      nbytes,
      8,
      256);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> dstHost(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(dstHost.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  EXPECT_EQ(dstHost[0], testValue) << "First element mismatch";
  EXPECT_EQ(dstHost[numInts / 2], testValue) << "Middle element mismatch";
  EXPECT_EQ(dstHost[numInts - 1], testValue) << "Last element mismatch";

  int errorCount = 0;
  for (std::size_t i = 0; i < numInts; ++i) {
    if (dstHost[i] != testValue) {
      ++errorCount;
    }
  }
  EXPECT_EQ(errorCount, 0) << "Total mismatches in 1MB transfer: "
                           << errorCount;

  CUDACHECK_TEST(cudaFree(transport_d));
}

// =============================================================================
// Peer Iteration Helper Tests
// =============================================================================

TEST_F(DeviceWindowTestFixture, PeerIterationHelpersRank0) {
  const int myRank = 0;
  const int nRanks = 4;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numPeers) << "numPeers() should return " << numPeers;
  EXPECT_EQ(results_h[1], 1) << "peerIndexToRank(0) should return 1";
  EXPECT_EQ(results_h[2], 2) << "peerIndexToRank(1) should return 2";
  EXPECT_EQ(results_h[3], 3) << "peerIndexToRank(2) should return 3";
}

TEST_F(DeviceWindowTestFixture, PeerIterationHelpersRank2) {
  const int myRank = 2;
  const int nRanks = 4;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numPeers);
  EXPECT_EQ(results_h[1], 0);
  EXPECT_EQ(results_h[2], 1);
  EXPECT_EQ(results_h[3], 3);
}

TEST_F(DeviceWindowTestFixture, PeerIterationHelpersRank7) {
  const int myRank = 7;
  const int nRanks = 8;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numPeers);
  std::vector<int> expectedPeerRanks(numPeers);
  std::iota(expectedPeerRanks.begin(), expectedPeerRanks.end(), 0);
  std::vector<int> actualPeerRanks(results_h.begin() + 1, results_h.end());
  EXPECT_EQ(actualPeerRanks, expectedPeerRanks);
}

TEST_F(DeviceWindowTestFixture, PeerIterationHelpersSinglePeer) {
  const int myRank = 0;
  const int nRanks = 2;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], 1);
  EXPECT_EQ(results_h[1], 1);
}

// =============================================================================
// Peer Index Conversion Roundtrip Tests
// =============================================================================

void verifyPeerIndexConversionRoundtrip(int myRank, int nRanks) {
  const int numPeers = nRanks - 1;
  const int numResults = 4 * numPeers + 2;

  DeviceBuffer resultsBuffer(numResults * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0xFF, numResults * sizeof(int)));

  test::testPeerIndexConversionRoundtrip(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> r(numResults);
  CUDACHECK_TEST(cudaMemcpy(
      r.data(), results_d, numResults * sizeof(int), cudaMemcpyDeviceToHost));

  int idx = 0;

  EXPECT_EQ(r[idx++], numPeers);

  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    int expectedPeerIndex = (rank < myRank) ? rank : (rank - 1);
    EXPECT_EQ(r[idx], expectedPeerIndex)
        << "rank_to_peer_index(" << rank << ") for myRank=" << myRank;
    idx++;
  }

  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    EXPECT_EQ(r[idx], rank)
        << "Roundtrip for rank " << rank << " myRank=" << myRank;
    idx++;
  }

  for (int i = 0; i < numPeers; ++i) {
    EXPECT_EQ(r[idx], i) << "Roundtrip for peer_index " << i
                         << " myRank=" << myRank;
    idx++;
  }

  EXPECT_EQ(r[idx++], 0)
      << "get_self_transport()->type should be SELF for myRank=" << myRank;

  for (int i = 0; i < numPeers; ++i) {
    EXPECT_EQ(r[idx], 1) << "get_peer_transport()->type should be P2P_NVL";
    idx++;
  }
}

TEST_F(DeviceWindowTestFixture, PeerIndexConversionRoundtripRank0Of4) {
  verifyPeerIndexConversionRoundtrip(0, 4);
}

TEST_F(DeviceWindowTestFixture, PeerIndexConversionRoundtripRank2Of4) {
  verifyPeerIndexConversionRoundtrip(2, 4);
}

TEST_F(DeviceWindowTestFixture, PeerIndexConversionRoundtripRank3Of4) {
  verifyPeerIndexConversionRoundtrip(3, 4);
}

TEST_F(DeviceWindowTestFixture, PeerIndexConversionRoundtripRank0Of2) {
  verifyPeerIndexConversionRoundtrip(0, 2);
}

TEST_F(DeviceWindowTestFixture, PeerIndexConversionRoundtripRank1Of2) {
  verifyPeerIndexConversionRoundtrip(1, 2);
}

TEST_F(DeviceWindowTestFixture, PeerIndexConversionRoundtripRank4Of8) {
  verifyPeerIndexConversionRoundtrip(4, 8);
}

} // namespace comms::pipes
