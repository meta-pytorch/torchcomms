// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include <vector>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/tests/DeviceSpanTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class DeviceSpanTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

// =============================================================================
// Basic Properties Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, BasicProperties) {
  const uint32_t size = 10;
  std::vector<uint32_t> hostData(size);
  std::iota(hostData.begin(), hostData.end(), 1); // 1, 2, 3, ..., 10

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer resultsBuffer(2 * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto results_d = static_cast<uint32_t*>(resultsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemcpy(
      data_d,
      hostData.data(),
      size * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testBasicProperties(data_d, size, results_d, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Basic properties test should have no errors";

  std::vector<uint32_t> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], size) << "size() should return " << size;
  EXPECT_EQ(results_h[1], 0) << "empty() should return false (0)";
}

TEST_F(DeviceSpanTestFixture, BasicPropertiesEmptySpan) {
  DeviceBuffer resultsBuffer(2 * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto results_d = static_cast<uint32_t*>(resultsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  // Pass nullptr and size 0 for empty span
  test::testBasicProperties(nullptr, 0, results_d, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "Empty span properties test should have no errors";

  std::vector<uint32_t> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], 0) << "size() should return 0 for empty span";
  EXPECT_EQ(results_h[1], 1) << "empty() should return true (1)";
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, ElementAccess) {
  const uint32_t size = 10;
  std::vector<uint32_t> hostData(size);
  std::iota(hostData.begin(), hostData.end(), 100); // 100, 101, ..., 109

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer resultsBuffer(2 * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto results_d = static_cast<uint32_t*>(resultsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemcpy(
      data_d,
      hostData.data(),
      size * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testElementAccess(data_d, size, results_d, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Element access test should have no errors";

  std::vector<uint32_t> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      2 * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], 100) << "front() should return first element";
  EXPECT_EQ(results_h[1], 109) << "back() should return last element";
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, Iterator) {
  const uint32_t size = 10;
  std::vector<uint32_t> hostData(size);
  std::iota(hostData.begin(), hostData.end(), 1); // 1, 2, 3, ..., 10

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer sumBuffer(sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto sum_d = static_cast<uint32_t*>(sumBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemcpy(
      data_d,
      hostData.data(),
      size * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testIterator(data_d, size, sum_d, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Iterator test should have no errors";

  uint32_t sum_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&sum_h, sum_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  uint32_t expectedSum = (size * (size + 1)) / 2; // Sum of 1 to 10 = 55
  EXPECT_EQ(sum_h, expectedSum)
      << "Range-based for loop should sum all elements";
}

// =============================================================================
// Subspan Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, Subspan) {
  const uint32_t size = 10;
  std::vector<uint32_t> hostData(size);
  std::iota(hostData.begin(), hostData.end(), 0); // 0, 1, 2, ..., 9

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer resultsBuffer(6 * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto results_d = static_cast<uint32_t*>(resultsBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemcpy(
      data_d,
      hostData.data(),
      size * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testSubspan(data_d, size, results_d, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Subspan test should have no errors";

  std::vector<uint32_t> results_h(6);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      6 * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));

  // results[0] = subspan(2, 3).size() = 3
  EXPECT_EQ(results_h[0], 3) << "subspan(2, 3).size() should be 3";
  // results[1] = subspan(2, 3)[0] = data[2] = 2
  EXPECT_EQ(results_h[1], 2) << "subspan(2, 3)[0] should be data[2]";
  // results[2] = subspan(2).size() = 10 - 2 = 8
  EXPECT_EQ(results_h[2], 8) << "subspan(2).size() should be 8";
  // results[3] = first(3).size() = 3
  EXPECT_EQ(results_h[3], 3) << "first(3).size() should be 3";
  // results[4] = last(3).size() = 3
  EXPECT_EQ(results_h[4], 3) << "last(3).size() should be 3";
  // results[5] = last(3)[0] = data[10-3] = data[7] = 7
  EXPECT_EQ(results_h[5], 7) << "last(3)[0] should be data[7]";
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, MakeDeviceSpan) {
  const uint32_t size = 5;
  std::vector<uint32_t> hostData = {10, 20, 30, 40, 50};

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemcpy(
      data_d,
      hostData.data(),
      size * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testMakeDeviceSpan(data_d, size, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "make_device_span factory function should work correctly";
}

// =============================================================================
// Const Conversion Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, ConstConversion) {
  const uint32_t size = 5;
  std::vector<uint32_t> hostData = {1, 2, 3, 4, 5};

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemcpy(
      data_d,
      hostData.data(),
      size * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testConstConversion(data_d, size, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "Implicit conversion from mutable to const span should work";
}

// =============================================================================
// Empty Span Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, EmptySpan) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testEmptySpan(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Empty span operations should work correctly";
}

// =============================================================================
// Mutable Span Write Tests
// =============================================================================

TEST_F(DeviceSpanTestFixture, MutableSpanWrite) {
  const uint32_t size = 5;

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto data_d = static_cast<uint32_t*>(dataBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  CUDACHECK_TEST(cudaMemset(data_d, 0, size * sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::testMutableSpanWrite(data_d, size, errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Writing through mutable span should work";

  // Verify data was written correctly
  std::vector<uint32_t> data_h(size);
  CUDACHECK_TEST(cudaMemcpy(
      data_h.data(), data_d, size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < size; i++) {
    EXPECT_EQ(data_h[i], i * 10)
        << "Element " << i << " should be " << i * 10 << " after write";
  }
}

// =============================================================================
// Host-side DeviceSpan Tests (metadata only - element access is device-only)
// =============================================================================

TEST_F(DeviceSpanTestFixture, HostSideConstruction) {
  const uint32_t size = 5;

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  auto data_d = static_cast<uint32_t*>(dataBuffer.get());

  // Test host-side construction with device pointer
  DeviceSpan<uint32_t> span(data_d, size);

  // Host can only access metadata
  EXPECT_EQ(span.size(), size);
  EXPECT_EQ(span.data(), data_d);
  EXPECT_FALSE(span.empty());
}

TEST_F(DeviceSpanTestFixture, HostSideEmptySpan) {
  // Default constructor
  DeviceSpan<uint32_t> emptySpan;

  EXPECT_EQ(emptySpan.size(), 0u);
  EXPECT_EQ(emptySpan.data(), nullptr);
  EXPECT_TRUE(emptySpan.empty());

  // Constructor with nullptr and 0
  DeviceSpan<uint32_t> emptySpan2(nullptr, 0);
  EXPECT_EQ(emptySpan2.size(), 0u);
  EXPECT_TRUE(emptySpan2.empty());
}

TEST_F(DeviceSpanTestFixture, HostSideConstConversion) {
  const uint32_t size = 3;

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  auto data_d = static_cast<uint32_t*>(dataBuffer.get());

  DeviceSpan<uint32_t> mutableSpan(data_d, size);

  // Implicit conversion to const (host can do this)
  DeviceSpan<const uint32_t> constSpan = mutableSpan;

  // Host can only check metadata
  EXPECT_EQ(constSpan.size(), mutableSpan.size());
  EXPECT_EQ(constSpan.data(), mutableSpan.data());
}

TEST_F(DeviceSpanTestFixture, HostSideMakeDeviceSpan) {
  const uint32_t size = 3;

  DeviceBuffer dataBuffer(size * sizeof(uint32_t));
  auto data_d = static_cast<uint32_t*>(dataBuffer.get());

  auto span = make_device_span(data_d, size);

  // Host can only access metadata
  EXPECT_EQ(span.size(), size);
  EXPECT_EQ(span.data(), data_d);
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
