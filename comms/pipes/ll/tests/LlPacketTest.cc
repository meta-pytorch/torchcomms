// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>

#include "comms/pipes/ll/LlPacket.cuh"
#include "comms/pipes/ll/tests/LlPacketTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class LlPacketTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

TEST_F(LlPacketTestFixture, LineSize) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll_line_size(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "LlLine size checks failed";
}

TEST_F(LlPacketTestFixture, FlagReadWrite) {
  DeviceBuffer lineBuffer(sizeof(LlLine));
  CUDACHECK_TEST(cudaMemset(lineBuffer.get(), 0, sizeof(LlLine)));

  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll_flag_read_write(lineBuffer.get(), errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Flag read/write round-trip failed";
}

TEST_F(LlPacketTestFixture, NumLines) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll_num_lines(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Num lines calculation failed";
}

TEST_F(LlPacketTestFixture, BufferSize) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll_buffer_size(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Buffer size calculation failed";
}

TEST_F(LlPacketTestFixture, BufferPayloadCapacity) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll_buffer_payload_capacity(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Buffer payload capacity calculation failed";
}

TEST_F(LlPacketTestFixture, FlagInitViaCudaMemset) {
  DeviceBuffer lineBuffer(sizeof(LlLine));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  CUDACHECK_TEST(
      cudaMemset(lineBuffer.get(), kLlMemsetInitByte, sizeof(LlLine)));

  test::test_ll_flag_init(lineBuffer.get(), errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "cudaMemset 0xFF should produce flags == kLlReadyToWrite";
}

TEST_F(LlPacketTestFixture, HostNumLinesCalculation) {
  EXPECT_EQ(ll_num_lines(0), 0u);
  EXPECT_EQ(ll_num_lines(1), 1u);
  EXPECT_EQ(ll_num_lines(7), 1u);
  EXPECT_EQ(ll_num_lines(8), 1u);
  EXPECT_EQ(ll_num_lines(9), 2u);
  EXPECT_EQ(ll_num_lines(16), 2u);
  EXPECT_EQ(ll_num_lines(17), 3u);
  EXPECT_EQ(ll_num_lines(65536), 8192u);

  EXPECT_EQ(ll_buffer_size(0), 0u);
  EXPECT_EQ(ll_buffer_size(8), 16u);
  EXPECT_EQ(ll_buffer_size(16), 32u);
  EXPECT_EQ(ll_buffer_size(65536), 131072u);

  EXPECT_EQ(ll_buffer_payload_capacity(0), 0u);
  EXPECT_EQ(ll_buffer_payload_capacity(16), 8u);
  EXPECT_EQ(ll_buffer_payload_capacity(32), 16u);
  EXPECT_EQ(ll_buffer_payload_capacity(131072), 65536u);
}

TEST_F(LlPacketTestFixture, HostCanUseLl) {
  // nbytes == 0 is always eligible
  EXPECT_TRUE(can_use_ll(nullptr, 0));
  EXPECT_TRUE(can_use_ll(reinterpret_cast<const void*>(uintptr_t(1)), 0));

  // Aligned pointer (0x100) + multiple of 8
  auto* aligned = reinterpret_cast<const void*>(uintptr_t(0x100));
  EXPECT_TRUE(can_use_ll(aligned, 8));
  EXPECT_TRUE(can_use_ll(aligned, 16));
  EXPECT_TRUE(can_use_ll(aligned, 1024));

  // Aligned pointer + NOT multiple of 8
  EXPECT_FALSE(can_use_ll(aligned, 1));
  EXPECT_FALSE(can_use_ll(aligned, 7));
  EXPECT_FALSE(can_use_ll(aligned, 9));

  // Misaligned pointer (0x101) + multiple of 8
  auto* misaligned = reinterpret_cast<const void*>(uintptr_t(0x101));
  EXPECT_FALSE(can_use_ll(misaligned, 8));
  EXPECT_FALSE(can_use_ll(misaligned, 16));

  // Misaligned + not multiple of 8
  EXPECT_FALSE(can_use_ll(misaligned, 9));
}

TEST_F(LlPacketTestFixture, HostLlFlag) {
  EXPECT_EQ(ll_flag(0), 1u);
  EXPECT_EQ(ll_flag(1), 2u);
  EXPECT_EQ(ll_flag(41), 42u);
  // Flag values should never be 0 or kLlReadyToWrite
  EXPECT_NE(ll_flag(0), 0u);
  EXPECT_NE(ll_flag(0), kLlReadyToWrite);
}

TEST_F(LlPacketTestFixture, CanUseLl) {
  DeviceBuffer alignedBuffer(256);
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_can_use_ll(
      static_cast<const char*>(alignedBuffer.get()), errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "can_use_ll device-side checks failed";
}

TEST_F(LlPacketTestFixture, LlFlag) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll_flag(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "ll_flag device-side checks failed";
}

} // namespace comms::pipes
