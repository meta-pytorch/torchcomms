// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>

#include "comms/prims/transport/ll/LlPacket.cuh"
#include "comms/prims/transport/ll/tests/LlPacketTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {

class LlPacketTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  template <typename F>
  void expect_no_device_errors(F&& launcher, const char* fail_msg) {
    DeviceBuffer errorCountBuffer(sizeof(uint32_t));
    auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
    CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));
    launcher(errorCount_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    uint32_t errorCount_h = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(errorCount_h, 0) << fail_msg;
  }
};

TEST_F(LlPacketTestFixture, LineSize) {
  expect_no_device_errors(test::test_ll_line_size, "LlLine size checks failed");
}

TEST_F(LlPacketTestFixture, FlagReadWrite) {
  DeviceBuffer lineBuffer(sizeof(LlLine));
  CUDACHECK_TEST(cudaMemset(lineBuffer.get(), 0, sizeof(LlLine)));
  expect_no_device_errors(
      [&](uint32_t* ec) {
        test::test_ll_flag_read_write(lineBuffer.get(), ec);
      },
      "Flag read/write round-trip failed");
}

TEST_F(LlPacketTestFixture, NumLines) {
  expect_no_device_errors(
      test::test_ll_num_lines, "Num lines calculation failed");
}

TEST_F(LlPacketTestFixture, BufferSize) {
  expect_no_device_errors(
      test::test_ll_buffer_size, "Buffer size calculation failed");
}

TEST_F(LlPacketTestFixture, BufferPayloadCapacity) {
  expect_no_device_errors(
      test::test_ll_buffer_payload_capacity,
      "Buffer payload capacity calculation failed");
}

TEST_F(LlPacketTestFixture, FlagInitViaCudaMemset) {
  DeviceBuffer lineBuffer(sizeof(LlLine));
  CUDACHECK_TEST(
      cudaMemset(lineBuffer.get(), kLlMemsetInitByte, sizeof(LlLine)));
  expect_no_device_errors(
      [&](uint32_t* ec) { test::test_ll_flag_init(lineBuffer.get(), ec); },
      "cudaMemset 0xFF should produce flags == kLlReadyToWrite");
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

  // buffer_num_lines checks (third parameter)

  // buffer_num_lines=0 (default) -- no buffer constraint
  EXPECT_TRUE(can_use_ll(aligned, 8, 0));

  // nbytes=0 with buffer_num_lines>0 -- early return, always eligible
  EXPECT_TRUE(can_use_ll(nullptr, 0, 64));

  // buffer >= message -- always fine
  EXPECT_TRUE(can_use_ll(aligned, 64, 8)); // 8 lines needed, 8 provided
  EXPECT_TRUE(can_use_ll(aligned, 64, 100)); // 8 lines needed, 100 provided

  // buffer < message but >= kLlLinesPerWarp -- fine (chunking)
  EXPECT_TRUE(can_use_ll(aligned, 1024, 32)); // 128 lines needed, 32 provided
  EXPECT_TRUE(can_use_ll(aligned, 1024, 64)); // 128 lines needed, 64 provided

  // buffer < message and < kLlLinesPerWarp -- rejected
  EXPECT_FALSE(can_use_ll(aligned, 1024, 16)); // 128 lines needed, 16 < 32
  EXPECT_FALSE(can_use_ll(aligned, 1024, 1)); // 128 lines needed, 1 < 32
}

TEST_F(LlPacketTestFixture, HostLlFlag) {
  // Basic values
  EXPECT_EQ(ll_flag(0), 1u);
  EXPECT_EQ(ll_flag(1), 2u);
  EXPECT_EQ(ll_flag(41), 42u);

  // Flag values should never be 0 or kLlReadyToWrite
  EXPECT_NE(ll_flag(0), 0u);
  EXPECT_NE(ll_flag(0), kLlReadyToWrite);

  // Boundary: step = kLlMaxFlag - 1 produces maximum valid flag
  EXPECT_EQ(ll_flag(static_cast<size_t>(kLlMaxFlag - 1)), kLlMaxFlag);

  // Boundary: step = kLlMaxFlag wraps back to 1
  EXPECT_EQ(ll_flag(static_cast<size_t>(kLlMaxFlag)), 1u);

  // Boundary: step = kLlMaxFlag + 1 wraps to 2
  EXPECT_EQ(ll_flag(static_cast<size_t>(kLlMaxFlag) + 1), 2u);

  // Boundary: step = 0xFFFFFFFE (would have been kLlReadyToWrite with old impl)
  EXPECT_EQ(ll_flag(static_cast<size_t>(0xFFFFFFFE)), 1u);
  EXPECT_NE(ll_flag(static_cast<size_t>(0xFFFFFFFE)), kLlReadyToWrite);

  // Boundary: step = 0xFFFFFFFF (would have wrapped to 0 with old impl)
  EXPECT_EQ(ll_flag(static_cast<size_t>(0xFFFFFFFF)), 2u);
  EXPECT_NE(ll_flag(static_cast<size_t>(0xFFFFFFFF)), 0u);
}

TEST_F(LlPacketTestFixture, CanUseLl) {
  DeviceBuffer alignedBuffer(256);
  expect_no_device_errors(
      [&](uint32_t* ec) {
        test::test_can_use_ll(
            static_cast<const char*>(alignedBuffer.get()), ec);
      },
      "can_use_ll device-side checks failed");
}

TEST_F(LlPacketTestFixture, LlFlag) {
  expect_no_device_errors(
      test::test_ll_flag, "ll_flag device-side checks failed");
}

} // namespace comms::prims
