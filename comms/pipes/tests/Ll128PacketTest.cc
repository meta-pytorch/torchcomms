// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>

#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/tests/Ll128PacketTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class Ll128PacketTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

TEST_F(Ll128PacketTestFixture, Alignment) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll128_packet_alignment(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Packet alignment/size checks failed";
}

TEST_F(Ll128PacketTestFixture, FlagReadWrite) {
  DeviceBuffer packetBuffer(sizeof(Ll128Packet));
  CUDACHECK_TEST(cudaMemset(packetBuffer.get(), 0, sizeof(Ll128Packet)));

  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll128_flag_read_write(packetBuffer.get(), errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Flag read/write round-trip failed";
}

TEST_F(Ll128PacketTestFixture, PayloadSize) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll128_packet_payload_size(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Payload size calculation failed";
}

TEST_F(Ll128PacketTestFixture, NumPackets) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll128_num_packets(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Num packets calculation failed";
}

TEST_F(Ll128PacketTestFixture, SlotPtr) {
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_ll128_slot_ptr(errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "Slot pointer addresses incorrect";
}

TEST_F(Ll128PacketTestFixture, FlagInitViaCudaMemset) {
  // Allocate a packet on device, memset to 0xFF, verify flag == -1
  DeviceBuffer packetBuffer(sizeof(Ll128Packet));
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  // 0xFF fills all bytes, int64_t all-ones = -1 in two's complement
  CUDACHECK_TEST(cudaMemset(
      packetBuffer.get(), kLl128MemsetInitByte, sizeof(Ll128Packet)));

  test::test_ll128_flag_init(packetBuffer.get(), errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0)
      << "cudaMemset 0xFF should produce flag == kLl128ReadyToWrite (-1)";
}

TEST_F(Ll128PacketTestFixture, HostPayloadSizeCalculation) {
  // Verify the host-side versions of the helper functions
  EXPECT_EQ(ll128_packet_payload_size(0, 0), 0u);
  EXPECT_EQ(ll128_packet_payload_size(0, 1), 1u);
  EXPECT_EQ(ll128_packet_payload_size(0, 120), 120u);
  EXPECT_EQ(ll128_packet_payload_size(0, 121), 120u);
  EXPECT_EQ(ll128_packet_payload_size(1, 121), 1u);

  EXPECT_EQ(ll128_num_packets(0), 0u);
  EXPECT_EQ(ll128_num_packets(1), 1u);
  EXPECT_EQ(ll128_num_packets(120), 1u);
  EXPECT_EQ(ll128_num_packets(121), 2u);
  EXPECT_EQ(ll128_num_packets(65536), 547u);

  EXPECT_EQ(ll128_buffer_size(0), 0u);
  EXPECT_EQ(ll128_buffer_size(120), 128u);
  EXPECT_EQ(ll128_buffer_size(121), 256u);
  EXPECT_EQ(ll128_buffer_size(65536), 547u * 128u);
}

TEST_F(Ll128PacketTestFixture, HostCanUseLl128) {
  // nbytes == 0 is always eligible regardless of pointer
  EXPECT_TRUE(can_use_ll128(nullptr, 0));
  EXPECT_TRUE(can_use_ll128(reinterpret_cast<const void*>(uintptr_t(1)), 0));

  // Aligned pointer (0x100) + multiple of 16
  auto* aligned = reinterpret_cast<const void*>(uintptr_t(0x100));
  EXPECT_TRUE(can_use_ll128(aligned, 16));
  EXPECT_TRUE(can_use_ll128(aligned, 32));
  EXPECT_TRUE(can_use_ll128(aligned, 1024));

  // Aligned pointer + NOT multiple of 16
  EXPECT_FALSE(can_use_ll128(aligned, 1));
  EXPECT_FALSE(can_use_ll128(aligned, 15));
  EXPECT_FALSE(can_use_ll128(aligned, 17));

  // Misaligned pointer (0x101) + multiple of 16
  auto* misaligned = reinterpret_cast<const void*>(uintptr_t(0x101));
  EXPECT_FALSE(can_use_ll128(misaligned, 16));
  EXPECT_FALSE(can_use_ll128(misaligned, 32));

  // Misaligned + not multiple of 16
  EXPECT_FALSE(can_use_ll128(misaligned, 17));
}

TEST_F(Ll128PacketTestFixture, CanUseLl128) {
  DeviceBuffer alignedBuffer(256); // cudaMalloc guarantees >= 256B alignment
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));
  auto* errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  test::test_can_use_ll128(
      static_cast<const char*>(alignedBuffer.get()), errorCount_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errorCount_h, 0) << "can_use_ll128 device-side checks failed";
}

} // namespace comms::pipes
