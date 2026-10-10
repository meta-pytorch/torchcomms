// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>

#include "comms/prims/transport/llx/LlPacket.cuh"
#include "comms/prims/transport/llx/tests/LlPacketTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {

// ---- compile-time geometry: the two tiers and the atomicity invariant ----
static_assert(
    Ll128PacketGeometry::kData == 120 && Ll128PacketGeometry::kFlag == 8);
static_assert(Ll128PacketGeometry::kPacketBytes % sizeof(uint64_t) == 0);
static_assert(
    Ll128PacketGeometry::kPacketBytes == 128 &&
    Ll128PacketGeometry::kThreadsPerPacket == 8);
static_assert(
    Ll128PacketGeometry::kFlagLane == 7 &&
    Ll128PacketGeometry::kPacketsPerWarp ==
        comms::device::kWarpSize / Ll128PacketGeometry::kThreadsPerPacket);
static_assert(LlPacketGeometry::kData == 4 && LlPacketGeometry::kFlag == 4);
static_assert(LlPacketGeometry::kPacketBytes % sizeof(uint64_t) == 0);
static_assert(
    LlPacketGeometry::kPacketBytes == 8 &&
    LlPacketGeometry::kThreadsPerPacket == 1);
static_assert(
    LlPacketGeometry::kFlagLane == 0 &&
    LlPacketGeometry::kPacketsPerWarp ==
        comms::device::kWarpSize / LlPacketGeometry::kThreadsPerPacket);

class LlPacketTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  uint32_t runKernel(void (*fn)(uint32_t*)) {
    DeviceBuffer errBuf(sizeof(uint32_t));
    auto* err_d = static_cast<uint32_t*>(errBuf.get());
    CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));
    fn(err_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    uint32_t err_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return err_h;
  }
};

TEST_F(LlPacketTestFixture, DeviceGeometry) {
  EXPECT_EQ(runKernel(test::test_ll_packet_geometry), 0u)
      << "Packet geometry constants wrong";
}

TEST_F(LlPacketTestFixture, DeviceSlotAndFlag) {
  DeviceBuffer p128(Ll128PacketGeometry::kPacketBytes);
  DeviceBuffer p8(LlPacketGeometry::kPacketBytes);
  CUDACHECK_TEST(cudaMemset(p128.get(), 0, Ll128PacketGeometry::kPacketBytes));
  CUDACHECK_TEST(cudaMemset(p8.get(), 0, LlPacketGeometry::kPacketBytes));

  DeviceBuffer errBuf(sizeof(uint32_t));
  auto* err_d = static_cast<uint32_t*>(errBuf.get());
  CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));

  test::test_ll_packet_slot_flag(p128.get(), p8.get(), err_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t err_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(err_h, 0u) << "ll_slot_ptr / flag round-trip wrong";
}

TEST_F(LlPacketTestFixture, HostNumPackets) {
  // LL128 (kData = 120) — matches legacy Ll128 math.
  EXPECT_EQ(Ll128PacketGeometry::num_packets(0), 0u);
  EXPECT_EQ(Ll128PacketGeometry::num_packets(1), 1u);
  EXPECT_EQ(Ll128PacketGeometry::num_packets(120), 1u);
  EXPECT_EQ(Ll128PacketGeometry::num_packets(121), 2u);
  EXPECT_EQ(Ll128PacketGeometry::num_packets(65536), 547u);
  // LL (kData = 4).
  EXPECT_EQ(LlPacketGeometry::num_packets(0), 0u);
  EXPECT_EQ(LlPacketGeometry::num_packets(1), 1u);
  EXPECT_EQ(LlPacketGeometry::num_packets(4), 1u);
  EXPECT_EQ(LlPacketGeometry::num_packets(5), 2u);
  EXPECT_EQ(LlPacketGeometry::num_packets(64), 16u);
}

TEST_F(LlPacketTestFixture, HostPayloadSize) {
  EXPECT_EQ(Ll128PacketGeometry::payload_size(0, 0), 0u);
  EXPECT_EQ(Ll128PacketGeometry::payload_size(0, 121), 120u);
  EXPECT_EQ(Ll128PacketGeometry::payload_size(1, 121), 1u);
  EXPECT_EQ(Ll128PacketGeometry::payload_size(1, 120), 0u); // out of range

  EXPECT_EQ(LlPacketGeometry::payload_size(0, 4), 4u);
  EXPECT_EQ(LlPacketGeometry::payload_size(0, 5), 4u);
  EXPECT_EQ(LlPacketGeometry::payload_size(1, 5), 1u);
  EXPECT_EQ(LlPacketGeometry::payload_size(1, 4), 0u); // out of range
}

TEST_F(LlPacketTestFixture, HostBufferSize) {
  EXPECT_EQ(Ll128PacketGeometry::buffer_size(0), 0u);
  EXPECT_EQ(Ll128PacketGeometry::buffer_size(120), 128u);
  EXPECT_EQ(Ll128PacketGeometry::buffer_size(121), 256u);

  EXPECT_EQ(LlPacketGeometry::buffer_size(0), 0u);
  EXPECT_EQ(LlPacketGeometry::buffer_size(4), 8u);
  EXPECT_EQ(LlPacketGeometry::buffer_size(5), 16u);
}

} // namespace comms::prims
