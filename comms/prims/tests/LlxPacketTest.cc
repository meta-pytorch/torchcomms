// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>

#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/tests/LlxPacketTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {

// ---- compile-time geometry: the 8 B tier + the atomicity invariant ----
static_assert(LlxPacketGeometry::kData == 4 && LlxPacketGeometry::kFlag == 4);
static_assert(
    LlxPacketGeometry::kPacketBytes == 8 &&
    LlxPacketGeometry::kThreadsPerPacket == 1 &&
    LlxPacketGeometry::kFlagLane == 0);
static_assert(std::is_same_v<LlxPacketGeometry::FlagType, uint32_t>);

// ---- sizing math is independently constructed (not copied from impl) ----
static_assert(LlxPacketGeometry::wire_bytes(0) == 0);
static_assert(LlxPacketGeometry::wire_bytes(4) == 8); // 1 packet
static_assert(LlxPacketGeometry::wire_bytes(5) == 16); // 2 packets
static_assert(LlxPacketGeometry::max_payload(16) == 8); // 2 packets * 4 B

class LlxPacketTest : public ::testing::Test {
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

TEST_F(LlxPacketTest, DeviceGeometry) {
  EXPECT_EQ(runKernel(test::test_llpacket_geometry), 0u)
      << "LlxPacket geometry constants wrong on device";
}

TEST_F(LlxPacketTest, DeviceAddressing) {
  DeviceBuffer p8(LlxPacketGeometry::kPacketBytes);
  CUDACHECK_TEST(cudaMemset(p8.get(), 0, LlxPacketGeometry::kPacketBytes));

  DeviceBuffer errBuf(sizeof(uint32_t));
  auto* err_d = static_cast<uint32_t*>(errBuf.get());
  CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));

  test::test_llpacket_addressing(p8.get(), err_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t err_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(err_h, 0u) << "slot()/flag_ptr() addressing wrong";
}

TEST_F(LlxPacketTest, HostPacketCount) {
  EXPECT_EQ(LlxPacketGeometry::packet_count(0), 0u);
  EXPECT_EQ(LlxPacketGeometry::packet_count(1), 1u);
  EXPECT_EQ(LlxPacketGeometry::packet_count(4), 1u);
  EXPECT_EQ(LlxPacketGeometry::packet_count(5), 2u);
}

TEST_F(LlxPacketTest, HostValidPayload) {
  EXPECT_EQ(LlxPacketGeometry::valid_payload(0, 5), 4u);
  EXPECT_EQ(LlxPacketGeometry::valid_payload(1, 5), 1u);
  EXPECT_EQ(LlxPacketGeometry::valid_payload(1, 4), 0u); // out of range
}

} // namespace comms::prims
