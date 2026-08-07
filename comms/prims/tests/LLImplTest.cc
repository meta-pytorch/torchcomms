// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/prims/core/LlxPacket.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/tests/LLImplTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {

// Memcpy opts into the LL protocol: it provides packet-aware sendLL/recvLL over
// LlxPacket geometry, so the detection traits used by the IBGDA transport's LL
// dispatch report it as LL-capable.
static_assert(has_sendLL_v<Memcpy, LlxPacketGeometry>);
static_assert(has_recvLL_v<Memcpy, LlxPacketGeometry>);

namespace {

using LaunchFn = void (*)(const char*, char*, char*, std::size_t, uint32_t*);

// Drive one pack->unpack round-trip on device for geometry P and verify the
// payload comes back byte-identical.
template <typename P>
void roundTrip(LaunchFn launch, std::size_t nbytes) {
  const std::size_t wire = P::wire_bytes(nbytes);

  std::vector<char> h_src(nbytes);
  for (std::size_t i = 0; i < nbytes; ++i) {
    h_src[i] = static_cast<char>(i * 131u + 7u);
  }

  DeviceBuffer src(nbytes);
  DeviceBuffer staging(wire);
  DeviceBuffer dst(nbytes);
  DeviceBuffer errBuf(sizeof(uint32_t));
  auto* err_d = static_cast<uint32_t*>(errBuf.get());

  CUDACHECK_TEST(
      cudaMemcpy(src.get(), h_src.data(), nbytes, cudaMemcpyHostToDevice));
  // Poison staging so a missed flag/decode surfaces as a mismatch.
  CUDACHECK_TEST(cudaMemset(staging.get(), 0xEE, wire));
  CUDACHECK_TEST(cudaMemset(dst.get(), 0, nbytes));
  CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));

  launch(
      static_cast<const char*>(src.get()),
      static_cast<char*>(staging.get()),
      static_cast<char*>(dst.get()),
      nbytes,
      err_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t err_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(err_h, 0u) << "payload round-trip mismatch at nbytes=" << nbytes;

  std::vector<char> h_dst(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(h_dst.data(), dst.get(), nbytes, cudaMemcpyDeviceToHost));
  EXPECT_EQ(h_dst, h_src) << "decoded payload differs at nbytes=" << nbytes;
}

} // namespace

class LLImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

TEST_F(LLImplTest, LlRoundTrip) {
  // Exact multiples and partial final packet (kData = 4).
  for (std::size_t n :
       {std::size_t(4),
        std::size_t(7),
        std::size_t(64),
        std::size_t(1000),
        std::size_t(4096)}) {
    roundTrip<LlxPacketGeometry>(test::test_ll_pack_unpack, n);
  }
}

TEST_F(LLImplTest, FlagRoundTrip) {
  DeviceBuffer p8(LlxPacketGeometry::kPacketBytes);
  CUDACHECK_TEST(cudaMemset(p8.get(), 0, LlxPacketGeometry::kPacketBytes));

  DeviceBuffer errBuf(sizeof(uint32_t));
  auto* err_d = static_cast<uint32_t*>(errBuf.get());
  CUDACHECK_TEST(cudaMemset(err_d, 0, sizeof(uint32_t)));

  test::test_ll_flag_roundtrip(p8.get(), err_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t err_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&err_h, err_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(err_h, 0u) << "store_flag/load_flag/is_flag_set round-trip wrong";
}

} // namespace comms::prims
