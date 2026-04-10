// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/torchcomms/transport/StagedRdmaTransport.h"
#include "comms/utils/cvars/nccl_cvars.h"

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace torch::comms;

class StagedBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclCvarInit();
    ASSERT_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);
    ASSERT_TRUE(ibverbx::ibvInit());
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    auto maybeDevices = ibverbx::IbvDevice::ibvGetDeviceList();
    ASSERT_TRUE(maybeDevices.hasValue()) << maybeDevices.error().errStr;
    ASSERT_GT(maybeDevices->size(), 0);
    devices_ = std::move(*maybeDevices);

    auto maybePd = devices_[0].allocPd();
    ASSERT_TRUE(maybePd.hasValue()) << maybePd.error().errStr;
    pd_.emplace(std::move(*maybePd));
  }

  void TearDown() override {
    pd_.reset();
    devices_.clear();
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaDeviceReset();
  }

  std::vector<ibverbx::IbvDevice> devices_;
  std::optional<ibverbx::IbvPd> pd_;
};

TEST_F(StagedBufferTest, AllocateAndDestroy) {
  const size_t bufSize = 4 * 1024 * 1024; // 4 MB
  {
    StagedBuffer buf(bufSize, 0, *pd_);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), bufSize);
    EXPECT_EQ(buf.cudaDev(), 0);
    EXPECT_NE(buf.lkey(), 0u);
    EXPECT_NE(buf.rkey(), 0u);
  }
  // Destruction should not crash
}

TEST_F(StagedBufferTest, SmallBuffer) {
  // Small but page-aligned buffer (cudaMalloc + dmabuf require page alignment)
  const size_t bufSize = 64 * 1024; // 64 KB
  StagedBuffer buf(bufSize, 0, *pd_);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), bufSize);
}

TEST_F(StagedBufferTest, LargeBuffer) {
  // 64 MB — the default staging buffer size used in production
  const size_t bufSize = 64 * 1024 * 1024;
  StagedBuffer buf(bufSize, 0, *pd_);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), bufSize);
  EXPECT_NE(buf.lkey(), 0u);
  EXPECT_NE(buf.rkey(), 0u);
}

TEST_F(StagedBufferTest, MoveConstruct) {
  const size_t bufSize = 1024 * 1024;
  StagedBuffer buf1(bufSize, 0, *pd_);
  void* origData = buf1.data();
  uint32_t origLkey = buf1.lkey();

  StagedBuffer buf2(std::move(buf1));
  EXPECT_EQ(buf2.data(), origData);
  EXPECT_EQ(buf2.lkey(), origLkey);
  EXPECT_EQ(buf2.size(), bufSize);

  // Moved-from object should have null data
  EXPECT_EQ(buf1.data(), nullptr);
}

TEST_F(StagedBufferTest, MoveAssign) {
  const size_t bufSize = 1024 * 1024;
  StagedBuffer buf1(bufSize, 0, *pd_);
  StagedBuffer buf2(bufSize * 2, 0, *pd_);

  void* origData1 = buf1.data();
  uint32_t origLkey1 = buf1.lkey();

  // Move-assign buf1 into buf2 — buf2's original resources should be freed
  buf2 = std::move(buf1);
  EXPECT_EQ(buf2.data(), origData1);
  EXPECT_EQ(buf2.lkey(), origLkey1);
  EXPECT_EQ(buf2.size(), bufSize);
  EXPECT_EQ(buf1.data(), nullptr);
}

TEST_F(StagedBufferTest, GpuDataReadWrite) {
  // Verify the GPU pointer is usable for cudaMemcpy (the staging use case)
  const size_t bufSize = 4096;
  StagedBuffer buf(bufSize, 0, *pd_);

  // Write a pattern to the staging buffer via host
  std::vector<uint8_t> hostSrc(bufSize, 0xAB);
  ASSERT_EQ(
      cudaMemcpy(buf.data(), hostSrc.data(), bufSize, cudaMemcpyHostToDevice),
      cudaSuccess);

  // Read it back and verify
  std::vector<uint8_t> hostDst(bufSize, 0);
  ASSERT_EQ(
      cudaMemcpy(hostDst.data(), buf.data(), bufSize, cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(hostSrc, hostDst);
}

TEST_F(StagedBufferTest, GpuD2DCopy) {
  // Verify D2D copy works (the core staging operation)
  const size_t bufSize = 4096;
  StagedBuffer buf(bufSize, 0, *pd_);

  // Allocate a separate GPU buffer (simulating a model tensor)
  void* srcGpu = nullptr;
  ASSERT_EQ(cudaMalloc(&srcGpu, bufSize), cudaSuccess);

  // Fill source with pattern
  std::vector<uint8_t> hostSrc(bufSize, 0xCD);
  ASSERT_EQ(
      cudaMemcpy(srcGpu, hostSrc.data(), bufSize, cudaMemcpyHostToDevice),
      cudaSuccess);

  // D2D copy: srcGpu → staging buffer
  ASSERT_EQ(
      cudaMemcpy(buf.data(), srcGpu, bufSize, cudaMemcpyDeviceToDevice),
      cudaSuccess);

  // Read staging buffer back to host and verify
  std::vector<uint8_t> hostDst(bufSize, 0);
  ASSERT_EQ(
      cudaMemcpy(hostDst.data(), buf.data(), bufSize, cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(hostSrc, hostDst);

  ASSERT_EQ(cudaFree(srcGpu), cudaSuccess);
}
