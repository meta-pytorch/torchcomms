// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestXPlatUtils.h"

template <int Threads>
__global__ void checksumKernel(
    const uint8_t* __restrict__ in,
    const uint32_t size,
    uint32_t* __restrict__ out);

class CtranAlgoChecksumTest : public ::testing::Test {
 public:
  CtranAlgoChecksumTest() = default;

  template <int threads>
  void launch_checksum_kernel(uint32_t* input, int count, uint32_t* output) {
    std::vector<void*> checksumArgs;
    checksumArgs.push_back(reinterpret_cast<void*>(&input));
    checksumArgs.push_back(reinterpret_cast<void*>(&count));
    checksumArgs.push_back(reinterpret_cast<void*>(&output));
    CUDACHECK_TEST(cudaLaunchKernel(
        reinterpret_cast<void*>(checksumKernel<threads>),
        1,
        threads,
        checksumArgs.data(),
        0,
        stream_));
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
  }

 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }

 protected:
  int cudaDev_{0};
  cudaStream_t stream_{0};
};

TEST_F(CtranAlgoChecksumTest, ChecksumMatch) {
  uint32_t* a = nullptr;
  uint32_t* a_checksum = nullptr;
  uint32_t* b = nullptr;
  uint32_t* b_checksum = nullptr;

  int count = 1048576;

  CUDACHECK_TEST(cudaMallocHost(&a, sizeof(uint32_t) * count));
  CUDACHECK_TEST(cudaMallocHost(&b, sizeof(uint32_t) * count));
  CUDACHECK_TEST(cudaMallocHost(&a_checksum, sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMallocHost(&b_checksum, sizeof(uint32_t)));

  *a_checksum = 0;
  *b_checksum = 0;

  for (int i = 0; i < count; ++i) {
    *(a + i) = i * i;
    *(b + i) = i * i;
  }

  launch_checksum_kernel<1024>(a, count, a_checksum);
  launch_checksum_kernel<1024>(b, count, b_checksum);
  EXPECT_EQ(*a_checksum, *b_checksum);

  *b_checksum = 0;
  *b += 1;
  launch_checksum_kernel<1024>(b, count, b_checksum);
  EXPECT_NE(*a_checksum, *b_checksum);

  CUDACHECK_TEST(cudaFreeHost(a));
  CUDACHECK_TEST(cudaFreeHost(b));
  CUDACHECK_TEST(cudaFreeHost(a_checksum));
  CUDACHECK_TEST(cudaFreeHost(b_checksum));
}

TEST_F(CtranAlgoChecksumTest, ChecksumThreads) {
  uint32_t* a = nullptr;
  uint32_t* a_checksum_512 = nullptr;
  uint32_t* a_checksum_1024 = nullptr;

  int count = 1048576;

  CUDACHECK_TEST(cudaMallocHost(&a, sizeof(uint32_t) * count));
  CUDACHECK_TEST(cudaMallocHost(&a_checksum_512, sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMallocHost(&a_checksum_1024, sizeof(uint32_t)));

  *a_checksum_512 = 0;
  *a_checksum_1024 = 0;

  for (int i = 0; i < count; ++i) {
    *(a + i) = i * i;
  }

  launch_checksum_kernel<512>(a, count, a_checksum_512);
  launch_checksum_kernel<1024>(a, count, a_checksum_1024);
  EXPECT_EQ(*a_checksum_512, *a_checksum_1024);

  CUDACHECK_TEST(cudaFreeHost(a));
  CUDACHECK_TEST(cudaFreeHost(a_checksum_512));
  CUDACHECK_TEST(cudaFreeHost(a_checksum_1024));
}
