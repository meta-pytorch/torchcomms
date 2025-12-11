// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranChecksum.h"
// FIXME [REBASE]: update the path once moved to fbcode/comms
#include "comms/ctran/gpe/tests/CtranChecksumUTKernels.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

class CtranChecksumTest : public ::testing::Test {
 public:
  int cudaDev{0};
  std::unique_ptr<TestCtranCommRAII> dummyCommRAII;
  CtranComm* dummyComm{nullptr};
  cudaStream_t stream_{0};
  CtranChecksumTest() = default;

 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev));

    ncclCvarInit();
    dummyCommRAII = createDummyCtranComm();
    dummyComm = dummyCommRAII->ctranComm.get();
  }

  void TearDown() override {}
};

TEST_F(CtranChecksumTest, AllGatherSampled) {
  size_t poolSize = 10;
  auto pool = std::make_unique<ChecksumPool>(poolSize);

  int opCount = 1;
  EXPECT_FALSE(
      ChecksumHandler<KernelConfig::KernelType::ALLGATHER>::isSampled(opCount));

  auto checksumSampleRateGuard =
      EnvRAII(NCCL_CTRAN_ALLGATHER_CHECKSUM_SAMPLE_RATE, 1);

  EXPECT_TRUE(
      ChecksumHandler<KernelConfig::KernelType::ALLGATHER>::isSampled(opCount));
}

TEST_F(CtranChecksumTest, AllGatherDummyLaunch) {
  size_t poolSize = 10;
  auto pool = std::make_unique<ChecksumPool>(poolSize);
  ChecksumItem* checksumItem = pool->pop();

  EXPECT_NE(checksumItem, nullptr);

  uint64_t dummyOpCount = 100;
  auto kernelConfig = KernelConfig(
      KernelConfig::KernelType::ALLGATHER, nullptr, "dummyAlgo", dummyOpCount);

  size_t count = 1024;
  auto datatype = commInt32;

  void* recvbuff = nullptr;
  CUDACHECK_TEST(cudaMallocHost(&recvbuff, sizeof(int)));

  uint32_t expectedSize =
      count * commTypeSize(datatype) * dummyComm->statex_->nRanks();

  ctranKernelSetAllGatherArgs(
      nullptr, recvbuff, datatype, count, nullptr, &kernelConfig.args);

  std::optional<ChecksumArgs> args =
      ChecksumHandler<KernelConfig::KernelType::ALLGATHER>::
          ctranFillChecksumArgs(kernelConfig, checksumItem, dummyComm);

  EXPECT_TRUE(args.has_value());
  EXPECT_EQ(args.value().buf, recvbuff);
  EXPECT_EQ(args.value().size, expectedSize);
  EXPECT_EQ(args.value().out, &checksumItem->checksum_);

  std::vector<void*> kernelArgs;
  kernelArgs.push_back(&args.value().buf);
  kernelArgs.push_back(&args.value().size);
  kernelArgs.push_back(&args.value().out);

  CUDACHECK_TEST(cudaLaunchKernel(
      reinterpret_cast<void*>(DummyChecksumKernel),
      1,
      1,
      kernelArgs.data(),
      0,
      stream_));
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));

  // Dummy kernel should have set the checksum to the size of the buffer
  EXPECT_EQ(checksumItem->checksum_, expectedSize);

  CUDACHECK_TEST(cudaFreeHost(recvbuff));
}
