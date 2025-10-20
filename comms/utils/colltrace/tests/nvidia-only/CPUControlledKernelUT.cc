// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <thread>

#include <folly/stop_watch.h>
#include <gtest/gtest.h>

#include "comms/utils/CudaRAII.h"
#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.cuh"
#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

namespace {
void waitKernelFlagValue(volatile KernelFlag* flag, KernelFlag expected) {
  folly::stop_watch<std::chrono::milliseconds> timer;
  auto timeout = std::chrono::milliseconds(5000); // 5 seconds timeout

  while (*flag != expected && timer.elapsed() < timeout) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (*flag != expected) {
    FAIL() << "Timeout waiting for flag to be set to " << expected;
  }
}
} // namespace

#define CUDACHECK_TEST(cmd)                  \
  do {                                       \
    cudaError_t e = cmd;                     \
    if (e != cudaSuccess) {                  \
      printf(                                \
          "Failed: Cuda error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

class CPUControlledKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a CUDA stream for testing
    stream_ = CudaStream();

    // Allocate host memory for the flag
    CUDACHECK_TEST(cudaMallocManaged(&sharedFlag_, sizeof(KernelFlag)));
    *sharedFlag_ = KERNEL_FLAG_SCHEDULED;
  }

  void TearDown() override {
    // Free allocated memory
    if (sharedFlag_) {
      CUDACHECK_TEST(cudaFree(const_cast<KernelFlag*>(sharedFlag_)));
      sharedFlag_ = nullptr;
    }
  }

  CudaStream stream_;
  volatile KernelFlag* sharedFlag_ = nullptr;
};

TEST_F(CPUControlledKernelTest, KernelSignalAndWait) {
  // Launch the kernel that will signal start and wait for CPU signal
  void* args[] = {&sharedFlag_};
  CudaEvent startEvent;
  CudaEvent endEvent;
  CUDACHECK_TEST(cudaEventRecord(startEvent.get(), stream_.get()));
  CUDACHECK_TEST(cudaLaunchKernel(
      (void*)waitCPUKernel, dim3(1), dim3(1), args, 0, stream_.get()));
  CUDACHECK_TEST(cudaEventRecord(endEvent.get(), stream_.get()));

  // Wait for the kernel to signal that it has started
  waitKernelFlagValue(sharedFlag_, KERNEL_FLAG_STARTED);
  EXPECT_EQ(cudaEventQuery(startEvent.get()), cudaSuccess);

  // Signal the kernel to terminate
  *sharedFlag_ = KERNEL_FLAG_TERMINATE;

  waitKernelFlagValue(sharedFlag_, KERNEL_FLAG_UNSET);
  EXPECT_EQ(cudaEventQuery(endEvent.get()), cudaSuccess);
}

TEST_F(CPUControlledKernelTest, MultipleKernelLaunches) {
  // Test launching the kernel multiple times in sequence

  void* args[] = {&sharedFlag_};
  CudaEvent startEvent;
  CudaEvent endEvent;

  for (int i = 0; i < 3; ++i) {
    CUDACHECK_TEST(cudaEventRecord(startEvent.get(), stream_.get()));
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)waitCPUKernel, dim3(1), dim3(1), args, 0, stream_.get()));
    CUDACHECK_TEST(cudaEventRecord(endEvent.get(), stream_.get()));

    // Wait for the kernel to signal that it has started
    waitKernelFlagValue(sharedFlag_, KERNEL_FLAG_STARTED);
    EXPECT_EQ(cudaEventQuery(startEvent.get()), cudaSuccess);

    // Signal the kernel to terminate
    *sharedFlag_ = KERNEL_FLAG_TERMINATE;

    waitKernelFlagValue(sharedFlag_, KERNEL_FLAG_UNSET);
    EXPECT_EQ(cudaEventQuery(endEvent.get()), cudaSuccess);
  }
}

TEST_F(CPUControlledKernelTest, KernelHighLevelClass) {
  // Launch the kernel that will signal start and wait for CPU signal
  CudaEvent startEvent;
  CudaEvent endEvent;
  CPUControlledKernel kernel{stream_.get()};

  CUDACHECK_TEST(cudaEventRecord(startEvent.get(), stream_.get()));
  kernel.launch();
  CUDACHECK_TEST(cudaEventRecord(endEvent.get(), stream_.get()));

  // Wait for the kernel to signal that it has started
  ASSERT_TRUE(kernel.waitKernelFlagValue(
      KERNEL_FLAG_STARTED, std::chrono::milliseconds(1000)));
  EXPECT_EQ(cudaEventQuery(startEvent.get()), cudaSuccess);

  // Signal the kernel to terminate
  ASSERT_TRUE(kernel.endKernel());

  EXPECT_EQ(kernel.getKernelFlagValue(), KERNEL_FLAG_UNSET);
  EXPECT_EQ(cudaEventQuery(endEvent.get()), cudaSuccess);
}

TEST_F(CPUControlledKernelTest, KernelMultipleClass) {
  // Launch the kernel that will signal start and wait for CPU signal
  CudaEvent startEvent;
  CudaEvent endEvent;
  CUDACHECK_TEST(cudaEventRecord(startEvent.get(), stream_.get()));
  CPUControlledKernel kernel1{stream_.get()};
  kernel1.launch();
  CPUControlledKernel kernel2{stream_.get()};
  kernel2.launch();
  CPUControlledKernel kernel3{stream_.get()};
  kernel3.launch();
  CPUControlledKernel kernel4{stream_.get()};
  kernel4.launch();
  CUDACHECK_TEST(cudaEventRecord(endEvent.get(), stream_.get()));

  kernel1.endKernel();
  kernel2.endKernel();
  kernel3.endKernel();
  kernel4.endKernel();

  EXPECT_EQ(cudaEventQuery(endEvent.get()), cudaSuccess);
  EXPECT_EQ(kernel1.getKernelFlagValue(), KERNEL_FLAG_UNSET);
  EXPECT_EQ(kernel2.getKernelFlagValue(), KERNEL_FLAG_UNSET);
  EXPECT_EQ(kernel3.getKernelFlagValue(), KERNEL_FLAG_UNSET);
  EXPECT_EQ(kernel4.getKernelFlagValue(), KERNEL_FLAG_UNSET);
}
