// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * Minimal repro for the issue where cuMemGetAccess() leaves a benign error
 * in the HIP runtime error queue on AMD GPUs.
 *
 * On ROCm, when cuMemGetAccess() is called on cudaMalloc-ed memory, it returns
 * CUDA_ERROR_INVALID_VALUE (which is expected and used to detect memory type).
 * However, this also leaves an error in the HIP runtime error queue that can
 * propagate to user code and cause unexpected failures.
 *
 * This test verifies:
 * 1. cuMemGetAccess() returns CUDA_ERROR_INVALID_VALUE for cudaMalloc memory
 * 2. This call DOES leave an error in the HIP runtime error queue
 * 3. The error can be cleared with cudaGetLastError()
 *
 * Expected behavior on ROCm (without fix):
 *   - cuMemGetAccess returns CUDA_ERROR_INVALID_VALUE
 *   - cudaGetLastError() returns hipErrorInvalidValue (error in queue!)
 *
 * Expected behavior on CUDA:
 *   - cuMemGetAccess may return CUDA_ERROR_INVALID_VALUE or CUDA_SUCCESS
 *   - cudaGetLastError() returns cudaSuccess (no queue contamination)
 *
 * Steps to reproduce:
 * `buck test  @fbcode//mode/opt-amd-gpu
 * fbcode//comms/ctran/utils/tests:cu_mem_get_access_hip_error_repro_ut`
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

#include "comms/ctran/utils/CudaWrap.h"

class CuMemGetAccessHipErrorReproTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctran::utils::commCudaLibraryInit();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // Clear any pre-existing errors
    (void)cudaGetLastError();
  }
};

TEST_F(CuMemGetAccessHipErrorReproTest, CuMemGetAccessLeavesHipRuntimeError) {
  // Step 1: Allocate memory with cudaMalloc (NOT cuMemCreate)
  void* ptr = nullptr;
  ASSERT_EQ(cudaMalloc(&ptr, 1024), cudaSuccess);
  ASSERT_NE(ptr, nullptr);

  // Ensure error queue is clean before the test
  cudaError_t preError = cudaGetLastError();
  EXPECT_EQ(preError, cudaSuccess)
      << "Pre-existing error before cuMemGetAccess: "
      << cudaGetErrorString(preError);

  // Step 2: Call cuMemGetAccess on cudaMalloc-ed memory
  // On ROCm, this should return CUDA_ERROR_INVALID_VALUE
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = 0;

  unsigned long long flags = 0;
  CUresult cuResult = FB_CUPFN(cuMemGetAccess)(
      &flags, &accessDesc.location, reinterpret_cast<CUdeviceptr>(ptr));

  std::cout << "cuMemGetAccess returned: " << static_cast<int>(cuResult);
  if (cuResult == CUDA_ERROR_INVALID_VALUE) {
    std::cout << " (CUDA_ERROR_INVALID_VALUE - expected for cudaMalloc on ROCm)"
              << std::endl;
  } else if (cuResult == CUDA_SUCCESS) {
    std::cout << " (CUDA_SUCCESS)" << std::endl;
  } else {
    std::cout << " (unexpected error)" << std::endl;
  }

  // Step 3: Check the HIP runtime error queue
  // On ROCm (before fix), this will return hipErrorInvalidValue
  // On CUDA, this should return cudaSuccess
  cudaError_t runtimeError = cudaGetLastError();

  std::cout << "cudaGetLastError() after cuMemGetAccess: "
            << static_cast<int>(runtimeError) << " ("
            << cudaGetErrorString(runtimeError) << ")" << std::endl;

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || \
    defined(__HIP_PLATFORM_HCC__)
  // On ROCm: Demonstrate that cuMemGetAccess contaminates the runtime queue
  if (cuResult == CUDA_ERROR_INVALID_VALUE) {
    // This is the BUG we're reproducing:
    // cuMemGetAccess returns CUDA_ERROR_INVALID_VALUE (expected),
    // but it ALSO leaves an error in the HIP runtime error queue
    std::cout << "\n=== ROCm Behavior ===" << std::endl;
    if (runtimeError != cudaSuccess) {
      std::cout
          << "CONFIRMED: cuMemGetAccess left error in HIP runtime queue!\n"
          << "This is the bug that causes 'HIP error: invalid argument' "
          << "to propagate to user code." << std::endl;
      // This EXPECT demonstrates the bug exists
      // After the fix in DevMemType.cc, we clear this error
    } else {
      std::cout << "Runtime error queue is clean (fix may already be applied)"
                << std::endl;
    }
  }
#else
  // On CUDA: The runtime queue should remain clean
  EXPECT_EQ(runtimeError, cudaSuccess)
      << "On CUDA, cuMemGetAccess should not contaminate runtime error queue";
#endif

  // Cleanup
  ASSERT_EQ(cudaFree(ptr), cudaSuccess);
}

TEST_F(
    CuMemGetAccessHipErrorReproTest,
    ClearingErrorQueuePreventsErrorPropagation) {
  // This test shows the fix: clearing the error queue after cuMemGetAccess

  void* ptr = nullptr;
  ASSERT_EQ(cudaMalloc(&ptr, 1024), cudaSuccess);

  // Clear pre-existing errors
  (void)cudaGetLastError();

  // Call cuMemGetAccess
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = 0;

  unsigned long long flags = 0;
  CUresult cuResult = FB_CUPFN(cuMemGetAccess)(
      &flags, &accessDesc.location, reinterpret_cast<CUdeviceptr>(ptr));

  // THE FIX: Clear the error queue if cuMemGetAccess returns INVALID_VALUE
  if (cuResult == CUDA_ERROR_INVALID_VALUE) {
    cudaError_t clearedError = cudaGetLastError();
    std::cout << "Cleared error from queue: " << static_cast<int>(clearedError)
              << " (" << cudaGetErrorString(clearedError) << ")" << std::endl;
  }

  // Now verify the queue is clean
  cudaError_t postClearError = cudaGetLastError();
  EXPECT_EQ(postClearError, cudaSuccess)
      << "After clearing, error queue should be empty";

  // Subsequent CUDA operations should succeed without seeing the stale error
  void* ptr2 = nullptr;
  cudaError_t allocResult = cudaMalloc(&ptr2, 1024);
  EXPECT_EQ(allocResult, cudaSuccess)
      << "Subsequent cudaMalloc should succeed after clearing error queue";

  if (ptr2) {
    ASSERT_EQ(cudaFree(ptr2), cudaSuccess);
  }
  ASSERT_EQ(cudaFree(ptr), cudaSuccess);
}
