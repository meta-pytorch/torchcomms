// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "alloc.h" // @manual
#include "comms/utils/memtrace/MemoryTrace.h"
#include "cudawrap.h" // @manual

using meta::comms::memtrace::MemoryTrace;

// The staged comm metadata must reach that comm's tracker on whichever path
// NCCL_CUMEM_ENABLE selects (one target per value), and the paired free must
// bring the comm's usage back down.
TEST(MemTraceAttributionTest, CallocAndFreeLandInStagedCommTracker) {
  ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  ASSERT_EQ(ncclSuccess, ncclCudaLibraryInit());

  CommLogData logMeta;
  logMeta.commHash = 0x1234abcd5678ef01;
  logMeta.commDesc = "memtrace_attribution_test";
  auto tracker = MemoryTrace::getOrCreate(logMeta.commHash);
  const int64_t usageBefore = tracker->getStats().currentUsage;

  constexpr size_t kCount = 1 << 20;
  int* ptr = nullptr;
  memLogMetaData = logMeta;
  ASSERT_EQ(ncclSuccess, ncclCudaCalloc(&ptr, kCount, nullptr));
  ASSERT_NE(nullptr, ptr);
  // CUMEM rounds the size up to the allocation granularity.
  EXPECT_GE(
      tracker->getStats().currentUsage - usageBefore,
      static_cast<int64_t>(kCount * sizeof(int)));

  memLogMetaData = logMeta;
  ASSERT_EQ(ncclSuccess, ncclCudaFree(ptr, nullptr));
  EXPECT_EQ(usageBefore, tracker->getStats().currentUsage);
}
