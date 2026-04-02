// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranUtUtils.h"
#include "comms/ctran/tests/CtranNcclTestUtils.h"

// Static helper instance for NCCL memory allocation
static ctran::CtranNcclTestHelpers ncclHelpers;

void* CtranBaseTest::prepareBuf(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments,
    size_t numSegments) {
  // Delegate to CtranNcclTestHelpers for all memory types
  return ncclHelpers.prepareBuf(bufSize, memType, segments, numSegments);
}

void CtranBaseTest::releaseBuf(
    void* buf,
    size_t bufSize,
    MemAllocType memType,
    size_t numSegments) {
  // Delegate to CtranNcclTestHelpers for all memory types
  ncclHelpers.releaseBuf(buf, bufSize, memType, numSegments);
}

void CtranBaseTest::allocDevArg(const size_t nbytes, void*& ptr) {
  CUDACHECK_ASSERT(cudaMalloc(&ptr, nbytes));
  // store argPtr to release at the end of test
  devArgs_.insert(ptr);
}

void CtranBaseTest::releaseDevArgs() {
  for (auto ptr : devArgs_) {
    CUDACHECK_TEST(cudaFree(ptr));
  }
  devArgs_.clear();
}

void CtranBaseTest::releaseDevArg(void* ptr) {
  cudaFree(ptr);
  devArgs_.erase(ptr);
}
