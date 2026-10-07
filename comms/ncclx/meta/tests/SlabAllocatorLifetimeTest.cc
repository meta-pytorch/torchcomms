// Copyright (c) Meta Platforms, Inc. and affiliates.

// Runs with NCCL_MEM_USE_SLAB_ALLOCATOR=1, set by the BUCK target.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "nccl.h"

namespace {

constexpr int kIters = 10;
// Channel metadata lands on at least one cuMem-granularity slab (>= 2 MiB) per
// comm, so leaking it across kIters comms is far above this tolerance.
constexpr int64_t kLeakToleranceBytes = 8LL << 20;
constexpr size_t kCount = 1024;

int64_t freeDeviceBytes() {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  EXPECT_EQ(cudaMemGetInfo(&freeBytes, &totalBytes), cudaSuccess);
  return static_cast<int64_t>(freeBytes);
}

// One rank per GPU, all driven from this process.
void initComms(ncclComm_t* comms, int nRanks, ncclConfig_t* config) {
  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < nRanks; ++r) {
    ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
    ASSERT_EQ(
        ncclCommInitRankConfig(&comms[r], nRanks, id, r, config), ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
}

void allReduce(ncclComm_t* comms, int nRanks) {
  std::vector<float*> bufs(nRanks);
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < nRanks; ++r) {
    ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&bufs[r], kCount * sizeof(float)), cudaSuccess);
    ASSERT_EQ(
        ncclAllReduce(
            bufs[r], bufs[r], kCount, ncclFloat, ncclSum, comms[r], nullptr),
        ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  for (int r = 0; r < nRanks; ++r) {
    ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaFree(bufs[r]), cudaSuccess);
  }
}

void destroyComms(ncclComm_t* comms, int nRanks) {
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < nRanks; ++r) {
    ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
}

} // namespace

TEST(SlabAllocatorLifetimeTest, DestroyReleasesChannelMetadata) {
  ncclComm_t comm;
  // Let one-time process state settle before measuring.
  initComms(&comm, 1, nullptr);
  allReduce(&comm, 1);
  destroyComms(&comm, 1);

  const int64_t before = freeDeviceBytes();
  for (int i = 0; i < kIters; ++i) {
    initComms(&comm, 1, nullptr);
    allReduce(&comm, 1);
    destroyComms(&comm, 1);
  }
  const int64_t leaked = before - freeDeviceBytes();
  EXPECT_LT(leaked, kLeakToleranceBytes)
      << "leaked " << leaked << " bytes over " << kIters << " comms";
}

TEST(SlabAllocatorLifetimeTest, SharedChildOutlivesParent) {
  constexpr int kNumRanks = 2;
  int nDevs = 0;
  ASSERT_EQ(cudaGetDeviceCount(&nDevs), cudaSuccess);
  if (nDevs < kNumRanks) {
    GTEST_SKIP() << "needs " << kNumRanks << " GPUs";
  }

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;
  ncclComm_t parents[kNumRanks];
  ncclComm_t children[kNumRanks];
  initComms(parents, kNumRanks, &config);
  allReduce(parents, kNumRanks);
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < kNumRanks; ++r) {
    ASSERT_EQ(
        ncclCommSplit(parents[r], 0, r, &children[r], &config), ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  destroyComms(parents, kNumRanks);

  // A single-rank comm never reads devPeers; the 2-rank ring kernel
  // dereferences the ones shared with the destroyed parent.
  allReduce(children, kNumRanks);
  destroyComms(children, kNumRanks);
}
