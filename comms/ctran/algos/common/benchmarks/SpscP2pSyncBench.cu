// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/common/SpscP2pSync.h"
#include "comms/ctran/algos/common/SpscP2pSyncDev.cuh"

using namespace ctran::algos;

//------------------------------------------------------------------------------
// Benchmark Kernel
//------------------------------------------------------------------------------

/**
 * Simplified benchmark kernel for SpscP2pSync synchronization.
 * This kernel focuses on sync-only performance (no data copy).
 *
 * @param myLocalRank: 0 for producer, 1 for consumer
 * @param numIter: number of sync iterations to perform
 * @param shmSync: pointer to shared memory sync structure (on consumer's
 * device)
 */
__global__ void
SpscP2pSyncBenchKernel(int myLocalRank, int numIter, SpscP2pSync* shmSync) {
  const auto isProducer = myLocalRank == 0;
  const auto isConsumer = myLocalRank == 1;
  int step = 1;

  for (auto x = 0; x < numIter; x++) {
    if (isProducer) {
      SpscP2pSyncDev::waitReady(shmSync);
      SpscP2pSyncDev::post(shmSync, step++);
    } else if (isConsumer) {
      SpscP2pSyncDev::waitPost(shmSync, step);
      SpscP2pSyncDev::complete(shmSync);
      step++;
    }
  }
}
