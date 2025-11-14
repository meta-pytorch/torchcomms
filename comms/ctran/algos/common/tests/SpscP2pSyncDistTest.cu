// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/common/SpscP2pSync.h"
#include "comms/ctran/algos/common/SpscP2pSyncDev.cuh"

using namespace ctran::algos;

__global__ void SpscP2pSyncTestKernel(
    int myLocalRank,
    int numIter,
    int count, // number of elements in data; set to 0 for sync-only perf test
    int* shmData, // count elements
    SpscP2pSync* shmSync,
    // received data from producer by consumer, total count * numIter elements
    // returned to host for test correctness check
    int* outData) {
  const auto isProducer = myLocalRank == 0;
  const auto isConsumer = myLocalRank == 1;
  int step = 1;
  for (auto x = 0; x < numIter; x++) {
    if (isProducer) {
      SpscP2pSyncDev::waitReady(shmSync);

      // producer puts data into shmData
      for (auto c = threadIdx.x; c < count; c += blockDim.x) {
        shmData[c] = x * count + c;
      }

      // notify consumer that data is ready (set flag to x+1)
      SpscP2pSyncDev::post(shmSync, step++);
    } else if (isConsumer) {
      SpscP2pSyncDev::waitPost(shmSync, step);

      // consumer copies data from shmData to outData
      auto* outDataIter = outData + x * count;
      for (auto c = threadIdx.x; c < count; c += blockDim.x) {
        outDataIter[c] = shmData[c];
      }
      SpscP2pSyncDev::complete(shmSync);
      step++;
    }
  }
}
