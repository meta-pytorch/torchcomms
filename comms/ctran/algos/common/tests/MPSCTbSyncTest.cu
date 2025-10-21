// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/MPSCTbSync.h"
#include "comms/ctran/algos/common/MPSCTbSyncDev.cuh"

using namespace ctran::algos;
__global__ void MPSCTbSyncTestKernel(
    int numProducers,
    int numIter,
    int count, // number of elements in data per producer
    int* shmData, // producer * count elements
    MPSCTbSync<>* sync,
    // received data from data by consumer, total numProducers * count * numIter
    // elements; returned to host for test correctness check
    int* outData) {
  const auto producerId = blockIdx.x;
  const auto isConsumer = blockIdx.x == numProducers;

  for (auto x = 0; x < numIter; x++) {
    if (isConsumer) {
      // wait for producers to put data into shmData
      MPSCTbSyncDev::waitPost(sync);

      auto* outDataIter = outData + x * numProducers * count;
      copy(outDataIter, shmData, numProducers * count, 0, 1);

      // notify producer to continue
      MPSCTbSyncDev::complete(sync);
    } else {
      // wait consumer copy out previous data
      MPSCTbSyncDev::waitReady(sync, producerId);

      // each producer puts data into shmData
      auto* dataP = shmData + producerId * count;
      for (auto c = threadIdx.x; c < count; c += blockDim.x) {
        dataP[c] = x * count * numProducers + producerId * count + c;
      }

      // notify consumer to copy out
      MPSCTbSyncDev::post(sync, producerId);
    }
  }
}

__global__ void SPSCTbSyncTestKernel(
    int numIter,
    int count, // number of elements in data
    int* shmData, // count elements
    MPSCTbSync<1>* postSync,
    MPSCTbSync<1>* completeSync,
    // received data from producer by consumer, total count * numIter elements
    // returned to host for test correctness check
    int* outData) {
  const auto isProducer = blockIdx.x == 0;
  const auto isConsumer = blockIdx.x == 1;
  const auto tid = threadIdx.x;
  int completeStep = 0;
  int postStep = 0;
  for (auto x = 0; x < numIter; x++) {
    if (isProducer) {
      if (tid == 0) {
        MPSCTbSyncDev::waitStep(completeSync, tid, 0, completeStep);
        completeStep++;
      }
      __syncthreads();
      // producer puts data into shmData
      for (auto c = threadIdx.x; c < count; c += blockDim.x) {
        shmData[c] = x * count + c;
      }
      __syncthreads();

      // notify consumer that data is ready (set flag to x+1)
      if (tid == 0) {
        postStep++;
        __threadfence_system();
        MPSCTbSyncDev::postStep(postSync, tid, postStep);
      }
    } else if (isConsumer) {
      if (tid == 0) {
        MPSCTbSyncDev::waitStep(postSync, tid, 0, postStep + 1);
        postStep++;
      }
      __syncthreads();

      // consumer copies data from shmData to outData
      auto* outDataIter = outData + x * count;
      for (auto c = threadIdx.x; c < count; c += blockDim.x) {
        outDataIter[c] = shmData[c];
      }
      __syncthreads();
      if (tid == 0) {
        completeStep++;
        MPSCTbSyncDev::postStep(completeSync, tid, completeStep);
      }
    }
  }
}
