// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/utils/DevAttribute.h"

/* Define a sync type used to synchronize between multiple producers and single
 * consumer. Each of the producers and consumer is a thread block. The producers
 * and consumer may sit in the same GPU or in two different GPUs via NVLink
 * access. If it is with different GPUs, the memory must be pre-mapped for
 * direct P2P access.
 * See device-side Synchronization APIs in MPSCTbSyncDev.cuh
 *
 * Code struture:
 * - struct definition in .h to be included by both host and device sides.
 * - Host side allocates the memory, and device side actually uses it. Device
 *   member functions defined in *Dev.cuh.*/
namespace ctran::algos {
template <int MaxNumProducers = CTRAN_ALGO_MAX_THREAD_BLOCKS>
struct alignas(16) MPSCTbSync {
  // constant values used in the sync
  enum Status { kUnset, kPosted };

  int numProducers;
  // only numProducers of flags is actually used
  int flags[MaxNumProducers];

  // Host side constructor, may be used together with cudaMemCpy to reset the
  // actual object in device memory
  MPSCTbSync(const int numProducers) : numProducers(numProducers) {
    reset();
  }

  DEVICE_ATTRIBUTE void reset() {
    for (unsigned int i = 0; i < numProducers; i++) {
      flags[i] = kUnset;
    }
  }
};
} // namespace ctran::algos
