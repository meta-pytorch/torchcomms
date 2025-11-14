// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

/* Define a sync type used to synchronize between single producers and single
 * consumer. Each of the producers and consumer is a thread block. The producers
 * and consumer may sit in the same GPU or in two different GPUs via NVLink
 * access. If it is with different GPUs, the memory must be pre-mapped for
 * direct P2P access.  See device-side Synchronization APIs in SPSCTbSyncDev.cuh
 *
 * Note that, using atomic would be a faster way for producer consumer in the
 * same GPU.
 *
 * Code struture:
 * - struct definition in .h to be included by both host and device sides.
 * - Host side allocates the memory, and device side actually uses it. Device
 *   member functions defined in *Dev.cuh.*/
namespace ctran::algos {
struct alignas(16) SpscP2pSync {
  // constant values used in the sync
  enum Status { kUnset = -1 };

  int flag{Status::kUnset};
};

} // namespace ctran::algos
