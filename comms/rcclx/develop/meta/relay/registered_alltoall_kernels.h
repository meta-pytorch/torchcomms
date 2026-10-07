/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace rcclx::relay {

constexpr int kRegisteredAllToAllActiveRanks = 4;
constexpr int kRegisteredAllToAllMaxHelpers = 4;
constexpr int kRegisteredAllToAllMaxRanks =
    kRegisteredAllToAllActiveRanks + kRegisteredAllToAllMaxHelpers;
constexpr int kRegisteredAllToAllThreads = 256;
constexpr int kRegisteredAllToAllMaxChunks = 128;

// Protocol words of one rank for one request, written by its peers. start[r]
// is the epoch rank r has started; done[r] the epoch at which r finished
// reading this rank's send buffer and staging. chunk[s][h][c] is the epoch at
// which source s staged chunk c of its rows for this rank on helper GPU h.
// probe holds the registration check's pattern; the kernel never touches it.
struct alignas(256) RegisteredAllToAllFlags {
  uint32_t start[kRegisteredAllToAllMaxRanks];
  uint32_t reserved0[64 - kRegisteredAllToAllMaxRanks];
  uint32_t done[kRegisteredAllToAllMaxRanks];
  uint32_t reserved1[64 - kRegisteredAllToAllMaxRanks];
  uint32_t calls;
  uint32_t ticket;
  uint32_t probe[4];
  uint32_t reserved2[58];
  uint32_t chunk[kRegisteredAllToAllActiveRanks][kRegisteredAllToAllMaxHelpers]
                [kRegisteredAllToAllMaxChunks];
};

// Row geometry of one (source, destination) pair, in bytes. Source row t of
// the pair (s, d) is at send[s] + d * sendPeerStride + t * sendRowStride and
// lands at recv[d] + s * recvPeerStride + t * recvRowStride.
struct RegisteredAllToAllGeometry {
  size_t rowBytes;
  size_t sendRowStride;
  size_t sendPeerStride;
  size_t recvRowStride;
  size_t recvPeerStride;
  int rows;
  // Rows [0, directRows) of every pair move directly; helper h relays rows
  // [directRows + h * relayRows, directRows + (h + 1) * relayRows).
  int directRows;
  int relayRows;
  int chunkRows;
  int helpers;
  int directCtasPerPeer;
  int relayCtasPerGroup;
};

struct RegisteredAllToAllArgs {
  const unsigned char* send[kRegisteredAllToAllActiveRanks];
  unsigned char* recv;
  // staging[s][h]: source s's staging on helper GPU h (allocated by s), one
  // relayRows x rowBytes slot per destination.
  unsigned char* staging[kRegisteredAllToAllActiveRanks]
                        [kRegisteredAllToAllMaxHelpers];
  RegisteredAllToAllFlags* activeFlags[kRegisteredAllToAllActiveRanks];
  RegisteredAllToAllGeometry geometry;
  int index;
};

// Bytes of one source's staging on one helper GPU.
size_t registeredAllToAllStagingBytes(
    const RegisteredAllToAllGeometry& geometry);

// Enqueues exactly one kernel and never synchronizes the stream.
hipError_t launchRegisteredAllToAllActive(
    const RegisteredAllToAllArgs& args,
    hipStream_t stream);

// One 16-byte read of the registration check, issued the way the exchange
// reads that memory: a plain vector load for data (send buffers, staging) or
// system-scope atomic loads for protocol words (flags).
struct RegisteredAllToAllProbe {
  const unsigned char* address;
  int atomic;
};

// Reads probes[i] into out[2i], out[2i + 1]. probes and out are device memory.
hipError_t launchRegisteredAllToAllProbe(
    const RegisteredAllToAllProbe* probes,
    int count,
    uint64_t* out,
    hipStream_t stream);

} // namespace rcclx::relay
