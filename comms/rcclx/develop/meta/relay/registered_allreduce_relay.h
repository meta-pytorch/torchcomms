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

#include "nccl.h"

namespace rcclx::relay {

// Large-message route of a two-rank registered all-reduce: part of each call
// is relayed through the node's other GPUs, with staging rings the two ranks
// allocate in those GPUs' HBM and no process on them.
struct RegisteredRelay;

// Collective over a two-rank comm, run at registration after the peer inputs
// are mapped. Sets *relay to null (and succeeds) when the route does not apply
// on both ranks: capacity below NCCL_REGISTERED_AR_RELAY_MIN_BYTES or no helper
// GPU both ranks can reach. Fails on both ranks if the relay state or its peer
// mappings cannot be created.
ncclResult_t registeredRelaySetup(
    ncclComm_t comm,
    size_t capacityBytes,
    RegisteredRelay** relay);

// Whether a `bytes` payload takes the relay route.
bool registeredRelayServes(const RegisteredRelay* relay, size_t bytes);

// Enqueues one BF16 SUM relay kernel: `mine` and `peer` are the two ranks'
// registered inputs (peer as mapped here), `output` this rank's registered
// output. Safe to capture into a graph.
hipError_t registeredRelayLaunch(
    RegisteredRelay* relay,
    const void* mine,
    const void* peer,
    void* output,
    size_t bytes,
    hipStream_t stream);

// Returns the request's pooled flags and staging to its communicator. Purely
// local; the caller must have established that neither rank still runs a
// relay kernel. Nothing is freed: relay memory lives until
// registeredRelayReleaseComm.
void registeredRelayRelease(RegisteredRelay* relay);

// Frees the communicator's pooled relay flags and staging and closes the peer
// imports of them (ncclCommDestroy, no live request); abort abandons them.
void registeredRelayReleaseComm(ncclComm_t comm);
void registeredRelayAbandonComm(ncclComm_t comm);

int registeredRelayHelperCount(const RegisteredRelay* relay);
uint64_t registeredRelayLaunchesForTest();
// Sets every lane's push and reduce counters and every slot's full and freed
// flags to `sequence` (local, no kernel may be running on either rank), so the
// next relay call starts at sequence + 1.
ncclResult_t registeredRelaySetSequenceForTest(
    RegisteredRelay* relay,
    uint32_t sequence);

} // namespace rcclx::relay
