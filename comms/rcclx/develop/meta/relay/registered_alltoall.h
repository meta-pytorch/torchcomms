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

#include "nccl.h"

namespace rcclx::relay {

// Persistent registered all-to-all among the first four ranks of a
// single-node communicator, optionally relayed through the remaining ranks.
// Destinations pull their direct rows from the sources' registered send
// buffers; relayed rows are pulled into a helper's staging buffer by the
// helper and then by the destination, which uses otherwise idle links. See
// ncclRegisteredAllToAllInit for the full contract.
struct RegisteredAllToAll;

ncclResult_t registeredAllToAllPublicInit(
    const void* sendbuff,
    void* recvbuff,
    const ncclRegisteredAllToAllLayout* layout,
    const ncclRegisteredAllToAllConfig* config,
    ncclComm_t comm,
    void** request);

ncclResult_t registeredAllToAllPublicExec(
    const void* sendbuff,
    void* recvbuff,
    hipStream_t stream,
    void* request);

ncclResult_t registeredAllToAllPublicFinalize(
    void* request,
    hipStream_t stream);

// Communicator teardown hooks: destroy must be refused while collective
// requests are live, and then frees the communicator's pooled flags, relay
// staging and peer imports; abort abandons them without collective cleanup.
bool registeredAllToAllCommHasLiveRequests(ncclComm_t comm);
void registeredAllToAllReleaseComm(ncclComm_t comm);
void registeredAllToAllAbandonComm(ncclComm_t comm);

// Per-request imports of peers' send buffers (0 once every request is
// finalized) and communicator-lifetime imports of pooled memory.
size_t registeredAllToAllLivePeerMappingsForTest();
size_t registeredAllToAllPooledMappingsForTest();

// Test hook: makes comm rank `reader`'s mapping check see stale data in comm
// rank `owner`'s send buffer from byte 2 MiB on (the page a stale import was
// observed to break), as a real stale mapping would. -1 disables it.
void registeredAllToAllSetStaleMappingForTest(int reader, int owner);

} // namespace rcclx::relay
