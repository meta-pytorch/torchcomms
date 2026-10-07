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

struct RegisteredAllReduce;

/**
 * Allocate a request object for the fixed-size registered all-reduce.
 *
 * A communicator contributes one fixed input buffer and one separate fixed
 * output buffer per rank. Registration is byte-oriented and independent of the
 * datatype and reduction selected by later executions. The object returned here
 * owns no peer mappings until registeredAllReduceInit() succeeds.
 */
ncclResult_t registeredAllReducePrepare(
    ncclComm_t comm,
    RegisteredAllReduce** request);

/**
 * Collectively publish the caller's fixed input buffer and this request's
 * internal state buffer, then retain peer mappings for later stream-only
 * execution.
 *
 * inputBase and outputBase are fixed for the request lifetime. capacityBytes is
 * the valid byte extent for both buffers; execute rejects any supported payload
 * larger than that capacity. inputBase and outputBase must be distinct and
 * 16-byte aligned, and capacityBytes a positive multiple of 16. Requires two
 * or four local MI350/gfx950 ranks on one node with RCCLX direct all-pairs
 * P2P; unsupported topologies are rejected collectively with
 * ncclInvalidArgument before peer mappings are created.
 *
 * Init drains every stream on the device, then temporarily writes 16 bytes at
 * every 2 MiB of the input (and its last 16 bytes) and restores them before
 * returning: every rank reads its peers' patterns through its mappings and
 * fails registration on all ranks with ncclSystemError if one is stale (a
 * fresh import of a re-used address was observed to resolve to an earlier
 * allocation's pages). Register input buffers that stay allocated for the
 * process lifetime.
 */
ncclResult_t registeredAllReduceInit(
    RegisteredAllReduce* request,
    const void* inputBase,
    void* outputBase,
    size_t capacityBytes);

/**
 * Enqueue exactly one all-reduce kernel on stream.
 *
 * count is in elements of datatype. input and output must be the fixed base
 * pointers supplied at init. Supports BF16 SUM at any payload that is a
 * positive multiple of 16 bytes within the capacity; four-rank 524288- and
 * 1048576-byte payloads use dedicated two-stage kernels, everything else a
 * one-shot kernel. Unsupported combinations are rejected at execution rather
 * than encoded in the API name. All ranks must
 * execute the same payload sizes in the same order, including graph replay
 * counts. Each rank must order its executions and replays on one stream so the
 * next start handshake remains a valid prior-call scratch-reuse fence. The HIP
 * default stream is supported. Execution enqueues exactly one kernel and does
 * not synchronize the stream; completion and cross-call ordering remain the
 * caller's responsibility until collective finalization.
 *
 * norm is optional and requires four ranks. When set (1 MiB payloads viewed
 * as 64 x 8192 rows), output
 * receives the gated-residual-norm epilogue's normalized rows instead of the
 * plain sum; see ncclRegisteredAllReduceGatedResidualNorm. Plain and epilogue
 * executions may be interleaved on one request.
 */
ncclResult_t registeredAllReduceExecute(
    RegisteredAllReduce* request,
    const void* input,
    void* output,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    const ncclRegisteredAllReduceGatedResidualNorm* norm,
    hipStream_t stream);

/**
 * Collectively tear down a request and delete it.
 *
 * Any HIP graph or graph executable that captured
 * registeredAllReduceExecute must be destroyed before this call, and
 * callers must pass graphsTeardownComplete=true to acknowledge that
 * precondition. stream must be the stream used for the final eager execution or
 * graph replay; the HIP default stream is valid. The function synchronizes it
 * internally, then runs collective
 * quiescence and mapping-close acknowledgements before exported storage is
 * freed.
 */
ncclResult_t registeredAllReduceFinalize(
    RegisteredAllReduce* request,
    hipStream_t stream,
    bool graphsTeardownComplete);

/**
 * Public C-ABI adapter for registered all-reduce. The opaque handle returned in
 * request is owned by the caller until ncclRegisteredAllReduceFinalize().
 * Normal communicator destruction rejects live requests because their safe
 * teardown is collective; abort invalidates any remaining handles without
 * entering a collective cleanup path.
 */
ncclResult_t registeredAllReducePublicInit(
    const void* sendbuff,
    void* recvbuff,
    size_t capacityBytes,
    ncclComm_t comm,
    void** request);

ncclResult_t registeredAllReducePublicExec(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    const ncclRegisteredAllReduceGatedResidualNorm* norm,
    hipStream_t stream,
    void* request);

ncclResult_t registeredAllReducePublicFinalize(
    void* request,
    hipStream_t stream);

bool registeredAllReduceCommHasLiveRequests(ncclComm_t comm);
// Frees the communicator's pooled state regions and closes the peer imports
// of them; called by ncclCommDestroy once no request is live.
void registeredAllReduceReleaseComm(ncclComm_t comm);
void registeredAllReduceAbandonComm(ncclComm_t comm);

size_t registeredAllReduceLiveRequestsForTest();
size_t registeredAllReduceLivePeerMappingsForTest();
size_t registeredAllReduceLiveStateAllocationsForTest();
// Communicator-lifetime imports of peers' pooled state regions.
size_t registeredAllReducePooledStateImportsForTest();
void registeredAllReduceSetIpcOpenFailureRankForTest(int rank);
// Makes `reader`'s mapping probe see stale data in `owner`'s input from byte
// 2 MiB on; -1, -1 clears it.
void registeredAllReduceSetStaleMappingForTest(int reader, int owner);

} // namespace rcclx::relay
