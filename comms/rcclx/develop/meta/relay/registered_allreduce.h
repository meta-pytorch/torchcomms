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
 * Requests whose capacity reaches NCCL_REGISTERED_AR_RELAY_MIN_BYTES (two
 * ranks, default 2 MiB) or NCCL_REGISTERED_AR_TP4_RELAY_MIN_BYTES (four ranks,
 * default 1 MiB + 16 B) also get a relay route at registration when the
 * node's other GPUs are visible: payloads of at least that size, without norm,
 * relay part of their traffic through those GPUs (staging the ranks allocate
 * in their HBM; nothing runs there). It is bitwise identical to the direct
 * result (rank order at two ranks, the two-stage kernel at four) and can be
 * captured like every other execution.
 *
 * norm is optional. It serves 1 MiB payloads viewed as 64 x 8192 rows on four
 * ranks, and 1 to 64 rows of 4608 elements on two or four ranks (the post-norm
 * weight may then be null). When set, output
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
// Relay helper GPUs of a public request (0 when it has no relay route, -1 for
// an unknown handle), and relay-route launches so far in this process.
int registeredAllReduceRelayHelpersForTest(void* request);
uint64_t registeredAllReduceRelayLaunchesForTest();
// Sets every CTA's start and midpoint epochs (its sequence counters and the
// start and midpoint words peers write into this rank's state region) to
// `epoch`, as after that many plain calls; relay words are left as they are.
// Rank-local: call on every rank with the same value while no execution runs.
ncclResult_t registeredAllReduceSetCtaEpochsForTest(
    RegisteredAllReduce* request,
    uint32_t epoch);
// Sets one epilogue's per-row call counter and the words peers write into
// this rank's state region for it to `epoch`, as after that many of its calls
// on every row: the 64 x 8192 epilogue's (calls, start, midpoint) or, with
// `rowEpilogue`, the row epilogue's (rowCalls, rowStart, quarter). The other
// epilogue's words are left as they are. Rank-local, like
// registeredAllReduceSetCtaEpochsForTest.
ncclResult_t registeredAllReduceSetRowEpochsForTest(
    RegisteredAllReduce* request,
    uint32_t epoch,
    bool rowEpilogue);
// Starts a public request's relay sequence counters at `sequence` (see
// registeredRelaySetSequenceForTest).
ncclResult_t registeredAllReduceSetRelaySequenceForTest(
    void* request,
    uint32_t sequence);
// Makes `reader`'s mapping probe see stale data in `owner`'s input from byte
// 2 MiB on; -1, -1 clears it.
void registeredAllReduceSetStaleMappingForTest(int reader, int owner);

} // namespace rcclx::relay
