// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "nccl.h"

struct ncclComm;

// CTRAN collective dispatch facade.
//
// These helpers concentrate the "redirect to CTRAN when enabled and applicable,
// otherwise fall through to the baseline path" decision that would otherwise be
// woven inline into every collective entry point in the forked upstream
// `collectives.cc`. Each forked collective keeps a single seam line that calls
// the matching helper, so the CTRAN integration surface lives here in
// NCCLX-only code rather than in the forked NCCL sources.
namespace ncclx {

// Each ctranTry* helper returns std::nullopt when the CTRAN fast path does not
// apply to this communicator/operation, in which case the caller runs the
// baseline path. Otherwise it returns the ncclResult_t of the CTRAN operation
// (which the caller should return directly).
std::optional<ncclResult_t> ctranTryAllGather(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTryAllReduce(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTryBroadcast(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTryReduceScatter(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTrySend(
    ncclComm* comm,
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTryRecv(
    ncclComm* comm,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTryAllToAll(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    cudaStream_t stream);

std::optional<ncclResult_t> ctranTryAllToAllv(
    ncclComm* comm,
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    cudaStream_t stream);

// Track a default (non-CTRAN) send/recv op so CTRAN can order it correctly at
// group end. No-op when CTRAN is not initialized on this communicator.
void ctranTrackDefaultSendRecv(ncclComm* comm);

// Device-initiated AllToAllv over the CTRAN pipes transport. Returns an error
// result when CTRAN pipes support is unavailable (or PRIMS is not built).
ncclResult_t ctranRunDeviceAllToAllv(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    const int64_t* sendcounts_d,
    const int64_t* recvcounts_d,
    ncclDataType_t datatype,
    cudaStream_t stream,
    int64_t sendcountsMultiplier,
    int64_t recvcountsMultiplier,
    const std::unordered_map<std::string, std::string>& hints);

} // namespace ncclx
