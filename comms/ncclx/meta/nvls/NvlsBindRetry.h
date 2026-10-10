// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>

#include "nccl.h"

struct ncclComm;

namespace ncclx::nvls {

// Test seam for the intra-node bootstrap and proxy calls on the NVLS retry
// path. Production code calls through gNvlsBootstrap, which defaults to the
// real functions; the retry unit tests replace the pointers per case. This
// keeps the tests independent of link style: --wrap only intercepts references
// linked into the test binary itself, which silently stops working when the
// library builds shared (dev).
struct NvlsBootstrapOps {
  ncclResult_t (*allGather)(void*, int*, int, int, void*, int);
  ncclResult_t (*barrier)(void*, int*, int, int, int);
  ncclResult_t (*broadcast)(void*, int*, int, int, int, void*, int);
  ncclResult_t (*getProxyFd)(ncclComm*, int, void*, int*);
};
extern NvlsBootstrapOps gNvlsBootstrap;

ncclResult_t collectiveBindResult(
    const ncclComm* comm,
    CUresult localResult,
    CUresult* collectiveResult);

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 32, 0)
ncclResult_t prepareBindRetry(
    ncclComm* comm,
    CUresult localResult,
    CUresult collectiveResult,
    int64_t bindAttempt,
    size_t ucsize,
    void** ucptr,
    CUmemGenericAllocationHandle* ucHandle,
    CUmemGenericAllocationHandle* mcHandle,
    int* allocMcHandle,
    bool* retried);
#else
// 2.32+ binds NVLS buffers into partitions of one multicast group, and the wait
// for every local rank happens in the cuMemMap of that group, so the retry of a
// transient Fabric Manager stall (CUDA_ERROR_SYSTEM_NOT_READY) moves there.
//
// Maps the group under the NVLS watchdog and agrees on the result across local
// ranks. On *retry the local map is undone, and the caller releases the group's
// VA range and handle, passes the cleanup result to finishNvlsTeamRetry, and
// rebuilds the group. Any other failure is reported and returned. *mapped
// records local map ownership, including on error, so the caller can clean up
// after an allgather failure or retry a failed unmap.
ncclResult_t multicastMapWithRetry(
    const ncclComm* comm,
    CUdeviceptr base,
    size_t size,
    CUmemGenericAllocationHandle mcHandle,
    int64_t attempt,
    bool* retry,
    int* mapped);

// Agree on teardown success before the barrier and configured backoff, so no
// rank rebuilds the group while a peer is exiting after a cleanup failure.
// cleanupOp names the failed operation (cuMemAddressFree/cuMemRelease) for the
// report; the caller tracks which step produced cleanupResult.
ncclResult_t finishNvlsTeamRetry(
    const ncclComm* comm,
    const char* cleanupOp,
    CUresult cleanupResult);
#endif

} // namespace ncclx::nvls
