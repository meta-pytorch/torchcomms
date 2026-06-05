// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "comms/utils/commSpecs.h"

class CtranComm;

namespace ctran::allreduce::fused {

bool is_supported_fused_type(commDataType_t datatype);
bool is_nccl_tests_sync_comm(CtranComm* comm);
int compute_p_min(CtranComm* comm);
int get_num_block_cap();
int compute_num_blocks(size_t totalBytes, int cap);

void* compute_phase2_buf(
    void* recvbuff,
    int localRank,
    size_t segmentBytes,
    bool participatesInIB);

} // namespace ctran::allreduce::fused

#if defined(ENABLE_PIPES)

#include "comms/ctran/algos/AllReduce/AllReduceV2Types.h"

namespace ctran::allreduce::fused {

commResult_t fill_common_kern_args(
    common::CommonKernArgs& args,
    const void* sendbuff,
    void* recvbuff,
    void* phase2Buf,
    size_t count,
    size_t segmentElems,
    int nNodes,
    int pMin,
    int nLocalRanks,
    int localRank,
    int numBlocks,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm);

commResult_t submit_fused_kernel(
    CtranComm* comm,
    cudaStream_t stream,
    const char* kernelName,
    uint64_t opCount,
    int numBlocks,
    int numThreads,
    void* algoArgs,
    const void* kernelFnPtr);

} // namespace ctran::allreduce::fused

#endif // ENABLE_PIPES
