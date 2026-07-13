// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <cuda_runtime.h>

#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"
#include "comms/utils/commSpecs.h"

class CtranComm;

namespace ctran::allreduce::fused {

bool is_supported_fused_type(commDataType_t datatype);
int compute_p_min(CtranComm* comm);
int get_num_block_cap();
int compute_num_blocks(size_t totalBytes, int cap);
int compute_num_blocks_ring(size_t segmentBytes, int cap);

void* compute_phase2_buf(
    void* recvbuff,
    int localRank,
    size_t segmentBytes,
    bool participatesInIB);

#if defined(ENABLE_PRIMS)

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
    CtranComm* comm,
    std::optional<std::vector<int>> ibPeers = std::nullopt);

commResult_t submit_fused_kernel(
    CtranComm* comm,
    cudaStream_t stream,
    const char* kernelName,
    uint64_t opCount,
    int numBlocks,
    int numThreads,
    void* algoArgs,
    const void* kernelFnPtr);

#endif // ENABLE_PRIMS

} // namespace ctran::allreduce::fused
