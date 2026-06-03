// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#if defined(ENABLE_PIPES)

#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/collectives/RingAllReduceLauncher.h"
#include "comms/pipes/collectives/RingUtils.h"

commResult_t ctranAllReducePipesFlatRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  return commInternalError;
}

#else

commResult_t ctranAllReducePipesFlatRing(
    const void*,
    void*,
    size_t,
    commDataType_t,
    commRedOp_t,
    CtranComm*,
    cudaStream_t,
    std::optional<std::chrono::milliseconds>) {
  return commInternalError;
}

#endif // ENABLE_PIPES
