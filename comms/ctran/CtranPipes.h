// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

#if defined(CTRAN_HAS_PRIMS)
#include <cuda_runtime.h>

#include "comms/ctran/prims/PipesTraceTypes.h"
#endif

class CtranComm;
class CtranAlgo;

inline size_t ctranEffectiveP2pNvlSharedDevbufSize(int nLocalRanks) {
  uint64_t size = NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE;
  if (NCCL_CTRAN_HIER_AG_OVERLAP_ENABLE && nLocalRanks > 1 &&
      NCCL_CTRAN_HIER_AG_NVL_SHARED_DEVBUF_SIZE > 0) {
    size = std::max(size, NCCL_CTRAN_HIER_AG_NVL_SHARED_DEVBUF_SIZE);
  }
  return static_cast<size_t>(size);
}

// Create and configure MultiPeerTransport on the CtranComm.
// exchange() is deferred to ctranInitPipesResources().
commResult_t ctranInitializePipes(CtranComm* comm);

// Wire SharedResource staging buffers as external data buffers to
// MultiPeerTransport and exchange handles. Must be called after both
// CtranAlgo (SharedResource) and MultiPeerTransport have been created.
commResult_t ctranInitPipesResources(CtranAlgo* algo);

namespace ctran {

#if defined(CTRAN_HAS_PRIMS)
commResult_t ctranPreparePipesTrace(
    CtranComm* comm,
    ctran::prims::PipesTraceHandle& trace);
void ctranEnqueuePipesTraceDrain(CtranComm* comm, cudaStream_t stream);
#endif

} // namespace ctran
