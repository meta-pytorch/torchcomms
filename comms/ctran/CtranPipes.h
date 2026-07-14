// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(ENABLE_PRIMS)
#include "comms/prims/trace/PipesTraceTypes.h"
#endif // defined(ENABLE_PRIMS)
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"

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

#if defined(ENABLE_PRIMS)
namespace ctran {

commResult_t ctranPreparePipesTrace(
    CtranComm* comm,
    comms::prims::PipesTraceHandle& trace);

} // namespace ctran
#endif // defined(ENABLE_PRIMS)
