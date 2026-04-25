// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware AllGather: when a regular ctranAllGather() is called during
// CUDA graph capture and this path is enabled, the collective is transparently
// converted to the persistent AllGatherP (window-based) algorithm.
//
// Flow:
//   1. ctranWinRegister() — register recvbuff as a window, exchange handles
//      with all peers. This is a collective CPU-side operation and is NOT
//      captured into the graph.
//   2. allGatherWinInit() — create persistent AGP state from window metadata.
//      Synchronous, no async handle exchange needed. Uses
//      StreamCaptureModeGuard to temporarily allow cudaHostAlloc
//      (blocked under cudaStreamCaptureModeGlobal used by PyTorch).
//   3. allGatherWinExec() — dry-run exec that IS captured into the graph.
//      CE copies (NVL intra-node) and GPE host-node callbacks (IB inter-node)
//      are recorded as graph nodes.
//   4. Register cleanup on comm's cudagraphDeferredCleanup.
//
// On graph replay, only the captured CE + host-node operations re-execute.
// The result is SM-free replay (no GPU kernels for NVL copies).
//
// Cleanup: Resources are registered for deferred cleanup at capture time
// (not at graph destruction). This ensures cleanup runs during comm
// destruction on the main thread, regardless of when or whether the graph
// is destroyed. Graph replay is guaranteed to finish before comm destroy.

#include <folly/ScopeGuard.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/cvars/nccl_cvars.h"

commResult_t ctranAllGatherCudagraphAware(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  const int nRanks = statex->nRanks();
  const size_t recvBytes = sendcount * commTypeSize(datatype) * nRanks;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "AllGather cudagraph-aware: converting to window-based AGP "
      "(sendcount={}, nRanks={}, recvBytes={})",
      sendcount,
      nRanks,
      recvBytes);

  // 1. Register recvbuff as a window and exchange handles with all peers.
  //    Collective and CPU-side — NOT captured into the graph.
  ctran::CtranWin* win = nullptr;
  FB_COMMCHECK(ctran::ctranWinRegister(recvbuff, recvBytes, comm, &win));

  // 2. Init persistent AGP from window metadata.
  //    Switch to relaxed capture mode so initResources() can cudaHostAlloc
  //    (blocked under cudaStreamCaptureModeGlobal used by PyTorch).
  //    Single-threaded-per-comm assumption (standard for NCCL).
  auto winGuard = folly::makeGuard([win]() { delete win; });

  CtranPersistentRequest* request = nullptr;
  {
    meta::comms::StreamCaptureModeGuard captureGuard{
        cudaStreamCaptureModeRelaxed};
    FB_COMMCHECK(ctran::allGatherWinInit(win, comm, stream, request));
  }

  // 3. Dry-run exec — CE copies and GPE host-node callbacks are captured.
  FB_COMMCHECK(ctran::allGatherWinExec(sendbuff, sendcount, datatype, request));

  // 4. Register cleanup on comm at capture time. Resources live for the
  //    comm's lifetime and are cleaned up during CtranComm::destroy() on
  //    the main thread. This avoids depending on retainUserObject callbacks
  //    which run on CUDA's internal thread where CUDA APIs are forbidden.
  winGuard.dismiss();
  comm->cudagraphDeferredCleanup.add([request, win]() {
    if (request) {
      ctran::allGatherWinDestroy(request);
      delete request;
    }
    if (win) {
      win->free(true /* skipBarrier */);
      delete win;
    }
  });

  return commSuccess;
}
