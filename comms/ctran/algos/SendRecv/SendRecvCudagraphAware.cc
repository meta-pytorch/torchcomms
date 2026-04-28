// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Cudagraph-aware SendRecv: when ctranGroupEndHook is called with
// algo=ctgraph during CUDA graph capture, this module pre-registers all
// send/recv buffers via globalRegisterWithPtr (local-persist pattern),
// then delegates to ctranGroupEndHookImpl(ctzcopy) for the actual GPE
// dispatch. The GPE host-node callbacks are captured into the graph and
// re-execute on replay with pre-registered buffers (searchRegHandle
// hits the fast path).
//
// Cleanup: registered buffers are deregistered via cudagraphDeferredCleanup
// during comm destruction. Uses folly::makeGuard for exception-safe cleanup
// on error paths.

#include <folly/ScopeGuard.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/utils/CudaRAII.h"

commResult_t ctranSendRecvCudagraphAware(
    std::deque<OpElem*>& opGroup,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  // Caller (ctranGroupEndHook) already verified active capture mode.
  const auto statex = comm->statex_.get();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "SendRecv cudagraph-aware: nOps {} commHash {:x} commDesc {} nRanks {} nLocalRanks {} nNodes {}",
      opGroup.size(),
      statex->commHash(),
      statex->commDesc(),
      statex->nRanks(),
      statex->nLocalRanks(),
      statex->nNodes());

  // Pre-register all send/recv buffers
  std::vector<std::pair<void*, size_t>> registeredBufs;
  {
    meta::comms::StreamCaptureModeGuard captureGuard{
        cudaStreamCaptureModeRelaxed};
    for (const auto* op : opGroup) {
      if (op->type == OpElem::opType::SEND) {
        const size_t nbytes = op->send.count * commTypeSize(op->send.datatype);
        FB_COMMCHECK(
            ctran::globalRegisterWithPtr(
                const_cast<void*>(op->send.sendbuff), nbytes, true, true));
        registeredBufs.emplace_back(
            const_cast<void*>(op->send.sendbuff), nbytes);
      } else if (op->type == OpElem::opType::RECV) {
        const size_t nbytes = op->recv.count * commTypeSize(op->recv.datatype);
        FB_COMMCHECK(
            ctran::globalRegisterWithPtr(
                op->recv.recvbuff, nbytes, true, true));
        registeredBufs.emplace_back(op->recv.recvbuff, nbytes);
      }
    }
  }

  auto cleanupGuard = folly::makeGuard([&registeredBufs]() {
    for (const auto& [buf, size] : registeredBufs) {
      ctran::globalDeregisterWithPtr(buf, size);
    }
  });

  // Dispatch to normal eager path with ctzcopy
  FB_COMMCHECK(
      ctranGroupEndHookImpl(opGroup, NCCL_SENDRECV_ALGO::ctzcopy, timeout));

  // Success — transfer cleanup to deferred
  cleanupGuard.dismiss();
  comm->cudagraphDeferredCleanup.add([registeredBufs]() {
    for (const auto& [buf, size] : registeredBufs) {
      ctran::globalDeregisterWithPtr(buf, size);
    }
  });

  return commSuccess;
}
