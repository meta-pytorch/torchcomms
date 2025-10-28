#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicPImpl.h"
#include "comms/ctran/algos/AllToAll/Types.h"

namespace ctran {
namespace alltoallp {
commResult_t prepareCudagraphAwareAllToAll(
    opFunc& opFunc,
    struct OpElem* op,
    PersistentObj& pObj) {
  pObj = std::make_unique<AlgoImpl>(op->comm_, op->stream);
  auto algoImplPtr = std::get<std::unique_ptr<AlgoImpl>>(pObj).get();
  if (!algoImplPtr) {
    return commSystemError;
  }

  FB_COMMCHECK(algoImplPtr->setPArgs(
      op->alltoall.recvbuff,
      op->alltoall.count * op->comm_->statex_->nRanks(),
      true /* skipCtrlMsg */,
      op->alltoall.datatype));

  // Exchange mem handles and record in pArgs. This will not be captured
  // by cudagraph.
  FB_COMMCHECK(algoImplPtr->init());

  // Replace gpe func by the persistent version (skip exchanging mem
  // handle); and OpGroup by the persistent op which has the remote
  // handles recorded.

  FB_COMMCHECK(algoImplPtr->updatePersistentFuncAndOp(opFunc, op));
  return commSuccess;
}

} // namespace alltoallp
namespace alltoallvdynamicp {
commResult_t prepareCudagraphAwareAllToAllvDynamic(
    opFunc& opFunc,
    struct OpElem* op,
    PersistentObj& pObj) {
  pObj = std::make_unique<AlgoImpl>(op->comm_, op->stream);
  auto algoImplPtr = std::get<std::unique_ptr<AlgoImpl>>(pObj).get();
  if (!algoImplPtr) {
    return commSystemError;
  }

  const int nRanks = op->comm_->statex_->nRanks();
  std::vector<void*> recvbuffs(nRanks);
  for (int i = 0; i < nRanks; i++) {
    recvbuffs[i] = op->alltoallv_dynamic.recvbuffs[i];
  }
  // FIXME: confirm if sendbuffs are also persistent, so we don't need to
  // search handle for sendbuffs every time
  algoImplPtr->pArgs = {
      .recvbuffs = recvbuffs,
      .maxSendCount = op->alltoallv_dynamic.maxSendcount,
      .maxRecvCount = op->alltoallv_dynamic.maxRecvcount,
      .datatype = op->alltoallv_dynamic.datatype,
  };

  // Exchange mem handles and record in pArgs. This will not be captured
  // by cudagraph.
  FB_COMMCHECK(algoImplPtr->init());

  // Replace gpe func by the persistent version (skip exchanging mem
  // handle); and OpGroup by the persistent op which has the remote
  // handles recorded.

  FB_COMMCHECK(algoImplPtr->updatePersistFuncAndOp(opFunc, op));
  return commSuccess;
}
} // namespace alltoallvdynamicp
} // namespace ctran
