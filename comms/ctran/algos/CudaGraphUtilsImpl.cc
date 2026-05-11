#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
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
} // namespace ctran
