// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/AllToAllvDedup/ExecCommon.h"

namespace ctran::alltoallvdedup {

thread_local uint64_t thOpCount = -1;
thread_local int thMyRank = -1;

ProgressState::ProgressState(int nNodes, int nLocalRanks) {
  preSendTrans.resize(nNodes, StepState());
  sendTrans.resize(nNodes, StepState());
  recvTrans.resize(nNodes, StepState());
  postRecvTrans.resize(nNodes, StepState());
  intraFwd.resize(nLocalRanks, StepState());
  recvCopy.resize(nLocalRanks, StepState());

  numSendSteps.resize(nNodes, 0);
  numRecvSteps.resize(nNodes, 0);

  sendTransReqs.resize(nNodes);
  remChkRSyncReqs.resize(nNodes);

  recvTransNotifies.resize(nNodes, CtranMapperNotify());
  remChkSSyncReqs.resize(nNodes);

  lastTransAck.resize(nNodes, 0);
}

} // namespace ctran::alltoallvdedup
