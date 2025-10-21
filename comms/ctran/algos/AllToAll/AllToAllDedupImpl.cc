// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllToAll/AllToAllDedupImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"

namespace {

inline commResult_t intraNodeSync(CtranComm*& comm) {
  std::vector<std::unique_ptr<CtranMapperRequest>> ibSyncSendReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibSyncRecvReqs;
  CtranMapperRequest* req = nullptr;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllToAllDedup.impl.fwdExchangeSync"));

  auto nRanks = comm->statex_->nRanks();
  auto rank = comm->statex_->rank();
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;
    if (comm->statex_->isSameNode(rank, peer)) {
      comm->ctran_->mapper->irecvCtrl(peer, &req);
      ibSyncRecvReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
      comm->ctran_->mapper->isendCtrl(peer, &req);
      ibSyncSendReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    }
  }
  while (!ibSyncRecvReqs.empty()) {
    FB_COMMCHECK(comm->ctran_->mapper->testSomeRequests(
        ibSyncRecvReqs, timestamp->putComplete));
  }
  while (!ibSyncSendReqs.empty()) {
    FB_COMMCHECK(comm->ctran_->mapper->testSomeRequests(
        ibSyncSendReqs, timestamp->putComplete));
  }
  return commSuccess;
}

inline commResult_t ctrlExchange(
    CtranComm*& comm,
    std::vector<std::unique_ptr<CtranMapperRequest>>& ibSendCtrlReqs,
    std::vector<std::unique_ptr<CtranMapperRequest>>& ibRecvCtrlReqs,
    std::vector<int>& ibSendPeers,
    std::vector<int>& ibRecvPeers,
    std::vector<std::unique_ptr<CtranMapperNotify>>& ibNotifyVec,
    std::vector<void*>& recvMemHdl,
    std::vector<void*>& recvBuffs,
    std::vector<void*>& remoteRecvBuffs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys) {
  ibSendCtrlReqs.reserve(ibRecvPeers.size());
  ibRecvCtrlReqs.reserve(ibSendPeers.size());
  ibNotifyVec.reserve(ibSendPeers.size());

  for (auto peer : ibRecvPeers) {
    int peerNode = comm->statex_->node(peer);
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->isendCtrl(
        recvBuffs[peerNode], recvMemHdl[peer], peer, &req));
    ibSendCtrlReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));

    // Initialize notify flag to receive from peer
    auto notify = std::make_unique<CtranMapperNotify>();
    FB_COMMCHECK(
        comm->ctran_->mapper->initNotify(peer, recvMemHdl[peer], notify.get()));
    ibNotifyVec.push_back(std::move(notify));
  }

  for (auto peer : ibSendPeers) {
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->irecvCtrl(
        &remoteRecvBuffs[peer], &remoteAccessKeys[peer], peer, &req));
    ibRecvCtrlReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  }
  return commSuccess;
}
} // namespace

commResult_t ctranAllToAllDedupExecImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;
  const auto statex = comm->statex_.get();
  const int myRank = statex->rank();
  const int nRanks = statex->nRanks();
  const int nNodes = statex->nNodes();
  const int localRank = statex->localRank();
  const int nLocalRanks = statex->nLocalRanks();
  auto sendCounts = op->alltoall_dedup.sendcounts;
  auto recvCounts = op->alltoall_dedup.recvcounts;
  auto sDispls = op->alltoall_dedup.sdispls;
  auto rDispls = op->alltoall_dedup.rdispls;
  auto sendbuff = op->alltoall_dedup.sendbuff;
  auto recvbuff = op->alltoall_dedup.recvbuff;
  auto datatype = op->alltoall_dedup.datatype;
  auto sendMemHdl = op->alltoall_dedup.sendHdl;
  auto recvMemHdl = op->alltoall_dedup.recvHdl;
  auto& remoteRecvBuffs = op->alltoall_dedup.remoteRecvBuffs;
  auto& remoteAccessKeys = op->alltoall_dedup.remoteAccessKeys;

  std::vector<void*> recvMemHdls(nRanks);
  std::vector<const void*> sendBuffs(nNodes);
  std::vector<void*> recvBuffs(nNodes);
  std::vector<int> ibRecvPeers, ibSendPeers;
  std::unordered_set<int> ibPeers;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibPutReqs, ibSyncRecvReqs,
      ibSyncSendReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibSendCtrlReqs,
      ibRecvCtrlReqs;
  std::vector<std::unique_ptr<CtranMapperNotify>> notifyVec;
  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAllToAllDedup.impl"));

  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  FB_COMMCHECK(intraNodeSync(comm));

  // Prepare buffers shifted with displacement.
  for (int peerNode = 0; peerNode < nNodes; peerNode++) {
    int peerRank = statex->localRankToRank(localRank, peerNode);
    if (sendCounts[peerNode]) {
      sendBuffs[peerNode] = static_cast<const char*>(sendbuff) +
          sDispls[peerNode] * commTypeSize(datatype);
      if (myRank != peerRank) {
        ibSendPeers.push_back(peerRank);
        ibPeers.insert(peerRank);
      }
    }
    if (recvCounts[peerNode]) {
      recvBuffs[peerNode] = static_cast<char*>(recvbuff) +
          rDispls[peerNode] * commTypeSize(datatype);
      if (myRank != peerRank) {
        ibRecvPeers.push_back(peerRank);
        ibPeers.insert(peerRank);
      }
    }
  }

  for (int i = 0; i < nRanks; i++) {
    recvMemHdls[i] = recvMemHdl;
  }

  FB_COMMCHECK(ctrlExchange(
      comm,
      ibSendCtrlReqs,
      ibRecvCtrlReqs,
      ibSendPeers,
      ibRecvPeers,
      notifyVec,
      recvMemHdls,
      recvBuffs,
      remoteRecvBuffs,
      remoteAccessKeys));

  // TODO: kick off puts immediately after ctrl msg done
  for (auto& req : ibRecvCtrlReqs) {
    comm->ctran_->mapper->waitRequest(req.get());
  }

  // issue network puts:
  // - Sender puts data for peers
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  ibPutReqs.reserve(ibSendPeers.size());
  for (auto peer : ibSendPeers) {
    int peerNode = peer / nLocalRanks;

    if (useProfiler) {
      timestamp->recvCtrl.emplace_back(CtranMapperTimestampPoint(peer));
    }
    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(comm->ctran_->mapper->iput(
        sendBuffs[peerNode],
        remoteRecvBuffs[peer],
        sendCounts[peerNode] * commTypeSize(datatype),
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl,
            .remoteAccessKey_ = remoteAccessKeys[peer],
            .notify_ = true /*notify*/},
        &req));
    ibPutReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    if (useProfiler) {
      timestamp->putIssued.emplace_back(CtranMapperTimestampPoint(peer));
    }
  }

  // Wait for all puts to complete
  while (!ibPutReqs.empty()) {
    FB_COMMCHECK(comm->ctran_->mapper->testSomeRequests(
        ibPutReqs, timestamp->putComplete, useProfiler /* recordTime */));
  }

  // start bcasts

  // start bcast for local node
  // TODO this should just be a copy, but leave for optimization
  op->alltoall_dedup.bcastElemMap[statex->node()]->post();
  // Wait for all receives (i.e., remote IB puts) to complete
  while (!notifyVec.empty()) {
    for (auto it = notifyVec.begin(); it != notifyVec.end();) {
      auto& notify = *it;
      bool completed = false;
      FB_COMMCHECK(comm->ctran_->mapper->checkNotify(notify.get(), &completed));
      if (completed) {
        op->alltoall_dedup.bcastElemMap[statex->node(notify->peer)]->post();
        it = notifyVec.erase(it);
      } else {
        it++;
      }
    }
  }

  for (auto& pair : op->alltoall_dedup.bcastElemMap) {
    pair.second->wait();
  }

  if (useProfiler) {
    comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
    comm->ctran_->mapper->reportProfiling();
  }

  return commSuccess;
}
