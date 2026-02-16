// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALL_TO_ALLV_DYNAMIC_IMPL_H_
#define CTRAN_ALL_TO_ALLV_DYNAMIC_IMPL_H_

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"

commResult_t ctranAllToAllvDynamicIbImpl(
    const void* const* sendbuffs,
    void* const* recvbuffs,
    size_t sendcountsLength,
    size_t maxSendcount,
    size_t maxRecvcount,
    commDataType_t datatype,
    OpElem::opType algoType,
    CtranComm* comm,
    std::unique_ptr<CtranMapperTimestamp> timestamp,
    KernelElem* elem,
    void* recvbuff = nullptr);

commResult_t setupKernelConfig(
    const size_t* sendcounts,
    size_t sendcountsLength,
    void* const* recvbuffs,
    size_t* actualRecvcounts,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config,
    KernelElem** elem);

commResult_t opIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup);

commResult_t setupGpeOp(
    const void* const* sendbuffs,
    void* const* recvbuffs,
    size_t sendcountsLength,
    size_t maxSendcount,
    size_t maxRecvcount,
    commDataType_t datatype,
    OpElem::opType opType,
    CtranComm* comm,
    uint64_t opCount,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup,
    KernelElem* elem,
    void* recvbuff = nullptr,
    bool combine = false);

template <typename PerfConfig = DefaultPerfCollConfig>
commResult_t peerPutNonContig(
    CtranComm* comm,
    const void* const* sendbuffs,
    std::vector<void*>& remoteRecvBuffs,
    size_t* sendCountsTmpbufCPU,
    size_t sendcountsLength,
    commDataType_t datatype,
    std::vector<void*>& tmpRegHdls,
    int nRanks,
    int myRank,
    std::unique_ptr<CtranMapperTimestamp> const& timestamp,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    std::vector<std::unique_ptr<CtranMapperRequest>>& ibPutReqs,
    std::vector<std::unique_ptr<CtranMapperRequest>>& ibRecvCtrlReqs,
    size_t maxRecvcount,
    size_t maxSendcount,
    bool combine,
    bool skipWaitRecvCtrl = false) {
  // Prepare basic info for nonContig send
  size_t* sendIndices = reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
      CtranAlgo::TmpbufType::SENDINDICES_TMPBUF_CPU));

  size_t* sendIndicesBlockLengthsTmpbufCPU =
      reinterpret_cast<size_t*>(comm->ctran_->algo->getTmpBuf(
          CtranAlgo::TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU));

  std::vector<size_t> sendIndicesBlockPos(nRanks);
  sendIndicesBlockPos[0] = 0;
  for (int i = 1; i < nRanks; i++) {
    sendIndicesBlockPos[i] =
        sendIndicesBlockPos[i - 1] + sendIndicesBlockLengthsTmpbufCPU[i - 1];
  }

  // Search handlers for the single contig sendbuff.
  void* sendMemHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      sendbuffs[0],
      maxSendcount * commTypeSize(datatype),
      &sendMemHdl,
      &localReg));
  if (localReg) {
    tmpRegHdls.push_back(sendMemHdl);
  }

  // Calculate the offset of each recvbuff, considering if it is 1st or 2nd
  // all2allv.
  std::vector<size_t> remoteRecvBuffsBytesOffset(sendcountsLength);
  remoteRecvBuffsBytesOffset[0] = 0;
  int numCountsPerRank = sendcountsLength / nRanks;
  for (int i = 1; i < sendcountsLength; i++) {
    if (combine && (i % numCountsPerRank == 0)) {
      remoteRecvBuffsBytesOffset[i] = 0;
    } else {
      remoteRecvBuffsBytesOffset[i] += remoteRecvBuffsBytesOffset[i - 1] +
          sendCountsTmpbufCPU[i - 1] * commTypeSize(datatype);
    }
  }

  size_t offsetRecvcounts = comm->ctran_->algo->getTmpBufOffset(
                                CtranAlgo::TmpbufType::RECVCOUNTS_TMPBUF) /
      sizeof(size_t);

  auto [sendCountsTmpbufGPU, sendcountsTmpbufRegHdl] =
      comm->ctran_->algo->getTmpBufInfo(
          CtranAlgo::TmpbufType::SENDCOUNTS_TMPBUF);
  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff
  // handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  while (!ibRecvCtrlReqs.empty()) {
    auto it = ibRecvCtrlReqs.begin();
    while (it != ibRecvCtrlReqs.end()) {
      auto& recvCtrlReq = *it;
      int peer = recvCtrlReq->peer;

      bool completed = false;
      if (skipWaitRecvCtrl) {
        completed = true;
      } else {
        FB_COMMCHECK(comm->ctran_->mapper->testRequest<PerfConfig>(
            recvCtrlReq.get(), &completed));
      }

      if (!completed) {
        it++;
        continue;
      }
      std::vector<CtranMapperPutMsg> puts;
      timestamp->recvCtrl.emplace_back(peer);

      auto [interNodeRemoteTmpbuff, interNodeRemoteTmpAccessKey] =
          comm->ctran_->algo->getInterNodeTmpBufInfo(peer);
      size_t* remoteTmpRecvCountsBufGPU =
          (size_t*)interNodeRemoteTmpbuff + offsetRecvcounts;

      // Allgather sendcounts
      // Skip sending sendcounts if it is second all2allv.
      // TODO: using hints instead of nonContigIndices to determine this.
      if (!combine) {
        puts.emplace_back(
            CtranMapperPutMsg{
                .sbuf = reinterpret_cast<size_t*>(sendCountsTmpbufGPU),
                .dbuf = &remoteTmpRecvCountsBufGPU[myRank * sendcountsLength],
                .len = sizeof(size_t) * sendcountsLength,
                .config =
                    CtranMapperConfig{
                        .memHdl_ = sendcountsTmpbufRegHdl,
                        .remoteAccessKey_ = interNodeRemoteTmpAccessKey,
                        .notify_ = false /*notify*/},
                .req = nullptr});
      }

      // Handle the corner case that all the metadata and data are not sent.
      // We then send an empty message to notify the peer.
      bool putNotifiedFlag = false;

      // iput for data
      int i = 0;
      while (i < sendIndicesBlockLengthsTmpbufCPU[peer]) {
        auto curIndex = sendIndices[sendIndicesBlockPos[peer] + i];
        auto totalSendcounts = sendCountsTmpbufCPU[curIndex];

        while (i + 1 < sendIndicesBlockLengthsTmpbufCPU[peer] &&
               sendIndices[sendIndicesBlockPos[peer] + i] + 1 ==
                   sendIndices[sendIndicesBlockPos[peer] + i + 1]) {
          totalSendcounts += sendCountsTmpbufCPU
              [sendIndices[sendIndicesBlockPos[peer] + i + 1]];
          i++;
          continue;
        }

        // Only notify the peer at the last message. If we notify every iput,
        // the peer may exist without receiving all the data.
        if (i == sendIndicesBlockLengthsTmpbufCPU[peer] - 1) {
          ibPutReqs.push_back(std::make_unique<CtranMapperRequest>());
          puts.emplace_back(
              CtranMapperPutMsg{
                  .sbuf = sendbuffs[curIndex],
                  .dbuf = (void*)(reinterpret_cast<uintptr_t>(
                                      remoteRecvBuffs[peer]) +
                                  remoteRecvBuffsBytesOffset[curIndex]),
                  .len = totalSendcounts * commTypeSize(datatype),
                  .config =
                      CtranMapperConfig{
                          .memHdl_ = sendMemHdl,
                          .remoteAccessKey_ = remoteAccessKeys[peer],
                          .notify_ = true /*notify*/},
                  .req = ibPutReqs.back().get()});
          putNotifiedFlag = true;
        } else {
          if (totalSendcounts == 0) {
            continue;
          }
          puts.emplace_back(
              CtranMapperPutMsg{
                  .sbuf = sendbuffs[curIndex],
                  .dbuf = (void*)(reinterpret_cast<uintptr_t>(
                                      remoteRecvBuffs[peer]) +
                                  remoteRecvBuffsBytesOffset[curIndex]),
                  .len = totalSendcounts * commTypeSize(datatype),
                  .config =
                      CtranMapperConfig{
                          .memHdl_ = sendMemHdl,
                          .remoteAccessKey_ = remoteAccessKeys[peer],
                          .notify_ = false /*notify*/},
                  .req = nullptr});
        }
        i++;
      }

      if (!putNotifiedFlag) {
        puts.emplace_back(
            CtranMapperPutMsg{
                .sbuf = reinterpret_cast<size_t*>(sendCountsTmpbufGPU),
                .dbuf = &remoteTmpRecvCountsBufGPU[myRank * sendcountsLength],
                .len = 0,
                .config =
                    CtranMapperConfig{
                        .memHdl_ = sendcountsTmpbufRegHdl,
                        .remoteAccessKey_ = interNodeRemoteTmpAccessKey,
                        .notify_ = true /*notify*/},
                .req = ibPutReqs.back().get()});
      }

      FB_COMMCHECK(
          comm->ctran_->mapper->iputBatch<PerfConfig>(std::move(puts), peer));
      timestamp->putIssued.emplace_back(peer);
      it = ibRecvCtrlReqs.erase(it);
    }
  }
  return commSuccess;
}

#endif
