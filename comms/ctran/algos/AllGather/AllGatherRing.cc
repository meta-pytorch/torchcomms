// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <deque>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/utils/cvars/nccl_cvars.h"

struct PutQElem {
  char* lAddr;
  char* rAddr;
  size_t size;
  void* hdl;
};

static const auto myAlgo = NCCL_ALLGATHER_ALGO::ctring;

static commResult_t impl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  size_t sendSize =
      op->allgather.sendcount * commTypeSize(op->allgather.datatype);
  CtranComm* comm = opGroup.front()->comm_;
  const auto& statex = comm->statex_;
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  void* memHdl;
  bool localMemReg;
  void* remoteRecvBuff;
  struct CtranMapperRemoteAccessKey remoteAccessKey;
  CtranMapper* mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger(allGatherAlgoName(myAlgo), op->opCount, comm);

  CtranMapperContext context(
      allGatherAlgoName(myAlgo), sendSize, sendSize * nRanks);
  mapper->setContext(std::move(context));

  if (nRanks == 1) {
    return commSuccess;
  }

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allGatherAlgoName(myAlgo)));

  std::unique_ptr<CtranMapperRequest> irecvReq;
  std::unique_ptr<CtranMapperRequest> isendReq;
  int left = (rank + nRanks - 1) % nRanks;
  int right = (rank + 1) % nRanks;

  size_t stepSize = sendSize <= NCCL_CTRAN_AG_RING_MIN_SPLIT_SIZE
      ? sendSize
      : sendSize / NCCL_CTRAN_AG_RING_NUM_SPLIT;
  size_t stepsPerBlock =
      std::max(1LU, (sendSize + stepSize - 1) / stepSize); // ceilDiv
  std::deque<PutQElem> putQ;
  std::deque<std::unique_ptr<CtranMapperRequest>> iputReqs;
  std::unique_ptr<CtranMapperNotify> notifyLeft = nullptr;
  uint64_t blockNum{0};
  uint64_t stepInBlock{0};

  FB_COMMCHECK(mapper->searchRegHandle(
      op->allgather.recvbuff, nRanks * sendSize, &memHdl, &localMemReg));

  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(
      mapper->irecvCtrl(&remoteRecvBuff, &remoteAccessKey, right, &req));
  irecvReq = std::unique_ptr<CtranMapperRequest>(req);

  FB_COMMCHECK(mapper->isendCtrl(op->allgather.recvbuff, memHdl, left, &req));
  isendReq = std::unique_ptr<CtranMapperRequest>(req);

  FB_COMMCHECK(mapper->waitRequest(irecvReq.get()));
  timestamp->recvCtrl.push_back(CtranMapperTimestampPoint(right));

  // Initialize notify flag to receive from left
  notifyLeft = std::make_unique<CtranMapperNotify>();
  FB_COMMCHECK(mapper->initNotify(left, memHdl, notifyLeft.get()));

  // Push addresses for first block onto deque
  for (int i = 0; i < stepsPerBlock; ++i) {
    size_t offset = rank * sendSize + i * stepSize;
    char* lAddr = (char*)op->allgather.recvbuff + offset;
    char* rAddr = (char*)remoteRecvBuff + offset;
    size_t size = std::min(stepSize, sendSize - i * stepSize);
    putQ.push_back({lAddr, rAddr, size, memHdl});
  }

  while (!putQ.empty() || !iputReqs.empty() || (blockNum < nRanks - 1)) {
    // Check for notifications from left and queue up corresponding sends
    while (true) {
      bool notifyRcvd{false};
      FB_COMMCHECK(mapper->checkNotify(notifyLeft.get(), &notifyRcvd));
      if (!notifyRcvd) {
        break;
      }
      // Don't queue send for final step
      if (blockNum < nRanks - 2) {
        int blockId = (rank - blockNum - 1 + nRanks) % nRanks;
        char* lAddr = (char*)op->allgather.recvbuff + blockId * sendSize +
            stepInBlock * stepSize;
        char* rAddr =
            (char*)remoteRecvBuff + blockId * sendSize + stepInBlock * stepSize;
        size_t size = std::min(stepSize, sendSize - stepInBlock * stepSize);
        putQ.push_back({lAddr, rAddr, size, memHdl});
      }
      if (stepInBlock == stepsPerBlock - 1) {
        ++blockNum;
      }
      stepInBlock = (stepInBlock + 1) % stepsPerBlock;
    }

    // Remove any completed puts from putQ, making room for new puts if possible
    while (!iputReqs.empty()) {
      bool done;
      FB_COMMCHECK(mapper->testRequest(iputReqs.front().get(), &done));
      if (done) {
        iputReqs.pop_front();
      } else {
        break;
      }
    }

    // Issue new puts if we're ready to
    while (!putQ.empty()) {
      CtranMapperRequest* req;
      const auto& e = putQ.front();
      // Always notify receiver and always get a cqe back
      FB_COMMCHECK(mapper->iput(
          e.lAddr,
          e.rAddr,
          e.size,
          right,
          CtranMapperConfig{
              .memHdl_ = e.hdl,
              .remoteAccessKey_ = remoteAccessKey,
              .notify_ = true},
          &req));
      iputReqs.emplace_back(req); // This will implicitly create unique_ptr
                                  // based on req and take ownership
      putQ.pop_front();
    }
  }

  FB_COMMCHECK(mapper->waitRequest(isendReq.get()));

  if (localMemReg) {
    FB_COMMCHECK(mapper->deregDynamic(memHdl));
  }

  mapper->timestamps.push_back(std::move(timestamp));
  mapper->reportProfiling();

  return commSuccess;
}

commResult_t ctranAllGatherRing(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      allGatherAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      -1,
      comm,
      stream);
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHER,
      stream,
      allGatherAlgoName(myAlgo),
      comm->ctran_->getOpCount());
  void* extraCopyBuff = nullptr;
  FB_COMMCHECK(prepareAllGatherArgs(
      opGroup,
      config,
      &extraCopyBuff,
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      comm,
      stream));
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      impl,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherCtranRing)));
  if (extraCopyBuff != nullptr) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff,
        extraCopyBuff,
        sendcount * commTypeSize(datatype) * comm->statex_->nRanks(),
        cudaMemcpyDefault,
        stream));
  }
  return commSuccess;
}
