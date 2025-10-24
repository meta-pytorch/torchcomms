// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>

#include <cuda.h>

#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#endif

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#include "comms/ctran/algos/AllReduce/AllReduceARGCommonDev.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/mapper/CtranMapper.h"

#include "comms/ctran/gpe/CtranGpe.h"

using namespace ctran::allreduce::arg;
static const std::unordered_map<
    std::tuple<commDataType_t, commDataType_t, commRedOp_t>,
    const void*,
    CtranTupleHash>
    typeToFunc = {
        ALLREDUCE_DEQUANT_FUNCMAP(
            commInt32,
            int,
            commInt32,
            int,
            ncclKernelAllReduceARG),
        ALLREDUCE_DEQUANT_FUNCMAP(
            commUint64,
            uint64_t,
            commUint64,
            uint64_t,
            ncclKernelAllReduceARG),
        ALLREDUCE_DEQUANT_FUNCMAP(
            commFloat32,
            float,
            commFloat32,
            float,
            ncclKernelAllReduceARG),
        ALLREDUCE_DEQUANT_FUNCMAP(
            commFloat64,
            double,
            commFloat64,
            double,
            ncclKernelAllReduceARG),
#if defined(__CUDA_BF16_TYPES_EXIST__)
        ALLREDUCE_DEQUANT_FUNCMAP(
            commBfloat16,
            __nv_bfloat16,
            commBfloat16,
            __nv_bfloat16,
            ncclKernelAllReduceARG),
        ALLREDUCE_DEQUANT_FUNCMAP(
            commBfloat16,
            __nv_bfloat16,
            commFloat32,
            float,
            ncclKernelAllReduceARG),
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
        ALLREDUCE_DEQUANT_FUNCMAP(
            commFloat8e5m2,
            __nv_fp8_e5m2,
            commFloat8e5m2,
            __nv_fp8_e5m2,
            ncclKernelAllReduceARG),
        ALLREDUCE_DEQUANT_FUNCMAP(
            commFloat8e4m3,
            __nv_fp8_e4m3,
            commFloat8e4m3,
            __nv_fp8_e4m3,
            ncclKernelAllReduceARG),
#endif
};

inline commResult_t searchRegHandle(
    CtranComm* comm,
    const void* buff,
    size_t bytes,
    void*& hdl,
    std::vector<void*>& tmpRegHdls) {
  bool localReg = false;
  FB_COMMCHECK(
      comm->ctran_->mapper->searchRegHandle(buff, bytes, &hdl, &localReg));
  if (localReg) {
    tmpRegHdls.push_back(hdl);
  }
  return commSuccess;
}

static inline commResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    const void* func,
    commDataType_t datatype,
    CtranComm* comm,
    KernelConfig& config) {
  // Allow user to customize thread block size if specified
  FB_CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
      (int*)&config.numBlocks,
      (int*)&config.numThreads,
      func,
      0 /* dynamicSMemSize */,
      0 /* blockSizeLimit */));
  if (config.numBlocks > NCCL_CTRAN_ALLREDUCE_ARG_MAX_NUM_THREAD_BLOCKS) {
    config.numBlocks = NCCL_CTRAN_ALLREDUCE_ARG_MAX_NUM_THREAD_BLOCKS;
  }
  if (config.numThreads > NCCL_CTRAN_ALLREDUCE_ARG_THREAD_BLOCK_SIZE) {
    config.numThreads = NCCL_CTRAN_ALLREDUCE_ARG_THREAD_BLOCK_SIZE;
  }

  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.args.collective.allreduce.sendbuff = sendbuff;
  config.args.collective.allreduce.recvbuff = recvbuff;
  config.args.collective.allreduce.datatype = datatype;
  config.args.collective.allreduce.count = count;
  config.args.collective.allreduce.tmpbuffSize =
      NCCL_CTRAN_INTERNODE_TMPBUF_SIZE;

  KernelElem* elem = nullptr;
  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
  config.args.collective.allreduce.kernelElems[kIntraAllToAll] = elem;

  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
  config.args.collective.allreduce.kernelElems[kLocalReduce] = elem;

  FB_COMMCHECK(comm->ctran_->gpe->allocKernelElems(1, config.numBlocks, &elem));
  config.args.collective.allreduce.kernelElems[kIntraAllGather] = elem;

  return commSuccess;
}

static inline commResult_t setupGpeOp(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    uint64_t opCount,
    void* sendMemHdl,
    void* recvMemHdl,
    std::vector<void*>& remoteRecvBuffs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    KernelConfig& config,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
      new OpElem(OpElem::opType::ALLREDUCE, stream, comm, opCount));
  op->allreduce.sendbuff = sendbuff;
  op->allreduce.recvbuff = recvbuff;
  op->allreduce.count = count;
  op->allreduce.datatype = datatype;
  op->allreduce.tmpbuffSize = config.args.collective.allreduce.tmpbuffSize;
  op->allreduce.kElemStepMap[kIntraAllToAll] =
      config.args.collective.allreduce.kernelElems[kIntraAllToAll];
  op->allreduce.kElemStepMap[kLocalReduce] =
      config.args.collective.allreduce.kernelElems[kLocalReduce];
  op->allreduce.kElemStepMap[kIntraAllGather] =
      config.args.collective.allreduce.kernelElems[kIntraAllGather];
  op->allreduce.sendHdl = sendMemHdl;
  op->allreduce.recvHdl = recvMemHdl;
  for (int i = 0; i < comm->statex_->nRanks(); i++) {
    op->allreduce.remoteRecvBuffs[i] = remoteRecvBuffs[i];
    op->allreduce.remoteAccessKeys[i].backend = remoteAccessKeys[i].backend;
    op->allreduce.remoteAccessKeys[i].ibKey = remoteAccessKeys[i].ibKey;
  }

  config.args.collective.allreduce.tmpbuff =
      comm->ctran_->algo->getTmpBuf(CtranAlgo::TmpbufType::INTERNODE_TMPBUF);

  for (int i = 0; i < comm->statex_->nLocalRanks(); i++) {
    if (i == comm->statex_->localRank()) {
      continue;
    }
    int peerGlobalRank = comm->statex_->localRankToRank(i);
    auto [remoteTmpRecvBuff, remoteTmpRecvAccessKey] =
        comm->ctran_->algo->getRemoteTmpBufInfo(peerGlobalRank);
    config.args.collective.allreduce.intraNodeRemoteTmpRecvBuffs[i] =
        remoteTmpRecvBuff;
    config.args.collective.allreduce.intraNodeRemoteRecvBuffs[i] =
        remoteRecvBuffs[peerGlobalRank];
  }
  opGroup.push_back(std::move(op));

  return commSuccess;
}

inline commResult_t alltoallOpImpl(
    OpElem* op,
    AllReduceARGContext& algoContext,
    void* sendMemHdl,
    const std::vector<int>& remoteIBPeers,
    std::unique_ptr<CtranMapperTimestamp>& timestamp,
    bool useProfiler) {
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  const void* sendbuff = op->allreduce.sendbuff;
  const auto datatype = op->allreduce.datatype;
  const auto myRank = statex->rank();
  const auto nRanks = statex->nRanks();
  const size_t typeSize = commTypeSize(datatype);

  KernelElem* elem = op->allreduce.kElemStepMap.at(kIntraAllToAll);
  // ping kernel to start intra node all to all
  elem->post();

  // step 2: inter node scatter
  CtranMapperRequest* req = nullptr;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibPutReqs(nRanks);
  // issue network puts:
  // - Sender puts data for peers, whenever received the remote recvbuff
  // handle
  // - Exit until all peers' put have been issued (putPeers becomes empty)
  for (const auto peer : remoteIBPeers) {
    auto [interNodeRemoteTmpbuff, interNodeRemoteTmpAccessKey] =
        comm->ctran_->algo->getRemoteTmpBufInfo(peer);

    if (useProfiler) {
      timestamp->recvCtrl.emplace_back(peer);
    }
    FB_COMMCHECK(comm->ctran_->mapper->iput(
        reinterpret_cast<const char*>(sendbuff) +
            getUserbuffOffset(algoContext, peer) * typeSize,
        reinterpret_cast<char*>(interNodeRemoteTmpbuff) +
            getTmpbuffOffset(algoContext, myRank) * typeSize,
        algoContext.stepCount * typeSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = sendMemHdl,
            .remoteAccessKey_ = interNodeRemoteTmpAccessKey,
            .notify_ = true /*notify*/},
        &req));
    ibPutReqs[peer] = std::unique_ptr<CtranMapperRequest>(req);
    if (useProfiler) {
      timestamp->putIssued.emplace_back(peer);
    }
  }

  // Wait for all puts to complete
  for (const auto& peer : remoteIBPeers) {
    const auto& ibPutReq = ibPutReqs[peer];
    // Process each request
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(ibPutReq.get()));
    CtranMapperNotify notify;
    FB_COMMCHECK(comm->ctran_->mapper->initNotify(peer, sendMemHdl, &notify));
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(&notify));
  }
  elem->wait();
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "finish the alltoall: totalStepCount {}",
      algoContext.totalStepCount);
  return commSuccess;
}

static inline commResult_t reduceOpImpl(
    OpElem* op,
    AllReduceARGContext& algoContext) {
  // step 3: local reduce
  KernelElem* elem = op->allreduce.kElemStepMap.at(kLocalReduce);
  elem->post();
  elem->wait();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "finish reduce: totalStepCount {}, stepCount {}",
      algoContext.totalStepCount,
      algoContext.stepCount);

  return commSuccess;
}

inline commResult_t allgatherOpImpl(
    OpElem* op,
    AllReduceARGContext& algoContext,
    std::vector<void*>& remoteRecvBuffs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys,
    void* sendMemHdl,
    void* recvMemHdl,
    std::vector<int>& remoteIBPeers,
    std::unique_ptr<CtranMapperTimestamp>& timestamp,
    bool useProfiler) {
  CtranComm* comm = op->comm_;
  const auto statex = comm->statex_.get();
  void* recvbuff = op->allreduce.recvbuff;
  const auto myRank = statex->rank();
  const auto nRanks = statex->nRanks();
  const auto typeSize = commTypeSize(op->allreduce.datatype);
  // step 4 intra-node all gather
  KernelElem* elem = op->allreduce.kElemStepMap.at(kIntraAllGather);
  // intra node all gather
  elem->post();

  CtranMapperRequest* req = nullptr;
  std::vector<std::unique_ptr<CtranMapperRequest>> ibPutReqs(nRanks);

  // step 4: inter node All Gather
  for (const auto peer : remoteIBPeers) {
    if (useProfiler) {
      timestamp->recvCtrl.emplace_back(peer);
    }
    FB_COMMCHECK(comm->ctran_->mapper->iput(
        static_cast<const char*>(recvbuff) +
            getUserbuffOffset(algoContext, myRank) * typeSize,
        reinterpret_cast<char*>(remoteRecvBuffs[peer]) +
            getUserbuffOffset(algoContext, myRank) * typeSize,
        algoContext.stepCount * typeSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = recvMemHdl,
            .remoteAccessKey_ = remoteAccessKeys[peer],
            .notify_ = true /*notify*/},
        &req));
    ibPutReqs[peer] = std::unique_ptr<CtranMapperRequest>(req);
    if (useProfiler) {
      timestamp->putIssued.emplace_back(peer);
    }
  }

  // Wait for all puts to complete
  for (const auto& peer : remoteIBPeers) {
    const auto& ibPutReq = ibPutReqs[peer];
    // Process each request
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(ibPutReq.get()));
    CtranMapperNotify notify;
    FB_COMMCHECK(comm->ctran_->mapper->initNotify(peer, sendMemHdl, &notify));
    FB_COMMCHECK(comm->ctran_->mapper->waitNotify(&notify));
  }
  elem->wait();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "finish the allgather: stepCount {}, displOffset {}",
      algoContext.stepCount,
      algoContext.displOffset);
  return commSuccess;
}

commResult_t exchangeUserbuff(
    CtranComm* comm,
    const void* sendbuff,
    void*& sendMemHdl,
    void* recvbuff,
    void*& recvMemHdl,
    std::vector<void*>& tmpRegHdls,
    size_t count,
    size_t typeSize,
    std::vector<void*>& remoteRecvBuffs,
    std::vector<struct CtranMapperRemoteAccessKey>& remoteAccessKeys) {
  const auto statex = comm->statex_.get();
  const auto myRank = statex->rank();
  const auto nRanks = statex->nRanks();
  std::unordered_set<int> ibPeers;

  CtranMapperEpochRAII epochRAII(comm->ctran_->mapper.get());
  // Search for the handle only when there are RecvPeers to avoid attempting
  // to search/register with a buffer size of 0.
  FB_COMMCHECK(searchRegHandle(
      comm, sendbuff, count * typeSize, sendMemHdl, tmpRegHdls));
  FB_COMMCHECK(searchRegHandle(
      comm, recvbuff, count * typeSize, recvMemHdl, tmpRegHdls));

  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (statex->isSameNode(peer, myRank)) {
      continue;
    }
    ibPeers.insert(peer);
  }

  // pre-connect all peers
  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    FB_COMMCHECK(comm->ctran_->mapper->preConnect(ibPeers));
  }

  remoteRecvBuffs.resize(nRanks);
  remoteAccessKeys.resize(nRanks);
  // exchange the recvbuff with inter and intra node peers
  FB_COMMCHECK(comm->ctran_->mapper->allGatherCtrl(
      recvbuff, recvMemHdl, remoteRecvBuffs, remoteAccessKeys));

  return commSuccess;
}

static const auto myAlgo = NCCL_ALLREDUCE_ALGO::ctarg;
static commResult_t opIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;
  const auto statex = comm->statex_.get();
  const auto myRank = statex->rank();
  const auto nRanks = statex->nRanks();
  const auto datatype = op->allreduce.datatype;
  const auto typeSize = commTypeSize(datatype);
  const void* sendbuff = op->allreduce.sendbuff;
  void* recvbuff = op->allreduce.recvbuff;
  const size_t count = op->allreduce.count;
  void* sendMemHdl = op->allreduce.sendHdl;
  void* recvMemHdl = op->allreduce.recvHdl;
  auto& remoteRecvBuffs = op->allreduce.remoteRecvBuffs;
  auto& remoteAccessKeys = op->allreduce.remoteAccessKeys;

  AllReduceARGContext allreduceContext = {
      .localRank = statex->localRank(),
      .nLocalRanks = statex->nLocalRanks(),
      .rank = statex->rank(),
      .nRanks = statex->nRanks(),
      .tmpbuffSize = op->allreduce.tmpbuffSize,
      .typeSize = typeSize,
      .count = count,
  };
  prepareContext(allreduceContext);

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "Prepare algo {}: sendbuff {}, recvbuff {}, count {}, totalStepCount {}, stepCount {}, nSteps {}, buffOffset {}, tmpbuffOffset {}",
      allReduceAlgoName(myAlgo),
      sendbuff,
      recvbuff,
      count,
      allreduceContext.totalStepCount,
      allreduceContext.stepCount,
      allreduceContext.nSteps,
      allreduceContext.buffOffset,
      allreduceContext.tmpbuffOffset);

  CtranAlgoLogger logger(allReduceAlgoName(myAlgo), op->opCount, comm);
  const bool useProfiler = NCCL_CTRAN_PROFILING != NCCL_CTRAN_PROFILING::none;

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allReduceAlgoName(myAlgo)));

  std::vector<int> remotePeers;
  // Populate remotePeers with IDs of remote peers
  for (int i = 0; i < nRanks; i++) {
    int peer = (myRank + i) % nRanks;
    if (statex->isSameNode(peer, myRank)) {
      continue;
    }
    remotePeers.emplace_back(peer);
  }

  // Assume all peers have the same message size and data type
  size_t sendSize = count * commTypeSize(datatype);
  CtranMapperContext context(allReduceAlgoName(myAlgo), sendSize, sendSize);
  comm->ctran_->mapper->setContext(std::move(context));

  for (int i = 0; i < allreduceContext.nSteps; i++) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "start the loop {}: count {}, totalStepCount {}, stepCount {}, buffOffset {}, tmpbuffOffset {}, displOffset {}",
        allReduceAlgoName(myAlgo),
        count,
        allreduceContext.totalStepCount,
        allreduceContext.stepCount,
        allreduceContext.buffOffset,
        allreduceContext.tmpbuffOffset,
        allreduceContext.displOffset);
    FB_COMMCHECK(alltoallOpImpl(
        op, allreduceContext, sendMemHdl, remotePeers, timestamp, useProfiler));

    // step 2: local reduce
    FB_COMMCHECK(reduceOpImpl(op, allreduceContext));

    FB_COMMCHECK(allgatherOpImpl(
        op,
        allreduceContext,
        remoteRecvBuffs,
        remoteAccessKeys,
        sendMemHdl,
        recvMemHdl,
        remotePeers,
        timestamp,
        useProfiler));

    updateContext(allreduceContext);
  }

  // special handling for remainder
  // we do allgather + local reduce for the remainder
  // Each rank reads all data from the other ranks and store to tmpbuf
  // Then local reduce is performed on the tmpbuf and write to the recvbuff
  if (allreduceContext.remCount) {
    updateContextForRemainder(allreduceContext);
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "remCount loop {}: count {}, totalStepCount {}, stepCount {}, buffOffset {}, tmpbuffOffset {}, displOffset {}",
        allReduceAlgoName(myAlgo),
        count,
        allreduceContext.totalStepCount,
        allreduceContext.stepCount,
        allreduceContext.buffOffset,
        allreduceContext.tmpbuffOffset,
        allreduceContext.displOffset);
    FB_COMMCHECK(alltoallOpImpl(
        op, allreduceContext, sendMemHdl, remotePeers, timestamp, useProfiler));

    // step 2: local reduce
    FB_COMMCHECK(reduceOpImpl(op, allreduceContext));
  }

  if (useProfiler) {
    comm->ctran_->mapper->timestamps.emplace_back(std::move(timestamp));
    comm->ctran_->mapper->reportProfiling();
  }
  return commSuccess;
}

commResult_t ctranAllReduceARG(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout) {
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_REDCOLL_INFO(
      allReduceAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      count,
      datatype,
      redOp,
      -1,
      comm,
      stream);
  if (count == 0) {
    return commSuccess;
  }

  auto reductionDatatype = datatype;
  // support bfloat16 (input) => float (reduction)
  if (NCCL_ALLREDUCE_TYPE == NCCL_ALLREDUCE_TYPE::ncclFloat32 &&
      datatype == commDataType_t::commBfloat16) {
    reductionDatatype = commFloat32;
  }

  auto key = std::make_tuple(datatype, reductionDatatype, redOp);
  auto it = typeToFunc.find(key);
  if (it == typeToFunc.end()) {
    // Key not exists; return error
    FB_ERRORRETURN(
        commInvalidArgument,
        "Key not exists for datatype {} reduce dataype and redOp {}",
        datatype,
        reductionDatatype,
        redOp);
  }
  const void* func = it->second;

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLREDUCE,
      stream,
      allReduceAlgoName(myAlgo),
      opCount);
  size_t typeSize = commTypeSize(datatype);
  void* sbuf = const_cast<void*>(sendbuff);
  void* dbuf = recvbuff;

  FB_COMMCHECK(comm->ctran_->algo->initTmpBufs());

  // FIXME: We perform an extra copy here before we submit to the GPE
  // thread.  Ideally we should be doing this copy inside the GPE
  // thread, but that requires two changes first: (1) our
  // searchRegHandle cannot try to dynamically register the buffer (as
  // that will fail); and (2) we need a copy kernel which does not
  // currently exist.

  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    sbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_SRC_TMPBUF);
    dbuf = comm->ctran_->algo->getTmpBuf(
        CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);
    FB_CUDACHECK(cudaMemcpyAsync(
        sbuf, sendbuff, count * typeSize, cudaMemcpyDefault, stream));
  }

  FB_COMMCHECK(setupKernelConfig(
      reinterpret_cast<const void*>(sbuf),
      dbuf,
      count,
      func,
      datatype,
      comm,
      config));

  std::vector<void*> tmpRegHdls;
  void *sendMemHdl = nullptr, *recvMemHdl = nullptr;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;

  // exchange the recvbuff with inter and intra node peers
  // temporarily put in the main threads.
  FB_COMMCHECK(exchangeUserbuff(
      comm,
      sbuf,
      sendMemHdl,
      dbuf,
      recvMemHdl,
      tmpRegHdls,
      count,
      typeSize,
      remoteRecvBuffs,
      remoteAccessKeys));

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      reinterpret_cast<const void*>(sbuf),
      dbuf,
      count,
      datatype,
      comm,
      stream,
      opCount,
      sendMemHdl,
      recvMemHdl,
      remoteRecvBuffs,
      remoteAccessKeys,
      config,
      opGroup));

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      reinterpret_cast<const void*>(func)));

  if (count * typeSize < CTRAN_MIN_REGISTRATION_SIZE) {
    FB_CUDACHECK(cudaMemcpyAsync(
        recvbuff, dbuf, count * typeSize, cudaMemcpyDefault, stream));
  }

  // deregister temporary registrations
  for (auto& hdl : tmpRegHdls) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(hdl));
  }

  return commSuccess;
}
