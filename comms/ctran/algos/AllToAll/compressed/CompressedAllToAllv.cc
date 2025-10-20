// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef ENABLE_META_COMPRESSION
#include <cuda_fp16.h>
#include <cstddef>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#include "comms/ctran/algos/AllToAll/compressed/CompressedAllToAllv.h"
#include "comms/ctran/algos/AllToAll/compressed/CompressedAllToAllvImpl.h"

static inline void logCompRatio(
    CtranComm* comm,
    size_t* ibSendBytes,
    size_t* ibCompSendBytes,
    size_t* totalSendBytes,
    size_t numChunks,
    cudaStream_t stream) {
  FB_CUDACHECKTHROW(cudaStreamSynchronize(stream));

  const auto statex = comm->statex_.get();
  const int myRank = statex->rank();
  const int myNode = statex->node();

  size_t totalIBbytes = 0;
  size_t totalIBCompBytes = 0;
  size_t totalBytes = 0;
  size_t totalCompBytes = 0;

  for (int i = 0; i < numChunks; i++) {
    const int peerNode = statex->node(i);
    totalIBbytes += ibSendBytes[i];
    totalIBCompBytes += ibCompSendBytes[i];
    totalBytes += totalSendBytes[i];
    if (myNode != peerNode) {
      totalCompBytes += ibCompSendBytes[i];
    } else {
      totalCompBytes += totalSendBytes[i];
    }
  }

  CLOGF(
      INFO,
      "[RANK {} NODE {}] IB compression ratio: {} | Total compression ratio: {}",
      myRank,
      myNode,
      (float)totalIBCompBytes / (float)totalIBbytes,
      (float)totalCompBytes / (float)totalBytes);
}

static inline void setRemoteSendBytes(
    std::vector<size_t>& sendBytes,
    std::vector<size_t>& remoteSendBytes,
    CtranComm* comm) {
  const auto statex = comm->statex_.get();
  const int myNode = statex->node();
  const int nRanks = statex->nRanks();
  for (int i = 0; i < nRanks; i++) {
    const int peerNode = statex->node(i);
    if (myNode != peerNode) {
      remoteSendBytes[i] = sendBytes[i];
    } else {
      remoteSendBytes[i] = 0;
    }
  }
}

#define RETURN_ALLTOALLV_IB_IMPL(perfconfig)                  \
  return ctranCompressedAllToAllvBootstrapIbImpl<perfconfig>( \
      op->alltoallv.sendbuff,                                 \
      op->alltoallv.sendcounts,                               \
      op->alltoallv.sdispls,                                  \
      op->alltoallv.recvbuff,                                 \
      op->alltoallv.recvcounts,                               \
      op->alltoallv.rdispls,                                  \
      op->alltoallv.datatype,                                 \
      comm,                                                   \
      std::move(timestamp));

#define RETURN_ALLTOALLV_FAST_IB_IMPL(perfconfig)    \
  return ctranCompressedAllToAllvIbImpl<perfconfig>( \
      op->alltoallv.sendbuff,                        \
      op->alltoallv.sendcounts,                      \
      op->alltoallv.sdispls,                         \
      op->alltoallv.recvbuff,                        \
      op->alltoallv.recvcounts,                      \
      op->alltoallv.rdispls,                         \
      op->alltoallv.datatype,                        \
      comm,                                          \
      std::move(timestamp));

static void* alltoallvKerns[commNumTypes] = {
    (void*)ncclKernelAllToAllv<int8_t>,
    (void*)ncclKernelAllToAllv<uint8_t>,
    (void*)ncclKernelAllToAllv<int32_t>,
    (void*)ncclKernelAllToAllv<uint32_t>,
    (void*)ncclKernelAllToAllv<int64_t>,
    (void*)ncclKernelAllToAllv<uint64_t>,
    (void*)ncclKernelAllToAllv<half>,
    (void*)ncclKernelAllToAllv<float>,
    (void*)ncclKernelAllToAllv<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    (void*)ncclKernelAllToAllv<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAllv<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAllv<__nv_fp8_e5m2>,
#endif
};

static auto myAlgo = NCCL_ALLTOALLV_ALGO::compCtran;

static inline void setAllToAllvAlgo(enum NCCL_ALLTOALLV_ALGO algo) {
  myAlgo = algo;
}

static inline commResult_t setupKernelConfig(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream,
    KernelConfig& config) {
  const auto statex = comm->statex_.get();
  // Unlike alltoall, we cannot automatically detect grid size because each rank
  // may see different counts; use static gridSize for now.
  config.numThreads = NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE;
  config.numBlocks = NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS;

  // Adjust gridSize to fit alltoallv kernel algorithm:
  // 1. gridSize must be even number, because we split blocks into two sets of
  //   groups, one for sends and the other for receives, each send and receive
  //   pair must use the same number of blocks
  if (config.numBlocks % 2) {
    config.numBlocks += 1;
  }
  // 2. gridSize must be <= CTRAN_ALGO_MAX_THREAD_BLOCKS, since internal
  //   states/flags holds at most CTRAN_ALGO_MAX_THREAD_BLOCKS blocks
  if (config.numBlocks < 2 || config.numBlocks > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    config.numBlocks = CTRAN_ALGO_MAX_THREAD_BLOCKS;
  }

  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.args.collective.alltoallv.sendbuff = sendbuff;
  config.args.collective.alltoallv.recvbuff = recvbuff;
  config.args.collective.alltoallv.datatype = datatype;
  config.args.collective.alltoallv.selfCount = sendcounts[statex->rank()];
  config.args.collective.alltoallv.selfSendDispl = sdispls[statex->rank()];
  config.args.collective.alltoallv.selfRecvDispl = rdispls[statex->rank()];

  // special case of ppn=1, simply set sendElemsList and recvElemsList to
  // nullptr, and numBlocks to 1 to let alltoallv kernel skip copies over NVLink
  // and only do self copy
  if (statex->nLocalRanks() == 1) {
    config.args.collective.alltoallv.sendElemsList = nullptr;
    config.args.collective.alltoallv.recvElemsList = nullptr;
    config.numBlocks = 1;
    return commSuccess;
  }

  // Pass number of thread block groups to kernel p2p elements
  // - Half blocks handle send, and the other handle receive
  // - Used in p2p elem to ensure ngroups number of inuse flags are checked when
  // reclaiming. This avoids cross-block sync in kernel
  const int ngroups = config.numBlocks / 2;
  comm->ctran_->gpe->allocKernelElems(
      statex->nLocalRanks() - 1,
      ngroups,
      &config.args.collective.alltoallv.sendElemsList);
  comm->ctran_->gpe->allocKernelElems(
      statex->nLocalRanks() - 1,
      ngroups,
      &config.args.collective.alltoallv.recvElemsList);

  // Ensure each rank sends to different peer at a time to avoid alltoone P2P
  // write congestion. For example, with localRanks = 4, the following
  // schedule is used:
  // - Round0:
  // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
  // - Round1:
  // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
  // - Round2:
  // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)
  KernelElem* sendElem = config.args.collective.alltoallv.sendElemsList;
  KernelElem* recvElem = config.args.collective.alltoallv.recvElemsList;
  for (int r = 0; r < statex->nLocalRanks() - 1; r++) {
    int sendPeer = (statex->localRank() + r + 1) % statex->nLocalRanks();
    int recvPeer = (statex->localRank() + statex->nLocalRanks() - r - 1) %
        statex->nLocalRanks();
    int sendPeerGlobal = statex->localRankToRank(sendPeer);
    int recvPeerGlobal = statex->localRankToRank(recvPeer);

    sendElem->staged.peerRank = sendPeer;
    sendElem->staged.count = sendcounts[sendPeerGlobal];
    sendElem->staged.displ = sdispls[sendPeerGlobal];
    sendElem = sendElem->next;

    recvElem->staged.peerRank = recvPeer;
    recvElem->staged.count = recvcounts[recvPeerGlobal];
    recvElem->staged.displ = rdispls[recvPeerGlobal];
    recvElem = recvElem->next;
  }

  return commSuccess;
}

static commResult_t opIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;

  CtranAlgoLogger logger(allToAllvAlgoName(myAlgo), op->opCount, comm);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allToAllvAlgoName(myAlgo)));

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    RETURN_ALLTOALLV_IB_IMPL(LowLatencyCollConfig);
  } else {
    RETURN_ALLTOALLV_IB_IMPL(DefaultPerfCollConfig);
  }
}

static commResult_t opFastIbImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  CtranComm* comm = opGroup.front()->comm_;

  CtranAlgoLogger logger(allToAllvAlgoName(myAlgo), op->opCount, comm);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp(allToAllvAlgoName(myAlgo)));

  if (NCCL_CTRAN_ENABLE_PRECONNECT) {
    RETURN_ALLTOALLV_FAST_IB_IMPL(LowLatencyCollConfig);
  } else {
    RETURN_ALLTOALLV_FAST_IB_IMPL(DefaultPerfCollConfig);
  }
}

static inline commResult_t setupGpeOp(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    uint64_t opCount,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  const auto statex = comm->statex_.get();
  // Passing op only when remote peers are present
  if (statex->nLocalRanks() < statex->nRanks()) {
    std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
        new OpElem(OpElem::opType::ALLTOALLV, comm, opCount));
    op->alltoallv.sendbuff = sendbuff;
    op->alltoallv.recvbuff = recvbuff;
    op->alltoallv.datatype = datatype;
    op->alltoallv.sendcounts.resize(statex->nRanks(), 0);
    op->alltoallv.sdispls.resize(statex->nRanks(), 0);
    op->alltoallv.recvcounts.resize(statex->nRanks(), 0);
    op->alltoallv.rdispls.resize(statex->nRanks(), 0);

    size_t totalSendCount = 0, totalRecvCount = 0;
    const int myNode = statex->node();
    for (int i = 0; i < statex->nRanks(); i++) {
      const int peerNode = statex->node(i);
      // GPE thread handles only remote peers
      if (myNode != peerNode) {
        op->alltoallv.sendcounts[i] = sendcounts[i];
        op->alltoallv.sdispls[i] = sdispls[i];
        op->alltoallv.recvcounts[i] = recvcounts[i];
        op->alltoallv.rdispls[i] = rdispls[i];

        totalSendCount += sendcounts[i];
        totalRecvCount += recvcounts[i];
      } else {
        // data to itself (i.e., HBM copy) will be handled by
        // ncclKernelAllToAllv kernel
        op->alltoallv.sendcounts[i] = 0;
        op->alltoallv.recvcounts[i] = 0;
      }
    }
    // if contains either non-zero send or receive, pass op
    if (totalSendCount || totalRecvCount) {
      opGroup.push_back(std::move(op));
    }
  }
  return commSuccess;
}

static inline commResult_t initCompManager(
    CtranComm* comm,
    commDataType_t dataType) {
  const auto statex = comm->statex_.get();
  auto myRank = statex->rank();
  auto nRanks = statex->nRanks();
  auto compressionManager = compression::CompressionManager::getInstance();

  if (compressionManager) {
    compressionManager->init(
        nRanks, myRank, commTypeSize(dataType), comm->ctran_.get());
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "{}: Compression manager created from rank: {}",
        allToAllvAlgoName(myAlgo),
        myRank);
  } else {
    CLOGF_SUBSYS(
        ERR,
        COLL,
        "{}: Compression manager not created from rank: {}",
        allToAllvAlgoName(myAlgo),
        myRank);

    return commSystemError;
  }

  return commSuccess;
}

// Perform compression only for remote IB peers, and perform no compression
// for intra-node peers. This is because there is little gain in performance
// for intra-node peers
static inline commResult_t compressRemote(
    const void* sendbuff,
    const size_t sendBytes[],
    const size_t sdispls[],
    CtranComm* comm,
    cudaStream_t stream) {
  if (sendbuff == nullptr) {
    return commSuccess;
  }

  auto statex = comm->statex_.get();
  size_t nRanks = statex->nRanks();
  size_t myRank = statex->rank();
  auto compressionManager = compression::CompressionManager::getInstance();

  // Perform compression
  {
    auto comp_status = compressionManager->compress(
        sendbuff, sendBytes, sdispls, nRanks, stream);
    if (comp_status != compression::compressionManagerStatus::Success) {
      CLOGF_SUBSYS(
          ERR,
          COLL,
          "{}: Compression failed from rank: {}",
          allToAllvAlgoName(myAlgo),
          myRank);
      return commInternalError;
    }
  }

  // Copy compressed byte counts from device to host
  {
    FB_CUDACHECKTHROW(cudaMemcpyAsync(
        compressionManager->getHostCompSendBytes(),
        compressionManager->getDeviceCompBytes(),
        nRanks * sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream));
  }

  return commSuccess;
}

commResult_t ctranCompressedAllToAllvWrapper(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t* recvcounts,
    const size_t rdispls[],
    commDataType_t datatype,
    IbImplType ibImplType,
    CtranComm* comm,
    cudaStream_t stream) {
  auto opCount = comm->ctran_->getOpCount();
  const auto statex = comm->statex_.get();
  auto myRank = statex->rank();
  auto nRanks = statex->nRanks();
  auto compressionManager = compression::CompressionManager::getInstance();

  CTRAN_COLL_INFO(
      allToAllvAlgoName(myAlgo).c_str(),
      sendbuff,
      recvbuff,
      0UL,
      datatype,
      -1,
      comm,
      stream);
  for (int i = 0; i < nRanks; i++) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "{}: opCount {} - sendcounts[{}] {} sdispls[{}] {} recvcounts[{}] {} rdispls[{}] {}",
        allToAllvAlgoName(myAlgo),
        opCount,
        i,
        sendcounts[i],
        i,
        sdispls[i],
        i,
        recvcounts[i],
        i,
        rdispls[i]);
  }

  // Convert every data and displs to byte for compression
  const size_t numChunks = comm->statex_->nRanks();
  std::vector<size_t> sendBytes(numChunks, 0);
  std::vector<size_t> sDisplsBytes(numChunks, 0);
  std::vector<size_t> recvBytes(numChunks, 0);
  std::vector<size_t> rDisplsBytes(numChunks, 0);
  {
    for (int i = 0; i < numChunks; i++) {
      sendBytes[i] = sendcounts[i] * commTypeSize(datatype);
      sDisplsBytes[i] = sdispls[i] * commTypeSize(datatype);
      recvBytes[i] = recvcounts[i] * commTypeSize(datatype);
      rDisplsBytes[i] = rdispls[i] * commTypeSize(datatype);
    }
  }

  // For local peers, we set the send bytes to 0
  std::vector<size_t> remoteSendBytes(nRanks, 0);
  setRemoteSendBytes(sendBytes, remoteSendBytes, comm);

  // Use one bytes for compression as well as communication
  commDataType_t compCommDataType = commUint8;
  if (initCompManager(comm, compCommDataType) != commSuccess) {
    CLOGF_SUBSYS(
        ERR,
        COLL,
        "{}: Compression manager initialization failed from rank: {}",
        allToAllvAlgoName(myAlgo),
        myRank);
    return commSystemError;
  }

  // Perform compression
  if (compressRemote(
          sendbuff, sendBytes.data(), sDisplsBytes.data(), comm, stream) !=
      commSuccess) {
    CLOGF_SUBSYS(
        ERR,
        COLL,
        "{}: Compression failed from rank: {}",
        allToAllvAlgoName(myAlgo),
        myRank);
    return commSystemError;
  }

  // prepare kernel config for self and NVL copies
  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::ALLTOALLV,
      stream,
      allToAllvAlgoName(myAlgo),
      opCount);
  FB_COMMCHECK(setupKernelConfig(
      sendbuff,
      sendBytes.data(),
      sDisplsBytes.data(),
      recvbuff,
      recvBytes.data(),
      rDisplsBytes.data(),
      compCommDataType,
      comm,
      stream,
      config));

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  FB_COMMCHECK(setupGpeOp(
      sendbuff,
      remoteSendBytes.data(),
      sDisplsBytes.data(),
      recvbuff,
      recvBytes.data(),
      rDisplsBytes.data(),
      compCommDataType,
      comm,
      opCount,
      opGroup));

  opFunc ibImpl = opFastIbImpl;
  {
    if (ibImplType == IbImplType::Bootstrap) {
      ibImpl = opIbImpl;
    } else if (ibImplType == IbImplType::IbExchange) {
      ibImpl = opFastIbImpl;
    }
  }

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      ibImpl,
      config,
      reinterpret_cast<void*>(alltoallvKerns[compCommDataType])));

  // TODO remove if bootstrapping is disabled
  FB_CUDACHECKTHROW(cudaStreamSynchronize(stream));

  for (int i = 0; i < nRanks; i++) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "{}: local rank [{}] remote rank [{}] send compressed bytes {} recv compressed bytes {}",
        allToAllvAlgoName(myAlgo),
        myRank,
        i,
        compressionManager->getHostCompSendBytes()[i],
        compressionManager->getHostCompRecvBytes()[i],
        myRank);
  }

  // Decompression
  std::vector<size_t> remoteDecompRecvBytes(nRanks, 0);
  for (int i = 0; i < nRanks; i++) {
    if (compressionManager->getHostCompRecvBytes()[i] != 0) {
      remoteDecompRecvBytes[i] = recvBytes[i];
    }
  }

  if (recvbuff != nullptr) {
    auto decomp_status = compressionManager->decompress(
        recvbuff,
        remoteDecompRecvBytes.data(),
        rDisplsBytes.data(),
        comm->statex_->nRanks(),
        stream);
    if (decomp_status != compression::compressionManagerStatus::Success) {
      CLOGF_SUBSYS(
          ERR,
          COLL,
          "{}: Decompression failed from rank: {}",
          allToAllvAlgoName(myAlgo),
          myRank);
      return commInternalError;
    }
  }

  if (NCCL_CTRAN_COMPRESSED_ENABLE_LOGGING) {
    logCompRatio(
        comm,
        remoteSendBytes.data(),
        compressionManager->getHostCompSendBytes(),
        sendBytes.data(),
        nRanks,
        stream);
  }

  return commSuccess;
}

commResult_t ctranCompressedAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t* recvcounts,
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  setAllToAllvAlgo(NCCL_ALLTOALLV_ALGO::compCtran);

  return ctranCompressedAllToAllvWrapper(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      IbImplType::IbExchange,
      comm,
      stream);
}

commResult_t ctranBootstrapCompressedAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t* recvcounts,
    const size_t rdispls[],
    commDataType_t datatype,
    CtranComm* comm,
    cudaStream_t stream) {
  setAllToAllvAlgo(NCCL_ALLTOALLV_ALGO::bsCompCtran);

  return ctranCompressedAllToAllvWrapper(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      IbImplType::Bootstrap,
      comm,
      stream);
}

// For now, we keep the same logic as alltoallv, but we will add data type check
// in the future
bool ctranCompressedAllToAllvSupport(CtranComm* comm) {
  bool ctranSupport = false;
  const auto statex = comm->statex_.get();
  if (ctranInitialized(comm)) {
    ctranSupport = true;
    // Check if all remote peers are supported by ctran
    // For intra-node peers, ctranAlgo supports copy based path;
    // for inter-node peers, we need a mapper backend to support.
    const int myNode = statex->node();
    for (int rank = 0; rank < statex->nRanks(); rank++) {
      if (statex->node(rank) != myNode &&
          comm->ctran_->mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
        ctranSupport = false;
        break;
      }
    }
  }

  return ctranSupport;
}
#endif
