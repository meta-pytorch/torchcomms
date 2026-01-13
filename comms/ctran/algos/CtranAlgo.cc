// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string.h>
#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/TmpBufSegManager.h"
#include "comms/pipes/ChunkState.cuh"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using comms::pipes::ChunkState;

CtranAlgo::CtranAlgo(CtranComm* comm, ICtran* ctran)
    : comm_(comm), ctran_(ctran) {
  all2allvDynamicMaxSendcounts =
      NCCL_CTRAN_ALLTOALLV_DYNAMIC_MAX_NUM_COUNTS_PER_PEER *
      comm_->statex_->nRanks();
  all2allvDynamicMaxNumSplitsPerRank =
      NCCL_CTRAN_ALLTOALLV_DYNAMIC_MAX_NUM_COUNTS_PER_PEER;

  // Always initialize kernel resources since the current impl requires barrier
  // among all local ranks. It should not be triggered on-demand at local
  // getDevState() call.
  // TODO: Properly move some heavy allocation to on-demand.
  FB_COMMCHECKTHROW_EX(initKernelResources(), comm_->logMetaData_);
  if (!comm->runtimeConn_) {
    FB_COMMCHECKIGNORE(initTmpBufs());
  }

  FB_CUDACHECKTHROW_EX(
      cudaHostAlloc(
          &this->sendCountsTmpbufCPU,
          sizeof(size_t) * all2allvDynamicMaxSendcounts,
          cudaHostAllocDefault),
      comm->logMetaData_);
  tmpbufSegments[TmpbufType::SENDCOUNTS_TMPBUF_CPU] = this->sendCountsTmpbufCPU;
  tmpbufSegmentOffsets[TmpbufType::SENDCOUNTS_TMPBUF_CPU] = 0;

  FB_CUDACHECKTHROW_EX(
      cudaHostAlloc(
          &this->sendIndicesTmpbufCPU,
          sizeof(size_t) * all2allvDynamicMaxSendcounts,
          cudaHostAllocDefault),
      comm->logMetaData_);
  tmpbufSegments[TmpbufType::SENDINDICES_TMPBUF_CPU] =
      this->sendIndicesTmpbufCPU;
  tmpbufSegmentOffsets[TmpbufType::SENDINDICES_TMPBUF_CPU] = 0;

  FB_CUDACHECKTHROW_EX(
      cudaHostAlloc(
          &this->sendIndicesBlockLengthsTmpbufCPU,
          sizeof(size_t) * comm_->statex_->nRanks(),
          cudaHostAllocDefault),
      comm->logMetaData_);
  tmpbufSegments[TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU] =
      this->sendIndicesBlockLengthsTmpbufCPU;
  tmpbufSegmentOffsets[TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU] = 0;

  FB_CUDACHECKTHROW_EX(
      cudaHostAlloc(
          &this->sendbuffsPtrTmpbufCPU,
          sizeof(void*) * all2allvDynamicMaxSendcounts,
          cudaHostAllocDefault),
      comm->logMetaData_);
  tmpbufSegments[TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU] =
      this->sendbuffsPtrTmpbufCPU;
  tmpbufSegmentOffsets[TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU] = 0;

  FB_COMMCHECKTHROW_EX(initializeCommAttributesMap(), comm_->logMetaData_);

  return;
}

CtranAlgo::~CtranAlgo() {
  collToVcConfigMap_.clear();

  if (this->sharedRes_) {
    FB_COMMCHECKIGNORE(this->sharedRes_->release());
    delete this->sharedRes_;
    this->sharedRes_ = nullptr;
  }

  // clean up intra node remote tmpbuffs resources
  FB_COMMCHECKIGNORE(deregRemoteTmpBufs());

  if (this->isResInitialized_) {
    FB_COMMCHECKIGNORE(
        ctran::utils::commCudaFree(
            this->devState_d_, &this->comm_->logMetaData_));
    this->devState_d_ = nullptr;
  }
  if (ctran_->mapper && this->tmpbufSegHdl) {
    FB_COMMCHECKIGNORE(ctran_->mapper->deregMem(
        this->tmpbufSegHdl, true /* skipRemRelease */));
  }
  if (this->tmpbuf) {
    if (comm_->memCache_) {
      FB_COMMCHECKTHROW_EX(
          comm_->memCache_->release({this->tmpBufKey}), comm_->logMetaData_);
    } else {
      // `ctran::utils::commCudaFree` automatically decides whether to use
      // commCuMemFree
      // or cudaFree to free buffer based on cuMem support
      FB_COMMCHECKTHROW_EX(
          ctran::utils::commCudaFree(this->tmpbuf, &this->comm_->logMetaData_),
          comm_->logMetaData_);
    }
  }
  if (this->sendCountsTmpbufCPU) {
    FB_CUDACHECKIGNORE(cudaFreeHost(this->sendCountsTmpbufCPU));
  }
  if (this->sendbuffsPtrTmpbufCPU) {
    FB_CUDACHECKIGNORE(cudaFreeHost(this->sendbuffsPtrTmpbufCPU));
  }
  if (this->sendIndicesTmpbufCPU) {
    FB_CUDACHECKIGNORE(cudaFreeHost(this->sendIndicesTmpbufCPU));
  }
  if (this->sendIndicesBlockLengthsTmpbufCPU) {
    FB_CUDACHECKIGNORE(cudaFreeHost(this->sendIndicesBlockLengthsTmpbufCPU));
  }

  if (this->allReduceDirectResource) {
    FB_COMMCHECKIGNORE(this->allReduceDirectResource->destroy());
  }

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

CtranAlgoDeviceState* CtranAlgo::getDevState() {
  FB_CHECKABORT(this->devState_d_ != nullptr, "CTRAN-ALGO: devState not ready");
  return this->devState_d_;
}

static const std::string kCtranAlgoInitResources{
    "CtranAlgoInitResources - lazy connect init"};

namespace {
inline size_t getPerPeerChunkStatesSize() {
  static size_t size = sizeof(ChunkState) * CTRAN_P2P_NVL_DEVMEM_MAX_CHUNKS;
  return size;
}
// Helper to calculate sync and staging buffer and chunkState pointers for a
// given peer
std::tuple<CtranAlgoDeviceSync*, void*, void*>
partitionDevShm(void* mappedDevShmPtr, int nLocalRanks, int pos) {
  char* regionPtr_d = reinterpret_cast<char*>(mappedDevShmPtr);
  void* bufBase_d =
      regionPtr_d + (nLocalRanks - 1) * sizeof(CtranAlgoDeviceSync);
  void* chunkStateBase_d = reinterpret_cast<char*>(bufBase_d) +
      (nLocalRanks - 1) * NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE;
  void* sync = regionPtr_d + pos * sizeof(CtranAlgoDeviceSync);
  void* buf = reinterpret_cast<char*>(bufBase_d) +
      pos * NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE;
  void* chunkState = reinterpret_cast<char*>(chunkStateBase_d) +
      pos * getPerPeerChunkStatesSize();
  return {reinterpret_cast<CtranAlgoDeviceSync*>(sync), buf, chunkState};
}
} // namespace

commResult_t CtranAlgo::initKernelResources() {
  const auto statex = comm_->statex_.get();
  CtranAlgoDeviceState tmpDevState;
  int nLocalRanks = statex->nLocalRanks();
  int localRank = statex->localRank();
  int rank = statex->rank();

  if (nLocalRanks > CTRAN_MAX_NVL_PEERS) {
    CLOGF(
        ERR,
        "CTRAN only supports NVL peers up to {}, but nLocalRanks is {}. "
        "This will likely cause seg fault or data corruption! "
        "Try set CTRAN_MAX_NVL_PEERS to be larger or equal to nLocalRanks via "
        "compile flag DCTRAN_MAX_NVL_PEERS.",
        CTRAN_MAX_NVL_PEERS,
        nLocalRanks);
    return commInternalError;
  }

  NcclScubaEvent scubaEvent(kCtranAlgoInitResources, &comm_->logMetaData_);
  scubaEvent.startAndRecord();

  memset(&tmpDevState, 0, sizeof(CtranAlgoDeviceState));

  // Initialize inter-process shared device buffer
  if (!this->sharedRes_) {
    this->sharedRes_ = new SharedResource(comm_);
  }

  // Check if shared memory per block (by opt in) is large enough to hold
  // CtranAlgoDeviceState since kernel will load the entire CtranAlgoDeviceState
  // to shared memory
  // NOTE: do not use cudaGetDeviceProperties to get the sharedMemPerBlockOptin
  // property value, because on AMD sharedMemPerBlockOptin returned by
  // cudaGetDeviceProperties is always 0 while cudaDeviceGetAttribute returns
  // the correct value. AMD bug report: https://ontrack.amd.com/browse/FBA-621
  int maxSharedMemOptin;
  FB_CUDACHECK(cudaDeviceGetAttribute(
      &maxSharedMemOptin,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      statex->cudaDev()));

  if (maxSharedMemOptin < sizeof(CtranAlgoDeviceState)) {
    CLOGF(
        ERR,
        "CTRAN-ALGO: sharedMemPerBlockOptin {} on device {} is smaller than the size of CtranAlgoDeviceState {}",
        maxSharedMemOptin,
        statex->cudaDev(),
        sizeof(CtranAlgoDeviceState));
    return commInternalError;
  }

  CLOGF(
      INFO,
      "CTRAN-ALGO: prepare device global state {} bytes (sharedMemPerBlockOptin {} bytes) on rank {} localRank {} nLocalRanks {} commHash {:x}",
      sizeof(CtranAlgoDeviceState),
      maxSharedMemOptin,
      rank,
      localRank,
      nLocalRanks,
      statex->commHash());

  // Copy basic comm info to device state for collective kernel to use
  statex->setupDev(tmpDevState.statex);

  tmpDevState.bufSize = NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE;
  tmpDevState.bcastBufSize = NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE;
  tmpDevState.enableTraceLog = NCCL_CTRAN_ENABLE_DEV_TRACE_LOG;
  tmpDevState.enableCancellableWaits = comm_->abortEnabled();

  // Setup pointers to bufstates and shared buffer of each peers' shared region
  // See description of bufState and buf memory locations in SharedResource.
  if (this->sharedRes_) {
    auto& remoteSyncsMap = tmpDevState.remoteSyncsMap;
    auto& localSyncsMap = tmpDevState.localSyncsMap;
    auto& remoteStagingBufsMap = tmpDevState.remoteStagingBufsMap;
    auto& localStagingBufsMap = tmpDevState.localStagingBufsMap;
    auto& remoteChunkStatesMap = tmpDevState.remoteChunkStatesMap;
    auto& localChunkStatesMap = tmpDevState.localChunkStatesMap;
    auto& peerBcastBufsMap = tmpDevState.peerBcastBufsMap;
    auto& peerAllToAllvDynamicBufsMap = tmpDevState.peerAllToAllvDynamicBufsMap;
    auto& alltoallvDynamicSendbuffsMap =
        tmpDevState.alltoallvDynamicSendbuffsMap;

    for (int i = 0; i < nLocalRanks; i++) {
      if (i == localRank) {
        localSyncsMap[i] = nullptr;
        remoteSyncsMap[i] = nullptr;
        localStagingBufsMap[i] = nullptr;
        remoteStagingBufsMap[i] = nullptr;
      } else {
        int localPos = LOCAL_RANK_TO_DEV_REGION_POS(i, localRank);
        auto [localSync, localBuf, localChunkState] = partitionDevShm(
            this->sharedRes_->mappedDevShmPtrs[localRank],
            nLocalRanks,
            localPos);
        localSyncsMap[i] = localSync;
        localStagingBufsMap[i] = localBuf;
        localChunkStatesMap[i] = localChunkState;

        int remotePos = LOCAL_RANK_TO_DEV_REGION_POS(localRank, i);
        auto [remoteSync, remoteBuf, remoteChunkState] = partitionDevShm(
            this->sharedRes_->mappedDevShmPtrs[i], nLocalRanks, remotePos);
        remoteSyncsMap[i] = remoteSync;
        remoteStagingBufsMap[i] = remoteBuf;
        remoteChunkStatesMap[i] = remoteChunkState;
      }

      // Next chunk is for bcastBuf
      peerBcastBufsMap[i] = (char*)this->sharedRes_->mappedDevShmPtrs[i] +
          (sizeof(CtranAlgoDeviceSync) + NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE +
           getPerPeerChunkStatesSize()) *
              (nLocalRanks - 1);
      CLOGF_TRACE(
          INIT,
          "CTRAN-ALGO: allocated local peerBcastBufsMap[{}] = {}, size {}",
          i,
          (void*)peerBcastBufsMap[i],
          tmpDevState.bcastBufSize);

      // Next chunk is for alltoallvDynamic
      peerAllToAllvDynamicBufsMap[i] = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(peerBcastBufsMap[i]) +
          tmpDevState.bcastBufSize);
    }
    // FIXME: need better management on shm resources. Like how we managed the
    // tmpbuf.
    char* alltoallvDynamicSendbuffsMap_d =
        (char*)(this->sharedRes_->devShmPtr) +
        (sizeof(CtranAlgoDeviceSync) + NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE +
         getPerPeerChunkStatesSize()) *
            (statex->nLocalRanks() - 1) +
        NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE +
        (CTRAN_ALGO_MAX_THREAD_BLOCKS / 2) * sizeof(size_t) *
            (all2allvDynamicMaxSendcounts + 1 +
             all2allvDynamicMaxNumSplitsPerRank) *
            statex->nLocalRanks();
    for (int i = 0; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; i++) {
      alltoallvDynamicSendbuffsMap[i] = reinterpret_cast<void**>(
          alltoallvDynamicSendbuffsMap_d +
          i * sizeof(void*) * all2allvDynamicMaxSendcounts);
    }
  }

  // Copy contents to device
  FB_COMMCHECK(
      ctran::utils::commCudaMalloc(
          &this->devState_d_,
          1,
          &this->comm_->logMetaData_,
          "initKernelResources"));
  FB_CUDACHECK(cudaMemcpy(
      this->devState_d_,
      &tmpDevState,
      sizeof(CtranAlgoDeviceState),
      cudaMemcpyHostToDevice));

  this->isResInitialized_ = true;

  scubaEvent.stopAndRecord();
  return commSuccess;
}

CtranAlgo::SharedResource::SharedResource(CtranComm* comm) {
  const auto statex = comm->statex_.get();
  this->comm_ = comm;
  int localRank = statex->localRank();
  int nLocalRanks = statex->nLocalRanks();

  // Create local shared memory region
  // The memory region on each owner rank is divided to (localRanks -1) sets of
  // bufState, buf, chunkState for each peer, excluding the owner. The format is
  // as below with N localRanks.
  // |bufState_0|bufState_1|...|bufState_N-2|buf_0|buf_1|...|buf_N-2|chunkState_0|chunkState_1|...|chunkState_N-2|
  std::vector<ctran::utils::CtranIpcDesc> ipcDescs(nLocalRanks);
  size_t shmSize =
      (sizeof(CtranAlgoDeviceSync) + NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE +
       getPerPeerChunkStatesSize()) *
          (nLocalRanks - 1) +
      NCCL_CTRAN_BCAST_NVL_SHARED_DEVBUF_SIZE;

  // Allocate extra buffer for AllToAllvDynamic count/indeciesblock/indices
  // (spaced used in aforementioned order) exchange.
  // FIXME: this much size is only needed for noncontig kernel.
  // Can add ifdef to reduce size for contig kernel, which only need 1 size_t
  // for one send/recv pair.
  shmSize += (CTRAN_ALGO_MAX_THREAD_BLOCKS / 2) * sizeof(size_t) *
      (all2allvDynamicMaxSendcounts + 1 + all2allvDynamicMaxNumSplitsPerRank) *
      nLocalRanks;

  DevMemType memType = DevMemType::kCumem;
  auto cuMemHandleType = ctran::utils::getCuMemAllocHandleType();

  if (!ctran::utils::getCuMemSysSupported()) {
    memType = DevMemType::kCudaMalloc;
    cuMemHandleType = CU_MEM_HANDLE_TYPE_NONE;
  }

  /* allocate extra buffer for AllToAllvDynamic sendbuffers. */
  shmSize += CTRAN_ALGO_MAX_THREAD_BLOCKS * sizeof(void*) *
      all2allvDynamicMaxSendcounts;

  // Throw exception if fails to allocate memory
  this->ipcMem_ = std::make_unique<ctran::utils::CtranIpcMem>(
      shmSize,
      statex->cudaDev(),
      &comm_->logMetaData_,
      "CtranAlgoSharedResource",
      memType,
      cuMemHandleType);
  void* devShmPtr = this->ipcMem_->getBase();
  this->devShmPtr = devShmPtr;

  FB_COMMCHECKTHROW_EX(
      this->ipcMem_->ipcExport(ipcDescs[localRank]), comm_->logMetaData_);

  // Initialize device state and chunk state for each peer
  for (int i = 0; i < nLocalRanks; i++) {
    // Skip owner itself
    if (i == localRank) {
      continue;
    }

    int pos = LOCAL_RANK_TO_DEV_REGION_POS(i, localRank);
    void* statePtr_d =
        reinterpret_cast<char*>(devShmPtr) + pos * sizeof(CtranAlgoDeviceSync);
    struct CtranAlgoDeviceSync syncInitialVal;
    for (int j = 0; j < CTRAN_ALGO_MAX_THREAD_BLOCKS; j++) {
      syncInitialVal.syncs[j].stepOnSameBlockIdx = CTRAN_ALGO_STEP_RESET;
    }
    FB_CUDACHECKTHROW_EX(
        cudaMemcpy(
            statePtr_d,
            &syncInitialVal,
            sizeof(CtranAlgoDeviceSync),
            cudaMemcpyHostToDevice),
        comm->logMetaData_);

    void* chunkStatePtr_d = reinterpret_cast<char*>(devShmPtr) +
        (nLocalRanks - 1) *
            (sizeof(CtranAlgoDeviceSync) +
             NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE) +
        pos * getPerPeerChunkStatesSize();
    std::vector<ChunkState> initStates(
        CTRAN_P2P_NVL_DEVMEM_MAX_CHUNKS, ChunkState());
    FB_CUDACHECKTHROW_EX(
        cudaMemcpy(
            chunkStatePtr_d,
            initStates.data(),
            getPerPeerChunkStatesSize(),
            cudaMemcpyHostToDevice),
        comm->logMetaData_);
  }

  // Exchange IPC handle with all local ranks
  // Note allGatherIntraNode can support allgather from all ranks in the same
  // nvl Domain even if they are cross node. TODO: maybe rename
  // allGatherIntraNode or create a new wrapper name to better reflect
  // this.tivegiinlkcghcehhtngelkjjcbunhdk
  auto resFuture = comm_->bootstrap_->allGatherIntraNode(
      ipcDescs.data(),
      sizeof(ctran::utils::CtranIpcDesc),
      localRank,
      nLocalRanks,
      statex->localRankToRanks());
  FB_COMMCHECKTHROW_EX(
      static_cast<commResult_t>(std::move(resFuture).get()),
      comm_->logMetaData_);

  // Setup mapped shared memory region pointers for all local ranks
  this->mappedDevShmPtrs.resize(nLocalRanks);

  for (int i = 0; i < nLocalRanks; ++i) {
    CLOGF_TRACE(INIT, "Received ipcDescs[{}]={}", i, ipcDescs[i].toString());
    if (localRank == i) {
      this->mappedDevShmPtrs[i] = devShmPtr;
    } else {
      std::unique_ptr<ctran::utils::CtranIpcRemMem> remMem =
          std::make_unique<ctran::utils::CtranIpcRemMem>(
              ipcDescs[i],
              statex->cudaDev(),
              &this->comm_->logMetaData_,
              "CtranAlgoSharedResource");

      this->mappedDevShmPtrs[i] = remMem->getBase();
      this->ipcRemMemMap_[i] = std::move(remMem);
    }
  }

  // Ensure all local ranks have imported remote memory.
  // This is required to ensure no one destroys the local memory handle while
  // other ranks are still importing, which may fail.
  resFuture = comm_->bootstrap_->barrierIntraNode(
      localRank, nLocalRanks, statex->localRankToRanks());
  FB_COMMCHECKTHROW_EX(
      static_cast<commResult_t>(std::move(resFuture).get()),
      comm_->logMetaData_);

  CLOGF(
      INFO,
      "CTRAN-ALGO: requested {} bytes (allocated {}) of device buffer as shared resource on rank {} localRank {}",
      shmSize,
      this->ipcMem_->getRange(),
      statex->rank(),
      statex->localRank());
  return;
}

commResult_t CtranAlgo::SharedResource::release() {
  const auto statex = comm_->statex_.get();
  // Release imported remote memory
  for (auto& [localRank, remMem] : this->ipcRemMemMap_) {
    FB_COMMCHECK(remMem->release());
  }

  // Release local memory
  FB_COMMCHECK(this->ipcMem_->free());

  // Reset pointers to avoid any future access
  for (int i = 0; i < statex->nLocalRanks(); i++) {
    this->mappedDevShmPtrs[i] = nullptr;
  }

  return commSuccess;
}

CtranAlgoLogger::CtranAlgoLogger(
    const std::string& name,
    const uint64_t opCount,
    const CtranComm* comm,
    std::optional<const ICtran*> ctran)
    : name(name), opCount_(opCount), comm_(comm), ctran_(ctran) {
  auto& statex = comm_->statex_;
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "{} GPE-START: opCount {} comm {} commHash {:x} Ctran {}",
      name,
      opCount_,
      (void*)comm_,
      statex->commHash(),
      (void*)ctran_.value_or(comm_->ctran_.get()));
}

CtranAlgoLogger::~CtranAlgoLogger() {
  auto& statex = comm_->statex_;
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "{} GPE-DONE: opCount {} comm {} commHash {:x} Ctran {}",
      name,
      opCount_,
      (void*)comm_,
      statex->commHash(),
      (void*)ctran_.value_or(comm_->ctran_.get()));
}

CtranAlgoRMALogger::CtranAlgoRMALogger(
    const std::string& name,
    const uint64_t opCount,
    const int peerRank,
    const ctran::CtranWin* win,
    const CtranComm* comm)
    : name_(name),
      opCount_(opCount),
      peerRank_(peerRank),
      win_(win),
      comm_(comm) {
  auto& statex = comm_->statex_;
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "{} GPE-START: opCount {} rank {} peer {} win {} comm {} commHash {:x} Ctran {}",
      name_,
      opCount_,
      statex->rank(),
      peerRank_,
      (void*)win_,
      (void*)comm_,
      statex->commHash(),
      (void*)comm_->ctran_.get());
}

CtranAlgoRMALogger::~CtranAlgoRMALogger() {
  auto& statex = comm_->statex_;
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "{} GPE-DONE: opCount {} rank {} peer {} win {} comm {} commHash {:x} Ctran {}",
      name_,
      opCount_,
      statex->rank(),
      peerRank_,
      (void*)win_,
      (void*)comm_,
      statex->commHash(),
      (void*)comm_->ctran_.get());
}

static const std::unordered_map<std::string, enum NCCL_ALLGATHER_ALGO>
    ctranAllGatherAlgoMap = {
        {"orig", NCCL_ALLGATHER_ALGO::orig},
        {"ctran", NCCL_ALLGATHER_ALGO::ctran},
        {"ctdirect", NCCL_ALLGATHER_ALGO::ctdirect},
        {"ctring", NCCL_ALLGATHER_ALGO::ctring},
        {"ctrd", NCCL_ALLGATHER_ALGO::ctrd},
        {"ctbrucks", NCCL_ALLGATHER_ALGO::ctbrucks}};

commResult_t ctranConfigCommAlgoOverride(CtranComm* comm) {
  if (!ctranInitialized(comm)) {
    return commSuccess;
  }

  if (std::strcmp(comm->config_.ncclAllGatherAlgo, "undefined") == 0) {
    return commSuccess;
  }

  auto it = ctranAllGatherAlgoMap.find(comm->config_.ncclAllGatherAlgo);
  if (it != ctranAllGatherAlgoMap.end()) {
    comm->ctran_->algo->setAllGatherAlgo(it->second);
  } else {
    CLOGF(
        WARN,
        "Invalid value for ncclAllGatherAlgo: {}",
        comm->config_.ncclAllGatherAlgo);
  }
  return commSuccess;
}

void CtranAlgo::setAllGatherAlgo(enum NCCL_ALLGATHER_ALGO algo) {
  allGatherAlgo = algo;
}

enum NCCL_ALLGATHER_ALGO CtranAlgo::getAllGatherAlgo() {
  if (allGatherAlgo.has_value()) {
    return allGatherAlgo.value();
  } else {
    return NCCL_ALLGATHER_ALGO;
  }
}

void CtranAlgo::setAllReduceAlgo(enum NCCL_ALLREDUCE_ALGO algo) {
  allReduceAlgo = algo;
}

enum NCCL_ALLREDUCE_ALGO CtranAlgo::getAllReduceAlgo() {
  if (allReduceAlgo.has_value()) {
    return allReduceAlgo.value();
  } else {
    return NCCL_ALLREDUCE_ALGO;
  }
}

commResult_t CtranAlgo::exchangePeerTmpbuf(int peer) {
  // if peer is out of range, we report an error
  if (comm_->statex_->nRanks() < peer) {
    CLOGF(WARN, "Invalid value for peer: {}", peer);
    return commInvalidArgument;
  }
  // if tmpbuffs are already exchanged, we don't need to do anything
  if (this->remoteTmpAccessKeys.size() == comm_->statex_->nRanks() &&
      this->remoteTmpAccessKeys[peer].backend != CtranMapperBackend::UNSET) {
    return commSuccess;
  }

  // create the vector of remoteTmpbuffs if not already created
  if (this->remoteTmpbuffs.size() != comm_->statex_->nRanks()) {
    this->remoteTmpbuffs.resize(comm_->statex_->nRanks());
    this->remoteTmpAccessKeys.resize(comm_->statex_->nRanks());
  }

  // skip if peer is the same as the rank
  if (comm_->statex_->rank() == peer) {
    return commSuccess;
  }

  CtranMapperRequest* remoteTmpbuffReq = nullptr;
  CtranMapperRequest* localTmpbuffReq = nullptr;
  FB_COMMCHECK(comm_->ctran_->mapper->irecvCtrl(
      &this->remoteTmpbuffs[peer],
      &this->remoteTmpAccessKeys[peer],
      peer,
      &remoteTmpbuffReq));
  FB_COMMCHECK(comm_->ctran_->mapper->isendCtrl(
      this->tmpbuf, this->tmpbufRegHdl, peer, &localTmpbuffReq));

  FB_COMMCHECK(comm_->ctran_->mapper->waitRequest(remoteTmpbuffReq));
  FB_COMMCHECK(comm_->ctran_->mapper->waitRequest(localTmpbuffReq));

  delete remoteTmpbuffReq;
  delete localTmpbuffReq;

  return commSuccess;
}

commResult_t CtranAlgo::exchangeInterNodeTmpbuf() {
  /* if tmpbuffs are already exchanged, we don't need to do anything */
  /* if we only have one node, we don't need to exchange tmpbuf handlers
     as GPU threads do not need tmpbuf handler to exchange infomation */
  if (!this->remoteTmpbuffs.empty() || comm_->statex_->nNodes() <= 1) {
    return commSuccess;
  }

  std::vector<std::unique_ptr<CtranMapperRequest>> interNodeRemoteTmpbuffReq(
      comm_->statex_->nRanks());
  std::vector<std::unique_ptr<CtranMapperRequest>> interNodeLocalTmpbuffReq(
      comm_->statex_->nRanks());

  this->remoteTmpbuffs.resize(comm_->statex_->nRanks());
  this->remoteTmpAccessKeys.resize(comm_->statex_->nRanks());

  for (int i = 0; i < comm_->statex_->nRanks(); i++) {
    if (!comm_->statex_->isSameNode(i, comm_->statex_->rank())) {
      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(comm_->ctran_->mapper->irecvCtrl(
          &this->remoteTmpbuffs[i], &this->remoteTmpAccessKeys[i], i, &req));
      interNodeRemoteTmpbuffReq[i] = std::unique_ptr<CtranMapperRequest>(req);

      FB_COMMCHECK(
          comm_->ctran_->mapper->isendCtrl(tmpbuf, tmpbufRegHdl, i, &req));
      interNodeLocalTmpbuffReq[i] = std::unique_ptr<CtranMapperRequest>(req);
    }
  }

  // Wait for all tmpbuff control messages to complete
  for (int i = 0; i < comm_->statex_->nRanks(); i++) {
    if (!comm_->statex_->isSameNode(i, comm_->statex_->rank())) {
      FB_COMMCHECK(comm_->ctran_->mapper->waitRequest(
          interNodeRemoteTmpbuffReq[i].get()));
      FB_COMMCHECK(comm_->ctran_->mapper->waitRequest(
          interNodeLocalTmpbuffReq[i].get()));
    }
  }

  return commSuccess;
}

commResult_t CtranAlgo::deregRemoteTmpBufs() {
  if (this->remoteTmpAccessKeys.empty() || !ctran_->mapper) {
    return commSuccess;
  }

  const int nRanks = comm_->statex_->nRanks();

  // cleanup the nvl resource
  for (int i = 0; i < nRanks; i++) {
    FB_COMMCHECK(ctran_->mapper->deregRemReg(&this->remoteTmpAccessKeys[i]));
  }

  return commSuccess;
}

using ctran::utils::TmpBufSegManager;
commResult_t CtranAlgo::initTmpBufs() {
  if (this->tmpbuf) {
    return commSuccess;
  }

  NcclScubaEvent scubaEvent(kCtranAlgoInitResources, &comm_->logMetaData_);
  scubaEvent.startAndRecord();

  TmpBufSegManager<TmpbufType, TmpbufType::NUM_TMPBUFS> segmentManager;

  // staging buffer for internode communication
  segmentManager.insert(
      TmpbufType::INTERNODE_TMPBUF, NCCL_CTRAN_INTERNODE_TMPBUF_SIZE);

  // temporary buffer for small message communication
  segmentManager.insert(
      TmpbufType::MIN_REG_SRC_TMPBUF, CTRAN_MIN_REGISTRATION_SIZE);
  segmentManager.insert(
      TmpbufType::MIN_REG_DST_TMPBUF, CTRAN_MIN_REGISTRATION_SIZE);

  // counts buffers for GPU resident collectives
  // To be improve: as all2allvDynamicMaxSendcounts size is large, can consider
  // to initilze it with algo type using ifdefine, as dynamic and split only
  // need to be nRanks.
  segmentManager.insert(
      TmpbufType::SENDCOUNTS_TMPBUF,
      sizeof(size_t) * all2allvDynamicMaxSendcounts);
  segmentManager.insert(
      TmpbufType::RECVCOUNTS_TMPBUF,
      sizeof(size_t) * all2allvDynamicMaxSendcounts);

  segmentManager.insert(
      TmpbufType::RING_TMP_SEND_BUF,
      NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS *
          NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE);
  segmentManager.insert(
      TmpbufType::RING_TMP_RECV_BUF,
      NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS *
          NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE);

  // request slab buffer from memory pool
  if (comm_->memCache_) {
    std::stringstream ss;
    ss << "Ctran::InitTmpBuf " << this->comm_->statex_->commHash();
    this->tmpBufKey = ss.str();

    FB_COMMCHECKTHROW_EX(
        comm_->memCache_->getCachedCuMemById(
            this->tmpBufKey,
            &this->tmpbuf,
            /*cuHandle=*/nullptr,
            segmentManager.totalLen,
            &this->comm_->logMetaData_,
            __func__),
        comm_->logMetaData_);
  } else {
    // `ctran::utils::commCudaMalloc` automatically decides whether to use cuMem
    // or cudaMalloc to allocate buffer based on cuMem support
    FB_COMMCHECKTHROW_EX(
        ctran::utils::commCudaMalloc(
            (char**)&this->tmpbuf,
            segmentManager.totalLen,
            &this->comm_->logMetaData_,
            "initTmpBufs"),
        comm_->logMetaData_);
  }
  FB_COMMCHECKTHROW_EX(
      ctran_->mapper->regMem(
          this->tmpbuf,
          segmentManager.totalLen,
          &this->tmpbufSegHdl,
          true,
          true,
          &this->tmpbufRegHdl),
      comm_->logMetaData_);

  // set offsets within the slab buffer for each tmpbuf type
  // note SENDCOUNTS_TMPBUF_CPU is a CPU type buffer, both its size and offset
  // would be zero in this loop if not skipped
  // TODO: move CPU buffers to separate segment management logic
  for (auto i = 0; i < (size_t)TmpbufType::NUM_TMPBUFS; i++) {
    if (i == (size_t)TmpbufType::SENDCOUNTS_TMPBUF_CPU ||
        i == (size_t)TmpbufType::SENDBUFFS_PTR_TMPBUF_CPU ||
        i == (size_t)TmpbufType::SENDINDICES_TMPBUF_CPU ||
        i == (size_t)TmpbufType::SENDINDICES_BLOCKLEN_TMPBUF_CPU) {
      continue;
    }
    const CtranAlgo::TmpbufType type = static_cast<CtranAlgo::TmpbufType>(i);
    const auto& x = segmentManager.getSegInfo(type);
    this->tmpbufSegments[type] = BUFOFFSET(this->tmpbuf, x.offset);
    this->tmpbufSegmentOffsets[type] = x.offset;
  }

  scubaEvent.stopAndRecord();

  return commSuccess;
}

size_t CtranAlgo::getTmpBufOffset(const TmpbufType type) {
  size_t offset = 0;

  auto it = tmpbufSegments.find(type);
  if (it != tmpbufSegments.end()) {
    offset = tmpbufSegmentOffsets.at(type);
  } else {
    FB_ERRORTHROW(
        commInternalError,
        "Failed to find tmpbuf for type {} during getTmpBufOffset",
        static_cast<int>(type));
  }

  return offset;
}

std::tuple<void*, void*> CtranAlgo::getTmpBufInfo(const TmpbufType type) {
  void* buf = nullptr;
  void* bufHdl = this->tmpbufRegHdl;

  auto it = tmpbufSegments.find(type);
  if (it == tmpbufSegments.end()) {
    FB_ERRORTHROW(
        commInternalError,
        "Failed to find tmpbuf for type {} during getTmpBufInfo",
        static_cast<int>(type));
  } else {
    buf = it->second;
  }

  if (type == TmpbufType::SENDCOUNTS_TMPBUF_CPU) {
    bufHdl = nullptr;
  }

  return std::make_tuple(buf, bufHdl);
}

void* CtranAlgo::getTmpBuf(const TmpbufType type) {
  auto [buf, _] = this->getTmpBufInfo(type);

  return buf;
}

std::tuple<void*, struct CtranMapperRemoteAccessKey>
CtranAlgo::getRemoteTmpBufInfo(int peer) {
  // ensure all tmp buffers are exchanged, will be a no-op if already done
  // we only do on demand exchange at here to avoid a race condition for
  // intra-node buff exchange.
  FB_COMMCHECKTHROW_EX(this->exchangePeerTmpbuf(peer), comm_->logMetaData_);

  return std::make_tuple(
      this->remoteTmpbuffs.at(peer), this->remoteTmpAccessKeys.at(peer));
}

std::tuple<void*, struct CtranMapperRemoteAccessKey>
CtranAlgo::getInterNodeTmpBufInfo(int peer) {
  // ensure all tmp buffers are exchanged, will be a no-op if already done
  FB_COMMCHECKTHROW_EX(this->exchangeInterNodeTmpbuf(), comm_->logMetaData_);

  return std::make_tuple(
      this->remoteTmpbuffs.at(peer), this->remoteTmpAccessKeys.at(peer));
}

ctran::algos::allreduce::AllReduceResourceRef&
CtranAlgo::getAllReduceDirectRes() {
  return this->allReduceDirectResource->getRef();
}

commResult_t CtranAlgo::initAllReduceDirectResource(
    int nBlocks,
    cudaStream_t stream) {
  if (this->allReduceDirectResource &&
      this->allReduceDirectResource->isInitialized()) {
    return commSuccess;
  }
  this->allReduceDirectResource =
      std::make_unique<ctran::algos::allreduce::AllReduceResourceImpl>(
          comm_->statex_.get(),
          comm_->ctran_->mapper.get(),
          &comm_->logMetaData_);
  this->allReduceDirectResource->initAllReduceDirectResourceAsync(
      nBlocks, stream);
  return commSuccess;
}

CollType CtranAlgo::getCollType(const std::string& algoStr) {
  if (algoStr == "alltoall") {
    return CollType::ALLTOALL;
  } else if (algoStr == "sendrecv") {
    return CollType::SENDRECV;
  } else if (algoStr == "allreduce") {
    return CollType::ALLREDUCE;
  }

  return CollType::UNKNOWN;
}

commResult_t CtranAlgo::initializeCommAttributesMap() {
  for (auto& commAttrKeyValuePair : NCCL_CTRAN_IB_QP_CONFIG_ALGO) {
    auto coll = getCollType(commAttrKeyValuePair.first);
    auto& statex = comm_->statex_;
    if (coll == CollType::UNKNOWN) {
      CLOGF(
          ERR,
          "CTRAN-ALGO: Unknown collective type {} in pimpl {} commHash {:x}, commDesc {}.",
          commAttrKeyValuePair.first,
          (void*)this,
          statex->commHash(),
          statex->commDesc());
      return commInternalError;
    }
    if (commAttrKeyValuePair.second.size() == kExpectedCommAttrLength) {
      CtranIbConfig config;
      config.numQps =
          std::stoi(commAttrKeyValuePair.second[qpConfigIndex::MAX_QPS]);
      config.qpScalingTh = stoul(
          commAttrKeyValuePair.second[qpConfigIndex::QP_SCALING_THRESHOLD]);
      config.qpMsgs =
          stoi(commAttrKeyValuePair.second[qpConfigIndex::MAX_QP_MSGS]);
      config.trafficClass =
          stoi(commAttrKeyValuePair.second[qpConfigIndex::TRAFFIC_CLASS]);

      if (commAttrKeyValuePair.second[qpConfigIndex::VC_MODE] == "spray") {
        config.vcMode = NCCL_CTRAN_IB_VC_MODE::spray;
      } else if (
          commAttrKeyValuePair.second[qpConfigIndex::VC_MODE] == "dqplb") {
        config.vcMode = NCCL_CTRAN_IB_VC_MODE::dqplb;
      } else {
        CLOGF(
            ERR,
            "CTRAN-ALGO: Invalid VC mode {} in pimpl {} commHash {:x}, commDesc {}.",
            commAttrKeyValuePair.second[qpConfigIndex::VC_MODE],
            (void*)this,
            statex->commHash(),
            statex->commDesc());
        return commInternalError;
      }
      collToVcConfigMap_[coll] = config;
    } else {
      CLOGF(
          ERR,
          "CTRAN-ALGO: Invalid collective->vc config specified for {} in pimpl {} commHash {:x}, commDesc {}. Expected {} parameters but received only {}",
          commAttrKeyValuePair.first,
          (void*)this,
          statex->commHash(),
          statex->commDesc(),
          kExpectedCommAttrLength,
          commAttrKeyValuePair.second.size());
      return commInternalError;
    }
  }
  return commSuccess;
}

CtranIbConfig* FOLLY_NULLABLE CtranAlgo::getCollToVcConfig(CollType coll) {
  return folly::get_ptr(collToVcConfigMap_, coll);
}
