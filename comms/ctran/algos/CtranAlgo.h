// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALGO_H_
#define CTRAN_ALGO_H_

#include <fmt/format.h>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceResourceImpl.h"
#include "comms/ctran/algos/CollUtils.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/utils/logger/Logger.h"

#define CTRAN_MAX_TOTAL_SENDBUFFS 2048
#define CTRAN_MAX_NUM_SPLITS_PER_RANK 1024

#define LOCAL_RANK_TO_DEV_REGION_POS(localRank, ownerLocalRank) \
  (localRank < ownerLocalRank ? localRank : localRank - 1)

enum CollType {
  ALLTOALL = 1,
  SENDRECV = 2,
  ALLREDUCE = 3,
  UNKNOWN = 4,
};
constexpr int kExpectedCommAttrLength = 5;

commResult_t ctranConfigCommAlgoOverride(CtranComm* comm);

class CtranAlgo {
 public:
  CtranAlgo(CtranComm* comm, ICtran* ctran);

  ~CtranAlgo();

  // intialize device state and SharedResource, if needed
  commResult_t initKernelResources();
  // Get device state
  CtranAlgoDeviceState* getDevState();

  // accessing allGatherAlgo
  void setAllGatherAlgo(enum NCCL_ALLGATHER_ALGO algo);
  enum NCCL_ALLGATHER_ALGO getAllGatherAlgo();
  // accessing allReduceAlgo
  void setAllReduceAlgo(enum NCCL_ALLREDUCE_ALGO algo);
  enum NCCL_ALLREDUCE_ALGO getAllReduceAlgo();

  // See Broadcast definition in Broadcast subdirectory
  bool supportBroadcast(
      std::optional<CtranMapperBackend> specifiedBackend) const;
  commResult_t broadcastBinomialTree(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      commDataType_t datatype,
      int root,
      std::shared_ptr<std::atomic_flag> cpuFlag);

  commResult_t initTmpBufs();
  commResult_t initAllReduceDirectResource(int nBlocks, cudaStream_t stream);
  ctran::algos::allreduce::AllReduceResourceRef& getAllReduceDirectRes();
  commResult_t exchangePeerTmpbuf(int peer);
  commResult_t exchangeInterNodeTmpbuf();

  enum class TmpbufType {
    // Temporary buffer for inter-node data staging
    INTERNODE_TMPBUF,

    // Temporary buffer to stage src data for small messages
    MIN_REG_SRC_TMPBUF,

    // Temporary buffer to stage dst data for small messages
    MIN_REG_DST_TMPBUF,

    // Temporary buffer to store sencdounts for inter-node alltoallv_dynamic
    SENDCOUNTS_TMPBUF,

    // Temporary buffer to store recvcounts for alltoallv_dynamic
    // It would be used to store actualrecvcounts for dynamic and split
    // store recvAllSplitLengths for split_non_contig
    RECVCOUNTS_TMPBUF,

    // Temporary buffer to store sendcounts in CPU memory for
    // alltoallv_dynamic
    SENDCOUNTS_TMPBUF_CPU,

    // Temporary buffer to store sendindices in CPU memory for
    // alltoallv_dynamic
    SENDINDICES_TMPBUF_CPU,

    // Temporary buffer to store sendindices in CPU memory for
    // alltoallv_dynamic
    SENDINDICES_BLOCKLEN_TMPBUF_CPU,

    // Temporary buffer to store sendbuffs pointers in CPU memory for
    // alltoallv_dynamic
    SENDBUFFS_PTR_TMPBUF_CPU,

    // Temporary buffer to hold partially/fully reduced results for
    // communication to peers in AllReduce Ring Algorithm
    RING_TMP_SEND_BUF,
    RING_TMP_RECV_BUF,

    NUM_TMPBUFS,
  };

  // Return a pointer to the tmpbuf segment of the requested type.
  void* getTmpBuf(const TmpbufType type);
  // Return a pair of pointers and handle to the tmpbuf segment of the requested
  // type.
  std::tuple<void*, void*> getTmpBufInfo(const TmpbufType type);
  size_t getTmpBufOffset(const TmpbufType type);

  std::tuple<void*, struct CtranMapperRemoteAccessKey> getRemoteTmpBufInfo(
      int peer);
  // dedicate function for inter-node tmpbuf exchange only.
  // This is to avoid a race condition between inter-node and intra-node
  // exchange.
  std::tuple<void*, struct CtranMapperRemoteAccessKey> getInterNodeTmpBufInfo(
      int peer);
  CollType getCollType(const std::string& collStr);
  commResult_t initializeCommAttributesMap();
  CtranIbConfig* getCollToVcConfig(CollType type);

 private:
  class SharedResource;

  commResult_t destroyDevState();
  commResult_t deregRemoteTmpBufs();

  CtranComm* comm_{nullptr};
  // Reference to the ctran object that owns this algo.
  ICtran* ctran_{nullptr};

  // Device buffer to store all states of
  // shared device buffers and comm info, accessed by kernels.
  CtranAlgoDeviceState* devState_d_{nullptr};

  SharedResource* sharedRes_{nullptr};
  bool isResInitialized_{false};
  // local tmpbuf block
  void* tmpbuf{nullptr};
  std::string tmpBufKey;
  // Temporary buffer to store sendcounts locally on CPU
  size_t* sendCountsTmpbufCPU{nullptr};
  size_t* sendIndicesTmpbufCPU{nullptr};
  size_t* sendIndicesBlockLengthsTmpbufCPU{nullptr};
  void** sendbuffsPtrTmpbufCPU{nullptr};

  void* tmpbufRegHdl{nullptr};
  void* tmpbufSegHdl{nullptr};
  std::unordered_map<TmpbufType, size_t> tmpbufSegmentOffsets;
  std::unordered_map<TmpbufType, void*> tmpbufSegments;
  // Store peers' tmpbuf and access key after exchangeTmpbuf
  std::vector<void*> remoteTmpbuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteTmpAccessKeys;

  std::optional<enum NCCL_ALLGATHER_ALGO> allGatherAlgo = std::nullopt;
  std::optional<enum NCCL_ALLREDUCE_ALGO> allReduceAlgo = std::nullopt;
  std::unordered_map<enum CollType, CtranIbConfig> collToVcConfigMap_;
  std::unique_ptr<ctran::algos::allreduce::AllReduceResourceImpl>
      allReduceDirectResource{nullptr};
};

class CtranAlgo::SharedResource {
 public:
  SharedResource(CtranComm* comm);
  ~SharedResource() = default;

  commResult_t release();

  std::vector<void*> mappedDevShmPtrs; // pointer to mapped device memory
                                       // regions of remote peers

  void* devShmPtr{nullptr}; // pointer to local device memory region
 private:
  CtranComm* comm_{nullptr};
  std::unique_ptr<ctran::utils::CtranIpcMem>
      ipcMem_; // local device memory region
  std::unordered_map<int, std::unique_ptr<ctran::utils::CtranIpcRemMem>>
      ipcRemMemMap_; // imported device memory from remote peers
};

class CtranAlgoLogger {
 public:
  CtranAlgoLogger(
      const std::string& name,
      const uint64_t opCount,
      const CtranComm* comm,
      std::optional<const ICtran*> ctran = std::nullopt);

  CtranAlgoLogger(
      const std::string& name,
      const uint64_t opCount,
      const CtranComm* comm,
      const ctran::CtranWin* win,
      std::optional<const ICtran*> ctran = std::nullopt);

  ~CtranAlgoLogger();

 private:
  const std::string name{"unknown"};
  const uint64_t opCount_{0};
  const CtranComm* comm_{nullptr};
  std::optional<const ICtran*> ctran_{std::nullopt};
};

class CtranAlgoRMALogger {
 public:
  CtranAlgoRMALogger(
      const std::string& name,
      const uint64_t opCount,
      const int peerRank,
      const ctran::CtranWin* win,
      const CtranComm* comm);

  ~CtranAlgoRMALogger();

 private:
  const std::string name_{"unknown"};
  const uint64_t opCount_{0};
  const int peerRank_{-1};
  const ctran::CtranWin* win_{nullptr};
  const CtranComm* comm_{nullptr};
};

class CtranPersistentRequest {
 public:
  enum Type {
    ALLGATHER_P,
    ALLTOALL_DEDUP,
    ALLTOALL_P,
    ALLTOALLV_DEDUP,
  };

  Type type;
  CtranComm* comm_{nullptr};
  std::unique_ptr<OpElem> op{nullptr};

  // Attach algorithm specific resource defined within each algorithm module
  void* algo{nullptr};

  cudaStream_t stream;
  void* segHdl{nullptr};

  CtranPersistentRequest(
      Type type,
      CtranComm* comm,
      OpElem* op,
      cudaStream_t stream)
      : type(type), comm_(comm), stream(stream) {
    this->op = std::unique_ptr<OpElem>(op);
  }

  CtranPersistentRequest(
      Type type,
      CtranComm* comm,
      std::unique_ptr<OpElem> op,
      cudaStream_t stream,
      void* segHdl)
      : type(type), comm_(comm), stream(stream), segHdl(segHdl) {
    this->op = std::move(op);
  }

  CtranPersistentRequest(Type type, CtranComm* comm, cudaStream_t stream)
      : type(type), comm_(comm), stream(stream) {}

  ~CtranPersistentRequest() = default;
};

template <>
struct fmt::formatter<CtranPersistentRequest::Type> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(CtranPersistentRequest::Type status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
#endif
