// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include <fmt/format.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/CtranExImpl.h"
#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/Broadcast/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/ReduceScatter/Types.h"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/window/CtranWin.h"

typedef commResult_t (*opFunc)(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup);

namespace ctran {
using PersistentObj = std::variant<
    std::monostate,
    std::unique_ptr<alltoallp::AlgoImpl>,
    std::unique_ptr<alltoallvdynamicp::AlgoImpl>>;
using PreLaunchGraphPrepareFn =
    commResult_t (*)(opFunc& opFunc, struct OpElem* op, PersistentObj& pObj);
} // namespace ctran

struct OpElem {
  enum opType {
    ALLGATHER,
    ALLGATHERP_INIT,
    ALLGATHERP,
    ALLREDUCE,
    SEND,
    RECV,
    ALLTOALL,
    ALLTOALLP,
    ALLTOALLV,
    ALLTOALLV_DYNAMIC,
    ALLTOALLV_DYNAMIC_SPLIT,
    ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
    ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG_P,
    ALLTOALL_DEDUP,
    ALLTOALLV_DEDUP,
    BROADCAST,
    REDUCESCATTER,
    PUTNOTIFY,
    WAITNOTIFY,
    PUTSIGNAL,
    WAITSIGNAL,
    SIGNAL,
    GET
  } type;
  cudaStream_t stream;

  CtranComm* comm_{nullptr};
  ICtran* ctran{nullptr};
  // Copied after collective called ctran->updateOpCount()
  // Upon collective submission, we should always use the copied opCount since
  // original opCount in comm may be updated by other threads.
  uint64_t opCount{0};
  // Whether the op is for device or host memory.
  // If true, stream must be a valid cuda stream; otherwise, it is unused.
  bool isDevice{true};

  // TCP Device Memory unpack pool for this operation.
  // Allocated by prepareUnpackConsumer() and freed during GPE kernel teardown.
  // Used by algorithm implementations to populate CtranMapperContext and
  // pass it down to CtranTcpDm::irecvConnected().
  void* unpackPool{nullptr};

  union {
    struct {
      const void* sendbuff;
      void* recvbuff;
      size_t sendcount;
      commDataType_t datatype;
      KernelElem* bcastElem;
    } allgather;
    struct {
      // reference to pre-initialized persistent arguments and resource
      void* pArgs;
    } allgatherp_init;
    struct {
      // reference to pre-initialized persistent arguments and resource
      void* pArgs;
      void* algoResource;
      // non-persistent
      const void* sendbuff;
      size_t count;
      commDataType_t datatype;
    } allgatherP;
    struct {
      const void* sendbuff;
      void* recvbuff;
      size_t count;
      commDataType_t datatype;
      commRedOp_t op;
      std::unordered_map<int, KernelElem*> kElemStepMap;
      size_t tmpbuffSize; // size of tmpbuff
      void* sendHdl;
      void* recvHdl;
      std::vector<void*> remoteRecvBuffs;
      std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;

      void* args;
      void* resource;
    } allreduce;
    struct {
      const void* sendbuff;
      std::atomic<void*>* recvbuff;
      struct CtranMapperRemoteAccessKey remoteAccessKey;
      size_t count;
      commDataType_t datatype;
      int peerRank;
      // for coordinating NVL put with kernel
      KernelElem* kElem;
    } send;
    struct {
      void* recvbuff;
      size_t count;
      commDataType_t datatype;
      int peerRank;
      // for coordinating NVL waitNotify with kernel
      KernelElem* kElem;
    } recv;
    struct {
      const void* sendbuff;
      void* recvbuff;
      size_t count;
      commDataType_t datatype;
    } alltoall;
    struct {
      // persistent
      void* pArgs;
      // non-persistent
      const void* sendbuff;
      size_t count;
    } alltoallP;
    struct {
      const void* sendbuff;
      std::vector<size_t> sendcounts;
      std::vector<size_t> sdispls;
      void* recvbuff;
      std::vector<size_t> recvcounts;
      std::vector<size_t> rdispls;
      commDataType_t datatype;
    } alltoallv;
    struct {
      const void* const* sendbuffs;
      void* const* recvbuffs;
      void* recvbuff;
      commDataType_t datatype;
      size_t sendcountsLength;
      size_t maxSendcount;
      size_t maxRecvcount;
      KernelElem* kElem;
      // Persistent args for persistent alltoallv_dynamic.
      void* pArgs;
      bool combine;
    } alltoallv_dynamic;
    struct {
      const void* sendbuff;
      const size_t* sendcounts;
      const size_t* sdispls;
      void* recvbuff;
      const size_t* recvcounts;
      const size_t* rdispls;
      commDataType_t datatype;
      std::unordered_map<int, KernelElem*> bcastElemMap;
      void* sendHdl;
      void* recvHdl;
      std::vector<void*> remoteRecvBuffs;
      std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;
    } alltoall_dedup;
    struct {
      // Reference to persistent algo fields
      void* pArgs;
      void* algoResource;
      void* algoConfig;
      void* perfTracer;
    } alltoallv_dedup_exec;
    struct {
      const void* sendbuff;
      void* recvbuff;
      size_t count;
      commDataType_t datatype;
      int root;
      std::unordered_map<int, KernelElem*> putNotifyMap;
      std::unordered_map<int, KernelElem*> waitNotifyMap;
    } broadcast;
    struct {
      const void* sendbuff;
      void* recvbuff;
      size_t recvcount;
      commDataType_t datatype;
      commRedOp_t redOp;
      std::vector<KernelElem*> intraReduce;
      KernelElem* interReduce;
    } reducescatter;
    struct {
      const void* sendbuff;
      size_t count;
      commDataType_t datatype;
      size_t targetDisp;
      int peerRank;
      ctran::CtranWin* win;
      bool notify;
    } putnotify;
    struct {
      int peerRank;
      ctran::CtranWin* win;
    } waitnotify;
    struct {
      const void* sendbuff;
      size_t targetDisp;
      size_t count;
      commDataType_t datatype;
      uint64_t* signalAddr;
      uint64_t signalVal;
      int peerRank;
      ctran::CtranWin* win;
    } putsignal;
    struct {
      const uint64_t* signalAddr;
      uint64_t cmpVal;
      ctran::CtranWin* win;
    } waitsignal;
    struct {
      uint64_t* signalAddr;
      uint64_t signalVal;
      int peerRank;
      ctran::CtranWin* win;
    } signal;
    struct {
      void* recvbuff;
      size_t targetDisp;
      size_t count;
      commDataType_t datatype;
      int peerRank;
      ctran::CtranWin* win;
    } get;
  };

 public:
  OpElem(enum opType type, CtranComm* comm, uint64_t opCount);

  OpElem(enum opType type, CtranComm* comm, ICtran* ctran, uint64_t opCount);

  OpElem(OpElem* op);

  OpElem(
      enum opType type,
      cudaStream_t stream,
      CtranComm* comm,
      uint64_t opCount);

  OpElem(
      enum opType type,
      cudaStream_t stream,
      CtranComm* comm,
      ICtran* ctran,
      uint64_t opCount);
  ~OpElem();

  void setStatus(KernelElem::ElemStatus status);
};

struct KernelConfig {
  enum KernelType {
    ALLGATHERP,
    ALLGATHERP_INIT,
    ALLGATHER,
    ALLREDUCE,
    SEND,
    RECV,
    SENDRECV,
    SEND_NOTIFY,
    RECV_NOTIFY,
    SENDRECV_NOTIFY,
    RECV_UNPACK,
    SENDRECV_UNPACK,
    SENDRECV_STAGED,
    SENDRECV_P2P,
    ALLTOALL,
    ALLTOALLV,
    ALLTOALLV_DYNAMIC,
    ALLTOALLV_DYNAMIC_SPLIT,
    ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
    ALLTOALL_DEDUP,
    ALLTOALLV_DEDUP,
    BROADCAST,
    BROADCAST_UNPACK,
    REDUCESCATTER,
    PUTNOTIFY,
    WAITNOTIFY,
    PUTSIGNAL,
    WAITSIGNAL,
    SIGNAL,
    GET
  } type;
  unsigned int numBlocks{1};
  unsigned int numThreads{1};

  cudaStream_t stream;
  CtranKernelArgs args;
  // Pointer to argument struct specific to each algorithm
  void* algoArgs{nullptr};
  void* unpackPool{nullptr};

  const std::string algoName;
  // Copied after collective called ctran->updateOpCount()
  // Upon collective submission, we should always use the copied opCount since
  // original opCount in comm may be updated by other threads.
  const uint64_t opCount;
  bool isDevice{true};

  // Experimental: allows one-sided communications, waitSignal and
  // multiWaitSignal, to run in parallel with other kernels when
  // launched on a single GPE thread.
  bool canConcurrent{false};

 public:
  KernelConfig(
      enum KernelType type,
      cudaStream_t stream,
      const std::string& algoName,
      const uint64_t opCount)
      : type(type), stream(stream), algoName(algoName), opCount(opCount) {};
  KernelConfig(
      enum KernelType type,
      cudaStream_t stream,
      const std::string& algoName,
      void* algoArgs,
      const uint64_t opCount)
      : type(type),
        stream(stream),
        algoArgs(algoArgs),
        algoName(algoName),
        opCount(opCount) {};
  std::string toString();
  ~KernelConfig() {};
};

template <>
struct fmt::formatter<KernelConfig::KernelType> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(KernelConfig::KernelType status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

class CtranGpe {
 public:
  CtranGpe(int cudaDev, CtranComm* comm);
  ~CtranGpe();

  // Submit device mem communication. A kernel will be launched and host side
  // func will be submitted to the GPE thread.
  // Completion of the operation is tracked by stream.
  commResult_t submit(
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      KernelConfig& kernelConfig,
      const void* ncclKernel,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt,
      ctran::PreLaunchGraphPrepareFn graphPrepareFn = nullptr);

  // Submit host mem communication. No kernel is launched, and only the host
  // side func will be submitted to the GPE thread. Also the op won't be
  // captured by cudagraph.
  // Completion of the operation is tracked by cpuFlag. cpuFlag can be nullptr,
  // indicating that the caller doesn't care about the completion of the
  // operation.
  commResult_t submitHost(
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      KernelConfig& kernelConfig,
      std::shared_ptr<std::atomic_flag> cpuFlag);

  // Allocate numElems number of p2pElem objects from internal pool.
  // When free objects are not enough, it will be in blocking wait and reclaim
  // inuse p2pElems till enough objects are available. Return commSuccess if all
  // elements are allocated, otherwise return commInternalError. Input
  // arguments:
  //   - numElems: number of p2pElem objects to be allocated
  //   - ngroups: number of thread block groups to use each p2pElem object
  // Output arguments:
  //   - elemsList: a C-style list of p2pElem objects being accessed in kernel
  commResult_t
  allocKernelElems(size_t numElems, int ngroups, KernelElem** elemsList);

  // Return number of inuse kernel elements.
  // Used to check potential kelem leak in UT due to inproper usage in ctran
  // algorithm.
  size_t numInUseKernelElems();

  // Return number of inuse kernel flags.
  // Used to check potential flag leak in UT due to inproper usage in ctran
  size_t numInUseKernelFlags();

  commResult_t allocGpeKernelSyncs(
      size_t count,
      int nworkers,
      std::vector<ctran::algos::GpeKernelSync*>& gpeKernelSyncs);

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

static inline void ctranKernelSetAllGatherArgs(
    const void* sendbuff,
    void* recvbuff,
    commDataType_t datatype,
    size_t count,
    CtranAlgoDeviceState* devState_d,
    CtranKernelArgs* args) {
  args->devState_d = devState_d;
  args->collective.allgather.sendbuff = sendbuff;
  args->collective.allgather.recvbuff = recvbuff;
  args->collective.allgather.datatype = datatype;
  args->collective.allgather.count = count;
}

extern __global__ void
ncclKernelNvlBarrier(int rank, int nLocalRanks, CtranAlgoDeviceState* devState);

template <typename T>
extern __global__ void ncclKernelAllGatherCtranDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allgather::KernelArgs args);

__global__ void ncclKernelAllGatherCtranRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allgather::KernelArgs args);

__global__ void ncclKernelAllGatherCtranRecDbl(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allgather::KernelArgs args);

template <typename T, commRedOp_t RedOp>
__global__ void ncclKernelAllReduceCtranDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allreduce::KernelArgs args);

extern __global__ void ncclKernelSend(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendArgs args);

template <bool UNPACK>
extern __global__ void ncclKernelRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelRecvArgs args);

template <bool UNPACK>
extern __global__ void ncclKernelSendRecv(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendRecvArgs args);

extern __global__ void ncclKernelSendNotifyOnly(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendArgs args);

extern __global__ void ncclKernelRecvNotifyOnly(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelRecvArgs args);

extern __global__ void ncclKernelSendRecvNotifyOnly(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernelSendRecvArgs args);

extern __global__ void ncclKernelSendRecvStaged(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernArgs args);

extern __global__ void ncclKernelSendRecvP2p(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::sendrecv::KernArgs args);

template <bool UNPACK>
__global__ void ncclKernelBroadcast(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::broadcast::KernelArgs args);

template <typename T>
extern __global__ void ncclKernelAllToAll(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoall::KernelArgs args);

template <typename T>
extern __global__ void ncclKernelAllToAllv(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoallv::KernelArgs args);

template <typename T>
extern __global__ void ncclKernelAllToAllvDynamic(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoallvdynamic::KernelArgs args);

template <typename T>
extern __global__ void ncclKernelAllToAllvDynamicSplit(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoallvdynamic::KernelArgs args);

template <typename T>
extern __global__ void ncclKernelAllToAllvDynamicSplitNonContig(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoallvdynamic::KernelArgs args);

template <typename T>
extern __global__ void ncclKernelAllToAllDedup(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::alltoalldedup::KernelArgs args);

template <typename T, commRedOp_t RedOp>
__global__ void ncclKernelReduceScatterDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::reducescatter::KernelArgs args);

template <typename T, commRedOp_t RedOp>
__global__ void ncclKernelReduceScatterRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::reducescatter::KernelArgs args);

template <typename T, commRedOp_t RedOp>
__global__ void ncclKernelReduceScatterRHD(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::reducescatter::KernelArgs args);

#endif
