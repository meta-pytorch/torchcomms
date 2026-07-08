// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_GPE_H_
#define CTRAN_GPE_H_

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include <fmt/format.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/Broadcast/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/ReduceScatter/Types.h"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/profiler/IGpeProfilerReporter.h"
#include "comms/ctran/utils/PinnedHostPool.h"
#include "comms/ctran/window/CtranWin.h"

typedef commResult_t (*opFunc)(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup);

// Pinned-host pool of GpeKernelSync slots. Declared here so consumers
// (notably the host-IB CB transport) can hold a borrowed pool pointer
// without pulling in CtranGpeImpl.h. Mirrored exactly in
// CtranGpeImpl.h:209 — both must alias the same instantiation.
using GpeKernelSyncPool = PinnedHostPool<ctran::algos::GpeKernelSync>;

namespace ctran {
using PersistentObj =
    std::variant<std::monostate, std::unique_ptr<alltoallp::AlgoImpl>>;
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
    DEVICE_ALLTOALLV,
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
  // Allocated by prepareUnpackConsumer(). Eager operations return it during
  // GPE kernel teardown; CUDA graph captures keep it until graph destruction.
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

#if !defined(__CUDACC__)
      // Ring algorithm host state. Owned by OpElem — constructed in
      // OpElem(), destroyed in ~OpElem(). Unused by ctdirect.
      ctran::allreduce::ring::HostArgs hostArgs;
      ctran::allreduce::ring::HostResource hostResource;
#endif
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
      const void* sendbuff;
      void* recvbuff;
      const int64_t* sendcounts_d; // device pointer
      const int64_t* recvcounts_d; // device pointer
      commDataType_t datatype;
    } device_alltoallv;
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
      uint64_t* signalCounter;
      int peerRank;
      ctran::CtranWin* win;
    } putsignal;
    struct {
      const uint64_t* signalAddr;
      uint64_t* signalCounter;
      ctran::CtranWin* win;
    } waitsignal;
    struct {
      // Remote peer's signal buffer slot for RDMA atomicSet (IB path).
      uint64_t* signalAddr;
      uint64_t* signalCounter;
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
    RECV_UNPACK,
    SENDRECV_UNPACK,
    SENDRECV_P2P,
    ALLTOALL,
    DEVICE_ALLTOALLV,
    ALLTOALLV,
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

  // Post-kernel cleanup callback. Populated by algorithm setup (e.g.,
  // SendRecv P2P useList path); transferred to CtranGpeCmd on submit,
  // invoked by GPE thread after the kernel signals completion via
  // kernelFlag.
  std::function<void()> postKernelCleanup{nullptr};

  // KernelElems marked persistent during graph capture by allocKernelElems().
  // submit() retains cleanup on the graph for the no-cmd (empty opGroup) path.
  // For the cmd path, ~OpElem handles free() and this vector is ignored.
  std::vector<KernelElem*> persistentKernelElems;

  const std::string algoName;
  // Copied after collective called ctran->updateOpCount()
  // Upon collective submission, we should always use the copied opCount since
  // original opCount in comm may be updated by other threads.
  const uint64_t opCount;
  bool isDevice{true};

  // Dynamic shared memory size override. 0 = use sizeof(CtranAlgoDeviceState).
  size_t dynamicSharedMemBytes{0};

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
  // Optional reporter injection for tests. The cvar
  // NCCL_CTRAN_GPE_PROFILING_ENABLE gates whether any reporter is wired
  // through to the profiler at all: when false (default), the reporter
  // is nulled regardless of what's passed here. When true, a passed
  // reporter is used as-is, and nullptr is replaced with the production
  // DefaultGpeProfilerReporter (Scuba flush).
  CtranGpe(
      int cudaDev,
      CtranComm* comm,
      std::unique_ptr<ctran::IGpeProfilerReporter> reporter = nullptr);
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
  commResult_t
  allocKernelElems(size_t numElems, int ngroups, KernelElem** elemsList);

  // Return number of inuse kernel elements.
  // Used to check potential kelem leak in UT due to inproper usage in ctran
  // algorithm.
  size_t numInUseKernelElems();

  // Return number of inuse kernel flags.
  // Used to check potential flag leak in UT due to inproper usage in ctran
  size_t numInUseKernelFlags();

  // Return number of inuse checksums.
  size_t numInUseChecksums();

  // Return number of inuse GpeKernelSync elements.
  // Used to verify that CUDA graph cmdDestroy callbacks have released all pool
  // elements before pool destruction (async cmdDestroy race).
  size_t numInUseGpeKernelSyncs();

  commResult_t allocGpeKernelSyncs(
      size_t count,
      int nworkers,
      std::vector<ctran::algos::GpeKernelSync*>& gpeKernelSyncs);

  // Borrowed pointer to the underlying pool of pinned-host GpeKernelSync
  // objects. Used by the host-IB CB transport (HostCbTransport) to
  // pop slot-syncs at construction and reset() them in its destructor.
  // Lifetime: pool is owned by CtranGpe; must outlive every consumer.
  // The mapper enforces this by clearing its per-peer host-transport
  // cache in setAtDestruction() before gpe is torn down.
  GpeKernelSyncPool* gpeKernelSyncPool();

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
