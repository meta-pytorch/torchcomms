// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_GPE_DEV_H_
#define CTRAN_GPE_DEV_H_

#include <fmt/format.h>
#include <stdint.h>

#include "comms/ctran/algos/CtranAlgoArgDev.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/commSpecs.h"

#ifdef CTRAN_DISABLE_TCPDM
#include "comms/ctran/backends/mock/CtranTcpDmBaseMock.h"
#else
#include "comms/tcp_devmem/unpack/batch_unpack_kernel.h"
#endif

// Used for ngroups value checking only. For H100, >128 is not possible.
#define MAX_NGROUPS (128)
#define CTRAN_MAX_TOTAL_RANK (128)

struct alignas(16) KernelElem {
  enum ElemStatus {
    RESET,
    INUSE, // marked as inuse when submitting with a GPE kernel
    POSTED, // posted to kernel
    REVOKED, // revoked by GPE thread after kernel launching (e.g., buffer is
             // allocated by cudaMalloc and does not qualify direct put via
             // NVL); kernel shall skip this op
    DONE, // optional state for kernel to handover back to GPE thread after
          // kernel side work
  };

  union {
    struct {
      size_t count{0};
      size_t displ{0};
      int peerRank{-1};
    } staged;
    struct {
      const void* sendbuff{nullptr};
      // addr mapped to remote receive buffer
      volatile uint64_t recvbuff{0};
      size_t nbytes{0};
      // actual number of groups used for each put
      int ngroups{0};
      bool notify{false};
      // kernel can notify peer once finished
      int peerLocalRank{-1};
    } putNotify;
    struct {
      const void* recvbuff{nullptr};
      size_t nbytes{0};
      // kernel can wait notify from peer
      int peerLocalRank{-1};
      // actual number of groups used for the remote put
      int ngroups{0};
    } waitNotify;
    alignas(16) volatile CtranAlgoDevReduceArg reduce;
    // Reduce with multiple strided blocks from multiple strided segment
    // starting from stridedSrc, final result is updated to dst.
    // - stride defines the distance bewteen the start of each
    //   block in number of elements. E.g., count=2, numBlocks=4, stride=4
    //   defines 4 blocks in stridedSrc, as [0,1], [4,5], [8,9], [12,13].
    // - dst is with blockCount elements.
    // - if inplaceBlockIdx is set >=0, tread dst as an inplace block at the
    //   specified index of stridedSrc.
    struct {
      size_t volatile blockCount{0}; // count in elements of each block
      int numBlocks{0}; // number of blocks in stridedSrc
      size_t volatile stride{0}; // stride in count of element
      void* volatile stridedSrc{nullptr};
      void* volatile dst{nullptr};
      int inplaceBlockIdx{-1};

      // Whether the kernel performs a memory fence after reduce.
      // It ensures data become visible to other device/host/NIC.
      bool flushMem{false};
      // Whether the kernel performs a barrier among nvectors of local ranks
      // after reduce. It ensures the local and all peer ranks have finished.
      bool barrier{false};
    } stridedReduce;
    struct {
      size_t nvectors{0};
      size_t volatile count{0};
      const void* volatile srcs[CTRAN_MAX_NVL_PEERS]{nullptr};
      void* volatile dst{nullptr};
    } localReduce;
    alignas(16) volatile CtranAlgoDevBcastArg bcast;
  };

  KernelElem(){};

  // number of thread blocks to launch the kernel.
  // Set by algorithm when submitting a GPE kernel; status update between GPE
  // and kernel need update with all groups
  int ngroups{0};
  // set to INUSE when submitting with a GPE kernel; set to RESET by when
  // finished use. See additional status in ElemStatus.
  volatile int status[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  // for posting the same elem multiple times
  volatile int stepDone{0};
  // allow kernel to access next element in the list
  KernelElem* next{nullptr};

  // CPU side calls to manage the lifetime of the element and coordinate with
  // kernel. Check if the element is free and ready to be reclaimed.
  bool isFree();

  // Free element from host side if it is unused (status == RESET), or posted
  // and completed (status == DONE) at end of collective.
  // - For REVOKED element, it is freed by kernel directly
  // - For RESET (already freed) element, it is a no-op
  // - For any other status (POST or INUSE), it indicates a leak since
  //   collective is responsible for handling all allocated elements. If found,
  //   program abort to raise bug early.
  // Called at ~OpElem when the associated GPE operation is released.
  void free();

  // Mark an element as unused. It allows free() to reclaim it at end of
  // collective.
  void unuse();

  // Set element status
  void setStatus(ElemStatus status);

  // Post an updated element to the kernel.
  void post(int groupId = -1);

  // Revoke a p2p element before post. Kernel shall skip the op.
  void revoke();

  // CPU side checks whether the element has finished (not yet freed)
  bool isComplete(int groupId = -1);

  // CPU side waits for the element to complete (not yet freed).
  // NOTE: it is risky to call it while outstanding network operations exist and
  // need make progress. It can be safely called only when algorithm ensures no
  // network progress is needed.
  void wait(int groupId = -1);
};

template <>
struct fmt::formatter<KernelElem::ElemStatus> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(KernelElem::ElemStatus status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

struct CtranKernelAllGatherArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t count;
  KernelElem* bcastElem;
};

#define ALLREDUCE_MAX_KERNEL_ELEMS (8)
struct CtranKernelAllReduceArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  commRedOp_t redOp;
  size_t count;
  size_t nSteps;
  void* tmpbuff;
  size_t tmpbuffSize;
  // IPC imported ptr to each of the local peers' tmpRecvBuff
  void* intraNodeRemoteTmpRecvBuffs[CTRAN_MAX_NVL_PEERS];
  // IPC imported ptr to each of the local peers' RecvBuff
  void* intraNodeRemoteRecvBuffs[CTRAN_MAX_NVL_PEERS];
  KernelElem* kernelElems[ALLREDUCE_MAX_KERNEL_ELEMS];
};
enum class AllReduceKernElemRole {
  kIntraReduceScatter,
  kInterReduceScatter,
  kIntraAllGather,
  kRemIntraReduce,
  kRemIntraBcast,
  kRemInterReduce
};

struct CtranKernelSendArgs {
  // List of send p2p elements each will be transferred via NVL copy
  KernelElem* putNotifyList;
  // used for checksum
  const void* sendbuff;
  commDataType_t datatype;
  size_t count;
};

struct CtranKernelRecvArgs {
  KernelElem* waitNotifyList;
  // used for checksum
  const void* recvbuff;
  commDataType_t datatype;
  size_t count;
  SQueues unpack; // TCP Device Memory
};

struct CtranKernelSendRecvArgs {
  KernelElem* putNotifyList;
  KernelElem* waitNotifyList;
  SQueues unpack; // TCP Device Memory
};

struct CtranKernelAllToAllArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  commDataType_t datatype;
};

struct CtranKernelAllToAllvArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t selfCount;
  size_t selfSendDispl;
  size_t selfRecvDispl;
  KernelElem* sendElemsList;
  KernelElem* recvElemsList;
};

struct CtranKernelAllToAllDedupArgs {
  KernelElem* bcastElemList;
  int numIbPeers;
};

struct CtranKernelAllToAllvDynamicArgs {
  void** sendbuffsPtrTmpbufCPU;
  const size_t* sendcounts;
  size_t* sendCountsTmpbufGPU;
  size_t* sendCountsTmpbufCPU;
  size_t sendcountsLength;
  size_t* recvCountsTmpbufGPU;
  size_t* actualRecvcounts;
  void* recvbuffsPtrGPU[CTRAN_MAX_TOTAL_RANK];
  commDataType_t datatype;
  KernelElem* kElem;
  union {
    struct {
      const void* sendbuff;
      void** sendbuffsPtrShmDev;
    } split;
    struct {
      const void* sendbuffsPtrGPU[CTRAN_MAX_TOTAL_RANK];
    } nonSplit;
  };
  union {
    struct {
      const size_t* sendIndices;
      size_t* sendIndicesTmpbufCPU;
      const size_t* sendIndicesBlockLengths;
      size_t* sendIndicesBlockLengthsTmpbufCPU;
      size_t maxSendIndicesBlockLength;
      size_t maxRecvcount;
      size_t maxSendcount;
    } nonContig;
    struct {
    } contig;
  };
};

struct CtranKernelBroadcastArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t count;
  KernelElem* putNotifyList;
  KernelElem* waitNotifyList;
  SQueues unpack; // TCP Device Memory
};

struct CtranKernelReduceScatterArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t recvcount;
  bool stageCopy;
  KernelElem* intraReduce;
  // Reuse single interReduce for number of interNode reduce steps
  int nStepsInterReduce;
  KernelElem* interReduce;
};

struct CtranKernelPutNotifyArgs {
  bool isDirect;
  int peerLocalRank;
};

struct CtranKernelWaitNotifyArgs {
  bool isDirect;
  int peerLocalRank;
};

struct CtranKernelGetArgs {
  bool isDirect;
  int peerLocalRank;
};

struct CtranKernelArgs {
  CtranAlgoDeviceState* devState_d{nullptr};
  union {
    CtranKernelAllGatherArgs allgather;
    CtranKernelAllReduceArgs allreduce;
    CtranKernelSendArgs send;
    CtranKernelRecvArgs recv;
    CtranKernelSendRecvArgs sendrecv;
    CtranKernelAllToAllArgs alltoall;
    CtranKernelAllToAllvArgs alltoallv;
    CtranKernelAllToAllvDynamicArgs alltoallv_dynamic;
    CtranKernelAllToAllDedupArgs alltoall_dedup;
    CtranKernelBroadcastArgs broadcast;
    CtranKernelReduceScatterArgs reducescatter;
    CtranKernelPutNotifyArgs putnotify;
    CtranKernelWaitNotifyArgs waitnotify;
    CtranKernelGetArgs get;
  } collective;
};

#endif
