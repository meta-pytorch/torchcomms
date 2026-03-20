// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h" // for CTRAN_MAX_NVL_PEERS
#include "comms/utils/commSpecs.h"

// Forward declarations
struct KernelElem;

namespace comms::pipes {
struct Transport;
}

#define CTRAN_MAX_TOTAL_RANK (128)

namespace ctran {

namespace alltoall {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  commDataType_t datatype;
};

} // namespace alltoall

namespace device_alltoallv_pipes {

struct KernArgs {
  const void* sendbuff;
  void* recvbuff;
  int nLocalRanks; // number of ranks on this node
  int myRank; // global rank of this process
  size_t elementSize; // bytes per element (commTypeSize(datatype))

  // Device pointers to split sizes (int64_t, indexed by global rank)
  const int64_t* sendcounts_d; // [nRanks] send counts per rank
  const int64_t* recvcounts_d; // [nRanks] recv counts per rank

  // Scaling factors for multi-dimensional tensors. For a tensor with shape
  // [N, D1, D2, ..., Dk], the split sizes are in units of dim-0 slices (rows),
  // and each row contains D1*D2*...*Dk elements. The kernel multiplies counts
  // by these factors to get actual element counts. Default is 1 (1D tensors).
  int64_t sendcountsMultiplier;
  int64_t recvcountsMultiplier;

  // Maps local rank index [0..nLocalRanks) to global rank
  int localRankToGlobalRank[CTRAN_MAX_NVL_PEERS];

  // Transport array from MultiPeerTransport, indexed by global rank
  comms::pipes::Transport* transports;

  // If true, use block-level scheduling (make_block_group) instead of
  // warp-level scheduling (make_warp_group). Block scheduling dedicates
  // all threads in a block to one peer; warp scheduling distributes
  // warps across peers for chunk-level pipelining.
  bool useBlockGroup;
};

} // namespace device_alltoallv_pipes

namespace alltoallv {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t selfCount;
  size_t selfSendDispl;
  size_t selfRecvDispl;
  KernelElem* sendElemsList;
  KernelElem* recvElemsList;
};

} // namespace alltoallv

namespace alltoallvdynamic {

struct KernelArgs {
  void** sendbuffsPtrTmpbufCPU{nullptr};
  const size_t* sendcounts{nullptr};
  size_t* sendCountsTmpbufGPU{nullptr};
  size_t* sendCountsTmpbufCPU{nullptr};
  size_t sendcountsLength{0};
  size_t* recvCountsTmpbufGPU{nullptr};
  size_t* actualRecvcounts{nullptr};
  void* recvbuffsPtrGPU[CTRAN_MAX_TOTAL_RANK]{};
  commDataType_t datatype{};
  KernelElem* kElem{nullptr};
  union {
    struct {
      const void* sendbuff{nullptr};
      void** sendbuffsPtrShmDev{nullptr};
    } split;
    struct {
      const void* sendbuffsPtrGPU[CTRAN_MAX_TOTAL_RANK]{};
    } nonSplit;
  };
  union {
    struct {
      const size_t* inputChunkIndices{nullptr};
      size_t* inputChunkIndicesTmpbufCPU{nullptr};
      const size_t* inputChunkCountPerRank{nullptr};
      size_t* inputChunkCountPerRankTmpbufCPU{nullptr};
      size_t maxInputChunkCountPerRank{0};
      size_t maxRecvcount{0};
      size_t maxSendcount{0};
      bool combine;
    } nonContig;
    struct {
    } contig;
  };

  // Default constructor needed because unions with non-trivial member
  // initializers have deleted default constructors
  KernelArgs() {
    // Unions are initialized by their first member by default
    // split and nonContig are already initialized above
  }
};

} // namespace alltoallvdynamic

namespace alltoalldedup {

struct KernelArgs {
  KernelElem* bcastElemList;
  int numIbPeers;
};

} // namespace alltoalldedup

namespace alltoallp {
class AlgoImpl;
} // namespace alltoallp

namespace alltoallvdynamicp {
class AlgoImpl;
} // namespace alltoallvdynamicp

} // namespace ctran
