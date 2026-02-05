// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/utils/commSpecs.h"

// Forward declaration
struct KernelElem;

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
