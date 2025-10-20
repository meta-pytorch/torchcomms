// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sys/types.h>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/MPSCTbSync.h"
#include "comms/ctran/algos/topo/CtranRingBuilder.h"

#define CTRAN_ALLREDUCE_BUFF_SIZE (1 << 22)
#define CTRAN_ALLREDUCE_STEPS (8)
#define CTRAN_ALLREDUCE_CHUNKSTEPS (CTRAN_ALLREDUCE_STEPS / 2)
#define CTRAN_ALLREDUCE_SLICESTEPS (CTRAN_ALLREDUCE_STEPS / 4)
#define NET_CONNECT (0x04)

namespace ctran::algos::allreduce {

// buff size sync between GPE and Kernel
struct NetSize {
  ssize_t size{-1};
};

// list of buff size sync between GPE and Kernel.
// used for pipelining
struct NetSizeMem {
  struct NetSize sizes[CTRAN_ALLREDUCE_STEPS];
};

// device side connection info
struct AllReduceConnInfo {
  void* buff; // Local for recv, remote for send
  int step{0}; // Keep where we are
  ctran::algos::MPSCTbSync<1>* post;
  ctran::algos::MPSCTbSync<1>* complete;
  int flags{0}; // Direct communication / other flags
  size_t stepSize; // Step size for the buffer
  struct NetSize* netSize;
};

// per peer connection info
// one connection could have one send and one recv.
struct alignas(16) AllReduceDevConn {
  struct AllReduceConnInfo send;
  struct AllReduceConnInfo recv;
};

// per block connection info
// one block could have multiple connections
struct alignas(16) AllReduceDevBlock {
  struct AllReduceDevConn** peers;
  struct ctran::algos::topo::CtranRing ring;
  int rank;
  int nRanks;
};

// per rank allreduce
struct alignas(16) AllReduceComm {
  struct AllReduceDevBlock blocks[CTRAN_ALGO_MAX_THREAD_BLOCKS];
};

struct BlockAllocation {
  int nBlocks;
  size_t bytesPerBlock[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  size_t gridOffsetSize[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  size_t chunkSize;
};

struct alignas(16) AllReduceKernelArgs {
  struct AllReduceComm* allReduceComm;
  const void* sendbuff;
  void* recvbuff;
  const size_t count;
  const int typeSize;
  struct BlockAllocation blockAlloc;
};

// share memory for allreduce kernel
struct alignas(16) AllReducShm {
  alignas(16) struct AllReduceDevBlock block;
  void* srcs[CTRAN_MAX_NVL_PEERS];
  void* dsts[CTRAN_MAX_NVL_PEERS];
  void* userInput;
  void* userOutput;
  int blockId;
};

} // namespace ctran::algos::allreduce
