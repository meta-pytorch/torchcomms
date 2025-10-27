// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/AllReduce/AllReduceDevTypes.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"

// data structure for allreduce GPE side processing
namespace ctran::algos::allreduce {
struct SyncCtrlMsg {
  uint32_t msgId;
};

struct SyncCtrlReq {
  SyncCtrlMsg msg;
  CtranMapperRequest req;
};

struct AllReduceBlockIdBuff {
  uint64_t* blockIdBuff;
  void* regHdl;
  uint64_t* remoteBlockIdBuff;
  struct CtranMapperRemoteAccessKey rKey;
};

struct AllReduceConnector {
  bool netConnect{false};
  struct AllReduceConnInfo conn;
  void* handle;
  void* remoteBuff;
  struct CtranMapperRemoteAccessKey rkey;
  struct AllReduceBlockIdBuff blockIdBuffType;
};

struct SendResources {
  struct ctran::algos::MPSCTbSync<1>* complete;
  struct ctran::algos::MPSCTbSync<1>* post;
  struct NetSize* netSize;
  int localRank;
  int remoteRank;
  uint64_t step{0};
  bool netConnect{false};

  const void* localBuff;
  void* remoteBuff;
  int buffSize;
  void* sendHandle;
  struct CtranMapperRemoteAccessKey rkey;

  struct AllReduceBlockIdBuff blockIdBuffType;
};

struct RecvResources {
  struct ctran::algos::MPSCTbSync<1>* complete;
  struct ctran::algos::MPSCTbSync<1>* post;
  struct NetSize* netSize;
  int localRank;
  int remoteRank;
  uint64_t step{0};
  bool netConnect{false};

  const void* localBuff;
  void* remoteBuff;
  int buffSize;
  void* recvHandle;
  struct AllReduceBlockIdBuff blockIdBuffType;
};

struct AllReducePeer {
  struct AllReduceConnector send{};
  struct AllReduceConnector recv{};
  int rank;
  int refCount;
  struct ctran::algos::topo::CtranRing ring;
};

struct AllReducePeerHost {
  std::vector<struct AllReducePeer> peers;
};

struct AllReduceGPESubArgs {
  int blockId;
  int nsteps;
  ssize_t chunkSize;
  int peer;

  uint64_t base{0};
  uint64_t posted{0};
  uint64_t received{0};
  uint64_t flushed{0};
  uint64_t transmitted{0};
  uint64_t done{0};
  uint64_t end{0};
  std::vector<std::unique_ptr<CtranMapperRequest>> requests;
  void* transportResources;
};

enum State {
  None = 0,
  Ready = 1,
  Progress = 2,
};

struct AllReduceGPEArgs {
  struct AllReduceGPESubArgs subArgs[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  int nsubs{0};
  int done{0};
  int sliceSteps{0};
  int chunkSteps{0};
  size_t chunkSize{0};
  size_t count{0};
  State state{State::None};
  std::unique_ptr<CtranMapperNotify> recvNotify;
  std::unique_ptr<CtranMapperNotify> posNotify;
  std::vector<std::unique_ptr<SyncCtrlReq>> bufSyncReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> sendReqs;

  uint64_t nsteps{0};
  uint64_t base{0};
  uint64_t posted{0};
  uint64_t received{0};
  uint64_t flushed{0};
  uint64_t transmitted{0};
  uint64_t completed{0};
  uint64_t* blockIdBuff;
  void* recvRouterHandle;
};

struct AllReduceGPERes {
  struct AllReducePeerHost* peers;
  struct BlockAllocation blockAlloc;
  SendResources* sendResources;
  RecvResources* recvResources;
};
} // namespace ctran::algos::allreduce
