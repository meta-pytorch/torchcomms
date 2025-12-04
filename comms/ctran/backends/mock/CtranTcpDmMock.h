// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/backends/mock/CtranTcpDmBaseMock.h"

namespace ctran {

class CtranTcpDm {
 public:
  CtranTcpDm(CtranComm* comm, CtranCtrlManager* ctrlMgr) {}
  ~CtranTcpDm() {}

  commResult_t preConnect(const std::unordered_set<int>& peerRanks) {
    return commInvalidUsage;
  }

  static commResult_t
  regMem(const void* buf, const size_t len, const int cudaDev, void** handle) {
    return commInvalidUsage;
  }

  static commResult_t deregMem(void* handle) {
    return commInvalidUsage;
  }

  commResult_t isend(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      CtranTcpDmRequest& req) {
    return commInvalidUsage;
  }

  commResult_t irecv(
      int peerRank,
      void* handle,
      void* data,
      size_t size,
      CtranTcpDmRequest& req,
      int unpackPoolId) {
    return commInvalidUsage;
  }

  commResult_t iput(
      const void* sbuf,
      void* dbuf,
      std::size_t len,
      int peerRank,
      void* tcpdmRegElem,
      bool notify,
      CtranTcpDmConfig* config,
      CtranTcpDmRequest* req) {
    return commInvalidUsage;
  }

  commResult_t irecvCtrlMsg(
      [[maybe_unused]] ControlMsg& msg,
      [[maybe_unused]] int peerRank,
      CtranTcpDmRequest& req) {
    return commInvalidUsage;
  }

  commResult_t isendCtrlMsg(
      [[maybe_unused]] const ControlMsg& msg,
      [[maybe_unused]] int peerRank,
      CtranTcpDmRequest& req) {
    return commInvalidUsage;
  }

  // irecv operations can not proceed unless the peer has been connected.
  // When there is no peer, irecv operations are queued and progress()
  // has to be called to make progress on them.
  commResult_t progress() {
    return commInvalidUsage;
  }

  // Export the location of GPU kernel consumer queues.
  commResult_t
  prepareUnpackConsumer(SQueues* sqs, size_t blocks, int* poolIndex = nullptr) {
    return commInvalidUsage;
  }
  commResult_t teardownUnpackConsumer(int poolIndex) {
    return commInvalidUsage;
  }
};
} // namespace ctran
