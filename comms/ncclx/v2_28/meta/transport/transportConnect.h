// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"

#include <folly/container/F14Map.h>
#include <array>
#include <sstream>
#include <string>

namespace ncclx {
// masks to indicate which algorithms, channels, connectors are
// required to be connected at runtime
struct ncclxPeerReConnInfo {
  std::array<bool, NCCL_NUM_ALGORITHMS> algoMask{false};
  std::array<uint64_t, NCCL_MAX_CONNS> sendChannelMask{0};
  std::array<uint64_t, NCCL_MAX_CONNS> recvChannelMask{0};

  void mark(bool isSend, int channelId, int connIndex, int algorithm) {
    if (isSend) {
      sendChannelMask[connIndex] |= (1UL << channelId);
    } else {
      recvChannelMask[connIndex] |= (1UL << channelId);
    }
    if (algorithm != NCCL_ALGO_UNDEF) {
      algoMask[algorithm] |= true;
    }
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "algoMask: ";
    for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
      ss << (algoMask[i] ? "1" : "0") << ", ";
    }
    for (int i = 0; i < NCCL_MAX_CONNS; i++) {
      ss << "sendChannelMask-" << i << ": " << std::hex << sendChannelMask[i]
         << ", ";
    }
    for (int i = 0; i < NCCL_MAX_CONNS; i++) {
      ss << "recvChannelMask-" << i << ": " << std::hex << recvChannelMask[i]
         << ", ";
    }
    return ss.str();
  }
};
// map from peer rank to its re-connection information
using ncclxPeerReConnInfoMap =
    folly::F14FastMap<int, std::unique_ptr<ncclxPeerReConnInfo>>;

// Perform transport pre-connect for all p2p peers, adopted from baseline
// ncclP2PPreconnectFunc
ncclResult_t p2pPreconnect(struct ncclComm* comm);
// Perform transport pre-connect for all p2p peers, adopted from basline
// ncclCollPreconnectFunc
ncclResult_t collPreconnect(
    struct ncclComm* comm,
    std::array<bool, NCCL_NUM_ALGORITHMS>& algoNeedConnect);

// Similar to ncclTransportRingConnect, but only connect ring on the assigned
// number of channels
ncclResult_t transportRingConnect(struct ncclComm* comm, int nChannels);

// Similar to ncclTransportTreeConnect, but only connect tree on the assigned
// number of channels
ncclResult_t transportTreeConnect(struct ncclComm* comm, int nChannels);

// Similar to ncclTransportPatConnect, but only connect binomial tree on the
// assigned number of channels
ncclResult_t transportPatConnect(struct ncclComm* comm, int nChannels);

// return true if the given task can be setup lazily
bool algoCanLazySetupChannel(struct ncclComm* comm, struct ncclTaskColl* task);

/* Check if algorithms are used for the given collective(s) and number of
 * channels need to be setup at runtime */
bool algoNeedConnect(struct ncclComm* comm, struct ncclTaskColl* task);

// Mark channels that need to be initialized in lazy mode
void p2pNeedConnect(
    struct ncclComm* comm,
    int peer,
    int channelId,
    bool isSendNotRecv);

/* Selectively copy channels' metadata to GPU memory for NCCL kernels to
 * access.This should only be used when NCCL_LAZY_SETUP_CHANNELS is enabled */
ncclResult_t devCommSetupChannels(ncclComm_t comm);

// setup channels up to maxChannelId for the given communicator
ncclResult_t setupChannels(struct ncclComm* comm, int maxChannelId);

/* Exchange transport information with peers to decide if a re-setup on
 * certain channels is needed for current plan due to buffer changes at
 * runtime. If so, reset the connection state and perform transport setup.
 * If skipReconnect is true, reserve the buffers and skip costly exchange. */
ncclResult_t transportReConnect(
    struct ncclComm* comm,
    uint64_t opCount,
    std::shared_ptr<void> peerReconnInfoMap,
    std::vector<std::string>& planBufKeys,
    bool skipReconnect);

// generate keys of cached buffers to current plan for given collective task
ncclResult_t addCollBufKeysToKernelPlan(
    struct ncclComm* comm,
    int channelId,
    struct ncclTaskColl* task,
    ncclKernelPlan* plan);

// generate keys of cached buffers to current plan for given channel and peer
ncclResult_t addP2PBufKeysToKernelPlan(
    struct ncclComm* comm,
    bool isSend,
    int channelId,
    int connIndex,
    int peerRank,
    ncclKernelPlan* plan,
    int algorithm = NCCL_ALGO_UNDEF);

}; // namespace ncclx
