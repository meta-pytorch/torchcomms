// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "bootstrap.h"
#include "channel.h"
#include "group.h"
#include "p2p.h"
#include "transport.h"

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Utils.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/EventsScubaUtil.h"
#include "meta/transport/transportConnect.h"
#include "meta/transport/transportExt.h"
#include "meta/transport/transportProxy.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx {

ncclResult_t transportRingConnect(struct ncclComm* comm, int nChannels) {
  if (!comm || comm->nRanks == 1) {
    return ncclSuccess;
  }
  // Set default value of useGdr and useNetPXN in first connection
  // note that ncclTransportP2pSetup below may update the values
  if (comm->algoConnectedChannels[NCCL_ALGO_RING] == 0) {
    comm->useGdr = true;
    comm->useNetPXN = false;
  }
  for (int c = comm->algoConnectedChannels[NCCL_ALGO_RING]; c < nChannels;
       c++) {
    connectionSummary ringSummary;
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECK(ncclTransportP2pConnect(
        comm,
        c,
        1,
        &channel->ring.prev,
        1,
        &channel->ring.next,
        0,
        &ringSummary));
    INFO(
        NCCL_INIT,
        "commDesc: %s set up P2P connections for rings with %s on channel %d",
        ctran::utils::parseCommDesc(comm->config.commDesc),
        ringSummary.toString().c_str(),
        c);
  }
  NCCLCHECK(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0));

  // exchange ring info in first connection, if needed
  if (comm->algoConnectedChannels[NCCL_ALGO_RING] == 0 &&
      (NCCL_LOCAL_REGISTER || NCCL_GRAPH_REGISTER)) {
    struct RingConnInfo {
      bool useNetPXN{false};
      bool useGdr{true};
    };
    std::vector<RingConnInfo> ringInfo(comm->nRanks);
    ringInfo.at(comm->rank).useGdr = comm->useGdr;
    ringInfo.at(comm->rank).useNetPXN = comm->useNetPXN;
    NCCLCHECK(bootstrapAllGather(
        comm->bootstrap, ringInfo.data(), sizeof(struct RingConnInfo)));
    for (int i = 0; i < comm->nRanks; ++i) {
      if (!ringInfo[i].useGdr) {
        comm->useGdr = false;
      }
      if (ringInfo[i].useNetPXN) {
        comm->useNetPXN = true;
      }
      if (!comm->useGdr && comm->useNetPXN) {
        break;
      }
    }
  }
  INFO(
      NCCL_INIT,
      "commDesc: %s connected rings from channel %d to %d, use ring PXN %d GDR %d",
      ctran::utils::parseCommDesc(comm->config.commDesc),
      comm->algoConnectedChannels[NCCL_ALGO_RING],
      nChannels - 1,
      comm->useNetPXN,
      comm->useGdr);
  comm->algoConnectedChannels[NCCL_ALGO_RING] = nChannels;
  // mark initAlgoChannels to be compatible with baseline NCCL
  comm->initAlgoChannels[NCCL_ALGO_RING] = true;
  return ncclSuccess;
}

ncclResult_t transportTreeConnect(struct ncclComm* comm, int nChannels) {
  if (!comm || comm->nRanks == 1) {
    return ncclSuccess;
  }
  for (int c = comm->algoConnectedChannels[NCCL_ALGO_TREE]; c < nChannels;
       c++) {
    connectionSummary treeUpwardSummary;
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECK(ncclTransportP2pConnect(
        comm,
        c,
        NCCL_MAX_TREE_ARITY,
        channel->tree.down,
        1,
        &channel->tree.up,
        0,
        &treeUpwardSummary));
    INFO(
        NCCL_INIT,
        "commDesc: %s set up P2P connections for tree downward connections with %s on channel %d",
        ctran::utils::parseCommDesc(comm->config.commDesc),
        treeUpwardSummary.toString().c_str(),
        c);
    connectionSummary treeDownwardSummary;
    NCCLCHECK(ncclTransportP2pConnect(
        comm,
        c,
        1,
        &channel->tree.up,
        NCCL_MAX_TREE_ARITY,
        channel->tree.down,
        0,
        &treeDownwardSummary));
    INFO(
        NCCL_INIT,
        "commDesc: %s set up P2P connections for tree upward connections with %s on channel %d",
        ctran::utils::parseCommDesc(comm->config.commDesc),
        treeDownwardSummary.toString().c_str(),
        c);
  }
  NCCLCHECK(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0));
  INFO(
      NCCL_INIT,
      "commDesc: %s connected Trees from channel %d to %d",
      ctran::utils::parseCommDesc(comm->config.commDesc),
      comm->algoConnectedChannels[NCCL_ALGO_TREE],
      nChannels - 1);
  comm->algoConnectedChannels[NCCL_ALGO_TREE] = nChannels;
  // mark initAlgoChannels to be compatible with baseline NCCL
  comm->initAlgoChannels[NCCL_ALGO_TREE] = true;
  return ncclSuccess;
}

ncclResult_t transportPatConnect(struct ncclComm* comm, int nChannels) {
  if (!comm || comm->nRanks == 1) {
    return ncclSuccess;
  }
  for (int mask = 1; mask < comm->nRanks; mask <<= 1) {
    int prevPeer = (comm->rank + mask) % comm->nRanks;
    int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
    for (int c = comm->algoConnectedChannels[NCCL_ALGO_PAT]; c < nChannels;
         c++) {
      connectionSummary rsSummary;
      NCCLCHECK(ncclTransportP2pConnect(
          comm,
          c,
          1,
          &prevPeer,
          1,
          &nextPeer,
          0,
          &rsSummary)); // ReduceScatter
      INFO(
          NCCL_INIT,
          "commDesc: %s set up P2P connections for RS binomial trees with %s on channel %d",
          ctran::utils::parseCommDesc(comm->config.commDesc),
          rsSummary.toString().c_str(),
          c);
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0));
    for (int c = comm->algoConnectedChannels[NCCL_ALGO_PAT]; c < nChannels;
         c++) {
      connectionSummary agSummary;
      NCCLCHECK(ncclTransportP2pConnect(
          comm,
          c,
          1,
          &nextPeer,
          1,
          &prevPeer,
          0,
          &agSummary)); // AllGather
      INFO(
          NCCL_INIT,
          "commDesc: %s set up P2P connections for AG binomial trees with %s on channel %d",
          ctran::utils::parseCommDesc(comm->config.commDesc),
          agSummary.toString().c_str(),
          c);
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0));
  }
  INFO(
      NCCL_INIT,
      "commDesc %s connected binomial trees channel from %d to %d",
      ctran::utils::parseCommDesc(comm->config.commDesc),
      comm->algoConnectedChannels[NCCL_ALGO_PAT],
      nChannels - 1);
  comm->algoConnectedChannels[NCCL_ALGO_PAT] = nChannels;
  // mark initAlgoChannels to be compatible with baseline NCCL
  comm->initAlgoChannels[NCCL_ALGO_PAT] = true;
  return ncclSuccess;
}

/* TODO: extend lazy channel setup to other algorithms, e.g., NVSL, currently
 * support Ring, Tree and PAT */
bool algoCanLazySetupChannel(struct ncclComm* comm, struct ncclTaskColl* task) {
  if (task->algorithm == NCCL_ALGO_RING || task->algorithm == NCCL_ALGO_TREE ||
      task->algorithm == NCCL_ALGO_PAT) {
    return true;
  } else {
    /* update nMaxChannelsNeedInit and algoMaxChannelsNeedConnect to ensure all
     * channels will be setup and selected algorithm will be connected */
    comm->planner.nMaxChannelsNeedInit = comm->nChannels;
    comm->planner.algoMaxChannelsNeedConnect.at(task->algorithm) =
        comm->nChannels;
    return false;
  }
}

bool algoNeedConnect(struct ncclComm* comm, struct ncclTaskColl* task) {
  if (comm->planner.nTasksColl > 1) {
    /* FIXME: cannot support aggregated collectives now because it may use more
     * channels at scheduling time, update nMaxChannelsNeedInit and
     * algoMaxChannelsNeedConnect in planner to ensure all channels will be
     * setup and the selected algorithm will be connected */
    comm->planner.nMaxChannelsNeedInit = comm->nChannels;
    comm->planner.algoMaxChannelsNeedConnect.at(task->algorithm) =
        comm->nChannels;
    return comm->nChannelsReady < comm->nChannels;
  }
  // update the maximal number of channels need to be initialized later
  if (task->nMaxChannels > comm->nChannelsReady) {
    comm->planner.nMaxChannelsNeedInit =
        std::max(comm->planner.nMaxChannelsNeedInit, task->nMaxChannels);
  }
  /* update the max number of channels need to be connected for the given
   * algorithm */
  comm->planner.algoMaxChannelsNeedConnect.at(task->algorithm) = std::max(
      comm->planner.algoMaxChannelsNeedConnect.at(task->algorithm),
      task->nMaxChannels);

  if (comm->planner.algoMaxChannelsNeedConnect.at(task->algorithm) >
      comm->algoConnectedChannels[task->algorithm]) {
    INFO(
        NCCL_INIT,
        "commDesc: %s commHash: %lx needs nChannels=%d (%d initialized) for op %s with %lu bytes using algo %s and protocol %s, (%d channels connected) %d total channels will be connected for this algo",
        ctran::utils::parseCommDesc(comm->config.commDesc),
        comm->commHash,
        task->nMaxChannels,
        comm->nChannelsReady,
        ncclFuncStr[task->func],
        task->count * ncclTypeSize(task->datatype),
        ncclAlgoToString(task->algorithm),
        ncclProtoToString(task->protocol),
        comm->algoConnectedChannels[task->algorithm],
        comm->planner.algoMaxChannelsNeedConnect.at(task->algorithm));
    return true;
  }
  return false;
}

void p2pNeedConnect(
    struct ncclComm* comm,
    int peer,
    int channelId,
    bool isSendNotRecv) {
  comm->planner.nMaxChannelsNeedInit =
      std::max(comm->planner.nMaxChannelsNeedInit, channelId + 1);
  if (isSendNotRecv) {
    comm->connectSend[peer] |= (1UL << channelId);
  } else {
    comm->connectRecv[peer] |= (1UL << channelId);
  }
  ncclGroupCommPreconnect(comm);
  INFO(
      NCCL_INIT,
      "commDesc %s: commHash: %lx Channel-%d try %s connection setup for peer %d, nMaxChannelsNeedInit: %d",
      ctran::utils::parseCommDesc(comm->config.commDesc),
      comm->commHash,
      channelId,
      isSendNotRecv ? "send" : "recv",
      peer,
      comm->planner.nMaxChannelsNeedInit);
}

ncclResult_t devCommSetupChannels(ncclComm_t comm) {
  // devCommSetup should be called at init time to cache devCommAndChans
  if (UNLIKELY(!comm->devCommAndChans.has_value())) {
    return ncclInternalError;
  }
  INFO(
      NCCL_INIT,
      "commDesc: %s commHash: %lx devCommSetupChannels: copy channels' metadata, comm->nChannelsReady=%d",
      ctran::utils::parseCommDesc(comm->config.commDesc),
      comm->commHash,
      comm->nChannelsReady);
  auto sampleGuardBegin = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("INIT");
  sampleGuardBegin.sample().setCommunicatorMetadata(
      comm ? &comm->logMetaData : nullptr);
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclKernelCommAndChannels tmpCommAndChans;
  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));
  struct ncclKernelCommAndChannels* devCommAndChans =
      comm->devCommAndChans.value();
  cudaStream_t deviceStream;

  NCCLCHECKGOTO(
      ncclStrongStreamAcquire(
          ncclCudaGraphNone(),
          &comm->sharedRes->deviceStream,
          /*concurrent=*/false,
          &deviceStream),
      ret,
      fail);

  // only copy metadata for channels have already initialized
  for (int c = 0; c < comm->nChannelsReady; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks =
        comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(
          ncclCudaMemcpyAsync(
              tmpCommAndChans.channels[c].ring.userRanks,
              comm->channels[c].ring.userRanks,
              nRanks,
              deviceStream),
          ret,
          fail);
    }
  }

  // Copy multiple channels at once should be better than multiple
  // cudaMemcpyAsyc calls given sizes are small
  ret = ncclCudaMemcpyAsync(
      devCommAndChans->channels,
      tmpCommAndChans.channels,
      comm->nChannelsReady,
      deviceStream);

exit:
  NCCLCHECK(ncclStrongStreamRelease(
      ncclCudaGraphNone(),
      &comm->sharedRes->deviceStream,
      /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
}

ncclResult_t setupChannels(struct ncclComm* comm, int maxNchannels) {
  INFO(
      NCCL_INIT,
      "commDesc: %s: commHash: %lx setup %d channels and copy metadata to GPU from channel %d to %d",
      ctran::utils::parseCommDesc(comm->config.commDesc),
      comm->commHash,
      (maxNchannels - comm->nChannelsReady),
      comm->nChannelsReady,
      maxNchannels - 1);
  NcclScubaEvent initEvent(&comm->logMetaData);
  initEvent.lapAndRecord("setupChannels START");
  for (int c = comm->nChannelsReady; c < maxNchannels; c++) {
    if (c < comm->nChannels) {
      NCCLCHECK(setupChannel(
          comm,
          c,
          comm->rank,
          comm->nRanks,
          comm->rings.value().data() + c * comm->nRanks));
    } else {
      // rest of channels are only for p2p, just initialize them
      NCCLCHECK(initChannel(comm, c));
    }
  }
  comm->nChannelsReady = maxNchannels;

  // copy channels' metadata to device
  NCCLCHECK(devCommSetupChannels(comm));
  /* If multiple ranks are on the same process, i.e., multiple-threaded
   * communicator, use local intra-node barrier to ensure no thread goes ahead
   * and launch kernels before finishes devCommSetupChannels, which performs
   * cuda memory allocation or copy, to avoid deadlock.
   */
  if (comm->intraRanks > 1) {
    NCCLCHECK(bootstrapIntraNodeBarrier(
        comm->bootstrap,
        comm->localRankToRank,
        comm->localRank,
        comm->localRanks,
        comm->localRankToRank[0]));
  }

  // all channels are setup for collectives, free rings graph which won't be
  // used anymore
  if (comm->nChannelsReady == comm->nChannels && comm->rings.has_value()) {
    comm->rings.reset();
  }

  initEvent.lapAndRecord("setupChannels COMPLETE");
  return ncclSuccess;
}

/* Leverage ncclTransportP2pSetup to setup resources and connect peers again */
ncclResult_t setupTransports(
    struct ncclComm* comm,
    bool p2pNeedReConnect,
    std::array<bool, NCCL_NUM_ALGORITHMS>& algoNeedReConnect) {
  // TODO: support other algorithms if needed
  if (algoNeedReConnect[NCCL_ALGO_RING]) {
    NCCLCHECK(
        transport::ncclTransportP2pSetupExt(
            comm,
            &comm->graphs[NCCL_ALGO_RING],
            /*connIndex=*/0,
            /*reSetup=*/true));
  }
  if (algoNeedReConnect[NCCL_ALGO_TREE]) {
    NCCLCHECK(
        transport::ncclTransportP2pSetupExt(
            comm,
            &comm->graphs[NCCL_ALGO_TREE],
            /*connIndex=*/0,
            /*reSetup=*/true));
  }
  if (p2pNeedReConnect) {
    NCCLCHECK(
        transport::ncclTransportP2pSetupExt(
            comm, /*graph=*/nullptr, /*connIndex=*/1, /*reSetup=*/true));
  }
  return ncclSuccess;
}

// helper function to reserve buffers in the provided peerReconnInfoMap
ncclResult_t reserveReqBufs(
    struct ncclComm* comm,
    ncclxPeerReConnInfoMap* peerInfoMap,
    std::vector<std::string>& reqBufKeys,
    bool skipReconnect) {
  for (auto& info : *peerInfoMap) {
    auto peerRank = info.first;
    auto myReconnInfo =
        reinterpret_cast<ncclxPeerReConnInfo*>(info.second.get());

    for (int connIndex = 0; connIndex < NCCL_MAX_CONNS; connIndex++) {
      auto sendChannelMask = myReconnInfo->sendChannelMask[connIndex];
      auto recvChannelMask = myReconnInfo->recvChannelMask[connIndex];
      // reset the masks and check again
      myReconnInfo->sendChannelMask[connIndex] = 0;
      myReconnInfo->recvChannelMask[connIndex] = 0;

      for (int channelId = 0; channelId < MAXCHANNELS; channelId++) {
        if (sendChannelMask & (1UL << channelId)) {
          auto key = transport::getTransportBufKey(
              comm, /*isSend=*/true, channelId, connIndex, peerRank);
          if (key.has_value()) {
            bool reserved = true;
            auto res = comm->memCache->reserve(key.value());
            if (res == commInProgress && skipReconnect) {
              // wait until the buffer is reserved if skipReconnect is true
              while (res == commInProgress) {
                comm->transportProxy_->progress();
                res = comm->memCache->reserve(key.value());
              }
            } else if (res != commSuccess) {
              myReconnInfo->mark(
                  /*isSend=*/true, channelId, connIndex, NCCL_ALGO_UNDEF);
              reserved = false;
            }
            if (reserved) {
              reqBufKeys.emplace_back(key.value());
            }
          }
        }
        if (recvChannelMask & (1UL << channelId)) {
          auto key = transport::getTransportBufKey(
              comm, /*isSend=*/false, channelId, connIndex, peerRank);
          if (key.has_value()) {
            bool reserved = true;
            auto res = comm->memCache->reserve(key.value());
            if (res == commInProgress && skipReconnect) {
              // wait until the buffer is reserved if skipReconnect is true
              while (res == commInProgress) {
                comm->transportProxy_->progress();
                res = comm->memCache->reserve(key.value());
              }
            } else if (res != commSuccess) {
              myReconnInfo->mark(
                  /*isSend=*/false, channelId, connIndex, NCCL_ALGO_UNDEF);
              reserved = false;
            }
            if (reserved) {
              reqBufKeys.emplace_back(key.value());
            }
          }
        }
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t transportReConnect(
    struct ncclComm* comm,
    uint64_t opCount,
    std::shared_ptr<void> peerReconnInfoMap,
    std::vector<std::string>& reqBufKeys,
    bool skipReconnect) {
  auto peerInfoMap =
      reinterpret_cast<ncclxPeerReConnInfoMap*>(peerReconnInfoMap.get());
  bool collNeedReConnect = false;
  bool p2pNeedReConnect = false;
  std::array<bool, NCCL_NUM_ALGORITHMS> algoNeedReConnect{false};
  int reconnBoostrapTagBase = static_cast<int>(comm->commHash + opCount);

  NcclScubaEvent initEvent(&comm->logMetaData);
  initEvent.lapAndRecord("transportReConnect START");
  auto exitGuard = folly::makeGuard([&] {
    peerInfoMap->clear();
    initEvent.lapAndRecord("transportReConnect COMPLETE");
  });

  // Reserve required buffers
  NCCLCHECK(reserveReqBufs(comm, peerInfoMap, reqBufKeys, skipReconnect));
  // If preconnect happened, add the reserved buffers to the plan to be released

  // later
  if (!comm->connSetupBufKeys.empty()) {
    reqBufKeys.insert(
        reqBufKeys.end(),
        comm->connSetupBufKeys.begin(),
        comm->connSetupBufKeys.end());
    comm->connSetupBufKeys.clear();
  }
  // early return if no need to re-connect
  if (skipReconnect) {
    return ncclSuccess;
  }
  // post all send ops first to let peers know my re-connect status
  for (auto& info : *peerInfoMap) {
    auto peerRank = info.first;
    auto myReconnInfo =
        reinterpret_cast<ncclxPeerReConnInfo*>(info.second.get());

    NCCLCHECK(bootstrapSend(
        comm->bootstrap,
        peerRank,
        reconnBoostrapTagBase,
        myReconnInfo,
        sizeof(ncclxPeerReConnInfo)));

    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "rank-{} sent reconnect state to peer-{} (tag {}): [{}]",
        comm->rank,
        peerRank,
        reconnBoostrapTagBase,
        myReconnInfo->toString());

    for (int connIndex = 0; connIndex < NCCL_MAX_CONNS; connIndex++) {
      auto sendChannelMask = myReconnInfo->sendChannelMask[connIndex];
      auto recvChannelMask = myReconnInfo->recvChannelMask[connIndex];
      if (connIndex == 1) {
        p2pNeedReConnect |= (sendChannelMask != 0 || recvChannelMask != 0);
      } else {
        collNeedReConnect |= (sendChannelMask != 0 || recvChannelMask != 0);
      }
      comm->connectSend[peerRank] |= sendChannelMask;
      comm->connectRecv[peerRank] |= recvChannelMask;
    }
    if (collNeedReConnect) {
      for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
        algoNeedReConnect[algo] |= myReconnInfo->algoMask[algo];
      }
    }
  }

  // wait for peers in this plan to get their state
  std::vector<std::string> resetKeys;
  for (auto& info : *peerInfoMap) {
    auto peerRank = info.first;
    ncclxPeerReConnInfo peerReconnInfo;

    NCCLCHECK(bootstrapRecv(
        comm->bootstrap,
        peerRank,
        reconnBoostrapTagBase,
        &peerReconnInfo,
        sizeof(ncclxPeerReConnInfo)));

    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "rank-{} received reconnect state from peer-{} (tag {}): [{}]",
        comm->rank,
        peerRank,
        reconnBoostrapTagBase,
        peerReconnInfo.toString());

    for (int connIndex = 0; connIndex < NCCL_MAX_CONNS; connIndex++) {
      auto sendChannelMask = peerReconnInfo.sendChannelMask[connIndex];
      auto recvChannelMask = peerReconnInfo.recvChannelMask[connIndex];
      if (connIndex == 1) {
        p2pNeedReConnect |= (sendChannelMask != 0 || recvChannelMask != 0);
      } else {
        collNeedReConnect |= (sendChannelMask != 0 || recvChannelMask != 0);
      }
      if (sendChannelMask != 0) {
        // if peer needs to re-connect, reset my p2pChannel Setup mask
        comm->connectRecv[peerRank] |= sendChannelMask;
      }
      if (recvChannelMask != 0) {
        // if peer needs to re-connect, reset my p2pChannel Setup mask
        comm->connectSend[peerRank] |= recvChannelMask;
      }
    }
    if (collNeedReConnect) {
      for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
        algoNeedReConnect[algo] |= peerReconnInfo.algoMask[algo];
      }
    }
  }

  if (collNeedReConnect || p2pNeedReConnect) {
    NCCLCHECK(setupTransports(comm, p2pNeedReConnect, algoNeedReConnect));
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "{}: comm {} re-connected all peers for current plan",
        __func__,
        comm->config.commDesc);
    reqBufKeys.insert(
        reqBufKeys.end(),
        comm->connSetupBufKeys.begin(),
        comm->connSetupBufKeys.end());
    comm->connSetupBufKeys.clear();
  }

  return ncclSuccess;
}

ncclResult_t addCollBufKeysToKernelPlan(
    struct ncclComm* comm,
    int channelId,
    struct ncclTaskColl* task,
    ncclKernelPlan* plan) {
  if (task->algorithm == NCCL_ALGO_RING) {
    // create buf keys for ring algorithm peers
    NCCLCHECK(addP2PBufKeysToKernelPlan(
        comm,
        true /* isSend */,
        channelId,
        0 /* connIndex */,
        comm->channels[channelId].ring.next,
        plan,
        NCCL_ALGO_RING));
    NCCLCHECK(addP2PBufKeysToKernelPlan(
        comm,
        false /* isSend */,
        channelId,
        0 /* connIndex */,
        comm->channels[channelId].ring.prev,
        plan,
        NCCL_ALGO_RING));

  } else if (task->algorithm == NCCL_ALGO_TREE) {
    if (comm->channels[channelId].tree.up != -1) {
      // create buffer keys for double binary tree
      NCCLCHECK(addP2PBufKeysToKernelPlan(
          comm,
          true /* isSend */,
          channelId,
          0 /* connIndex */,
          comm->channels[channelId].tree.up,
          plan,
          NCCL_ALGO_TREE));
      NCCLCHECK(addP2PBufKeysToKernelPlan(
          comm,
          false /* isSend */,
          channelId,
          0 /* connIndex */,
          comm->channels[channelId].tree.up,
          plan,
          NCCL_ALGO_TREE));
    }

    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      if (comm->channels[channelId].tree.down[i] == -1) {
        continue;
      }
      NCCLCHECK(addP2PBufKeysToKernelPlan(
          comm,
          true /* isSend */,
          channelId,
          0 /* connIndex */,
          comm->channels[channelId].tree.down[i],
          plan,
          NCCL_ALGO_TREE));
      NCCLCHECK(addP2PBufKeysToKernelPlan(
          comm,
          false /* isSend */,
          channelId,
          0 /* connIndex */,
          comm->channels[channelId].tree.down[i],
          plan,
          NCCL_ALGO_TREE));
    }
  } else {
    WARN(
        "algorithm %s is not supported by NCCL MEM CACHE currently",
        ncclAlgoToString(task->algorithm));
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

ncclResult_t addP2PBufKeysToKernelPlan(
    struct ncclComm* comm,
    bool isSend,
    int channelId,
    int connIndex,
    int peerRank,
    ncclKernelPlan* plan,
    int algorithm) {
  // plan must be a valid pointer
  if (!plan) {
    return ncclInternalError;
  }
  auto key = transport::getTransportBufKey(
      comm, isSend, channelId, connIndex, peerRank);

  if (key.has_value()) {
    plan->bufKeys.push_back(key.value());
  }

  // intialize the map if not yet
  if (plan->peerReconnInfoMap == nullptr) {
    plan->peerReconnInfoMap = std::make_shared<ncclxPeerReConnInfoMap>();
  }
  // Only record if the peer is in the same node
  // TODO: support inter-node once net transport is ready to share buffers
  if (comm->rankToNode[peerRank] == comm->node) {
    auto peerInfoMap = reinterpret_cast<ncclxPeerReConnInfoMap*>(
        plan->peerReconnInfoMap.get());
    if (!peerInfoMap->contains(peerRank)) {
      peerInfoMap->insert({peerRank, std::make_unique<ncclxPeerReConnInfo>()});
    }
    // toggle the channel mask to reserve the buffer later
    if (key.has_value()) {
      peerInfoMap->at(peerRank)->mark(isSend, channelId, connIndex, algorithm);
    }
  }

  return ncclSuccess;
}

ncclResult_t p2pPreconnect(struct ncclComm* comm) {
  INFO(
      NCCL_INIT,
      "commDesc: %s new p2p send/recv needs to connect",
      ctran::utils::parseCommDesc(comm->config.commDesc));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }
  // setup channels if needed before setup transport
  if (comm->lazySetupChannels &&
      comm->nChannelsReady < comm->planner.nMaxChannelsNeedInit) {
    NCCLCHECK(ncclx::setupChannels(comm, comm->planner.nMaxChannelsNeedInit));
  }
  NcclScubaEvent initEvent(&comm->logMetaData);
  initEvent.lapAndRecord("p2pPreconnectFunc START");
  auto exitGuard = folly::makeGuard(
      [&] { initEvent.lapAndRecord("p2pPreconnectFunc COMPLETE"); });
  NCCLCHECK(ncclTransportP2pSetup(comm, nullptr, 1));

  INFO(
      NCCL_INIT,
      "commDesc: %s new p2p send/recv finished preconnect",
      ctran::utils::parseCommDesc(comm->config.commDesc));
  return ncclSuccess;
}

ncclResult_t collPreconnect(
    struct ncclComm* comm,
    std::array<bool, NCCL_NUM_ALGORITHMS>& algoNeedConnect) {
  ncclResult_t ret = ncclSuccess;
  INFO(
      NCCL_INIT,
      "commDesc: %s new collective needs to connect",
      ctran::utils::parseCommDesc(comm->config.commDesc));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }
  // setup channels if needed before setup transport
  if (comm->lazySetupChannels &&
      comm->nChannelsReady < comm->planner.nMaxChannelsNeedInit) {
    NCCLCHECK(ncclx::setupChannels(comm, comm->planner.nMaxChannelsNeedInit));
  }

  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {
    if (algoNeedConnect[i]) {
      NcclScubaEvent initEvent(&comm->logMetaData);
      const auto collpreStage =
          std::string("CollPreconnectFunc ") + std::string(ncclAlgoToString(i));
      initEvent.lapAndRecord(collpreStage + " START");
      switch (i) {
        case NCCL_ALGO_RING: {
          NCCLCHECK(ncclTransportRingConnect(comm));
          break;
        }
        case NCCL_ALGO_TREE: {
          NCCLCHECK(ncclTransportTreeConnect(comm));
          break;
        }
        case NCCL_ALGO_NVLS: {
          /* If we are using NVLS_TREE algo, we must mark NVLS algo to set up
           * NVLS intra-node buffer */
          NCCLCHECK(ncclNvlsBufferSetup(comm));
          break;
        }
        case NCCL_ALGO_NVLS_TREE: {
          NCCLCHECK(ncclNvlsTreeConnect(comm));
          break;
        }
        case NCCL_ALGO_COLLNET_CHAIN: {
          NCCLCHECK(ncclCollNetChainBufferSetup(comm));
          break;
        }
        case NCCL_ALGO_COLLNET_DIRECT: {
          NCCLCHECK(ncclCollNetDirectBufferSetup(comm));
          break;
        }
        case NCCL_ALGO_PAT: {
          NCCLCHECK(ncclTransportPatConnect(comm));
          break;
        }
        default: {
          ret = ncclInternalError;
          break;
        }
      }
      initEvent.lapAndRecord(collpreStage + " COMPLETE");
    }
  }
  INFO(
      NCCL_INIT,
      "commDesc: %s new collective finished preconnect",
      ctran::utils::parseCommDesc(comm->config.commDesc));

  return ret;
}
}; // namespace ncclx
