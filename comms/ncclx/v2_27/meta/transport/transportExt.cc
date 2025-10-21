// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "bootstrap.h"
#include "transport.h"

#include "comms/ctran/memory/Utils.h"
#include "comms/utils/logger/EventsScubaUtil.h"
#include "meta/transport/transportExt.h"
#include "meta/wrapper/MetaFactory.h"

namespace ncclx::transport {

std::mutex transportSetupMutex;

namespace {
// Copy the static function from src/transport.cc
template <int type>
inline ncclResult_t selectTransport(
    struct ncclComm* comm,
    struct ncclTopoGraph* graph,
    struct ncclConnect* connect,
    int channelId,
    int peer,
    int connIndex,
    int* transportType) {
  struct ncclPeerInfo* myInfo = comm->peerInfo + comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo + peer;
  struct ncclConnector* connector = (type == 1)
      ? comm->channels[channelId].peers[peer]->send + connIndex
      : comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t = 0; t < NTRANSPORTS; t++) {
    struct ncclTransport* transport = ncclTransports[t];
    struct ncclTransportComm* transportComm =
        type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(
          comm,
          graph,
          myInfo,
          peerInfo,
          connect,
          connector,
          channelId,
          connIndex));
      if (transportType) {
        *transportType = t;
      }
      return ncclSuccess;
    }
  }
  WARN(
      "No transport found for rank %d[%lx] -> rank %d[%lx]",
      myInfo->rank,
      myInfo->busId,
      peerInfo->rank,
      peerInfo->busId);
  return ncclSystemError;
}

// helper function to obtain send and recv peers used by given
// algorithm and channel
inline void getPeers(
    struct ncclComm* comm,
    int channelId,
    int algorithm,
    std::vector<int>& sendPeers,
    std::vector<int>& recvPeers) {
  if (comm->channels[channelId].peers) {
    if (algorithm == NCCL_ALGO_RING) {
      sendPeers.push_back(comm->channels[channelId].ring.next);
      recvPeers.push_back(comm->channels[channelId].ring.prev);
    } else if (algorithm == NCCL_ALGO_TREE) {
      if (comm->channels[channelId].tree.up != -1) {
        sendPeers.push_back(comm->channels[channelId].tree.up);
        recvPeers.push_back(comm->channels[channelId].tree.up);
      }
      for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
        if (comm->channels[channelId].tree.down[i] == -1) {
          continue;
        }
        sendPeers.push_back(comm->channels[channelId].tree.down[i]);
        recvPeers.push_back(comm->channels[channelId].tree.down[i]);
      }
    } else {
      WARN(
          "algorithm %s is not supported in getPeers",
          ncclAlgoToString(algorithm));
    }
  }
}
/* helper function to reset the connection state and resources when we require
 * setup/connection again due to buffer changes on particular
 * send and/or recv connection for  given peer's   */
ncclResult_t resetTransport(
    struct ncclComm* comm,
    uint64_t sendChannelMask,
    uint64_t recvChannelMask,
    int connIndex,
    int peerRank) {
  for (int channelId = 0; channelId < MAXCHANNELS; channelId++) {
    if (comm->channels[channelId].peers == nullptr) {
      continue;
    }
    auto channelPeer = comm->channels[channelId].peers[peerRank];
    if (sendChannelMask & (1UL << channelId)) {
      if (channelPeer->send[connIndex].connected) {
        channelPeer->send[connIndex].connected = 0;
        INFO(
            NCCL_INIT,
            "Channel %d: Reset send connection to peer %d, connIndex: %d",
            channelId,
            peerRank,
            connIndex);
      }
    }
    if (recvChannelMask & (1UL << channelId)) {
      if (channelPeer->recv[connIndex].connected) {
        channelPeer->recv[connIndex].connected = 0;
        INFO(
            NCCL_INIT,
            "Channel %d: Reset recv connection from peer %d, connIndex: %d",
            channelId,
            peerRank,
            connIndex);
      }
    }
  }
  return ncclSuccess;
}

} // namespace

// helper function to check a given channelId, connIndex, peerRank is
// initialized and connected. If so, return the corresponding key used to
// reserve/release the internal buffer
std::optional<std::string> getTransportBufKey(
    struct ncclComm* comm,
    bool isSend,
    int channelId,
    int connIndex,
    int peerRank) {
  if (comm->channels[channelId].peers) {
    auto conn = (isSend)
        ? &comm->channels[channelId].peers[peerRank]->send[connIndex]
        : &comm->channels[channelId].peers[peerRank]->recv[connIndex];
    if (conn->connected && conn->proxyConn.connection) {
      // FIXME: conn->proxyConn.connection may not always be valid,
      // bool isP2p = (conn->proxyConn.connection->transport ==
      // TRANSPORT_P2P);
      struct ncclTransportComm* p2pTcomm = (isSend)
          ? &ncclTransports[TRANSPORT_P2P]->send
          : &ncclTransports[TRANSPORT_P2P]->recv;
      bool isP2p = (conn->transportComm == p2pTcomm);
      // TODO: support other transport (i.e., net)
      // for p2p sender, WRITE protocol no longer needs to reserve buffer
      if (!isP2p || (isP2p && isSend && conn->conn.flags & NCCL_P2P_WRITE)) {
        return std::nullopt;
      }
      // p2p buffers are allocated in ProxySetup, and net buffers allocated
      // in ProxyConnect
      auto allocCallsite = (isP2p) ? "ProxySetup" : "ProxyConnect";

      return fmt::format(
          "{}:{:#x}",
          ncclx::memory::genKey(
              allocCallsite, isP2p, isSend, channelId, connIndex, peerRank),
          comm->commHash);
    }
  }

  return std::nullopt;
}

ncclResult_t getTransportBufKeys(
    ncclComm* comm,
    struct ncclTopoGraph* graph,
    int connIndex,
    std::vector<std::string>& keys) {
  auto nChannels = (connIndex == 1) ? comm->p2pnChannels : comm->nChannels;
  std::vector<int> sendPeers;
  std::vector<int> recvPeers;
  if (!graph) {
    // no graph provided, check all peers
    for (int i = 1; i < comm->nRanks; i++) {
      int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
      int sendPeer = (comm->rank + i) % comm->nRanks;
      if (comm->rank != sendPeer) {
        sendPeers.push_back(sendPeer);
      }
      if (comm->rank != recvPeer) {
        recvPeers.push_back(recvPeer);
      }
    }
  }

  for (int c = 0; c < nChannels; c++) {
    auto connPeers = comm->channels[c].peers;
    if (connPeers) {
      // if a graph is provided, only need to check peers in the graph
      if (graph) {
        sendPeers.clear();
        recvPeers.clear();
        if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
          getPeers(comm, c, NCCL_ALGO_RING, sendPeers, recvPeers);
        } else if (
            graph->pattern == NCCL_TOPO_PATTERN_TREE ||
            graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
            graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) {
          getPeers(comm, c, NCCL_ALGO_TREE, sendPeers, recvPeers);
        } else {
          WARN(
              "graph pattern %d is not supported in using buffer pool",
              graph->pattern);
          return ncclSuccess;
        }
      }
      for (auto sendPeer : sendPeers) {
        auto sendBufKey = getTransportBufKey(
            comm,
            /*isSend=*/true,
            c,
            connIndex,
            sendPeer);
        if (sendBufKey.has_value()) {
          keys.push_back(sendBufKey.value());
        }
      }
      for (auto recvPeer : recvPeers) {
        auto recvBufKey =
            getTransportBufKey(comm, /*isSend=*/false, c, connIndex, recvPeer);
        if (recvBufKey.has_value()) {
          keys.push_back(recvBufKey.value());
        }
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTransportP2pSetupExt(
    struct ncclComm* comm,
    struct ncclTopoGraph* graph,
    int connIndex,
    bool reSetup) {
  std::lock_guard<std::mutex> lock(transportSetupMutex);
  auto sampleGuardBegin = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("INIT");
  sampleGuardBegin.sample().setCommunicatorMetadata(
      comm ? &comm->logMetaData : nullptr);
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  ncclResult_t ret = ncclSuccess;
  struct ncclConnect**
      data; // Store intermediate send/recvData structs for connect
  struct ncclConnect** recvData =
      nullptr; // Points to entries inside data for given recv connection within
               // a channel
  struct ncclConnect** sendData =
      nullptr; // Points to entries inside data for given send connection within
               // a channel
  int done = 0;
  auto maxPeers = NCCL_CONNECT_ROUND_MAX_PEERS;
  size_t connInfoSize = sizeof(struct ncclConnInfo);

  NCCLCHECK(ncclCalloc(&data, maxPeers));
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);
  cudaStream_t hostStream, deviceStream;

  if (!comm->channelMetadataOnHost) {
    NCCLCHECKGOTO(
        ncclStrongStreamAcquire(
            ncclCudaGraphNone(),
            &comm->sharedRes->hostStream,
            /*concurrent=*/false,
            &hostStream),
        ret,
        fail);
    NCCLCHECKGOTO(
        ncclStrongStreamAcquire(
            ncclCudaGraphNone(),
            &comm->sharedRes->deviceStream,
            /*concurrent=*/false,
            &deviceStream),
        ret,
        fail);
  }

  // If this is a re-setup, we need to reset the connected transports to
  // avoid resource leaks
  if (reSetup) {
    // For re-setup, only need to copy new buffer pointers
    connInfoSize = sizeof(char*) * NCCL_NUM_PROTOCOLS;
    for (int i = 0; i < comm->nRanks; i++) {
      if (i != comm->rank) {
        NCCLCHECKGOTO(
            resetTransport(
                comm, comm->connectSend[i], comm->connectRecv[i], connIndex, i),
            ret,
            fail);
      }
    }
  }
  // First time initialization
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];
    uint64_t sendMask = comm->connectSend[sendPeer];

    // Data[i] contains all ncclConnect information for all send and receive
    // connections with a given send and recv peer This data is packed in the
    // array based on the number of sendChannels and recvChannels connected with
    // these peers The first N entries contain recvData, connection information
    // for recv connections The next M entries contain sendData, connection
    // information for send connections It's not guaranteed that each entry of
    // data has the same number of total or send/recv specific connections
    int p = i - (done + 1);
    if (recvMask || sendMask) {
      if (data[p] == nullptr) {
        NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      } else {
        memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect));
      }
    }
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    int type;
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (recvMask & (1UL << c)) {
        NCCLCHECKGOTO(
            selectTransport<0>(
                comm,
                graph,
                recvData[p] + recvChannels++,
                c,
                recvPeer,
                connIndex,
                &type),
            ret,
            fail);
      }
    }
    sendData[p] = recvData[p] + recvChannels;
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (sendMask & (1UL << c)) {
        NCCLCHECKGOTO(
            selectTransport<1>(
                comm,
                graph,
                sendData[p] + sendChannels++,
                c,
                sendPeer,
                connIndex,
                &type),
            ret,
            fail);
      }
    }

    if (sendPeer == recvPeer) {
      if (recvChannels + sendChannels) {
        NCCLCHECKGOTO(
            bootstrapSend(
                comm->bootstrap,
                recvPeer,
                bootstrapTag,
                data[p],
                sizeof(struct ncclConnect) * (recvChannels + sendChannels)),
            ret,
            fail);
        NCCLCHECKGOTO(
            bootstrapRecv(
                comm->bootstrap,
                recvPeer,
                bootstrapTag,
                data[p],
                sizeof(struct ncclConnect) * (recvChannels + sendChannels)),
            ret,
            fail);
        sendData[p] = data[p];
        recvData[p] = data[p] + sendChannels;
      }
    } else {
      if (recvChannels) {
        NCCLCHECKGOTO(
            bootstrapSend(
                comm->bootstrap,
                recvPeer,
                bootstrapTag,
                recvData[p],
                sizeof(struct ncclConnect) * recvChannels),
            ret,
            fail);
      }
      if (sendChannels) {
        NCCLCHECKGOTO(
            bootstrapSend(
                comm->bootstrap,
                sendPeer,
                bootstrapTag,
                sendData[p],
                sizeof(struct ncclConnect) * sendChannels),
            ret,
            fail);
        NCCLCHECKGOTO(
            bootstrapRecv(
                comm->bootstrap,
                sendPeer,
                bootstrapTag,
                sendData[p],
                sizeof(struct ncclConnect) * sendChannels),
            ret,
            fail);
      }
      if (recvChannels) {
        NCCLCHECKGOTO(
            bootstrapRecv(
                comm->bootstrap,
                recvPeer,
                bootstrapTag,
                recvData[p],
                sizeof(struct ncclConnect) * recvChannels),
            ret,
            fail);
      }
    }

    if (i - done == maxPeers || i == comm->nRanks - 1) {
      // Loop until all channels with all ranks have been connected
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j = done + 1; j <= i; j++) {
          recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          sendPeer = (comm->rank + j) % comm->nRanks;
          recvMask = comm->connectRecv[recvPeer];
          sendMask = comm->connectSend[sendPeer];

          p = j - (done + 1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c = 0; c < MAXCHANNELS; c++) {
            if (sendMask & (1UL << c)) {
              struct ncclConnector* conn =
                  comm->channels[c].peers[sendPeer]->send + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(
                    conn->transportComm->connect(
                        comm,
                        sendData[p] + sendDataOffset,
                        1,
                        comm->rank,
                        conn),
                    ret,
                    fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  if (comm->channelMetadataOnHost) {
                    std::memcpy(
                        &comm->channels[c]
                             .devPeersHostPtr[sendPeer]
                             ->send[connIndex],
                        &conn->conn,
                        connInfoSize);
                  } else {
                    /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is
                     * a device memory access. */
                    CUDACHECKGOTO(
                        cudaMemcpyAsync(
                            &comm->channels[c]
                                 .devPeersHostPtr[sendPeer]
                                 ->send[connIndex],
                            &conn->conn,
                            connInfoSize,
                            cudaMemcpyDefault,
                            hostStream),
                        ret,
                        fail);
                  }
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }

            // Start with recv channels
            if (recvMask & (1UL << c)) {
              struct ncclConnector* conn =
                  comm->channels[c].peers[recvPeer]->recv + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(
                    conn->transportComm->connect(
                        comm,
                        recvData[p] + recvDataOffset,
                        1,
                        comm->rank,
                        conn),
                    ret,
                    fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  if (comm->channelMetadataOnHost) {
                    std::memcpy(
                        &comm->channels[c]
                             .devPeersHostPtr[recvPeer]
                             ->recv[connIndex],
                        &conn->conn,
                        connInfoSize);
                  } else {
                    /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is
                     * a device memory access. */
                    CUDACHECKGOTO(
                        cudaMemcpyAsync(
                            &comm->channels[c]
                                 .devPeersHostPtr[recvPeer]
                                 ->recv[connIndex],
                            &conn->conn,
                            connInfoSize,
                            cudaMemcpyDefault,
                            hostStream),
                        ret,
                        fail);
                  }
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
          }
        }
      }
      done = i;
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after
   * connection setup and start to destroy the connection after returning from
   * this function; however, the others might still be trying to connect and
   * import the buffer. No sync can lead to invalid shmem/cuda buffer. In
   * addition, we also clear all connect masks and free each connectInfo array
   */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      if (comm->connectSend[sendPeer] != 0UL) {
        NCCLCHECKGOTO(
            bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, nullptr, 0),
            ret,
            fail);
      }
      if (comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(
            bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, nullptr, 0),
            ret,
            fail);
      }
      if (comm->connectSend[sendPeer] != 0UL) {
        NCCLCHECKGOTO(
            bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, nullptr, 0),
            ret,
            fail);
      }
      if (comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(
            bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, nullptr, 0),
            ret,
            fail);
      }
    } else {
      if (comm->connectSend[sendPeer] != 0UL ||
          comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(
            bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, nullptr, 0),
            ret,
            fail);
        NCCLCHECKGOTO(
            bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, nullptr, 0),
            ret,
            fail);
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

exit:
  for (int i = 0; i < maxPeers; ++i) {
    if (data[i]) {
      free(data[i]);
    }
  }
  free(data);
  if (sendData) {
    free(sendData);
  }
  if (recvData) {
    free(recvData);
  }

  if (!comm->channelMetadataOnHost) {
    NCCLCHECK(ncclStreamWaitStream(
        deviceStream, hostStream, comm->sharedRes->scratchEvent));
    NCCLCHECK(ncclStrongStreamRelease(
        ncclCudaGraphNone(),
        &comm->sharedRes->hostStream,
        /*concurrent=*/false));
    NCCLCHECK(ncclStrongStreamRelease(
        ncclCudaGraphNone(),
        &comm->sharedRes->deviceStream,
        /*concurrent=*/false));
  }
  return ret;
fail:
  goto exit;
}

ncclResult_t getP2pSyncBufPtr(
    struct ncclComm* comm,
    bool isSend,
    int channelId,
    int connIndex,
    int rank,
    void** ptr,
    ncclIpcDesc* ipcDesc,
    size_t* maxSize,
    size_t* offset) {
  constexpr size_t slotSize =
      std::max(sizeof(struct ncclSendMem), sizeof(struct ncclRecvMem));

  // Allocate max possible size for a communicatorand each
  // channel-connection-peer-send/recv pair will get a unique slab, e.g., in a
  // case of 8 GPUs within a host using 64 channels, each slot is 4KiB, it will
  // result in 8 * 64 * 2 * 4KiB * 2 = 8MiB per communicator, if p2p transport
  // is used. Minimum size is 2MiB as required by cuMem miniminal granularity.
  int nMaxChannels = std::max(
      std::min(comm->nChannels, comm->p2pnChannels * comm->p2pnChannelsPerPeer),
      MAXCHANNELS);
  *maxSize = std::max(
      comm->maxLocalRanks * nMaxChannels * NCCL_MAX_CONNS * slotSize * 2,
      1UL << 21);
  ALIGN_SIZE((*maxSize), CUDA_IPC_MIN);
  // Grab the cached buffer per communicator
  NCCLCHECK(ncclx::memory::metaAllocateShareableBuffer(
      *maxSize,
      0,
      ipcDesc,
      ptr,
      kP2pSyncBufKey,
      comm->memCache,
      &comm->logMetaData));

  *offset = getP2pSyncBufSlot(
      comm->maxLocalRanks, isSend, nMaxChannels, channelId, connIndex, rank);

  return ncclSuccess;
}

} // namespace ncclx::transport
