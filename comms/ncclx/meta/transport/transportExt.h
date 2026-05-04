// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "meta/NcclMemoryUtils.h"

// [META]: Extension to store local/remote memory for synchronization, i.e.,
// fifo information, used in p2p.cc
struct p2pSyncResources {
  struct ncclSendMem* sendDevMem{nullptr};
  struct ncclRecvMem* recvDevMem{nullptr};
  void* sendMemIpc{nullptr};
  void* recvMemIpc{nullptr};
  struct ncclComm* comm{nullptr};
};

namespace ncclx {
inline enum NCCL_CHANNEL_METADATA_LOCATION getChannelMetadataLoc() {
  auto val = NCCL_CHANNEL_METADATA_LOCATION;
  if (NCCL_CHANNEL_METADATA_LOCATION == NCCL_CHANNEL_METADATA_LOCATION::unset) {
    val = (NCCL_USE_MEM_CACHE) ? NCCL_CHANNEL_METADATA_LOCATION::host
                               : NCCL_CHANNEL_METADATA_LOCATION::device;
  }
  CLOGF_SUBSYS(
      INFO, ENV, "NCCL_CHANNEL_METADATA_LOCATION={}", static_cast<int>(val));
  return val;
}
inline bool useTransportExt() {
  return NCCL_USE_TRANSPORT_EXT || NCCL_USE_MEM_CACHE ||
      getChannelMetadataLoc() == NCCL_CHANNEL_METADATA_LOCATION::host;
}
} // namespace ncclx

namespace ncclx::transport {

extern std::mutex transportSetupMutex;
constexpr const char* kP2pSyncBufKey = "p2pSyncBuf";

std::optional<std::string> getTransportBufKey(
    struct ncclComm* comm,
    bool isSend,
    int channelId,
    int connIndex,
    int peerRank);

// helper function to get the keys used to cache all internal buffers assigned
// to baseline transport, e.g., p2p, net, etc.
ncclResult_t getTransportBufKeys(
    ncclComm* comm,
    struct ncclTopoGraph* graph,
    int connIndex,
    std::vector<std::string>& keys);

ncclResult_t ncclTransportP2pSetupExt(
    struct ncclComm* comm,
    struct ncclTopoGraph* graph,
    int connIndex,
    bool reSetup = false);

inline size_t getP2pSyncBufSlot(
    int maxLocalRanks,
    bool isSend,
    size_t nMaxChannels,
    int channelId,
    int connIndex,
    int rank) {
  constexpr size_t slotSize =
      std::max(sizeof(struct ncclSendMem), sizeof(struct ncclRecvMem));
  // locate the slot for the given channelId, connIndex, and peerRank
  return isSend * (NCCL_MAX_CONNS * nMaxChannels * slotSize * maxLocalRanks) +
      connIndex * (nMaxChannels * slotSize * maxLocalRanks) +
      channelId * (slotSize * maxLocalRanks) + rank * slotSize;
}

inline ncclResult_t releaseP2pSyncBuf(struct ncclComm* comm) {
  return metaCommToNccl(comm->memCache->release(
      {fmt::format("{}:{:#x}", kP2pSyncBufKey, comm->commHash)}));
}
/* Get a peer sharable buffer pointer from internal pool used for p2p
 * transport's synchronization between peers. Each p2p connection will get a
 * slab for the given channel-connIndex-rank-send/recv connection. The entire
 * buffer is shared within the communicator, and fully released only when the
 * communicator is destroyed.
 */
ncclResult_t getP2pSyncBufPtr(
    struct ncclComm* comm,
    bool isSend,
    int channelId,
    int connIndex,
    int rank,
    void** ptr,
    ncclIpcDesc* ipcDesc,
    size_t* maxSize,
    size_t* offset);

} // namespace ncclx::transport
