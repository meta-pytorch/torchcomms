// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "transport.h" // @manual

// Transport type detected from NCCL connector's transportComm pointer.
enum class NcclTransportType {
  kP2P,
  kSHM,
  kNET,
  kCollNet,
  kUnknown,
};

// Detect the transport type of an NCCL connector by comparing its
// transportComm pointer against the global transport objects. This is
// the same method NCCL uses internally (see enqueue.cc).
//
// In RE multihost mode, NCCL may use NET transport for same-node peers
// due to /dev/shm device ID mismatch across container sandboxes. Using
// this function instead of hostname-based isSameNode gives the actual
// transport NCCL selected, matching the callsite format in memory logging.
inline NcclTransportType getTransportType(
    const struct ncclConnector* connector,
    bool isSend) {
  if (connector == nullptr || !connector->connected) {
    return NcclTransportType::kUnknown;
  }
  const struct ncclTransportComm* tc = connector->transportComm;
  if (isSend) {
    if (tc == &p2pTransport.send) {
      return NcclTransportType::kP2P;
    } else if (tc == &shmTransport.send) {
      return NcclTransportType::kSHM;
    } else if (tc == &netTransport.send) {
      return NcclTransportType::kNET;
    } else if (tc == &collNetTransport.send) {
      return NcclTransportType::kCollNet;
    }
  } else {
    if (tc == &p2pTransport.recv) {
      return NcclTransportType::kP2P;
    } else if (tc == &shmTransport.recv) {
      return NcclTransportType::kSHM;
    } else if (tc == &netTransport.recv) {
      return NcclTransportType::kNET;
    } else if (tc == &collNetTransport.recv) {
      return NcclTransportType::kCollNet;
    }
  }
  return NcclTransportType::kUnknown;
}
