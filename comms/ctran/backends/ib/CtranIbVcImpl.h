// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/utils/commSpecs.h"

namespace ctran::ib {

constexpr int MAX_CONTROL_MSGS{128};
constexpr int MAX_SEND_WR{256};
constexpr int MAX_PAYLOAD_SIZE{4096};

struct CtranIbRemoteQpInfo {
  enum ibverbx::ibv_mtu mtu;
  uint32_t qpn;
  uint8_t port;
  int linkLayer;
  union {
    struct {
      uint64_t spn;
      uint64_t iid;
    } eth;
    struct {
      uint16_t lid;
    } ib;
  } u;
};

// Fix-sized payload buffer for IB transport to prepare and register the
// temporary buffers for control messages
struct CtrlPacket {
  int type{0}; // for callback check
  size_t size{0}; // size of actual data in payload
  char payload[MAX_PAYLOAD_SIZE];

  inline void
  copyFrom(const int srcType, const void* srcPayload, const size_t srcSize) {
    FB_CHECKABORT(
        srcSize <= sizeof(payload),
        "Unexpected payload size {} > packet max payload size {}",
        srcSize,
        sizeof(payload));

    memcpy(payload, srcPayload, srcSize);
    size = srcSize;
    type = srcType;
  }

  inline void copyTo(void* dstPayload, const size_t dstSize) {
    FB_CHECKABORT(
        size == dstSize,
        "Unexpected packet payload size {} != input payload size {}",
        size,
        dstSize);

    memcpy(dstPayload, payload, dstSize);
  }

  inline void copyFrom(const CtrlPacket& src) {
    type = src.type;
    size = src.size;
    memcpy(payload, src.payload, src.size);
  }

  inline void copyTo(CtrlPacket& dst) {
    dst.type = type;
    dst.size = size;
    memcpy(dst.payload, payload, size);
  }

  inline size_t getPacketSize() const {
    return offsetof(CtrlPacket, payload) +
        size; // transfer header + actual size of payload
  }

  std::string toString() const {
    return fmt::format(
        "addr {} type {} payloadSize {} packetSize {}",
        (void*)this,
        type,
        size,
        getPacketSize());
  }
};

folly::Expected<ibverbx::IbvQp, ibverbx::Error> ctranIbQpCreate(
    const ibverbx::IbvPd* ibvPd,
    ibverbx::ibv_cq* cq);

folly::Expected<folly::Unit, ibverbx::Error>
ctranIbQpInit(ibverbx::IbvQp& ibvQp, int port, int qp_access_flags);

folly::Expected<folly::Unit, ibverbx::Error> ctranIbQpRTR(
    const CtranIbRemoteQpInfo& remoteQpInfo,
    ibverbx::IbvQp& ibvQp,
    uint8_t trafficClass);

folly::Expected<folly::Unit, ibverbx::Error> ctranIbQpRTS(
    ibverbx::IbvQp& ibvQp);

} // namespace ctran::ib
