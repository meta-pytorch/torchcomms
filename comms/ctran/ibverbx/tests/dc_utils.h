// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <ostream>
#include <string>

#include "comms/ctran/ibverbx/IbvAh.h"
#include "comms/ctran/ibverbx/IbvCq.h"
#include "comms/ctran/ibverbx/IbvPd.h"
#include "comms/ctran/ibverbx/IbvQp.h"
#include "comms/ctran/ibverbx/IbvSrq.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// DC Authentication Key
constexpr uint64_t DC_KEY = 0x1234;

// mlx5 is the only supported NIC for DC
inline const std::string kNicPrefix("mlx5_");

constexpr uint8_t kPortNum = 1;

#if defined(USE_FE_NIC)
constexpr int kGidIndex = 1;
#else
constexpr int kGidIndex = 3;
#endif

struct DcBusinessCard {
  int mtu{5}; // IBV_MTU_4096 = 5, use int to avoid enum forward decl issues
  uint32_t dctNum{0}; // DCT number for receiving
  uint8_t port{0};
  uint64_t subnetPrefix{0};
  uint64_t interfaceId{0};
  int32_t rank{-1};
  // Memory registration info for RDMA_WRITE
  uint64_t remoteAddr{0};
  uint32_t rkey{0};
};

std::ostream& operator<<(std::ostream& out, DcBusinessCard const& card);

// helper functions using ibverbx types

folly::Expected<IbvSrq, Error>
createSRQ(IbvPd& pd, int maxWr = 256, int maxSge = 1);

folly::Expected<IbvQp, Error> createDCI(IbvPd& pd, IbvCq& cq);

folly::Expected<IbvQp, Error>
createDCT(IbvPd& pd, IbvCq& cq, IbvSrq& srq, uint64_t dcKey = DC_KEY);

folly::Expected<folly::Unit, Error>
transitionDCIToRts(IbvQp& qp, uint8_t port, ibv_mtu mtu);

folly::Expected<folly::Unit, Error>
transitionDCTToRtr(IbvQp& qp, uint8_t port, ibv_mtu mtu);

folly::Expected<IbvAh, Error> createAddressHandle(
    IbvPd& pd,
    const DcBusinessCard& remoteCard,
    uint8_t sgidIndex = kGidIndex);

folly::Expected<folly::Unit, Error> pollCqForCompletions(
    int rank,
    IbvCq& cq,
    int expectedCompletions,
    int timeoutMs = 30000);

} // namespace ibverbx
