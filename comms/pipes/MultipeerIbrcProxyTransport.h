// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/IbTransportConfig.h"

namespace comms::pipes {

/**
 * Host-side owner for the CPU-proxy IBRC backend.
 *
 * This slice only establishes ownership and mode dispatch. Resource setup
 * (ibverbX device/PD/CQ/QP, descriptor rings, and progress threads) is added
 * in later slices.
 */
class MultipeerIbrcProxyTransport {
 public:
  MultipeerIbrcProxyTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbTransportConfig& config);

  ~MultipeerIbrcProxyTransport();

  MultipeerIbrcProxyTransport(const MultipeerIbrcProxyTransport&) = delete;
  MultipeerIbrcProxyTransport& operator=(const MultipeerIbrcProxyTransport&) =
      delete;
  MultipeerIbrcProxyTransport(MultipeerIbrcProxyTransport&&) = delete;
  MultipeerIbrcProxyTransport& operator=(MultipeerIbrcProxyTransport&&) =
      delete;

  void exchange();

  int my_rank() const {
    return myRank_;
  }

  int n_ranks() const {
    return nRanks_;
  }

 private:
  const int myRank_;
  const int nRanks_;
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  MultipeerIbTransportConfig config_;
};

} // namespace comms::pipes
