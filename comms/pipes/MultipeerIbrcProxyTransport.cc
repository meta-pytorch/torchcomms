// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbrcProxyTransport.h"

#include <stdexcept>
#include <utility>

#include <glog/logging.h>

namespace comms::pipes {

MultipeerIbrcProxyTransport::MultipeerIbrcProxyTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbTransportConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  if (nRanks_ <= 0) {
    throw std::invalid_argument(
        "MultipeerIbrcProxyTransport: nRanks must be positive");
  }
  if (myRank_ < 0 || myRank_ >= nRanks_) {
    throw std::invalid_argument(
        "MultipeerIbrcProxyTransport: myRank out of range");
  }
  if (!bootstrap_) {
    throw std::invalid_argument(
        "MultipeerIbrcProxyTransport: bootstrap must not be null");
  }

  VLOG(1) << "MultipeerIbrcProxyTransport: rank " << myRank_ << "/" << nRanks_
          << " created skeleton backend";
}

MultipeerIbrcProxyTransport::~MultipeerIbrcProxyTransport() = default;

void MultipeerIbrcProxyTransport::exchange() {
  throw std::runtime_error(
      "MultipeerIbrcProxyTransport::exchange: IBRC proxy resource setup is "
      "not implemented yet");
}

} // namespace comms::pipes
