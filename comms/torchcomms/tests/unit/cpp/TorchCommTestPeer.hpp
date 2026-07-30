// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "comms/torchcomms/TorchComm.hpp"

namespace torch::comms::test {

class TorchCommTestPeer {
 public:
  static std::shared_ptr<TorchComm> create(
      std::shared_ptr<TorchCommBackend> backend,
      std::vector<int> ranks,
      const std::string& name) {
    return std::shared_ptr<TorchComm>(
        new TorchComm(name, std::move(backend), std::move(ranks)));
  }
};

} // namespace torch::comms::test
