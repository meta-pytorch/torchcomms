// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/HealthCheck.hpp"

#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace torch::comms {
namespace {

// Static registration of the torchcomms_health_check control plane handler.
// Returns {"healthy": true/false} based on whether any communicator's
// watchdog has detected a timeout.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
c10d::control_plane::RegisterHandler torchcommsHealthCheckRegistration(
    "torchcomms_health_check",
    [](const c10d::control_plane::Request& /* req */,
       c10d::control_plane::Response& res) {
      bool timed_out = TorchCommsHealthCheck::get()->isTimedOut();
      std::string json =
          timed_out ? R"({"healthy": false})" : R"({"healthy": true})";
      res.setContent(std::move(json), "application/json");
      res.setStatus(200);
    });

} // namespace
} // namespace torch::comms
