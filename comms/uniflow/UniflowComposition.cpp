// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/TransportComposition.h"
#include "comms/uniflow/Uniflow.h"

#include <utility>

namespace uniflow {

UniflowAgent::UniflowAgent(
    const UniflowAgentConfig& config,
    std::unique_ptr<controller::Client> client,
    std::unique_ptr<controller::Server> server)
    : UniflowAgent(
          config,
          makeDefaultTransportProviders(
              config.deviceId,
              makeFactoryOptions(config)),
          std::move(client),
          std::move(server)) {}

} // namespace uniflow
