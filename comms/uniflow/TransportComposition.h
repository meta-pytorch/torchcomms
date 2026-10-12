// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/MultiTransport.h"

namespace uniflow {

/// Selects built-in providers using this endpoint's device and NIC settings.
TransportProviderList makeDefaultTransportProviders(
    int deviceId,
    const MultiTransportFactoryOptions& options);

} // namespace uniflow
