/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "registered_allreduce_relay_args.h"

namespace rcclx::relay {

// Enqueues exactly one kernel of directLanes + 2 * helpers * lanes CTAs.
hipError_t launchRegisteredRelayAllReduce(
    const RegisteredRelayArgs& args,
    hipStream_t stream);

} // namespace rcclx::relay
