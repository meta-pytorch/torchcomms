// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/executor/EpollEventBase.h"
#include "comms/uniflow/executor/LockFreeQueuePolicy.h"

namespace uniflow {

/// EventBase using a lock-free MPSC queue for dispatch.
///
/// Usage:
///   LockFreeEventBase evb;
///   std::thread t([&evb] { evb.loop(); });
///   evb.dispatch([]() noexcept { /* runs on loop thread */ });
///   evb.stop();
///   t.join();
using LockFreeEventBase = EpollEventBase<LockFreeQueuePolicy>;

} // namespace uniflow
