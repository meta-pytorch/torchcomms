// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/executor/EpollEventBase.h"
#include "comms/uniflow/executor/MutexQueuePolicy.h"

namespace uniflow {

/// EventBase using a mutex-guarded double-buffered queue for dispatch.
///
/// Usage:
///   MutexEventBase evb;
///   std::thread t([&evb] { evb.loop(); });
///   evb.dispatch([]() noexcept { /* runs on loop thread */ });
///   evb.stop();
///   t.join();
using MutexEventBase = EpollEventBase<MutexQueuePolicy>;

} // namespace uniflow
