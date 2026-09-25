// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <optional>

#include "comms/utils/colltrace/CollTraceInterface.h"

// Kept out of nccl.h: this is a Meta-internal question about a communicator,
// and the public header is the NCCL API.

namespace meta::comms::ncclx {

// Describe a collective captured into a CUDA graph, by the comm id its
// events carry and the id its replays report. Empty when no feed stamps that
// comm id, or when the feed does not recognise the collective.
std::optional<meta::comms::colltrace::CapturedCollDescription>
describeCapturedCollective(uint64_t commId, uint64_t capturedCollId);

} // namespace meta::comms::ncclx
