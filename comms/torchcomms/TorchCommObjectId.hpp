// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

namespace torch {
namespace comms {

// Get the next unique object ID (for pickle identity preservation)
// Used by windows, batches, and other objects that need unique IDs
uint64_t next_object_id();

} // namespace comms
} // namespace torch
