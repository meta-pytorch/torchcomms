// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommObjectId.hpp"

#include <atomic>

namespace torch {
namespace comms {

// Global counter for object IDs (used for pickle identity preservation)
// Used by windows, batches, and other objects that need unique IDs
static std::atomic<uint64_t> g_object_id_counter{1};

uint64_t next_object_id() {
  return g_object_id_counter.fetch_add(1);
}

} // namespace comms
} // namespace torch
