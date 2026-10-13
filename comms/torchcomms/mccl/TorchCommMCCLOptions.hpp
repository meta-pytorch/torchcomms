// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// DEPRECATED: This header provides a legacy ReconfigureOptions type in the
// torch::comms::mccl namespace. New code should use
// torch::comms::ReconfigureOptions from TorchCommTypes.hpp instead.
// Field mapping to torch::comms::ReconfigureOptions:
//   uuid (string)  -> uuid (int64_t, via std::stoll)
//   urls            -> handles
//   kvPairs         -> hints
//   timeout         -> timeout (unchanged)

#include <chrono>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace torch::comms::mccl {

// A InitURL is an opaque handle for members of a communicator to discover
// each other.
using InitURL = std::string;

// Every time a communicator is initialized, pass in a new UUID to identify
// this new instance of the communicator. Every time we init-destroy-init, we
// must pass a different UUID.
using InitInstanceUUID = std::string;

struct ReconfigureOptions {
  // Uniquely identifies this instance of the communicator. The uuid
  // must not have been used previously on this communicator.
  InitInstanceUUID uuid;

  // Represents the members that will participate in this communicator. Each URL
  // represents a rank in the communicator. MCCL supports two regimes
  // * vector<InitUrl> urls: MCCL guarantees that assigned ranks correspond to
  // position of URL in the vector
  // * unordered_set<InitUrl> urls: MCCL will determine the rank assignment
  // based on internal considerations, no external rank order is respected.
  std::variant<std::unordered_set<InitURL>, std::vector<InitURL>> urls;

  // How long to allow init to take before we fail with an error.
  std::optional<std::chrono::milliseconds> timeout{std::nullopt};

  // Additional configs to use, implementation-specific
  std::unordered_map<std::string, std::string> kvPairs;
};

} // namespace torch::comms::mccl
