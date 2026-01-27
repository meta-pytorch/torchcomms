// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/TorchCommUtils.hpp"

namespace torch::comms {

CommOptions::CommOptions() {
  // Check environment variables for options
  abort_process_on_timeout_or_error =
      env_to_value<bool>("TORCHCOMM_ABORT_ON_ERROR", true);

  // Get timeout from environment variable - using 600s as default, which is the
  // same timeout as what ProcessGroupNCCL uses
  uint64_t timeout_seconds =
      env_to_value<uint64_t>("TORCHCOMM_TIMEOUT_SECONDS", 600);
  timeout = std::chrono::seconds(timeout_seconds);

  // Initialize hints to empty map (don't read from environment)
  hints = std::unordered_map<std::string, std::string>();
}

bool CommOptions::operator==(const CommOptions& other) const {
  return (
      abort_process_on_timeout_or_error ==
          other.abort_process_on_timeout_or_error &&
      timeout == other.timeout && hints == other.hints);
}

} // namespace torch::comms
