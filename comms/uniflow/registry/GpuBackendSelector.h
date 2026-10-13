// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "comms/uniflow/Result.h"

namespace uniflow {

inline constexpr std::array<std::string_view, 3> kGpuBackendNames{
    "nvidia",
    "amd",
    "mtia"};

/// Validates a configured vendor name without discovering or selecting a GPU.
Result<std::string> parseGpuBackend(std::string_view backend);

/// Throws invalid_argument for an invalid explicit request; unset stays unset.
void validateGpuBackendRequest(const std::optional<std::string>& requested);

/// Resolves GPU configuration and atomically claims one vendor per process.
/// Instances provide isolated state; production uses gpuBackendSelector().
class GpuBackendSelector {
 public:
  /// Reads UNIFLOW_GPU_BACKEND when resolving an unclaimed, implicit request.
  explicit GpuBackendSelector(std::string buildDefault);

  /// Uses a fixed environment value for isolated configuration.
  GpuBackendSelector(
      std::string buildDefault,
      std::optional<std::string> environment);

  /// Explicit request, environment, then build default; does not claim a
  /// vendor. Once claimed, implicit requests reuse that vendor without reading
  /// the env.
  Result<std::string> resolve(
      const std::optional<std::string>& requested = std::nullopt);

  /// Claims a preview at GPU initialization without reading configuration
  /// again. Implicit requests reuse a competing claim; explicit conflicts fail.
  /// Use the returned vendor to look up callbacks, rather than the preview.
  Result<std::string> claim(
      std::string_view candidate,
      const std::optional<std::string>& requested = std::nullopt);

  std::optional<std::string> selectedBackend() const;

 private:
  // Caller holds mutex_ and selectedBackend_ has a value.
  Result<std::string> checkSelectedBackend(
      const std::optional<std::string>& requested) const;

  const std::string buildDefault_;
  const bool readEnvironment_{true};
  const std::optional<std::string> environment_;
  mutable std::mutex mutex_;
  std::optional<std::string> selectedBackend_;
};

/// The single common-runtime selector; callers must not create private copies.
GpuBackendSelector& gpuBackendSelector();

} // namespace uniflow
