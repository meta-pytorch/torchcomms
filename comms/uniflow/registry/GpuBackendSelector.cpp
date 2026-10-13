// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/GpuBackendSelector.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#ifndef UNIFLOW_DEFAULT_GPU_BACKEND
#define UNIFLOW_DEFAULT_GPU_BACKEND "nvidia"
#endif

namespace uniflow {
namespace {

std::optional<std::string> readGpuBackendEnvironment() {
  const auto* backend = std::getenv("UNIFLOW_GPU_BACKEND");
  return backend ? std::optional<std::string>(backend) : std::nullopt;
}

} // namespace

Result<std::string> parseGpuBackend(std::string_view backend) {
  if (std::find(kGpuBackendNames.begin(), kGpuBackendNames.end(), backend) !=
      kGpuBackendNames.end()) {
    return std::string(backend);
  }
  return Err(
      ErrCode::InvalidArgument,
      "Invalid GPU backend '" + std::string(backend) +
          "'; expected nvidia, amd, or mtia.");
}

void validateGpuBackendRequest(const std::optional<std::string>& requested) {
  if (requested) {
    auto parsed = parseGpuBackend(*requested);
    if (!parsed) {
      throw std::invalid_argument(parsed.error().message());
    }
  }
}

GpuBackendSelector::GpuBackendSelector(std::string buildDefault)
    : buildDefault_(std::move(buildDefault)) {}

GpuBackendSelector::GpuBackendSelector(
    std::string buildDefault,
    std::optional<std::string> environment)
    : buildDefault_(std::move(buildDefault)),
      readEnvironment_(false),
      environment_(std::move(environment)) {}

Result<std::string> GpuBackendSelector::resolve(
    const std::optional<std::string>& requested) {
  if (requested) {
    auto parsed = parseGpuBackend(*requested);
    CHECK_RETURN(parsed);
  }
  std::lock_guard lock(mutex_);
  if (selectedBackend_) {
    return checkSelectedBackend(requested);
  }
  auto configured = requested;
  if (!configured) {
    configured = readEnvironment_ ? readGpuBackendEnvironment() : environment_;
  }
  return parseGpuBackend(configured ? *configured : buildDefault_);
}

Result<std::string> GpuBackendSelector::claim(
    std::string_view candidate,
    const std::optional<std::string>& requested) {
  auto parsed = parseGpuBackend(candidate);
  CHECK_RETURN(parsed);
  if (requested) {
    auto explicitBackend = parseGpuBackend(*requested);
    CHECK_RETURN(explicitBackend);
  }
  std::lock_guard lock(mutex_);
  if (selectedBackend_) {
    return checkSelectedBackend(requested);
  }
  if (requested && *requested != parsed.value()) {
    return Err(
        ErrCode::InvalidArgument,
        "Resolved GPU backend '" + parsed.value() +
            "' does not match the explicit request '" + *requested + "'.");
  }
  selectedBackend_ = std::move(parsed).value();
  return *selectedBackend_;
}

std::optional<std::string> GpuBackendSelector::selectedBackend() const {
  std::lock_guard lock(mutex_);
  return selectedBackend_;
}

Result<std::string> GpuBackendSelector::checkSelectedBackend(
    const std::optional<std::string>& requested) const {
  if (requested && *requested != *selectedBackend_) {
    return Err(
        ErrCode::InvalidArgument,
        "GPU backend '" + *requested +
            "' conflicts with the selected backend '" + *selectedBackend_ +
            "'.");
  }
  return *selectedBackend_;
}

GpuBackendSelector& gpuBackendSelector() {
  static GpuBackendSelector selector(UNIFLOW_DEFAULT_GPU_BACKEND);
  return selector;
}

} // namespace uniflow
