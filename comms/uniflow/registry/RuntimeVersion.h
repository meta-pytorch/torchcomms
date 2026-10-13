// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "comms/uniflow/Result.h"

namespace uniflow {

inline constexpr std::string_view kRuntimePackageVersion{"0.1.0.dev0"};
inline constexpr uint32_t kRuntimeAbiVersion{1};

/// Process-local identity of the runtime that owns the production registries.
const void* runtimeIdentity() noexcept;

/// Compatibility requirements compiled into a native component.
struct RuntimeVersion {
  std::string packageVersion{kRuntimePackageVersion};
  uint32_t abiVersion{kRuntimeAbiVersion};
  const void* identity{runtimeIdentity()};
};

RuntimeVersion runtimeVersion();

/// Rejects a different package, registry ABI or process-local runtime.
Status validateRuntimeVersion(const RuntimeVersion& version);

} // namespace uniflow
