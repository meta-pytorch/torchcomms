// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>

namespace meta::comms {

/**
 * Get the string environment variable value
 *
 * @param name the name of the environment variable
 * @return the value of the environment variable if it exists, otherwise
 * std::nullopt
 */
std::optional<std::string> getStrEnv(const std::string& name);

} // namespace meta::comms
