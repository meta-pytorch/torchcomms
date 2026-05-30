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

/**
 * Get a boolean environment variable value, parsed leniently per the
 * cvar convention: "1" / "true" / "yes" / "on" (case-insensitive) → true;
 * "0" / "false" / "no" / "off" → false; missing / empty / unparsable →
 * defaultValue.
 */
bool getBoolEnv(const char* name, bool defaultValue);

} // namespace meta::comms
