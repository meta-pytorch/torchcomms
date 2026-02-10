// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <folly/File.h>

namespace comms::logger {

// Generate Scuba log file name following Logarithm's expected format.
// Format:
// {logFilePrefix}/dedicated_log_structured_json.perfpipe_{tableName}.Rank_{rank}.{uniqueSuffix}.scribe
// See:
// https://www.internalfb.com/code/configerator/[master]/source/datainfra/logarithm/transport/logarithm_conda_custom_transport.cinc
std::string getScubaFileName(
    const std::string& logFilePrefix,
    const std::string& tableName);

// Create the Scuba log file, creating directories if needed.
// Returns nullopt if file creation fails.
// If appendMode is true, the file is opened with O_APPEND flag (used by MCCL).
// If appendMode is false, the file is opened without O_APPEND (original NCCLX
// behavior).
std::optional<folly::File> createScubaFile(
    const std::string& fileName,
    bool appendMode = false);

// Get current timestamp in seconds since epoch.
// Used for Scuba "time" field which requires seconds precision.
int64_t getTimestamp();

// Get current timestamp in milliseconds since epoch.
// Useful for trace context generation and correlation.
int64_t getTimestampMs();

// Get current timestamp in microseconds since epoch.
// Useful for high-precision timing in operation traces.
int64_t getTimestampUs();

} // namespace comms::logger
