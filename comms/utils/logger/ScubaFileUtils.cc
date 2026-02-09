// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/logger/ScubaFileUtils.h"

#include <chrono>
#include <filesystem>

#include <fmt/format.h>
#include <folly/logging/xlog.h>

#include "comms/utils/RankUtils.h"
#include "comms/utils/StrUtils.h"

namespace comms::logger {

std::string getScubaFileName(
    const std::string& logFilePrefix,
    const std::string& tableName) {
  auto globalRank = RankUtils::getGlobalRank().value_or(-1);
  return fmt::format(
      "{}/dedicated_log_structured_json.perfpipe_{}.Rank_{}.{}.scribe",
      logFilePrefix,
      tableName,
      globalRank,
      getUniqueFileSuffix());
}

std::optional<folly::File> createScubaFile(
    const std::string& fileName,
    bool appendMode) {
  try {
    // Extract the directory path and create the directory if it doesn't exist
    std::filesystem::path filePath{fileName};
    std::filesystem::path dirPath{filePath.parent_path()};
    std::filesystem::create_directories(dirPath);
    int flags = O_CREAT | O_WRONLY | (appendMode ? O_APPEND : 0);
    return folly::File(fileName, flags, 0644);
  } catch (const std::exception& e) {
    XLOG(WARNING) << "Failed to create scuba file " << fileName << ": "
                  << e.what();
  }
  return std::nullopt;
}

int64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t getTimestampMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t getTimestampUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

} // namespace comms::logger
