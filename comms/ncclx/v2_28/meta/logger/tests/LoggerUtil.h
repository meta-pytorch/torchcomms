// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <filesystem>
#include <string>

#include "comms/utils/cvars/nccl_cvars.h"

inline std::string getScubaFile(const std::string& scubaTable) {
  static std::string scubaLogFile{};
  // Need to do this here because scuba logger is lazily initialized on first
  // sample.
  for (const auto& entry :
       std::filesystem::directory_iterator(NCCL_SCUBA_LOG_FILE_PREFIX)) {
    if (entry.is_regular_file() &&
        entry.path().string().starts_with(
            NCCL_SCUBA_LOG_FILE_PREFIX +
            "/dedicated_log_structured_json.perfpipe_" + scubaTable)) {
      scubaLogFile = entry.path().string();
      break;
    }
  }
  if (scubaLogFile.empty()) {
    throw std::runtime_error("No log file found");
  }
  return scubaLogFile;
}

inline std::string getCommEventScubaFile(
    const std::string scubaTable = "nccl_structured_logging") {
  return getScubaFile(scubaTable);
}

inline std::string getMemoryEventScubaFile(
    const std::string scubaTable = "nccl_memory_logging") {
  return getScubaFile(scubaTable);
}
