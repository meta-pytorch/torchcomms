// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/CommsLogging.h"

#include <unistd.h>
#include <array>
#include <cstdio>
#include <sstream>
#include <utility>

#include <folly/Synchronized.h>
#include <folly/synchronization/CallOnce.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/ErrorStackUtil.h"
#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/NcclScubaSample.h"

namespace {

std::string getHostName(const char delim) {
  constexpr int maxlen = HOST_NAME_MAX + 1;
  char hostname[maxlen];
  if (gethostname(hostname, maxlen) != 0) {
    return "unknown";
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) {
    i++;
  }
  hostname[i] = '\0';
  return std::string{hostname};
}

struct ProcMetaData {
  ProcMetaData() : hostname(getHostName('.')), pid(getpid()) {}

  std::string hostname;
  int pid;
};

ProcMetaData& getProcMetaData() {
  /*
   * Retain process metadata for the process lifetime because formatting may
   * run during static destruction.
   */
  static auto* metaData = new ProcMetaData{};
  return *metaData;
}

struct LastErrorInfo {
  std::string lastErrorMessage;
  // Legacy per-frame error chain, still appended by v2_27/v2_29's debug.cc via
  // appendErrorToStack(). Kept for backward compatibility.
  // TODO: remove once ncclx v2_29 is retired -- only v2_29 populates this;
  // v2_30 is native-stack-only (lastErrorNativeStack), so this field and the
  // getLastCommsError() fallback that reads it become dead.
  std::vector<std::string> lastErrorStack;
  // Native symbolized stack captured at the error site by logErrorToScuba().
  // Preferred by getLastCommsError() when present.
  std::vector<std::string> lastErrorNativeStack;
};

folly::Synchronized<LastErrorInfo>& getLastCommsErrorStorage() {
  /*
   * Error callbacks remain reachable from logger threads during static
   * destruction, so this storage must not register an exit-time destructor.
   */
  static auto* storage = new folly::Synchronized<LastErrorInfo>{};
  return *storage;
}

constexpr char toAsciiUpper(char value) {
  return value >= 'a' && value <= 'z' ? value - ('a' - 'A') : value;
}

bool asciiCaseInsensitiveEqual(std::string_view left, std::string_view right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (toAsciiUpper(left[index]) != toAsciiUpper(right[index])) {
      return false;
    }
  }
  return true;
}

uint64_t getSubSystemMaskForName(std::string_view name) {
  struct NamedMask {
    std::string_view name;
    uint64_t mask;
  };
  static constexpr std::array<NamedMask, 18> kNamedMasks{{
      {"INIT", meta::comms::logger::INIT},
      {"COLL", meta::comms::logger::COLL},
      {"P2P", meta::comms::logger::P2P},
      {"SHM", meta::comms::logger::SHM},
      {"NET", meta::comms::logger::NET},
      {"GRAPH", meta::comms::logger::GRAPH},
      {"TUNING", meta::comms::logger::TUNING},
      {"ENV", meta::comms::logger::ENV},
      {"ALLOC", meta::comms::logger::ALLOC},
      {"CALL", meta::comms::logger::CALL},
      {"PROXY", meta::comms::logger::PROXY},
      {"NVLS", meta::comms::logger::NVLS},
      {"BOOTSTRAP", meta::comms::logger::BOOTSTRAP},
      {"REG", meta::comms::logger::REG},
      {"PROFILE", meta::comms::logger::PROFILE},
      {"RAS", meta::comms::logger::RAS},
      {"DESTROY", meta::comms::logger::DESTROY},
      {"ALL", static_cast<uint64_t>(meta::comms::logger::ALL)},
  }};
  for (const auto& namedMask : kNamedMasks) {
    if (asciiCaseInsensitiveEqual(name, namedMask.name)) {
      return namedMask.mask;
    }
  }
  return 0;
}

} // namespace

namespace meta::comms::logger {

void initProcMetaData() {
  (void)getProcMetaData();
}

uint64_t parseDebugSubsysMask(const char* ncclDebugSubsysEnv) {
  if (ncclDebugSubsysEnv == nullptr) {
    return INIT | BOOTSTRAP | ENV;
  }
  std::string_view input{ncclDebugSubsysEnv};
  const bool invert = !input.empty() && input.front() == '^';
  if (invert) {
    input.remove_prefix(1);
  }
  uint64_t maskResult = invert ? ~0ULL : 0ULL;
  while (!input.empty()) {
    const auto delimiter = input.find(',');
    const auto token = input.substr(0, delimiter);
    const auto mask = getSubSystemMaskForName(token);
    if (mask) {
      if (invert) {
        maskResult &= ~mask;
      } else {
        maskResult |= mask;
      }
    }
    if (delimiter == std::string_view::npos) {
      break;
    }
    input.remove_prefix(delimiter + 1);
  }
  return maskResult;
}

std::string parseDebugFile(const char* ncclDebugFileEnv) {
  if (ncclDebugFileEnv == nullptr) {
    return {};
  }
  initProcMetaData();

  int c = 0;
  char debugFn[PATH_MAX + 1] = "";
  char* dfn = debugFn;
  while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
    if (ncclDebugFileEnv[c++] != '%') {
      *dfn++ = ncclDebugFileEnv[c - 1];
      continue;
    }
    switch (ncclDebugFileEnv[c++]) {
      case '%':
        *dfn++ = '%';
        break;
      case 'h':
        dfn += snprintf(
            dfn,
            PATH_MAX + 1 - (dfn - debugFn),
            "%s",
            getProcMetaData().hostname.c_str());
        break;
      case 'p':
        dfn += snprintf(
            dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", getProcMetaData().pid);
        break;
      default:
        *dfn++ = '%';
        if ((dfn - debugFn) < PATH_MAX) {
          *dfn++ = ncclDebugFileEnv[c - 1];
        }
        break;
    }
    if ((dfn - debugFn) > PATH_MAX) {
      dfn = debugFn + PATH_MAX;
    }
  }
  *dfn = '\0';
  return std::string{debugFn};
}

LogLevel getLoggerDebugLevel(std::string_view level) {
  if (level.empty()) {
    return LogLevel::NONE;
  }
  if (level == "VERSION") {
    return LogLevel::VERSION;
  } else if (level == "ERROR") {
    return LogLevel::ERROR;
  } else if (level == "WARN") {
    return LogLevel::WARN;
  } else if (level == "INFO") {
    return LogLevel::INFO;
  } else if (level == "ABORT") {
    return LogLevel::ABORT;
  } else if (level == "TRACE") {
    return LogLevel::TRACE;
  } else if (level == "NONE") {
    return LogLevel::NONE;
  }
  return LogLevel::NONE;
}

LogLevel getNcclLoggerDebugLevel(std::string_view level) {
  if (asciiCaseInsensitiveEqual(level, "VERSION")) {
    return LogLevel::VERSION;
  } else if (asciiCaseInsensitiveEqual(level, "ERROR")) {
    return LogLevel::ERROR;
  } else if (asciiCaseInsensitiveEqual(level, "WARN")) {
    return LogLevel::WARN;
  } else if (asciiCaseInsensitiveEqual(level, "INFO")) {
    return LogLevel::INFO;
  } else if (asciiCaseInsensitiveEqual(level, "ABORT")) {
    return LogLevel::ABORT;
  } else if (asciiCaseInsensitiveEqual(level, "TRACE")) {
    return LogLevel::TRACE;
  }
  return LogLevel::NONE;
}

bool parseDebugLoggingAsync(const char* value, bool valueWhenUnset) {
  if (value == nullptr) {
    return valueWhenUnset;
  }
  const std::string_view input{value};
  if (asciiCaseInsensitiveEqual(input, "1") ||
      asciiCaseInsensitiveEqual(input, "Y") ||
      asciiCaseInsensitiveEqual(input, "YES") ||
      asciiCaseInsensitiveEqual(input, "T") ||
      asciiCaseInsensitiveEqual(input, "TRUE")) {
    return true;
  }
  if (asciiCaseInsensitiveEqual(input, "0") ||
      asciiCaseInsensitiveEqual(input, "N") ||
      asciiCaseInsensitiveEqual(input, "NO") ||
      asciiCaseInsensitiveEqual(input, "F") ||
      asciiCaseInsensitiveEqual(input, "FALSE")) {
    return false;
  }
  return true;
}

detail::ProcessMetadata detail::getLogProcessMetadata() {
  const auto& metadata = getProcMetaData();
  return {.hostname = metadata.hostname, .processId = metadata.pid};
}

void initThreadMetaData(std::string_view threadName) {
  static thread_local folly::once_flag threadNameFlag;
  folly::call_once(threadNameFlag, [&]() { setSpdlogThreadName(threadName); });
}

void detail::setLastErrorFromLegacyLog(std::string_view message) {
  auto lockedError = getLastCommsErrorStorage().wlock();
  lockedError->lastErrorMessage.assign(message);
  lockedError->lastErrorNativeStack.clear();
}

void logErrorToScuba(
    const std::string& message,
    const int code,
    const std::string& errorName,
    const std::vector<std::string>& stack) {
  auto sampleGuard = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("ERROR");
  auto& sample = sampleGuard.sample();
  sample.setError(message, stack);
  if (code != 0) {
    sample.addNormal("error_code", fmt::format("{}:{}", code, errorName));
  }
}

void setLastError(const std::string& message, std::vector<std::string> stack) {
  auto w = getLastCommsErrorStorage().wlock();
  w->lastErrorMessage = message;
  w->lastErrorNativeStack = std::move(stack);
}

void logCommErrorToScuba(commResult_t code, const std::string& message) {
  if (!NCCL_SCUBA_LOG_ERROR_ENABLED) {
    return;
  }
  std::vector<std::string> stack;
  if (NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED) {
    stack = captureNativeErrorStack();
  }
  logErrorToScuba(
      message,
      static_cast<int>(code),
      ::meta::comms::commCodeToString(code),
      stack);
}

std::string getLastCommsError() {
  std::ostringstream ss;
  {
    auto lastCommsErrorRLocked = getLastCommsErrorStorage().rlock();
    ss << lastCommsErrorRLocked->lastErrorMessage << "\nNCCL Stack trace:";
    const auto& stackTrace =
        !lastCommsErrorRLocked->lastErrorNativeStack.empty()
        ? lastCommsErrorRLocked->lastErrorNativeStack
        : lastCommsErrorRLocked->lastErrorStack;
    for (const auto& stack : stackTrace) {
      ss << '\n' << stack;
    }
  }
  return ss.str();
}

void appendErrorToStack(std::string error) {
  getLastCommsErrorStorage().wlock()->lastErrorStack.push_back(
      std::move(error));
}

} // namespace meta::comms::logger
