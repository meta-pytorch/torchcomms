// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LoggingFormat.h"

#include <unistd.h>
#include <array>
#include <sstream>
#include <string_view>

#include <fmt/format.h>
#include <folly/String.h>
#include <folly/Synchronized.h>
#include <folly/logging/LogCategory.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/LogName.h>
#include <folly/synchronization/CallOnce.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/CommsLogFormatter.h"
#include "comms/utils/logger/ErrorStackUtil.h"
#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/NcclScubaSample.h"
#include "comms/utils/logger/SpdlogLogger.h"

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
  std::string hostname;
  int pid{};
  folly::once_flag initFlag;
};

ProcMetaData& getProcMetaData() {
  /*
   * Retain process metadata for the process lifetime because formatting may run
   * during static destruction. Keeping the initialization guard here also
   * prevents its destruction from racing with initialization.
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
  // Error callbacks remain reachable from logger threads during static
  // destruction, so this storage must not register an exit-time destructor.
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
  static constexpr std::array<NamedMask, 17> kNamedMasks{{
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
      {"ALL", static_cast<uint64_t>(meta::comms::logger::ALL)},
  }};
  for (const auto& namedMask : kNamedMasks) {
    if (asciiCaseInsensitiveEqual(name, namedMask.name)) {
      return namedMask.mask;
    }
  }
  return 0;
}
} // Anonymous namespace

namespace meta::comms::logger {

folly::LogLevel loggerLevelToFollyLogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::NONE:
    case LogLevel::VERSION:
      return folly::LogLevel::FATAL;
    case LogLevel::ERROR:
      return folly::LogLevel::ERR;
    case LogLevel::WARN:
      return folly::LogLevel::WARN;
    case LogLevel::INFO:
      return folly::LogLevel::INFO;
    case LogLevel::ABORT:
    case LogLevel::TRACE:
      return folly::LogLevel::DBG;
    default:
      return folly::LogLevel::UNINITIALIZED;
  }
}

std::string_view getGlogLevelName(folly::LogLevel level) {
  if (level < folly::LogLevel::INFO) {
    return "VERBOSE";
  } else if (level < folly::LogLevel::WARN) {
    return "INFO";
  } else if (level < folly::LogLevel::ERR) {
    return "WARN";
  } else if (level < folly::LogLevel::CRITICAL) {
    return "ERROR";
  } else if (level < folly::LogLevel::DFATAL) {
    return "CRITICAL";
  }
  return "FATAL";
}

folly::StringPiece getCategoryNthParent(folly::StringPiece category, int n) {
  for (auto i = 0; i < n; i++) {
    category = ::folly::LogName::getParent(category);
  }
  return category;
}

/* Parse the DEBUG_SUBSYS env var
 * This can be a comma separated list such as INIT,COLL
 * or ^INIT,COLL etc
 */
uint64_t parseDebugSubsysMask(const char* ncclDebugSubsysEnv) {
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
      case '%': // Double %
        *dfn++ = '%';
        break;
      case 'h': // %h = hostname
        dfn += snprintf(
            dfn,
            PATH_MAX + 1 - (dfn - debugFn),
            "%s",
            getProcMetaData().hostname.c_str());
        break;
      case 'p': // %p = pid
        dfn += snprintf(
            dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", getProcMetaData().pid);
        break;
      default: // Echo everything we don't understand
        *dfn++ = '%';
        if ((dfn - debugFn) < PATH_MAX) {
          *dfn++ = ncclDebugFileEnv[c - 1];
        }
        break;
    }
    if ((dfn - debugFn) > PATH_MAX) {
      // snprintf wanted to overfill the buffer: set dfn to the end
      // of the buffer (for null char) and it will naturally exit
      // the loop.
      dfn = debugFn + PATH_MAX;
    }
  }
  *dfn = '\0';
  return std::string{debugFn};
}

LogLevel getLoggerDebugLevel(std::string_view level) {
  // If the env var is empty, then we default to log nothing
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
  // TODO: Add a warning here
  return LogLevel::NONE;
}

void initProcMetaData() {
  auto& metaData = getProcMetaData();
  folly::call_once(metaData.initFlag, [&metaData]() {
    metaData.hostname = getHostName('.');
    metaData.pid = getpid();
  });
}

void initThreadMetaData(std::string_view threadName) {
  static thread_local folly::once_flag threadNameFlag;
  folly::call_once(threadNameFlag, [&]() { setSpdlogThreadName(threadName); });
}

std::string NcclLogFormatter::formatMessage(
    const folly::LogMessage& message,
    const folly::LogCategory* /* handlerCategory */) {
  initProcMetaData();

  bool isErrorMessage = message.getLevel() >= folly::LogLevel::ERR;
  if (isErrorMessage) {
    // Errors are recorded to Scuba at their call sites (ncclMetaDebugLogError
    // for NCCL, CERR for CTRAN), each of which captures a fresh native stack
    // via logErrorToScuba(). Clear any stale cached native stack here so
    // getLastCommsError() does not pair this message with an unrelated stack;
    // the call site's logErrorToScuba(), which runs after this formatter,
    // re-sets it when present, and a bare XLOG(ERR) correctly falls back to the
    // legacy per-frame chain.
    auto lockedError = getLastCommsErrorStorage().wlock();
    lockedError->lastErrorMessage = message.getMessage();
    lockedError->lastErrorNativeStack.clear();
  }

  // At least for now, formatter is called in the same thread as the logging
  // thread. So we don't need to worry about getting the information of another
  // thread here.
  int cudaDev = threadContextFn_();

  const auto basename = message.getFileBaseName();
  return formatCommsLogMessage(
      getGlogLevelName(message.getLevel()),
      message.getMessage(),
      {.timestamp = message.getTimestamp(),
       .threadId = message.getThreadID(),
       .filename = std::string_view{basename.data(), basename.size()},
       .lineNumber = message.getLineNumber(),
       .hostname = getProcMetaData().hostname,
       .processId = getProcMetaData().pid,
       .threadContext = cudaDev,
       .threadName = getLogThreadName(),
       .prefix = prefix_});
}

void logErrorToScuba(
    const std::string& message,
    const int code,
    const std::string& errorName,
    const std::vector<std::string>& stack) {
  // Build one Scuba record for the whole error; the guard flushes it to
  // nccl_structured_logging on scope exit and keeps the sticky-context columns.
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
    // Prefer the native captured stack; fall back to the legacy per-frame
    // chain (still populated by v2_27/v2_29) when no native stack is present.
    // TODO: remove the lastErrorStack fallback once ncclx v2_29 is retired --
    // v2_30 is native-only, so this can collapse to lastErrorNativeStack.
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

// TODO: remove once ncclx v2_29 is retired (see appendErrorToStack decl in
// LoggingFormat.h) -- v2_30/ctran use captureNativeErrorStack() +
// setLastError().
void appendErrorToStack(std::string error) {
  getLastCommsErrorStorage().wlock()->lastErrorStack.push_back(
      std::move(error));
}

NcclLogFormatter::NcclLogFormatter(
    std::string prefix,
    std::function<int(void)> threadContextFn)
    : prefix_(std::move(prefix)),
      threadContextFn_(std::move(threadContextFn)) {};

} // namespace meta::comms::logger
