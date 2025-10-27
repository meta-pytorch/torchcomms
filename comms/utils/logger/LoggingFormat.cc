// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LoggingFormat.h"

#include <unistd.h>
#include <cstring>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/LogName.h>
#include <folly/synchronization/CallOnce.h>

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
} procMetaData;
folly::once_flag procMetaDataInitFlag;
static thread_local std::string myThreadName = "main";

// Using char array to ensure compatibility with NCCL GetLastError API
static folly::Synchronized<std::array<char, 1024>> lastCommsError{
    std::array<char, 1024>{'\0'}};

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
  uint64_t maskResult{0};

  int invert = 0;
  if (ncclDebugSubsysEnv[0] == '^') {
    invert = 1;
    ncclDebugSubsysEnv++;
  }
  maskResult = invert ? ~0ULL : 0ULL;
  char* ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
  // Fixme: this is not thread safe, rewrite with folly::split
  char* subsys = strtok(ncclDebugSubsys, ",");
  while (subsys != nullptr) {
    uint64_t mask = 0;
    if (strcasecmp(subsys, "INIT") == 0) {
      mask = INIT;
    } else if (strcasecmp(subsys, "COLL") == 0) {
      mask = COLL;
    } else if (strcasecmp(subsys, "P2P") == 0) {
      mask = P2P;
    } else if (strcasecmp(subsys, "SHM") == 0) {
      mask = SHM;
    } else if (strcasecmp(subsys, "NET") == 0) {
      mask = NET;
    } else if (strcasecmp(subsys, "GRAPH") == 0) {
      mask = GRAPH;
    } else if (strcasecmp(subsys, "TUNING") == 0) {
      mask = TUNING;
    } else if (strcasecmp(subsys, "ENV") == 0) {
      mask = ENV;
    } else if (strcasecmp(subsys, "ALLOC") == 0) {
      mask = ALLOC;
    } else if (strcasecmp(subsys, "CALL") == 0) {
      mask = CALL;
    } else if (strcasecmp(subsys, "PROXY") == 0) {
      mask = PROXY;
    } else if (strcasecmp(subsys, "NVLS") == 0) {
      mask = NVLS;
    } else if (strcasecmp(subsys, "BOOTSTRAP") == 0) {
      mask = BOOTSTRAP;
    } else if (strcasecmp(subsys, "REG") == 0) {
      mask = REG;
    } else if (strcasecmp(subsys, "PROFILE") == 0) {
      mask = PROFILE;
    } else if (strcasecmp(subsys, "RAS") == 0) {
      mask = RAS;
    } else if (strcasecmp(subsys, "ALL") == 0) {
      mask = ALL;
    }
    if (mask) {
      if (invert) {
        maskResult &= ~mask;
      } else {
        maskResult |= mask;
      }
    }
    subsys = strtok(nullptr, ",");
  }
  free(ncclDebugSubsys);
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
            procMetaData.hostname.c_str());
        break;
      case 'p': // %p = pid
        dfn += snprintf(
            dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", procMetaData.pid);
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
  folly::call_once(procMetaDataInitFlag, []() {
    procMetaData.hostname = getHostName('.');
    procMetaData.pid = getpid();
  });
}

void initThreadMetaData(std::string_view threadName) {
  static thread_local folly::once_flag threadNameFlag;
  folly::call_once(threadNameFlag, [&]() { myThreadName = threadName; });
}

std::string NcclLogFormatter::formatMessage(
    const folly::LogMessage& message,
    const folly::LogCategory* /* handlerCategory */) {
  initProcMetaData();

  bool isErrorMessage = message.getLevel() >= folly::LogLevel::ERR;
  if (isErrorMessage) {
    auto lastErrorLocked = lastCommsError.wlock();
    std::snprintf(
        lastErrorLocked->data(),
        lastErrorLocked->size(),
        "%s",
        message.getMessage().c_str());
  }

  auto timeSinceEpoch = message.getTimestamp().time_since_epoch();
  auto epochSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(timeSinceEpoch);
  std::chrono::microseconds usecs =
      std::chrono::duration_cast<std::chrono::microseconds>(timeSinceEpoch) -
      epochSeconds;

  // At least for now, formatter is called in the same thread as the logging
  // thread. So we don't need to worry about getting the information of another
  // thread here.
  int cudaDev = threadContextFn_();

  auto basename = message.getFileBaseName();
  // Format: <Glog format> <hostname>:<pid>:<tid> [<threadCtx>][<threadName>]
  // <prefix> <logLevel>
  // Example: W0414 11:46:56.369712 4115466 Logger.cc:25]
  // devvm2605:4115466:4115466 [-1][main] NCCL WARN
  auto header = fmt::format(
      "{}{:%m%d %H:%M:%S}.{:06d} {:5d} {}:{}] {}:{}:{} [{}][{}] {} {} ",
      getGlogLevelName(message.getLevel())[0],
      message.getTimestamp(),
      usecs.count(),
      message.getThreadID(),
      basename,
      message.getLineNumber(),
      procMetaData.hostname,
      procMetaData.pid,
      message.getThreadID(),
      cudaDev,
      myThreadName,
      prefix_,
      getGlogLevelName(message.getLevel()));

  // The fixed portion of the header takes up 31 bytes.
  //
  // The variable portions that we can't account for here include the line
  // number and the thread ID (just in case it is larger than 6 digits long).
  // Here we guess that 40 bytes will be long enough to include room for this.
  //
  // If this still isn't long enough the string will grow as necessary, so the
  // code will still be correct, but just slightly less efficient than if we
  // had allocated a large enough buffer the first time around.
  size_t headerLengthGuess = 90 + basename.size();

  // Format the data into a buffer.
  std::string buffer;
  std::string_view msgData{message.getMessage()};
  if (message.containsNewlines()) {
    // If there are multiple lines in the log message, add a header
    // before each one.

    buffer.reserve(
        ((header.size() + 1) * message.getNumNewlines()) + msgData.size());

    size_t idx = 0;
    while (true) {
      auto end = msgData.find('\n', idx);
      if (end == std::string_view::npos) {
        end = msgData.size();
      }

      buffer.append(header);
      auto line = msgData.substr(idx, end - idx);
      buffer.append(line.data(), line.size());
      buffer.push_back('\n');

      if (end == msgData.size()) {
        break;
      }
      idx = end + 1;
    }
  } else {
    buffer.reserve(headerLengthGuess + msgData.size());
    buffer.append(header);
    buffer.append(msgData.data(), msgData.size());
    buffer.push_back('\n');
  }

  return buffer;
}

const char* getLastCommsError() {
  return lastCommsError.rlock()->data();
}

NcclLogFormatter::NcclLogFormatter(
    std::string prefix,
    std::function<int(void)> threadContextFn)
    : prefix_(std::move(prefix)),
      threadContextFn_(std::move(threadContextFn)) {};

} // namespace meta::comms::logger
