// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

/**
 * Logging for ibverbx, which needs to emit diagnostics without depending on any
 * particular host logger. Kept here, beside Expected.h and ScopeGuard.h, so
 * that ibverbx and its OSS/CMake build stay self-contained.
 *
 * By default a line is formatted with fmt and written to stderr with a single
 * write, so concurrent ranks and threads cannot tear each other's output.
 *
 * A host that already owns a logger installs setLogSink() once during its own
 * initialization and thereafter receives every line, gating and routing it
 * itself. That keeps the dependency pointing host -> ibverbx: ibverbx needs no
 * build dependency on the host, and a host that installs nothing still gets
 * usable output.
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include <fmt/core.h>

namespace ibverbx::logging {

// Ordered by severity; shouldLog() compares against it.
enum class Level {
  DBG,
  INFO,
  WARN,
  ERR,
  FATAL,
};

// Receives one fully formatted message, without a trailing newline. Called on
// the thread that logged, so an implementation must be thread-safe.
using LogSink = void (*)(
    Level level,
    const char* file,
    int line,
    const char* function,
    const std::string& message);

namespace detail {

/*
 * The installed sink, or nullptr while ibverbx owns delivery. Function-local so
 * there is exactly one per linkage image even though this header is inline.
 *
 * Atomic because a host may install its sink after ibverbx threads are already
 * logging -- ncclx initializes its logger while another host in the same
 * process already drives ibverbx.
 */
inline std::atomic<LogSink>& logSink() {
  static std::atomic<LogSink> sink{nullptr};
  return sink;
}

inline LogSink loadLogSink() {
  return logSink().load(std::memory_order_acquire);
}

} // namespace detail

// Install a sink, or pass nullptr to return to the built-in stderr delivery.
inline void setLogSink(LogSink sink) {
  detail::logSink().store(sink, std::memory_order_release);
}

constexpr const char* levelName(const Level level) {
  switch (level) {
    case Level::DBG:
      return "DBG";
    case Level::INFO:
      return "INFO";
    case Level::WARN:
      return "WARN";
    case Level::ERR:
      return "ERR";
    case Level::FATAL:
      return "FATAL";
  }
  return "UNKNOWN";
}

/*
 * Whether a level is worth formatting. Only gates the built-in stderr path: an
 * installed sink sees every level and applies the host's own threshold, so a
 * host stays in sole charge of which ibverbx lines it keeps.
 *
 * Inline and checked before formatting, so a dropped line costs one comparison
 * and evaluates none of its arguments.
 */
inline bool shouldLog(const Level level) {
  return detail::loadLogSink() != nullptr || level >= Level::INFO;
}

/*
 * Emit one line, then terminate if it was FATAL. Aborting here rather than at
 * the call site means a FATAL cannot be written without also terminating.
 * stderr is flushed first so the message explaining the crash is never left
 * buffered.
 */
inline void logMessage(
    const Level level,
    const char* const file,
    const int line,
    const char* const function,
    const std::string& message) {
  if (const LogSink sink = detail::loadLogSink()) {
    sink(level, file, line, function, message);
  } else {
    // One write per line: two ranks sharing a terminal must not interleave.
    const std::string rendered = fmt::format(
        "{}:{} {}] IBVERBX {} {}\n",
        file,
        line,
        function,
        levelName(level),
        message);
    std::fwrite(rendered.data(), 1, rendered.size(), stderr);
    if (level >= Level::WARN) {
      std::fflush(stderr);
    }
  }

  if (level == Level::FATAL) {
    std::fflush(stderr);
    std::abort();
  }
}

template <typename... Args>
void logf(
    const Level level,
    const char* const file,
    const int line,
    const char* const function,
    fmt::format_string<Args...> format,
    Args&&... args) {
  logMessage(
      level,
      file,
      line,
      function,
      fmt::format(format, std::forward<Args>(args)...));
}

} // namespace ibverbx::logging

/*
 * Log at a bare level token, e.g. IBVERBX_LOG(INFO, "n={}", n). FATAL
 * terminates inside logMessage().
 *
 * The gate is in the macro rather than in logf() so a dropped line does not
 * even evaluate its arguments. FATAL always passes it.
 */
#define IBVERBX_LOG(level, ...)                                            \
  do {                                                                     \
    if (::ibverbx::logging::shouldLog(::ibverbx::logging::Level::level)) { \
      ::ibverbx::logging::logf(                                            \
          ::ibverbx::logging::Level::level,                                \
          __FILE__,                                                        \
          __LINE__,                                                        \
          __func__,                                                        \
          __VA_ARGS__);                                                    \
    }                                                                      \
  } while (false)

/*
 * Log only when `cond` holds. The condition is evaluated first and the message
 * arguments not at all when it is false, which is what keeps a FATAL assertion
 * on a hot path free.
 */
#define IBVERBX_LOG_IF(level, cond, ...) \
  do {                                   \
    if ((cond)) {                        \
      IBVERBX_LOG(level, __VA_ARGS__);   \
    }                                    \
  } while (false)
