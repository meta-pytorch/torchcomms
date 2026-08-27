// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/SpdlogLogger.h"

#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/details/periodic_worker.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "comms/utils/logger/CommsLogFormatter.h"

namespace meta::comms::logger {

void reportCommsLoggingFailureToStderr(const char* level) noexcept {
  std::fprintf(stderr, "%s: communications logging failed\n", level);
  std::fflush(stderr);
}

[[noreturn]] void abortAfterCommsLoggingFailure() noexcept {
  reportCommsLoggingFailureToStderr("FATAL");
  std::abort();
}

spdlog::level::level_enum loggerLevelToSpdlogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::NONE:
    case LogLevel::VERSION:
      return spdlog::level::off;
    case LogLevel::ERROR:
      return spdlog::level::err;
    case LogLevel::WARN:
      return spdlog::level::warn;
    case LogLevel::INFO:
      return spdlog::level::info;
    case LogLevel::ABORT:
      return spdlog::level::debug;
    case LogLevel::TRACE:
      return spdlog::level::trace;
  }
  return spdlog::level::off;
}
namespace {

constexpr size_t kAsyncQueueSize = 8192;
constexpr size_t kAsyncThreadCount = 1;
/*
 * Bounds how much buffered file output an abnormal exit can lose. Folly wrote
 * through to the fd, so without a periodic flush the migrated path would keep
 * recent lines only in the stdio buffer.
 */
constexpr auto kPeriodicFlushInterval = std::chrono::seconds{1};
// Guard the full callback chain so it cannot recurse through another context
// logger and eventually re-enter the originating callback.
thread_local bool errorCallbackInProgress = false;

class CommsThreadPoolState final {
 public:
  class Lease final {
   public:
    Lease() = default;
    Lease(
        std::shared_lock<std::shared_mutex> lock,
        std::shared_ptr<spdlog::details::thread_pool> threadPool)
        : lock_(std::move(lock)), threadPool_(std::move(threadPool)) {}

    explicit operator bool() const {
      return threadPool_ != nullptr;
    }

    const std::shared_ptr<spdlog::details::thread_pool>& threadPool() const {
      return threadPool_;
    }

   private:
    // Destroy the strong reference before releasing the shared shutdown lock.
    std::shared_lock<std::shared_mutex> lock_;
    std::shared_ptr<spdlog::details::thread_pool> threadPool_;
  };

  CommsThreadPoolState()
      : threadPool_{std::make_shared<spdlog::details::thread_pool>(
            kAsyncQueueSize,
            kAsyncThreadCount)} {}

  Lease acquire() const {
    if (stopping_.load(std::memory_order_acquire)) {
      return {};
    }
    std::shared_lock lock{leaseMutex_};
    if (stopping_.load(std::memory_order_acquire)) {
      return {};
    }
    return {std::move(lock), threadPool_};
  }

  void stop() {
    {
      std::lock_guard lock{lifecycleMutex_};
      stopping_.store(true, std::memory_order_release);
    }
    lifecycleCv_.notify_all();

    // The exclusive lock rejects new leases and waits for existing calls to
    // release theirs before destroying, draining, and joining the pool.
    std::unique_lock lock{leaseMutex_};
    auto threadPool = std::move(threadPool_);
    threadPool.reset();
  }

  void waitForShutdownToStart() const {
    std::unique_lock lock{lifecycleMutex_};
    lifecycleCv_.wait(
        lock, [this]() { return stopping_.load(std::memory_order_acquire); });
  }

 private:
  std::atomic<bool> stopping_{false};
  mutable std::mutex lifecycleMutex_;
  mutable std::condition_variable lifecycleCv_;
  mutable std::shared_mutex leaseMutex_;
  std::shared_ptr<spdlog::details::thread_pool> threadPool_;
};

class CommsThreadPoolStopper final {
 public:
  explicit CommsThreadPoolStopper(CommsThreadPoolState& state)
      : state_(state) {}
  ~CommsThreadPoolStopper() {
    state_.stop();
  }

  CommsThreadPoolStopper(const CommsThreadPoolStopper&) = delete;
  CommsThreadPoolStopper& operator=(const CommsThreadPoolStopper&) = delete;
  CommsThreadPoolStopper(CommsThreadPoolStopper&&) = delete;
  CommsThreadPoolStopper& operator=(CommsThreadPoolStopper&&) = delete;

 private:
  CommsThreadPoolState& state_;
};

CommsThreadPoolState& getCommsThreadPoolState() {
  static auto* state = new CommsThreadPoolState{};
  static const CommsThreadPoolStopper stopper{*state};
  return *state;
}

/*
 * Deliberately leaked along with the logger-owned static state below. A process
 * that exits without aborting its NCCL communicator leaves the proxy,
 * CollTrace, and watchdog threads logging while __cxa_atexit runs, so a logger
 * with a destructor would be freed out from under them. Leaking behind a
 * pointer registers no destructor, so the object outlives every thread that can
 * reach it.
 */
class PeriodicSinkFlusher final {
 public:
  PeriodicSinkFlusher()
      : worker_{std::make_unique<spdlog::details::periodic_worker>(
            [this]() noexcept { flushRegisteredSinks(); },
            kPeriodicFlushInterval)} {}

  // Joins the flush thread so it cannot touch sinks or stdio during exit. The
  // flusher itself survives, so registerSink() from a running thread stays
  // safe.
  void stopFlushing() {
    worker_.reset();
  }

  void registerSink(const std::shared_ptr<spdlog::sinks::sink>& sink) {
    std::lock_guard lock{mutex_};
    for (auto it = sinks_.begin(); it != sinks_.end();) {
      if (const auto registered = it->lock()) {
        if (registered == sink) {
          return;
        }
        ++it;
      } else {
        it = sinks_.erase(it);
      }
    }
    sinks_.push_back(sink);
  }

 private:
  void flushRegisteredSinks() noexcept {
    std::vector<std::shared_ptr<spdlog::sinks::sink>> sinks;
    {
      std::lock_guard lock{mutex_};
      for (auto it = sinks_.begin(); it != sinks_.end();) {
        if (auto sink = it->lock()) {
          sinks.push_back(std::move(sink));
          ++it;
        } else {
          it = sinks_.erase(it);
        }
      }
    }

    for (const auto& sink : sinks) {
      try {
        sink->flush();
      } catch (...) {
        // A sink failure must not terminate the process from this worker.
      }
    }
  }

  std::mutex mutex_;
  std::vector<std::weak_ptr<spdlog::sinks::sink>> sinks_;
  // Declared last so it is joined before the state its callback reads.
  std::unique_ptr<spdlog::details::periodic_worker> worker_;
};

// Stops the flush thread at exit without destroying the flusher it points at.
class PeriodicFlushStopper final {
 public:
  explicit PeriodicFlushStopper(PeriodicSinkFlusher& flusher)
      : flusher_(flusher) {}
  ~PeriodicFlushStopper() {
    flusher_.stopFlushing();
  }

  PeriodicFlushStopper(const PeriodicFlushStopper&) = delete;
  PeriodicFlushStopper& operator=(const PeriodicFlushStopper&) = delete;
  PeriodicFlushStopper(PeriodicFlushStopper&&) = delete;
  PeriodicFlushStopper& operator=(PeriodicFlushStopper&&) = delete;

 private:
  PeriodicSinkFlusher& flusher_;
};

PeriodicSinkFlusher& getPeriodicSinkFlusher() {
  static auto* flusher = new PeriodicSinkFlusher{};
  // Constructed after the logger and spdlog registry, so atexit's LIFO order
  // joins the flush thread before spdlog's exit-time teardown begins.
  static const PeriodicFlushStopper stopper{*flusher};
  return *flusher;
}

class ErrorCallbackGuard {
 public:
  ErrorCallbackGuard() {
    errorCallbackInProgress = true;
  }
  ~ErrorCallbackGuard() {
    errorCallbackInProgress = false;
  }

  ErrorCallbackGuard(const ErrorCallbackGuard&) = delete;
  ErrorCallbackGuard& operator=(const ErrorCallbackGuard&) = delete;
  ErrorCallbackGuard(ErrorCallbackGuard&&) = delete;
  ErrorCallbackGuard& operator=(ErrorCallbackGuard&&) = delete;
};

class StderrRoutingSink final : public spdlog::sinks::sink {
 public:
  StderrRoutingSink()
      : sink_(std::make_shared<spdlog::sinks::stderr_color_sink_mt>()) {}

  void log(const spdlog::details::log_msg& message) override {
    if (should_log(message.level) &&
        shouldWriteCommsLogToStderr(
            std::string_view{message.payload.data(), message.payload.size()})) {
      sink_->log(message);
    }
  }

  void flush() override {
    sink_->flush();
  }

  void set_pattern(const std::string& pattern) override {
    sink_->set_pattern(pattern);
  }

  void set_formatter(std::unique_ptr<spdlog::formatter> formatter) override {
    sink_->set_formatter(std::move(formatter));
  }

 private:
  std::shared_ptr<spdlog::sinks::sink> sink_;
};

std::unique_ptr<spdlog::formatter> makeFormatter() {
  return std::make_unique<spdlog::pattern_formatter>(
      "%v", spdlog::pattern_time_type::local, "");
}

std::string getHostName() {
  char hostname[HOST_NAME_MAX + 1];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    return "unknown";
  }
  hostname[HOST_NAME_MAX] = '\0';
  if (auto* domain = std::strchr(hostname, '.')) {
    *domain = '\0';
  }
  return hostname;
}

std::string_view getSpdlogLevelName(spdlog::level::level_enum level) {
  switch (level) {
    case spdlog::level::trace:
    case spdlog::level::debug:
      return "VERBOSE";
    case spdlog::level::info:
      return "INFO";
    case spdlog::level::warn:
      return "WARN";
    case spdlog::level::err:
      return "ERROR";
    case spdlog::level::critical:
      return "CRITICAL";
    case spdlog::level::off:
    case spdlog::level::n_levels:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

std::string_view getBaseName(const char* filename) {
  if (filename == nullptr) {
    return {};
  }
  const auto* slash = std::strrchr(filename, '/');
  return slash == nullptr ? filename : slash + 1;
}

uint64_t getCurrentThreadId() {
  static thread_local const auto threadId =
      static_cast<uint64_t>(::syscall(SYS_gettid));
  return threadId;
}

struct NamedLoggerRegistry {
  std::shared_mutex mutex;
  std::map<std::string, std::unique_ptr<CommsSpdlogLogger>, std::less<>>
      loggers;
};

std::shared_ptr<spdlog::logger> createLogger(std::string name) {
  auto threadPoolLease = getCommsThreadPoolState().acquire();
  auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
  std::shared_ptr<spdlog::logger> logger;
  if (threadPoolLease) {
    logger = std::make_shared<spdlog::async_logger>(
        std::move(name),
        std::move(sink),
        threadPoolLease.threadPool(),
        spdlog::async_overflow_policy::overrun_oldest);
  } else {
    // A logger first referenced during exit cannot use the stopped async pool.
    logger = std::make_shared<spdlog::logger>(std::move(name), std::move(sink));
  }
  logger->set_formatter(makeFormatter());
  return logger;
}

} // namespace

void shutdownSpdlogForFatal() {
  getCommsThreadPoolState().stop();
}

namespace testing {

bool holdAsyncThreadPoolLeaseForTesting(const std::function<void()>& callback) {
  auto lease = getCommsThreadPoolState().acquire();
  if (!lease) {
    return false;
  }
  callback();
  return true;
}

void waitForAsyncThreadPoolShutdownForTesting() {
  getCommsThreadPoolState().waitForShutdownToStart();
}

bool asyncThreadPoolLeaseAvailableForTesting() {
  return static_cast<bool>(getCommsThreadPoolState().acquire());
}

} // namespace testing

bool shouldWriteCommsLogToStderr(std::string_view formattedMessage) {
  /*
   * CRITICAL and FATAL share spdlog's critical level. The formatter begins
   * with the legacy level name so stderr routing can still distinguish them.
   */
  if (formattedMessage.empty()) {
    return false;
  }
  const auto levelInitial = formattedMessage.front();
  return levelInitial == 'W' || levelInitial == 'E' || levelInitial == 'F';
}

CommsSpdlogLogger::CommsSpdlogLogger()
    : CommsSpdlogLogger(std::string{kCommsLoggerName}) {}

CommsSpdlogLogger::CommsSpdlogLogger(std::string name)
    : logger_(createLogger(std::move(name))) {
  outputSink_ = std::make_shared<spdlog::sinks::dist_sink_mt>(logger_->sinks());
  logger_->sinks() = {outputSink_};
  synchronousLogger_ =
      std::make_shared<spdlog::logger>(logger_->name(), outputSink_);
  synchronousLogger_->set_level(spdlog::level::trace);
  storeConfiguration(std::make_shared<const Configuration>());
}

CommsLogStreamBase::CommsLogStreamBase(
    CommsSpdlogLogger& logger,
    spdlog::source_loc location,
    spdlog::level::level_enum level)
    : logger_(logger), location_(location), level_(level) {}

std::ostream& CommsLogStreamBase::stream() {
  return stream_;
}

void CommsLogStreamBase::log() {
  logger_.log(location_, level_, stream_.str());
}

[[noreturn]] void CommsLogStreamBase::logFatalAndAbort() noexcept {
  try {
    logger_.flush();
    shutdownSpdlogForFatal();
    logger_.logFatal(location_, stream_.str());
  } catch (...) {
    abortAfterCommsLoggingFailure();
  }
  std::abort();
}

CommsLogStream::CommsLogStream(
    CommsSpdlogLogger& logger,
    spdlog::source_loc location,
    spdlog::level::level_enum level)
    : CommsLogStreamBase(logger, location, level) {}

CommsLogStream::~CommsLogStream() noexcept {
  try {
    log();
  } catch (...) {
    reportCommsLoggingFailureToStderr("ERROR");
  }
}

CommsFatalLogStream::CommsFatalLogStream(
    CommsSpdlogLogger& logger,
    spdlog::source_loc location)
    : CommsLogStreamBase(logger, location, spdlog::level::critical) {}

[[noreturn]] CommsFatalLogStream::~CommsFatalLogStream() noexcept {
  logFatalAndAbort();
}

std::string_view CommsSpdlogLogger::getLevelName(
    spdlog::level::level_enum level) {
  return getSpdlogLevelName(level);
}

bool CommsSpdlogLogger::should_log(spdlog::level::level_enum level) const {
  return logger_->should_log(level);
}

const std::string& CommsSpdlogLogger::name() const {
  return logger_->name();
}

void CommsSpdlogLogger::set_level(spdlog::level::level_enum level) {
  logger_->set_level(level);
  /*
   * logger_ applies the runtime gate before either delivery path. The
   * synchronous logger stays at trace so it accepts enabled synchronous
   * messages while logFatal can bypass the primary gate.
   */
}

bool CommsSpdlogLogger::usesAsyncLogging() const {
  return loadConfiguration()->asyncLogging;
}

void CommsSpdlogLogger::flush() {
  // Once the async pool is gone, spdlog reports the failed post through its
  // error handler and drops the flush, so use the synchronous path instead.
  const auto threadPoolLease = getCommsThreadPoolState().acquire();
  if (threadPoolLease) {
    logger_->flush();
  }
  if (!usesAsyncLogging() || !threadPoolLease) {
    synchronousLogger_->flush();
  }
}

void CommsSpdlogLogger::log(
    spdlog::source_loc location,
    spdlog::level::level_enum level,
    std::string_view message) {
  if (should_log(level)) {
    logFormatted(location, level, getLevelName(level), message, false);
  }
}

void CommsSpdlogLogger::logSynchronous(
    spdlog::source_loc location,
    spdlog::level::level_enum level,
    std::string_view message) {
  if (should_log(level)) {
    logFormatted(location, level, getLevelName(level), message, true);
  }
}

void CommsSpdlogLogger::logFatal(
    spdlog::source_loc location,
    std::string_view message) {
  logFormatted(location, spdlog::level::critical, "FATAL", message, true);
}

void CommsSpdlogLogger::configure(
    std::string prefix,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback,
    bool asyncLogging) {
  if (!threadContextFn) {
    threadContextFn = []() { return 0; };
  }
  std::shared_ptr<const Configuration> configuration =
      std::make_shared<const Configuration>(Configuration{
          std::move(prefix),
          std::move(threadContextFn),
          std::move(errorCallback),
          asyncLogging});
  storeConfiguration(std::move(configuration));

  /*
   * The file sink buffers in user space and spdlog does not flush by default,
   * so errors would otherwise reach the log file only once the buffer filled.
   * Set unconditionally: configure() is public and may switch a logger back to
   * synchronous, which must not inherit the previous flush level.
   */
  logger_->flush_on(asyncLogging ? spdlog::level::err : spdlog::level::off);
  if (asyncLogging) {
    getPeriodicSinkFlusher().registerSink(outputSink_);
  }
}

std::shared_ptr<const CommsSpdlogLogger::Configuration>
CommsSpdlogLogger::loadConfiguration() const {
#if defined(__cpp_lib_atomic_shared_ptr) && \
    __cpp_lib_atomic_shared_ptr >= 201711L
  return configuration_.load(std::memory_order_acquire);
#else
  return std::atomic_load_explicit(&configuration_, std::memory_order_acquire);
#endif
}

void CommsSpdlogLogger::storeConfiguration(
    std::shared_ptr<const Configuration> configuration) {
#if defined(__cpp_lib_atomic_shared_ptr) && \
    __cpp_lib_atomic_shared_ptr >= 201711L
  configuration_.store(std::move(configuration), std::memory_order_release);
#else
  std::atomic_store_explicit(
      &configuration_, std::move(configuration), std::memory_order_release);
#endif
}

void CommsSpdlogLogger::configureOutput(std::string_view logFilePath) {
  std::vector<std::shared_ptr<spdlog::sinks::sink>> sinks;
  if (logFilePath.empty()) {
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  } else {
    const auto path = std::string{logFilePath};
    std::shared_ptr<spdlog::sinks::sink> fileSink;
    try {
      fileSink =
          std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, false);
    } catch (const spdlog::spdlog_ex& error) {
      throw spdlog::spdlog_ex(
          "Failed to open comms log file '" + path + "': " + error.what());
    }
    sinks.push_back(std::move(fileSink));
    sinks.push_back(std::make_shared<StderrRoutingSink>());
  }
  for (auto& sink : sinks) {
    sink->set_formatter(makeFormatter());
  }
  outputSink_->set_sinks(std::move(sinks));
}

void CommsSpdlogLogger::logFormatted(
    spdlog::source_loc location,
    spdlog::level::level_enum level,
    std::string_view levelName,
    std::string_view message,
    bool bypassLevelGate) {
  // Leaked for the reason PeriodicSinkFlusher documents: threads that outlive
  // exit-time destruction still format messages through here.
  static const auto* hostname = new std::string{getHostName()};
  static const auto processId = getpid();
  const auto configuration = loadConfiguration();
  if (level >= spdlog::level::err && configuration->errorCallback &&
      !errorCallbackInProgress) {
    ErrorCallbackGuard guard;
    try {
      configuration->errorCallback(message);
    } catch (...) {
    }
  }
  const auto formatted = formatCommsLogMessage(
      levelName,
      message,
      {std::chrono::system_clock::now(),
       getCurrentThreadId(),
       getBaseName(location.filename),
       static_cast<unsigned int>(location.line),
       *hostname,
       processId,
       configuration->threadContextFn(),
       getLogThreadName(),
       configuration->prefix});
  const auto threadPoolLease = getCommsThreadPoolState().acquire();
  if (bypassLevelGate || !configuration->asyncLogging || !threadPoolLease) {
    synchronousLogger_->log(location, level, formatted);
    synchronousLogger_->flush();
  } else {
    logger_->log(location, level, formatted);
  }
}

CommsSpdlogLogger& getSpdlogLogger() {
  // Leaked; see PeriodicSinkFlusher for why nothing here may have a destructor.
  static auto* logger = new CommsSpdlogLogger{};
  return *logger;
}

CommsSpdlogLogger& getSpdlogLogger(std::string_view loggerName) {
  if (loggerName == kCommsLoggerName) {
    return getSpdlogLogger();
  }
  // Leaked as a unit: destroying the map would destroy every named logger, and
  // destroying the mutex would strand any thread still looking one up.
  static auto* registry = new NamedLoggerRegistry{};
  {
    std::shared_lock lock{registry->mutex};
    if (const auto it = registry->loggers.find(loggerName);
        it != registry->loggers.end()) {
      return *it->second;
    }
  }
  std::unique_lock lock{registry->mutex};
  if (const auto it = registry->loggers.find(loggerName);
      it != registry->loggers.end()) {
    return *it->second;
  }
  auto name = std::string{loggerName};
  auto logger = std::make_unique<CommsSpdlogLogger>(name);
  const auto it =
      registry->loggers.emplace(std::move(name), std::move(logger)).first;
  return *it->second;
}

CommsSpdlogLogger& getSpdlogLoggerForFatal(
    std::string_view loggerName) noexcept {
  try {
    return getSpdlogLogger(loggerName);
  } catch (...) {
    abortAfterCommsLoggingFailure();
  }
}

void configureSpdlogLogger(
    std::string prefix,
    std::function<int(void)> threadContextFn) {
  getSpdlogLogger().configure(std::move(prefix), std::move(threadContextFn));
}

void configureSpdlogLogger(
    std::string_view loggerName,
    std::string prefix,
    std::string_view logFilePath,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback,
    bool asyncLogging) {
  auto& logger = getSpdlogLogger(loggerName);
  logger.configure(
      std::move(prefix),
      std::move(threadContextFn),
      std::move(errorCallback),
      asyncLogging);
  logger.configureOutput(logFilePath);
}

void configureCommsAndNamedSpdlogLoggers(
    std::string_view loggerName,
    std::string logPrefix,
    std::string_view logFilePath,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback,
    bool asyncLogging,
    spdlog::level::level_enum logLevel,
    bool configureCommsLogger) {
  if (configureCommsLogger || loggerName == kCommsLoggerName) {
    configureSpdlogLogger(
        kCommsLoggerName,
        "COMM",
        logFilePath,
        threadContextFn,
        errorCallback,
        asyncLogging);
    getSpdlogLogger().set_level(logLevel);
  }

  if (loggerName != kCommsLoggerName) {
    configureSpdlogLogger(
        loggerName,
        std::move(logPrefix),
        logFilePath,
        std::move(threadContextFn),
        std::move(errorCallback),
        asyncLogging);
    getSpdlogLogger(loggerName).set_level(logLevel);
  }
}

void setSpdlogThreadName(std::string_view name) {
  setLogThreadName(name);
}

} // namespace meta::comms::logger
