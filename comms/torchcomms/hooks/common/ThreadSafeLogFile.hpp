// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace torch::comms {

class ThreadSafeLogFile {
 public:
  // The flush cadence bounds are constructor parameters so callers (and tests)
  // can tune or pin them; the defaults match the production clog logger.
  explicit ThreadSafeLogFile(
      int flushEveryLines = 1024,
      std::chrono::milliseconds flushInterval = std::chrono::milliseconds{100})
      : flushEveryLines_(flushEveryLines), flushInterval_(flushInterval) {}

  ThreadSafeLogFile(const ThreadSafeLogFile&) = delete;
  ThreadSafeLogFile& operator=(const ThreadSafeLogFile&) = delete;
  ThreadSafeLogFile(ThreadSafeLogFile&&) = delete;
  ThreadSafeLogFile& operator=(ThreadSafeLogFile&&) = delete;

  void open(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.open(path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open log file: " + path);
    }
  }

  void writeLine(const std::string& line) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.write(line.data(), line.size());
    file_ << '\n';
    // Flush on a bounded line/time cadence rather than per line: this runs on
    // the collective-launch thread, where a per-line flush is a write(2)
    // syscall on every launch. The flush is opportunistic — it fires on a later
    // writeLine once a bound is crossed, not on a background timer — so a hard
    // crash can lose the still-buffered tail; a caller that needs the log on
    // disk while the process is live calls flush().
    const auto now = std::chrono::steady_clock::now();
    if (++lines_since_flush_ >= flushEveryLines_ ||
        now - last_flush_ >= flushInterval_) {
      file_.flush();
      lines_since_flush_ = 0;
      last_flush_ = now;
    }
  }

  // Force any buffered lines to the OS immediately, for consumers/tests that
  // read the log while the process is still running.
  void flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
      file_.flush();
      lines_since_flush_ = 0;
      last_flush_ = std::chrono::steady_clock::now();
    }
  }

  ~ThreadSafeLogFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
      file_.flush();
      file_.close();
    }
  }

 private:
  const int flushEveryLines_;
  const std::chrono::milliseconds flushInterval_;

  std::ofstream file_;
  std::mutex mutex_;
  int lines_since_flush_ = 0;
  std::chrono::steady_clock::time_point last_flush_ =
      std::chrono::steady_clock::now();
};

} // namespace torch::comms
