// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

namespace comms::prims::detail {

/*
 * Executes at most one cleanup task asynchronously. While that task is queued
 * or running, callers retain ownership of subsequent tasks and can reclaim
 * their resources synchronously instead of growing an unbounded backlog.
 */
template <typename Task>
class BoundedCleanupExecutor final {
  static_assert(std::is_nothrow_move_constructible_v<Task>);
  static_assert(std::is_nothrow_invocable_v<Task&>);

 public:
  BoundedCleanupExecutor() : worker_([this] { run(); }) {}

  ~BoundedCleanupExecutor() {
    shutdown();
  }

  BoundedCleanupExecutor(const BoundedCleanupExecutor&) = delete;
  BoundedCleanupExecutor& operator=(const BoundedCleanupExecutor&) = delete;
  BoundedCleanupExecutor(BoundedCleanupExecutor&&) = delete;
  BoundedCleanupExecutor& operator=(BoundedCleanupExecutor&&) = delete;

  bool tryEnqueue(Task& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || occupied_) {
      return false;
    }
    task_.emplace(std::move(task));
    occupied_ = true;
    cv_.notify_one();
    return true;
  }

  void drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !occupied_; });
  }

  void shutdown() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  void run() noexcept {
    while (true) {
      std::optional<Task> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stopping_ || task_.has_value(); });
        if (!task_.has_value()) {
          return;
        }
        task.emplace(std::move(*task_));
        task_.reset();
      }

      (*task)();
      task.reset();

      {
        std::lock_guard<std::mutex> lock(mutex_);
        occupied_ = false;
      }
      cv_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::optional<Task> task_;
  bool stopping_{false};
  bool occupied_{false};
  std::thread worker_;
};

} // namespace comms::prims::detail
