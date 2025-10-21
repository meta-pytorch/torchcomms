// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <folly/Synchronized.h>
#include <queue>

namespace ctran::utils {

/* Thread safe Single Producer Single Consumer Queue.
 * - The producer thread can enqueue a task and notify the consumer thread to
 *   wake up.
 * - The consumer thread waits on call to dequeue, and is waken up when a
 *   task is available.
 * - Unlike folly::MPMCQueue, this queue provides unlimited capacity, so
 *   producer would never block on enqueue. */
template <typename T>
class SPSCQueue {
 private:
  struct Queue {
    std::queue<std::unique_ptr<T>> queue;
  };

  folly::Synchronized<Queue, std::mutex> q_;
  std::condition_variable cv_;

 public:
  // Enqueue a task and notify the consumer thread to wake up.
  inline void enqueue(std::unique_ptr<T> cmd) {
    { q_.lock()->queue.push(std::move(cmd)); }
    cv_.notify_one();
  }

  // Dequeue a task by the consumer thread.
  // If the queue is empty, the calling thread will sleep until receive
  // a wakeup signal when a task is enqueued.
  inline std::unique_ptr<T> dequeue() {
    auto lk = q_.lock();
    cv_.wait(lk.as_lock(), [&lk] { return !lk->queue.empty(); });

    auto cmd = std::move(lk->queue.front());
    lk->queue.pop();
    return cmd;
  }
};
} // namespace ctran::utils
