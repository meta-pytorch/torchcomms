// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <optional>
#include <type_traits>
#include <utility>

namespace uniflow {

/// Wait-free multi-producer, single-consumer unbounded (MPSC) queue.
///
/// Based on Dmitry Vyukov's MPSC queue. Producers enqueue at the tail via an
/// atomic exchange (wait-free). The consumer dequeues from the head by
/// following next pointers (lock-free). A permanent sentinel node is
/// re-inserted when consuming the last item, giving the consumer a node
/// to park on.
///
/// Thread safety:
///   - push() is thread-safe and wait-free (multiple producers)
///   - pop(), empty() are single-consumer only
template <typename T>
  requires(std::is_nothrow_move_constructible_v<T>)
class MpscQueue {
 private:
  struct Node {
    T value{};
    std::atomic<Node*> next{nullptr};

    Node() = default;
    explicit Node(T v) : value(std::move(v)) {}
  };

 public:
  MpscQueue() noexcept : sentinel_(), tail_(&sentinel_), head_(&sentinel_) {}

  ~MpscQueue() {
    // Walk the linked list and delete all heap-allocated nodes.
    // Caller must ensure all producers have finished before destruction.
    Node* node = head_;
    while (node != nullptr) {
      Node* next = node->next.load(std::memory_order_relaxed);
      if (node != &sentinel_) {
        ::delete node;
      }
      node = next;
    }
  }

  MpscQueue(const MpscQueue&) = delete;
  MpscQueue& operator=(const MpscQueue&) = delete;
  MpscQueue(MpscQueue&&) = delete;
  MpscQueue& operator=(MpscQueue&&) = delete;

  /// Enqueue a value. Thread-safe, wait-free.
  ///
  /// The atomic exchange always completes in one step (no CAS retry),
  /// making this wait-free for all producers.
  void push(T value) noexcept {
    enqueue(::new Node(std::move(value)));
  }

  /// Pop one item. Returns the value if an item was dequeued, or
  /// std::nullopt if the queue is empty or an enqueue is in progress
  /// (caller should retry).
  ///
  /// Single-consumer only.
  std::optional<T> pop() noexcept {
    Node* node = head_;
    // ACQUIRE is important: pairs with producers' prev->next.store(RELEASE).
    // If this returns non-null, we can safely read the enqueued node's `value`.
    Node* next = node->next.load(std::memory_order_acquire);

    // If head is the sentinel, skip past it.
    if (node == &sentinel_) {
      if (!next) {
        return std::nullopt;
      }
      head_ = next;
      node = next;
      // Again: ACQUIRE so that observing `next != nullptr` implies we see that
      // node's contents.
      next = node->next.load(std::memory_order_acquire);
    }

    // Fast path: next node exists, consume current node.
    if (next) {
      head_ = next;
      // After we've advanced head_, nobody else will touch `node` (single
      // consumer). The `value` move does not need atomics; the ordering was
      // provided by the acquire load that made `node` reachable in the first
      // place.
      auto value = std::move(node->value);
      ::delete node;
      return value;
    }

    // node has no next. There are two cases:
    //  (1) node really is the last node in the queue (tail == node)
    //  (2) some producer has already advanced tail_ to a new node (exchange
    //  done), but has not yet linked node->next (prev->next.store not done).
    //
    // ACQUIRE here is the usual pairing with producers' exchange(...,
    // RELEASE). This check is only for progress/emptiness; correctness
    // still hinges on the next-pointer release/acquire.
    if (node != tail_.load(std::memory_order_acquire)) {
      return std::nullopt;
    }

    // node is (believed to be) the last item. Re-insert the sentinel so the
    // consumer can "park" on it after consuming node.
    //
    // RELAXED is fine: the publication of sentinel happens via enqueue()'s
    // prev->next.store(RELEASE) before, and no producer will touch it anymore.
    sentinel_.next.store(nullptr, std::memory_order_relaxed);
    enqueue(&sentinel_);

    // The sentinel (or another producer's node) may now be linked after node.
    // ACQUIRE again to synchronize with that linking store.
    next = node->next.load(std::memory_order_acquire);
    if (next) {
      head_ = next;
      auto value = std::move(node->value);
      ::delete node;
      return value;
    }

    // The sentinel enqueue hasn't fully linked yet, or another producer
    // is mid-link. Return nullopt; caller retries.
    return std::nullopt;
  }

  /// Check if the queue is empty.
  ///
  /// Single-consumer only.
  bool empty() const noexcept {
    // If head_ is the sentinel, skip past it.
    Node* node = head_;
    if (node == &sentinel_) {
      node = node->next.load(std::memory_order_relaxed);
    }
    return node == nullptr;
  }

 private:
  void enqueue(Node* node) noexcept {
    Node* prev = tail_.exchange(node, std::memory_order_release);
    prev->next.store(node, std::memory_order_release);
  }

  // Sentinel is declared first so it is constructed before tail_/head_
  // reference it.
  Node sentinel_;
  // Align to separate cache lines to prevent false sharing between
  // the producer-hot tail and consumer-hot head.
  alignas(64) std::atomic<Node*> tail_;
  alignas(64) Node* head_;
};

} // namespace uniflow
