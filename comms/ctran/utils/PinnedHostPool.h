// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
PinnedHostPool is a pre-allocated memory pool of pinned host objects allocated
using cudaHostAlloc. It is NOT thread-safe.
*/

#pragma once

#include <list>
#include <stack>

#include "comms/ctran/utils/Checks.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

/*
PinnedHostItem is the concept/interface for pinned host objects. All pinned host
objects must implement the following functions:
- reset() to reset the object to its initial state, where inUse should be false
- name() to return the name of the object, used for logging
- inUse() to return whether the object is in use, when not in use, it will be
reclaimed by the pool.
- onPop() is called when the object is popped from the pool, it should make the
object in use.
*/
template <typename T>
concept PinnedHostItem = requires(T t) {
  { t.reset() } -> std::same_as<void>;
  { T::name() } -> std::same_as<const char*>;
  { t.inUse() } -> std::same_as<bool>;
  { t.onPop() } -> std::same_as<void>;
};

template <PinnedHostItem T>
class PinnedHostPool {
 public:
  PinnedHostPool() = delete;

  explicit PinnedHostPool(size_t capacity) : capacity_(capacity) {
    FB_CUDACHECKTHROW_EX_NOCOMM(cudaHostAlloc(
        &this->memPtr_, this->capacity_ * sizeof(T), cudaHostAllocDefault));

    for (int i = 0; i < capacity_; ++i) {
      T* item = reinterpret_cast<T*>(this->memPtr_) + i;
      item->reset();
      this->freeItems_.push(item);
    }
  }

  ~PinnedHostPool() {
    this->reclaim();
    if (this->inuseItems_.size()) {
      CLOGF(
          WARN,
          "CTRAN-GPE: Internal {} pool has {} inuse items, indicating same amount of unfinished kernel",
          T::name(),
          this->inuseItems_.size());
    }
    FB_CUDACHECKIGNORE(cudaFreeHost(this->memPtr_));

    // Dot not throw exception in destructor to avoid early termination in stack
    // unwind. See discussion in
    // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
  }

  T* pop() {
    if (this->freeItems_.size() == 0) {
      CLOGF(
          WARN,
          "CTRAN-GPE: Internal {} pool ran out of available items",
          T::name());
      return nullptr;
    }

    T* item = this->freeItems_.top();
    this->freeItems_.pop();
    item->onPop();
    this->inuseItems_.push_back(item);
    CLOGF_TRACE(
        COLL,
        "CTRAN-GPE: Pop {} {}, {} free, {} inuse",
        T::name(),
        (void*)item,
        this->size(),
        this->inuseItems_.size());
    return item;
  }

  void reclaim() {
    auto it = this->inuseItems_.begin();
    while (it != this->inuseItems_.end()) {
      auto item = *it;
      if (!item->inUse()) {
        it = this->inuseItems_.erase(it);
        item->reset();
        this->freeItems_.push(item);
        CLOGF_TRACE(
            COLL,
            "CTRAN-GPE: Reclaimed {} {}, {} free",
            T::name(),
            (void*)item,
            this->size());
      } else {
        it++;
      }
    }
  }

  size_t size() {
    return this->freeItems_.size();
  }

  size_t capacity() {
    return capacity_;
  }

 private:
  std::stack<T*> freeItems_;
  std::list<T*> inuseItems_;
  const size_t capacity_{0};
  void* memPtr_{nullptr};

  PinnedHostPool(const PinnedHostPool&) = delete;
  PinnedHostPool& operator=(const PinnedHostPool&) = delete;
};
