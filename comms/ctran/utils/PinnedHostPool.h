// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
PinnedHostPool is a pre-allocated memory pool of pinned host objects allocated
using cudaHostAlloc. It is NOT thread-safe.
*/

#pragma once

#include <list>
#include <stack>
#include <vector>

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

  explicit PinnedHostPool(size_t startCapacity) : chunkSize_(startCapacity) {
    allocChunk();
  }

  ~PinnedHostPool() {
    this->reclaim();
    if (this->inuseItems_.size()) {
      CLOGF(
          INFO,
          "CTRAN-GPE: Internal {} pool has {} inuse items, indicating same amount of unfinished kernel",
          T::name(),
          this->inuseItems_.size());
    }
    for (void* chunk : chunks_) {
      FB_CUDACHECKIGNORE(cudaFreeHost(chunk));
    }

    // Do not throw exception in destructor to avoid early termination in stack
    // unwind. See discussion in
    // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
  }

  T* pop() {
    if (this->freeItems_.size() == 0) {
      this->reclaim();
    }

    if (this->freeItems_.size() == 0) {
      CLOGF(
          INFO,
          "CTRAN-GPE: {} pool exhausted ({} capacity), growing by {}",
          T::name(),
          capacity_,
          chunkSize_);
      allocChunk();
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
  void allocChunk() {
    void* mem = nullptr;
    FB_CUDACHECKTHROW_EX_NOCOMM(
        cudaHostAlloc(&mem, chunkSize_ * sizeof(T), cudaHostAllocDefault));
    chunks_.push_back(mem);

    for (size_t i = 0; i < chunkSize_; ++i) {
      T* item = reinterpret_cast<T*>(mem) + i;
      item->reset();
      this->freeItems_.push(item);
    }
    capacity_ += chunkSize_;
  }

  std::stack<T*> freeItems_;
  std::list<T*> inuseItems_;
  std::vector<void*> chunks_;
  const size_t chunkSize_{0};
  size_t capacity_{0};

  PinnedHostPool(const PinnedHostPool&) = delete;
  PinnedHostPool& operator=(const PinnedHostPool&) = delete;
};
