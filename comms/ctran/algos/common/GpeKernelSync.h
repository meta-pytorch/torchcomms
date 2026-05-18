// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/utils/MemFence.h"

namespace ctran::algos {

struct alignas(16) GpeKernelSync {
  enum Status {
    kUnset = -1,
  };

  // number of thread blocks to handle the GpeKernelSync.
  // Set by algorithm when submitting a GPE kernel; status update between GPE
  // and kernel need update with all workers
  unsigned int nworkers{0};

  // Set flag with value defined in GpeKernelSync::Status or a positive stepId
  // starting from zero. Each flag is used by a separate worker (often a
  // threadBlock).
  int postFlag[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  int completeFlag[CTRAN_ALGO_MAX_THREAD_BLOCKS];

  GpeKernelSync(unsigned int nworkers) : nworkers(nworkers) {
    resetStatus();
  }

  inline void resetStatus() {
    for (unsigned int i = 0; i < nworkers; i++) {
      postFlag[i] = kUnset;
      completeFlag[i] = kUnset;
    }
  }

  // Check if all workers on kernel side has completed the specified step.
  //
  // Pairs with the kernel's st_release_sys_global on completeFlag in
  // GpeKernelSyncDev::complete(). On aarch64 (GB200) the load-acquire fence
  // after the polling load is required: aarch64 permits load->load
  // reordering, so without dmb ishld here, subsequent loads of data
  // released-by-this-flag (e.g. via the chained post() to another sync)
  // are not guaranteed to observe the producer's writes.
  inline bool isComplete(const int step) {
    bool allComplete = true;
    for (unsigned int i = 0; i < nworkers; i++) {
      volatile int* flag = &completeFlag[i];
      int val = *flag;
      // Kernel handles posted request in sequential, thus >= step indicates
      // completion of the checking step.
      allComplete &= (val >= step);
      if (!allComplete) {
        break;
      }
    }
    if (allComplete) {
      // Acquire only after a successful observation; failed polls do not
      // synchronise with the producer.
      wcAcquireFence();
    }
    return allComplete;
  }

  inline void waitComplete(const int step) {
    for (unsigned int i = 0; i < nworkers; i++) {
      // Kernel handles posted request in sequential, thus >= step indicates
      // that the checking step has been posted.
      volatile int* flag = &completeFlag[i];
      while (*flag < step)
        ;
    }
    // Acquire after observing all workers' completion releases.
    wcAcquireFence();
  }

  // Post a request to kernel with specified step
  inline void post(const int step) {
    // ensure in-order exec of postFlag store after any data update
    // before post
    wcStoreFence();
    // Post to all thread blocks (workers) on kernel side
    for (unsigned int i = 0; i < nworkers; i++) {
      volatile int* flag = &postFlag[i];
      *flag = step;
    }
  }

  // Implements PinnedHostItem concept
  // { T::name() } -> std::same_as<const char*>;
  static const char* name() {
    return "GpeKernelSync";
  }

  // Implements PinnedHostItem concept
  // { t.inUse() } -> std::same_as<bool>;
  bool inUse() {
    return inuse;
  }

  // Implements PinnedHostItem concept
  // { t.reset() } -> std::same_as<void>;
  void reset() {
    resetStatus();
    inuse = false;
  }

  // Implements PinnedHostItem concept
  // { t.onPop() } -> std::same_as<void>;
  void onPop() {
    resetStatus();
    inuse = true;
  }

 private:
  volatile bool inuse{false};
};
} // namespace ctran::algos
