// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstddef>

#include "comms/ctran/algos/CtranAlgoDev.h"

// TODO: Each KernelArg struct has common field. Consider merge them and put
// into a parent class to avoid duplication. We need revisit it after migrated
// all KernelArg structs from GPE.
struct alignas(16) CtranAlgoDevReduceArg {
  size_t nsrcs{0};
  size_t ndsts{0};
  size_t count{0};
  const void* srcs[CTRAN_MAX_NVL_PEERS]{nullptr};
  void* dsts[CTRAN_MAX_NVL_PEERS]{nullptr};
  // Set isFinal to true to perform AVG. It should be true only at the last
  // step of any partial reduce involved algorithm (e.g., Ring).
  bool isFinal{false};
  // Whether the kernel performs a memory fence after reduce.
  // It ensures data become visible to other device/host/NIC.
  bool flushMem{false};
  // Whether the kernel performs a barrier among nvectors of local ranks
  // after reduce. It ensures the local and all peer ranks have finished.
  bool barrier{false};
};

struct alignas(16) CtranAlgoDevBcastArg {
  size_t nvectors{0};
  size_t count{0};
  const void* src{nullptr};
  void* dsts[CTRAN_MAX_NVL_PEERS]{nullptr};
  // Whether the kernel performs a memory fence after bcast.
  // It ensures data become visible to other device/host/NIC.
  bool flushMem{false};
  // Whether the kernel performs a barrier among nvectors of local ranks
  // after bcast. It ensures the local and all peer ranks have finished the
  // bcast, e.g., as an allgather.
  bool barrier{false};
};
