// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/CudaRAII.h"

namespace meta::comms::colltrace {

class CachedCudaEvent {
 public:
  explicit CachedCudaEvent(CudaEvent event);

  ~CachedCudaEvent();

  // delete copy constructor
  CachedCudaEvent(const CachedCudaEvent&) = delete;
  CachedCudaEvent& operator=(const CachedCudaEvent&) = delete;

  // We could use default move constructor and move assignment operator due to
  // underlying CudaEvent properly handle move constructor and move assignment
  CachedCudaEvent(CachedCudaEvent&& other) noexcept;
  CachedCudaEvent& operator=(CachedCudaEvent&& other) noexcept;

  cudaEvent_t get() const;
  const CudaEvent& getRef() const;

 private:
  CudaEvent event_;
  bool isMoved_{false};
};

class CudaEventPool {
 public:
  static CachedCudaEvent getEvent();

  // Should only be called inside the destructor of CachedCudaEvent
  static void returnEvent(CudaEvent event);
};

} // namespace meta::comms::colltrace
