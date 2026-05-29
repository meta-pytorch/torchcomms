// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Device-compatible placement new/delete for CUDA device-only bitcode
// compilation. The standard <new> header does not provide __device__
// qualified versions, which ncclx device headers require (gin__funcs.h).

#ifndef TORCHCOMMS_IR_DEVICE_NEW_H_
#define TORCHCOMMS_IR_DEVICE_NEW_H_

inline __device__ void* operator new(decltype(sizeof(0)), void* p) noexcept {
  return p;
}
inline __device__ void operator delete(void*, void*) noexcept {}

#endif // TORCHCOMMS_IR_DEVICE_NEW_H_
