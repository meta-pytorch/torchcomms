// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

enum DevMemType {
  kCudaMalloc = 0,
  kManaged = 1,
  kHostPinned = 2,
  kHostUnregistered = 3,
  kCumem = 4,
};

inline const char* devMemTypeStr(DevMemType memType) {
  switch (memType) {
    case kCudaMalloc:
      return "cudaMalloc";
    case kManaged:
      return "managed";
    case kHostPinned:
      return "hostPinned";
    case kHostUnregistered:
      return "hostUnregistered";
    case kCumem:
      return "cuMem";
    default:
      return "unknown";
  }
}
