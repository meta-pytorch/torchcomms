// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/commSpecs.h"

// Forward declaration
struct KernelElem;

namespace ctran::allgather {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t count;
  KernelElem* bcastElem;
};

} // namespace ctran::allgather
