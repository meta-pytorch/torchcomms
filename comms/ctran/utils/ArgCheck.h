// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/utils/commSpecs.h"

namespace ctran {

commResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname);

} // namespace ctran

namespace ctran::utils {
#define ARGCHECK_NULL_COMM(ptr, name)                                   \
  do {                                                                  \
    if (ptr == nullptr) {                                               \
      FB_ERRORRETURN(commInvalidArgument, "{} argument is NULL", name); \
    }                                                                   \
  } while (0)
} // namespace ctran::utils
