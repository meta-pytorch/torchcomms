// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <folly/logging/xlog.h>

#include "comms/ctran/utils/ArgCheck.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran {

commResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == nullptr) {
    CLOGF(ERR, "{} : {} argument is NULL", opname, ptrname);
    return commInvalidArgument;
  }
  return commSuccess;
}

} // namespace ctran
