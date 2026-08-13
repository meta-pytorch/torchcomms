// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/utils/ArgCheck.h"
#include "comms/ctran/utils/CtranLogUtils.h"

namespace ctran {

commResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == nullptr) {
    CTRAN_ERR(commInvalidArgument, "{} : {} argument is NULL", opname, ptrname);
    return commInvalidArgument;
  }
  return commSuccess;
}

} // namespace ctran
