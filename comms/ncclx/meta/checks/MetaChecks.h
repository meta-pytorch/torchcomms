// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// NCCLX-only additions that have no upstream NCCL equivalent, hoisted out of
// the forked upstream `checks.h` and pulled back in by a single include from
// it.
//
// Only *purely additive* symbols live here (helpers and macros upstream does
// not define). The in-place customizations of upstream check macros (routing
// WARN -> WARN_WITH_SCUBA, adding CUDA-error capture, etc.) are deliberately
// LEFT in `checks.h`: overriding them here with #undef/#define would silently
// swallow any future upstream change to those macro bodies, whereas keeping the
// edits in place makes such a change surface as a rebase merge conflict that a
// human must reconcile.
//
// Do not include this header directly; include "checks.h".

#include <utility>

#include "debug.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

constexpr const char* ncclCodeToString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError:
      return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    // No dedicated message; listed explicitly (falling through to default) to
    // satisfy -Wswitch-enum without changing behavior.
    case ncclTimeout:
    case ncclNumResults:
    default:
      return "unknown result code";
  }
}

// Report a CUDA error to colltrace for analyzer consumption
#define COMMDUMP_REPORT_CUDA_ERROR(err)                                    \
  do {                                                                     \
    if (NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES > 0) {                 \
      ProcessGlobalErrorsUtil::CudaError cudaErr;                          \
      cudaErr.errorString = cudaGetErrorString(err);                       \
      cudaErr.errorCode = static_cast<int>(err);                           \
      cudaErr.scaleupDomain = ProcessGlobalErrorsUtil::getScaleupDomain(); \
      cudaErr.localHostname = ProcessGlobalErrorsUtil::getHostname();      \
      ProcessGlobalErrorsUtil::addCudaError(std::move(cudaErr));           \
    }                                                                      \
  } while (false)

#define CHECKABORT(statement, ...)          \
  do {                                      \
    if (!(statement)) {                     \
      WARN("Check failed: %s", #statement); \
      WARN(__VA_ARGS__);                    \
      abort();                              \
    }                                       \
  } while (0)

// Use of abort should be aware of potential memory leak risk
// and place a signal handler to catch it and trigger termination processing
#define CUDACHECKABORT(cmd)                                         \
  do {                                                              \
    cudaError_t err = cmd;                                          \
    if (err != cudaSuccess) {                                       \
      ERR_WITH_SCUBA("Cuda failure '%s'", cudaGetErrorString(err)); \
      COMMDUMP_REPORT_CUDA_ERROR(err);                              \
      abort();                                                      \
    }                                                               \
  } while (false)

#define SYSCHECKVAL(call, name, retval)                                \
  do {                                                                 \
    SYSCHECKSYNC(call, name, retval);                                  \
    if (retval == -1) {                                                \
      ERR_WITH_SCUBA("Call to " name " failed : %s", strerror(errno)); \
      return ncclSystemError;                                          \
    }                                                                  \
  } while (false)
