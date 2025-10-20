/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"

constexpr const char* ncclCodeToString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN_WITH_SCUBA("Cuda failure '%s'", cudaGetErrorString(err)); \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, RES, label) do {                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN_WITH_SCUBA("Cuda failure '%s'", cudaGetErrorString(err)); \
        RES = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd) do {  \
    cudaError_t err = cmd;         \
    if( err != cudaSuccess ) {     \
        INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        (void) cudaGetLastError(); \
    }                              \
} while(false)

// Use of abort should be aware of potential memory leak risk
// and place a signal handler to catch it and trigger termination processing
#define CUDACHECKABORT(cmd)                              \
  do {                                                   \
    cudaError_t err = cmd;                               \
    if (err != cudaSuccess) {                            \
      ERR_WITH_SCUBA("Cuda failure '%s'", cudaGetErrorString(err)); \
      abort();                                           \
    }                                                    \
  } while (false)

#include <errno.h>
// Check system calls
#define SYSCHECK(statement, name) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN_WITH_SCUBA("Call to " name " failed: %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKVAL(call, name, retval)                     \
  do {                                                      \
    SYSCHECKSYNC(call, name, retval);                       \
    if (retval == -1) {                                     \
      ERR_WITH_SCUBA("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(statement, name, retval) do { \
  retval = (statement); \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN_WITH_SCUBA("Call to " name " failed: %s", strerror(errno)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN_WITH_SCUBA("Call to " name " failed: %s", strerror(retval)); \
    return ncclSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN_WITH_SCUBA("Call to " name " failed: %s", strerror(retval)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    WARN_WITH_SCUBA("%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define NEQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    WARN_WITH_SCUBA("%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    WARN_WITH_SCUBA("%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define EQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    WARN_WITH_SCUBA("%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN_WITH_SCUBA("%s:%d -> %d", __FILE__, __LINE__, RES);    \
    return RES; \
  } \
} while (0)

#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN_WITH_SCUBA("%s:%d -> %d", __FILE__, __LINE__, RES);    \
    goto label; \
  } \
} while (0)

// Report failure but clear error and continue
#define NCCLCHECKIGNORE(call)                          \
  do {                                                 \
    ncclResult_t RES = call;                           \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      WARN_WITH_SCUBA(                                 \
          "%s:%s:%d -> %d (%s)",                       \
          __FILE__,                                    \
          __func__,                                    \
          __LINE__,                                    \
          RES,                                         \
          ncclCodeToString(RES));                      \
    }                                                  \
  } while (0)

#define CHECKABORT(statement, ...)   \
  do {                               \
    if (!(statement)) {              \
      WARN("Check failed: %s", #statement); \
      WARN(__VA_ARGS__);             \
      abort();                       \
    }                                \
  } while (0);

#define NCCLWAIT(call, cond, abortFlagPtr) do {         \
  uint32_t* tmpAbortFlag = (abortFlagPtr);     \
  ncclResult_t RES = call;                \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) WARN_WITH_SCUBA(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    return ncclInternalError;             \
  }                                       \
  if (__atomic_load(tmpAbortFlag, __ATOMIC_ACQUIRE)) NEQCHECK(*tmpAbortFlag, 0); \
} while (!(cond))

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) do { \
  uint32_t* tmpAbortFlag = (abortFlagPtr);             \
  RES = call;                             \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) WARN_WITH_SCUBA(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    goto label;                           \
  }                                       \
  if (__atomic_load(tmpAbortFlag, __ATOMIC_ACQUIRE)) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
} while (!(cond))

#define NCCLCHECKTHREAD(a, args) do { \
  if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
    WARN_WITH_SCUBA("%s:%d -> %d [Async thread]", __FILE__, __LINE__, (args)->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    WARN_WITH_SCUBA("%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

#endif
