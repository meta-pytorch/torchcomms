/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"

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
  case ncclTimeout:
    return "NCCL operation timed out";
  case ncclNumResults:
  default:
    return "unknown result code";
  }
}

// Check CUDA RT calls
#define CUDACHECK(cmd) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      ERR(ncclUnhandledCudaError, "Cuda failure '%s'", cudaGetErrorString(err)); \
      (void)cudaGetLastError(); \
      return ncclUnhandledCudaError; \
    } \
  } while (false)

#define CUDACHECKGOTO(cmd, RES, label) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      ERR(ncclUnhandledCudaError, "Cuda failure '%s'", cudaGetErrorString(err)); \
      (void)cudaGetLastError(); \
      RES = ncclUnhandledCudaError; \
      goto label; \
    } \
  } while (false)

// Use of abort should be aware of potential memory leak risk
// and place a signal handler to catch it and trigger termination processing
#define CUDACHECKABORT(cmd) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      ERR(ncclUnhandledCudaError, "Cuda failure '%s'", cudaGetErrorString(err)); \
      abort(); \
    } \
  } while (false)

// fmt is required: WARN expands to a printf-format function, so an
// argument-less CHECKABORT would expand to a format-less WARN.
#define CHECKABORT(statement, fmt, ...) \
  do { \
    if (!(statement)) { \
      WARN("Check failed: %s", #statement); \
      WARN(fmt, ##__VA_ARGS__); \
      abort(); \
    } \
  } while (0)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd) \
  do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
      INFO_LOC(NCCL_ALL, "Cuda failure '%s'", cudaGetErrorString(err)); \
      (void)cudaGetLastError(); \
    } \
  } while (false)

// Use inline function to clear CUDA error inside expressions
static inline cudaError_t cuda_clear(cudaError_t err) {
  if (err != cudaSuccess) (void)cudaGetLastError();
  return err;
}

// Check if cudaSuccess & clear CUDA error
#define CUDASUCCESS(cmd) cuda_clear(cmd) == cudaSuccess
// Clear CUDA error, return CUDA return code
#define CUDACLEARERROR(cmd) cuda_clear(cmd)

#include <errno.h>
// Check system calls
#define SYSCHECK(statement, name) \
  do { \
    int retval; \
    SYSCHECKSYNC((statement), name, retval); \
    if (retval == -1) { \
      ERR(ncclSystemError, "Call to " name " failed: %s", strerror(errno)); \
      return ncclSystemError; \
    } \
  } while (false)

#define SYSCHECKSYNC(statement, name, retval) \
  do { \
    retval = (statement); \
    if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
      INFO_LOC(NCCL_ALL, "Call to " name " returned %s, retrying", strerror(errno)); \
    } else { \
      break; \
    } \
  } while (true)

#define SYSCHECKGOTO(statement, name, RES, label) \
  do { \
    int retval; \
    SYSCHECKSYNC((statement), name, retval); \
    if (retval == -1) { \
      ERR(ncclSystemError, "Call to " name " failed: %s", strerror(errno)); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) \
  do { \
    int retval = (statement); \
    if (retval != 0) { \
      ERR(ncclSystemError, "Call to " name " failed: %s", strerror(retval)); \
      return ncclSystemError; \
    } \
  } while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) \
  do { \
    int retval = (statement); \
    if (retval != 0) { \
      ERR(ncclSystemError, "Call to " name " failed: %s", strerror(retval)); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

#define NEQCHECK(statement, value) \
  do { \
    if ((statement) != value) { \
      /* Print the back trace*/ \
      ERR(ncclSystemError, "-> %d (%s)", ncclSystemError, strerror(errno)); \
      return ncclSystemError; \
    } \
  } while (0)

#define NEQCHECKGOTO(statement, value, RES, label) \
  do { \
    if ((statement) != value) { \
      /* Print the back trace*/ \
      RES = ncclSystemError; \
      ERR(ncclSystemError, "-> %d (%s)", RES, strerror(errno)); \
      goto label; \
    } \
  } while (0)

#define EQCHECK(statement, value) \
  do { \
    if ((statement) == value) { \
      /* Print the back trace*/ \
      ERR(ncclSystemError, "-> %d (%s)", ncclSystemError, strerror(errno)); \
      return ncclSystemError; \
    } \
  } while (0)

#define EQCHECKGOTO(statement, value, RES, label) \
  do { \
    if ((statement) == value) { \
      /* Print the back trace*/ \
      RES = ncclSystemError; \
      ERR(ncclSystemError, "-> %d (%s)", RES, strerror(errno)); \
      goto label; \
    } \
  } while (0)

// Propagate errors up
#define NCCLCHECK(call) \
  do { \
    ncclResult_t RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      /* Print the back trace*/ \
      if (ncclDebugNoWarn == 0) WARN("-> %d", RES); \
      return RES; \
    } \
  } while (0)

#define NCCLCHECKGOTO(call, RES, label) \
  do { \
    RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      /* Print the back trace*/ \
      if (ncclDebugNoWarn == 0) WARN("-> %d", RES); \
      goto label; \
    } \
  } while (0)

// Report failure but continue - useful for cleanup paths where we want to
// attempt all cleanup steps. Preserves the first error in RES.
#define NCCLCHECKIGNORE(call, RES) \
  do { \
    ncclResult_t TMPRES = call; \
    if (TMPRES != ncclSuccess && TMPRES != ncclInProgress) { \
      if (ncclDebugNoWarn == 0) WARN("-> %d (%s)", TMPRES, ncclCodeToString(TMPRES)); \
      if (RES == ncclSuccess) RES = TMPRES; \
    } \
  } while (0)

#define NCCLCHECKNOWARN(call, FLAGS) \
  do { \
    ncclResult_t RES; \
    NOWARN(RES = call, FLAGS); \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      return RES; \
    } \
  } while (0)

#define NCCLCHECKGOTONOWARN(call, RES, label, FLAGS) \
  do { \
    NOWARN(RES = call, FLAGS); \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      goto label; \
    } \
  } while (0)

#define NCCLWAIT(call, cond, abortFlagPtr) \
  do { \
    uint32_t* tmpAbortFlag = (abortFlagPtr); \
    ncclResult_t RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      if (ncclDebugNoWarn == 0) WARN("-> %d", RES); \
      return ncclInternalError; \
    } \
    if (COMPILER_ATOMIC_LOAD(tmpAbortFlag, std::memory_order_acquire)) NEQCHECK(*tmpAbortFlag, 0); \
  } while (!(cond))

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) \
  do { \
    uint32_t* tmpAbortFlag = (abortFlagPtr); \
    RES = call; \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      if (ncclDebugNoWarn == 0) WARN("-> %d", RES); \
      goto label; \
    } \
    if (COMPILER_ATOMIC_LOAD(tmpAbortFlag, std::memory_order_acquire)) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
  } while (!(cond))

#define NCCLCHECKTHREAD(a, args) \
  do { \
    if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
      WARN("-> %d [Async thread]", (args)->ret); \
      return args; \
    } \
  } while (0)

#define CUDACHECKTHREAD(a) \
  do { \
    cudaError_t err = (a); \
    if (err != cudaSuccess) { \
      ERR(ncclUnhandledCudaError, "Cuda failure '%s' [Async thread]", cudaGetErrorString(err)); \
      args->ret = ncclUnhandledCudaError; \
      return args; \
    } \
  } while (0)

// Common thread creation implementation with error handling
#define STDTHREADCREATE_IMPL(var, func, error_action, ...) \
  do { \
    try { \
      (var) = std::thread(func, __VA_ARGS__); \
    } catch (const std::exception& e) { \
      WARN("Thread creation failed: %s", e.what()); \
      error_action; \
    } \
  } while (0)

#define STDTHREADCREATE(var, func, ...) STDTHREADCREATE_IMPL(var, func, return ncclSystemError, __VA_ARGS__)

#define STDTHREADCREATE_GOTO(var, func, RES, label, ...) \
  STDTHREADCREATE_IMPL( \
    var, func, \
    do { \
      RES = ncclSystemError; \
      goto label; \
    } while (0), \
    __VA_ARGS__)

#define NEW_NOTHROW(var, x) \
  do { \
    (var) = new (std::nothrow) x{}; \
    if (!(var)) { \
      WARN("Allocation failed"); \
      return ncclSystemError; \
    } \
  } while (0)

#define NEW_NOTHROW_GOTO(var, x, RES, label) \
  do { \
    (var) = new (std::nothrow) x{}; \
    if (!(var)) { \
      WARN("Allocation failed"); \
      RES = ncclSystemError; \
      goto label; \
    } \
  } while (0)

#endif
