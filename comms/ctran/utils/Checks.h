// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <folly/Format.h>

#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

/**
 * Error check macros.
 * We use logging level following the rules below:
 * - ERR: report critical error and error stack trace
 * - WARN: report warning and continue (e.g., in *IGNORE macros)
 *
 * Rules to use the macros:
 * - If is to catch a potential Ctran internal bug, use macro that aborts
 * - For erros that should be returned to user (e.g., system error, bad input)
 *   + If the function returns a error code, use macro that returns the error
 *     code
 *   + If the function must return void, e.g., constructor, use macro that
 *     throws exception
 *   + If the function must return void and is unsafe to throw exception, use
 *     macro that ignores
 */

#define FB_CUDACHECK_RETURN(cmd, ret)                                    \
  do {                                                                   \
    cudaError_t err = cmd;                                               \
    if (err != cudaSuccess) {                                            \
      auto errStr = cudaGetErrorString(err);                             \
      CLOGF(ERR, "Cuda failure {} '{}'", static_cast<int>(err), errStr); \
      ErrorStackTraceUtil::logErrorMessage(                              \
          "Cuda Error: " + std::string(errStr));                         \
      return ret;                                                        \
    }                                                                    \
  } while (false)

#define FB_CUDACHECK(cmd) FB_CUDACHECK_RETURN(cmd, commUnhandledCudaError)

#define FB_CUDACHECKTHROW(cmd)                                      \
  do {                                                              \
    cudaError_t err = cmd;                                          \
    if (err != cudaSuccess) {                                       \
      CLOGF(                                                        \
          ERR,                                                      \
          "{}:{} Cuda failure {}",                                  \
          __FILE__,                                                 \
          __LINE__,                                                 \
          cudaGetErrorString(err));                                 \
      (void)cudaGetLastError();                                     \
      throw std::runtime_error(                                     \
          std::string("Cuda failure: ") + cudaGetErrorString(err)); \
    }                                                               \
  } while (false)

#define FB_CUDACHECKGOTO(cmd, RES, label)                     \
  do {                                                        \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
      CLOGF(ERR, "Cuda failure {}", cudaGetErrorString(err)); \
      RES = commUnhandledCudaError;                           \
      goto label;                                             \
    }                                                         \
  } while (false)

#define FB_CUCHECKGOTO(cmd, RES, label)                   \
  do {                                                    \
    CUresult err = cmd;                                   \
    if (err != CUDA_SUCCESS) {                            \
      const char* errStr;                                 \
      cuGetErrorString(err, &errStr);                     \
      CLOGF(ERR, "Cuda failure {}", std::string(errStr)); \
      RES = commUnhandledCudaError;                       \
      goto label;                                         \
    }                                                     \
  } while (false)

#define FB_CUDACHECKGUARD(cmd, RES)                           \
  do {                                                        \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
      CLOGF(ERR, "Cuda failure {}", cudaGetErrorString(err)); \
      RES = commUnhandledCudaError;                           \
      return;                                                 \
    }                                                         \
  } while (false)

// Report failure but clear error and continue
#define FB_CUDACHECKIGNORE(cmd)     \
  do {                              \
    cudaError_t err = cmd;          \
    if (err != cudaSuccess) {       \
      CLOGF(                        \
          WARN,                     \
          "{}:{} Cuda failure {}",  \
          __FILE__,                 \
          __LINE__,                 \
          cudaGetErrorString(err)); \
      (void)cudaGetLastError();     \
    }                               \
  } while (false)

// Use of abort should be aware of potential memory leak risk
// and place a signal handler to catch it and trigger termination processing
#define FB_CUDACHECKABORT(cmd)                                \
  do {                                                        \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
      CLOGF(ERR, "Cuda failure {}", cudaGetErrorString(err)); \
      abort();                                                \
    }                                                         \
  } while (false)

#define FB_SYSCHECK(statement, name)                              \
  do {                                                            \
    int retval;                                                   \
    FB_SYSCHECKSYNC((statement), name, retval);                   \
    if (retval == -1) {                                           \
      CLOGF(ERR, "Call to " name " failed: {}", strerror(errno)); \
      return commSystemError;                                     \
    }                                                             \
  } while (false)

#define FB_SYSCHECKVAL(call, name, retval)                         \
  do {                                                             \
    FB_SYSCHECKSYNC(call, name, retval);                           \
    if (retval == -1) {                                            \
      CLOGF(ERR, "Call to " name " failed : {}", strerror(errno)); \
      return commSystemError;                                      \
    }                                                              \
  } while (false)

#define FB_SYSCHECKSYNC(statement, name, retval)                             \
  do {                                                                       \
    retval = (statement);                                                    \
    if (retval == -1 &&                                                      \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {       \
      CLOGF(ERR, "Call to " name " returned {}, retrying", strerror(errno)); \
    } else {                                                                 \
      break;                                                                 \
    }                                                                        \
  } while (true)

#define FB_SYSCHECKGOTO(statement, name, RES, label)              \
  do {                                                            \
    int retval;                                                   \
    FB_SYSCHECKSYNC((statement), name, retval);                   \
    if (retval == -1) {                                           \
      CLOGF(ERR, "Call to " name " failed: {}", strerror(errno)); \
      RES = commSystemError;                                      \
      goto label;                                                 \
    }                                                             \
  } while (0)

#define FB_SYSCHECKTHROW(cmd)                                                  \
  do {                                                                         \
    int err = cmd;                                                             \
    if (err != 0) {                                                            \
      auto errstr = folly::errnoStr(err);                                      \
      CLOGF(ERR, "{}:{} -> {} ({})", __FILE__, __LINE__, err, errstr.c_str()); \
      throw std::runtime_error(std::string("System error: ") + errstr);        \
    }                                                                          \
  } while (0)

#define FB_SYSCHECKRETURN(cmd, retval)                                         \
  do {                                                                         \
    int err = cmd;                                                             \
    if (err != 0) {                                                            \
      auto errstr = folly::errnoStr(err);                                      \
      CLOGF(ERR, "{}:{} -> {} ({})", __FILE__, __LINE__, err, errstr.c_str()); \
      return retval;                                                           \
    }                                                                          \
  } while (0)

// Pthread calls don't set errno and never return EINTR.
#define FB_PTHREADCHECK(statement, name)                           \
  do {                                                             \
    int retval = (statement);                                      \
    if (retval != 0) {                                             \
      CLOGF(ERR, "Call to " name " failed: {}", strerror(retval)); \
      return commSystemError;                                      \
    }                                                              \
  } while (0)

#define FB_PTHREADCHECKGOTO(statement, name, RES, label)           \
  do {                                                             \
    int retval = (statement);                                      \
    if (retval != 0) {                                             \
      CLOGF(ERR, "Call to " name " failed: {}", strerror(retval)); \
      RES = commSystemError;                                       \
      goto label;                                                  \
    }                                                              \
  } while (0)

#define FB_NEQCHECK(statement, value) \
  do {                                \
    if ((statement) != value) {       \
      /* Print the back trace*/       \
      CLOGF(                          \
          ERR,                        \
          "{}:{} -> {} ({})",         \
          __FILE__,                   \
          __LINE__,                   \
          commSystemError,            \
          strerror(errno));           \
      return commSystemError;         \
    }                                 \
  } while (0)

#define FB_NEQCHECKGOTO(statement, value, RES, label)                         \
  do {                                                                        \
    if ((statement) != value) {                                               \
      /* Print the back trace*/                                               \
      RES = commSystemError;                                                  \
      CLOGF(                                                                  \
          ERR, "{}:{} -> {} ({})", __FILE__, __LINE__, RES, strerror(errno)); \
      goto label;                                                             \
    }                                                                         \
  } while (0)

#define FB_EQCHECK(statement, value) \
  do {                               \
    if ((statement) == value) {      \
      /* Print the back trace*/      \
      CLOGF(                         \
          ERR,                       \
          "{}:{} -> {} ({})",        \
          __FILE__,                  \
          __LINE__,                  \
          commSystemError,           \
          strerror(errno));          \
      return commSystemError;        \
    }                                \
  } while (0)

#define FB_EQCHECKGOTO(statement, value, RES, label)                          \
  do {                                                                        \
    if ((statement) == value) {                                               \
      /* Print the back trace*/                                               \
      RES = commSystemError;                                                  \
      CLOGF(                                                                  \
          ERR, "{}:{} -> {} ({})", __FILE__, __LINE__, RES, strerror(errno)); \
      goto label;                                                             \
    }                                                                         \
  } while (0)

// Propagate errors up
#define FB_COMMCHECK(call)                                \
  do {                                                    \
    commResult_t RES = call;                              \
    if (RES != commSuccess && RES != commInProgress) {    \
      CLOGF(ERR, "{}:{} -> {}", __FILE__, __LINE__, RES); \
      return RES;                                         \
    }                                                     \
  } while (0)

// Propagate errors up for ibverbx
#define FOLLY_EXPECTED_CHECK(RES) \
  do {                            \
    if (RES.hasError()) {         \
      CLOGF(                      \
          ERR,                    \
          "{}:{} -> {}, {}",      \
          __FILE__,               \
          __LINE__,               \
          RES.error().errNum,     \
          RES.error().errStr);    \
      return commSystemError;     \
    }                             \
  } while (0)

#define FOLLY_EXPECTED_CHECKTHROW(RES)                                  \
  do {                                                                  \
    if (RES.hasError()) {                                               \
      CLOGF(                                                            \
          ERR,                                                          \
          "{}:{} -> {} ({})",                                           \
          __FILE__,                                                     \
          __LINE__,                                                     \
          RES.error().errNum,                                           \
          RES.error().errStr);                                          \
      throw std::runtime_error(                                         \
          std::string("COMM internal failure: ") + RES.error().errStr); \
    }                                                                   \
  } while (0)

#define FOLLY_EXPECTED_CHECKGOTO(RES, label) \
  do {                                       \
    if (RES.hasError()) {                    \
      CLOGF(                                 \
          ERR,                               \
          "{}:{} -> {} ({})",                \
          __FILE__,                          \
          __LINE__,                          \
          RES.error().errNum,                \
          RES.error().errStr);               \
      goto label;                            \
    }                                        \
  } while (0)

#define FB_COMMCHECKTHROW(cmd)                         \
  do {                                                 \
    commResult_t RES = cmd;                            \
    if (RES != commSuccess && RES != commInProgress) { \
      CLOGF(                                           \
          ERR,                                         \
          "{}:{} -> {} ({})",                          \
          __FILE__,                                    \
          __LINE__,                                    \
          RES,                                         \
          ::meta::comms::commCodeToString(RES));       \
      throw std::runtime_error(                        \
          std::string("COMM internal failure: ") +     \
          ::meta::comms::commCodeToString(RES));       \
    }                                                  \
  } while (0)

#define FB_COMMCHECKTHROW_EX(cmd, rank, commHash)      \
  do {                                                 \
    commResult_t RES = cmd;                            \
    if (RES != commSuccess && RES != commInProgress) { \
      CLOGF(                                           \
          ERR,                                         \
          "{}:{} -> {} ({})",                          \
          __FILE__,                                    \
          __LINE__,                                    \
          RES,                                         \
          ::meta::comms::commCodeToString(RES));       \
      throw ctran::utils::Exception(                   \
          std::string("COMM internal failure: ") +     \
              ::meta::comms::commCodeToString(RES),    \
          RES,                                         \
          rank,                                        \
          commHash);                                   \
    }                                                  \
  } while (0)

#define FB_COMMCHECKGOTO(call, RES, label)                \
  do {                                                    \
    RES = call;                                           \
    if (RES != commSuccess && RES != commInProgress) {    \
      CLOGF(ERR, "{}:{} -> {}", __FILE__, __LINE__, RES); \
      goto label;                                         \
    }                                                     \
  } while (0)

// Report failure but clear error and continue
#define FB_COMMCHECKIGNORE(call)                       \
  do {                                                 \
    commResult_t RES = call;                           \
    if (RES != commSuccess && RES != commInProgress) { \
      CLOGF(                                           \
          WARN,                                        \
          "{}:{}:{} -> {} ({})",                       \
          __FILE__,                                    \
          __func__,                                    \
          __LINE__,                                    \
          RES,                                         \
          ::meta::comms::commCodeToString(RES));       \
    }                                                  \
  } while (0)

#define FB_CHECKABORT(statement, ...)             \
  do {                                            \
    if (!(statement)) {                           \
      CLOGF(ERR, "Check failed: {}", #statement); \
      CLOGF(ERR, ##__VA_ARGS__);                  \
      abort();                                    \
    }                                             \
  } while (0);

#define FB_CHECKTHROW(statement, ...)                                          \
  do {                                                                         \
    if (!(statement)) {                                                        \
      CLOGF(                                                                   \
          ERR, "Check failed: {} - {}", #statement, fmt::format(__VA_ARGS__)); \
      throw std::runtime_error(                                                \
          fmt::format(                                                         \
              "Check failed: {} - {}", #statement, fmt::format(__VA_ARGS__))); \
    }                                                                          \
  } while (0)

#define FB_COMMWAIT(call, cond, abortFlagPtr)             \
  do {                                                    \
    uint32_t* tmpAbortFlag = (abortFlagPtr);              \
    commResult_t RES = call;                              \
    if (RES != commSuccess && RES != commInProgress) {    \
      CLOGF(ERR, "{}:{} -> {}", __FILE__, __LINE__, RES); \
      return commInternalError;                           \
    }                                                     \
    if (__atomic_load(tmpAbortFlag, __ATOMIC_ACQUIRE))    \
      FB_NEQCHECK(*tmpAbortFlag, 0);                      \
  } while (!(cond))

#define FB_COMMWAITGOTO(call, cond, abortFlagPtr, RES, label) \
  do {                                                        \
    uint32_t* tmpAbortFlag = (abortFlagPtr);                  \
    RES = call;                                               \
    if (RES != commSuccess && RES != commInProgress) {        \
      CLOGF(ERR, "{}:{} -> {}", __FILE__, __LINE__, RES);     \
      goto label;                                             \
    }                                                         \
    if (__atomic_load(tmpAbortFlag, __ATOMIC_ACQUIRE))        \
      FB_NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label);          \
  } while (!(cond))

#define FB_COMMCHECKTHREAD(a, args)                                            \
  do {                                                                         \
    if (((args)->ret = (a)) != commSuccess && (args)->ret != commInProgress) { \
      CLOGF_SUBSYS(                                                            \
          ERR,                                                                 \
          INIT,                                                                \
          "{}:{} -> {} [Async thread]",                                        \
          __FILE__,                                                            \
          __LINE__,                                                            \
          (args)->ret);                                                        \
      return args;                                                             \
    }                                                                          \
  } while (0)

#define FB_CUDACHECKTHREAD(a)             \
  do {                                    \
    if ((a) != cudaSuccess) {             \
      CLOGF_SUBSYS(                       \
          ERR,                            \
          INIT,                           \
          "{}:{}} -> {}} [Async thread]", \
          __FILE__,                       \
          __LINE__,                       \
          args->ret);                     \
      args->ret = commUnhandledCudaError; \
      return args;                        \
    }                                     \
  } while (0)

#define FB_COMMARGCHECK(statement, ...)                     \
  do {                                                      \
    if (!(statement)) {                                     \
      CLOGF(ERR, ##__VA_ARGS__);                            \
      return ErrorStackTraceUtil::log(commInvalidArgument); \
    }                                                       \
  } while (0);

#define FB_ERRORRETURN(error, ...)                                  \
  do {                                                              \
    CLOGF(ERR, ##__VA_ARGS__);                                      \
    ErrorStackTraceUtil::logErrorMessage(fmt::format(__VA_ARGS__)); \
    return error;                                                   \
  } while (0)

#define FB_ERRORTHROW(error, ...)                \
  do {                                           \
    CLOGF(ERR, ##__VA_ARGS__);                   \
    throw std::runtime_error(                    \
        std::string("COMM internal failure: ") + \
        ::meta::comms::commCodeToString(error)); \
  } while (0)
