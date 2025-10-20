// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Synchronized.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {
// NOTE: use XLOGF instead of CLOGF to avoid dependency on logging util. It
// should use the same format (e.g., CTRAN ERROR prefix) following the logging
// initialized in the caller file.
#define CTRAN_ASYNC_ERR_HANDLE_IMPL(asyncErr, e)                    \
  do {                                                              \
    const auto errLog = fmt::format(                                \
        "{}: Encountered exception: {}", asyncErr->desc, e.what()); \
    if (asyncErr->abortOnError) {                                   \
      /* FATAL will abort with error stack */                       \
      XLOGF(FATAL, "{}; aborting", errLog);                         \
    } else {                                                        \
      XLOGF(ERR, "{}; setting async error flag", errLog);           \
      /* TODO: expose also error stack to user */                   \
      asyncErr->setAsyncException(e);                               \
    }                                                               \
  } while (0)

#define CTRAN_ASYNC_ERR_GUARD(asyncErr, code)                          \
  try {                                                                \
    code;                                                              \
  } catch (const ctran::utils::Exception& e) {                         \
    CTRAN_ASYNC_ERR_HANDLE_IMPL(asyncErr, e);                          \
  } catch (const std::runtime_error& e) {                              \
    /*TODO: replace remaining runtime_error with Exception */          \
    CTRAN_ASYNC_ERR_HANDLE_IMPL(                                       \
        asyncErr, ctran::utils::Exception(e.what(), commRemoteError)); \
  }

#define CTRAN_ASYNC_ERR_HANDLE_IMPL_FAULT_TOLERANCE(comm, e) \
  do {                                                       \
    CTRAN_ASYNC_ERR_HANDLE_IMPL(comm->getAsyncError(), e);   \
    if (comm->abortEnabled()) {                              \
      XLOGF(ERR, "Fault tolerance enabled; aborting");       \
      comm->setAbort();                                      \
    } else {                                                 \
      throw;                                                 \
    }                                                        \
  } while (0)

#define CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(comm, code)          \
  try {                                                            \
    code;                                                          \
  } catch (const ctran::utils::Exception& e) {                     \
    CTRAN_ASYNC_ERR_HANDLE_IMPL_FAULT_TOLERANCE(comm, e);          \
  } catch (const std::runtime_error& e) {                          \
    /*TODO: replace remaining runtime_error with Exception */      \
    /*TODO(T238821628): improve from simple commRemoteError */     \
    CTRAN_ASYNC_ERR_HANDLE_IMPL_FAULT_TOLERANCE(                   \
        comm, ctran::utils::Exception(e.what(), commRemoteError)); \
  }

class AsyncError {
 private:
  folly::Synchronized<Exception> asyncEx_{Exception()};

 public:
  const bool abortOnError;
  const std::string desc{"undefined"};

  AsyncError(bool abortOnError, const std::string& desc)
      : abortOnError(abortOnError), desc(desc){};

  inline void setAsyncException(const Exception& e) {
    asyncEx_ = e;
  }

  inline commResult_t getAsyncResult() const {
    return asyncEx_.rlock()->result();
  }

  inline Exception getAsyncException() const {
    return asyncEx_.copy();
  }
};

} // namespace ctran::utils
