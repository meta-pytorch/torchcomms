// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Synchronized.h>

#include "comms/ctran/utils/AbortUtils.h"
#include "comms/ctran/utils/CtranLogger.h"
#include "comms/ctran/utils/Exception.h"

namespace ctran::utils {
// Use the named CTRAN logger so these header macros do not inherit the caller's
// logging category.
#define CTRAN_ASYNC_ERR_HANDLE_IMPL(asyncErr, e)                    \
  do {                                                              \
    const auto errLog = fmt::format(                                \
        "{}: Encountered exception: {}", asyncErr->desc, e.what()); \
    if (asyncErr->abortOnError) {                                   \
      /* FATAL will abort with error stack */                       \
      CTRAN_LOG(FATAL, "{}; aborting", errLog);                     \
    } else {                                                        \
      CTRAN_LOG(ERR, "{}; setting async error flag", errLog);       \
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

#define CTRAN_ASYNC_ERR_HANDLE_IMPL_FAULT_TOLERANCE(comm, e, opType, opCount) \
  do {                                                                        \
    CTRAN_ASYNC_ERR_HANDLE_IMPL(comm->getAsyncError(), e);                    \
    if (comm->abortEnabled()) {                                               \
      CTRAN_LOG(                                                              \
          ERR,                                                                \
          "Fault tolerance enabled; marking communicator aborted "            \
          "(opType={}, opCount={}) on rank {} commHash {:x}",                 \
          opType,                                                             \
          opCount,                                                            \
          comm->logMetaData_.rank,                                            \
          comm->logMetaData_.commHash);                                       \
      comm->setAbort(                                                         \
          comms::fault_tolerance::AbortInfo{                                  \
              .reason = ctran::utils::abortReason(e.result()),                \
              .context = fmt::format(                                         \
                  "op_type={} op_count={} rank={} comm_hash={} error={}",     \
                  opType,                                                     \
                  opCount,                                                    \
                  comm->logMetaData_.rank,                                    \
                  comm->logMetaData_.commHash,                                \
                  e.what()),                                                  \
          });                                                                 \
    } else {                                                                  \
      throw;                                                                  \
    }                                                                         \
  } while (0)

#define CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(comm, code, opType, opCount) \
  try {                                                                    \
    code;                                                                  \
  } catch (const ctran::utils::Exception& e) {                             \
    CTRAN_ASYNC_ERR_HANDLE_IMPL_FAULT_TOLERANCE(comm, e, opType, opCount); \
  } catch (const std::runtime_error& e) {                                  \
    /*TODO: replace remaining runtime_error with Exception */              \
    /*TODO(T238821628): improve from simple commRemoteError */             \
    CTRAN_ASYNC_ERR_HANDLE_IMPL_FAULT_TOLERANCE(                           \
        comm,                                                              \
        ctran::utils::Exception(e.what(), commRemoteError),                \
        opType,                                                            \
        opCount);                                                          \
  }

class AsyncError {
 private:
  folly::Synchronized<Exception> asyncEx_{Exception()};

 public:
  const bool abortOnError;
  const std::string desc{"undefined"};

  AsyncError(bool abortOnError, const std::string& desc)
      : abortOnError(abortOnError), desc(desc) {};

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
