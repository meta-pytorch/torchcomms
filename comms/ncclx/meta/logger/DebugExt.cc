// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdarg>
#include <cstdio>
#include <vector>

#include "core.h"
#include "debug.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ErrorStackUtil.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

// These are Meta's logging implementations, kept out of the baseline debug.cc.
// Guarded to v2_30+ since older versions keep their own copy in debug.cc.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)

void ncclMetaDebugLogError(
    ncclResult_t code,
    unsigned long flags,
    const char* file,
    const char* func,
    int line,
    const char* fmt,
    ...) {
  // Format the message once (same vsnprintf pattern as ncclMetaDebugLog).
  size_t logLen = 0;
  va_list vargs;
  va_start(vargs, fmt);
  logLen += std::vsnprintf(nullptr, 0, fmt, vargs);
  va_end(vargs);

  std::vector<char> buffer(logLen + 1); // +1 for null terminator
  va_start(vargs, fmt);
  // vsnprintf copy at most buf_size - 1 characters
  std::vsnprintf(buffer.data(), buffer.size(), fmt, vargs);
  va_end(vargs);

  // Emit the folly ERR log via the common path so glog still shows 'E' and
  // honors level/subsys filtering. Delegating also preserves the
  // ncclDebugNoWarn downgrade and the ncclLastError save consistently with
  // every other logging macro.
  ncclMetaDebugLog(
      NCCL_LOG_ERROR, flags, file, func, line, "%s", buffer.data());

  const std::string message{buffer.data()};
  // Capture the expensive native stack once and share it across all reporters.
  const auto stack = ::meta::comms::logger::captureNativeErrorStack();
  // ncclGetLastError state (always) and the process-global error store
  // (gated inside by NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES) are
  // independent of Scuba error logging.
  ::meta::comms::logger::setLastError(message, stack);
  ProcessGlobalErrorsUtil::addErrorAndStackTrace(message, stack);
  // Scuba error record: hidden, env-gated; skip when downgraded via
  // ncclDebugNoWarn.
  if (ncclDebugNoWarn == 0 && NCCL_SCUBA_LOG_ERROR_ENABLED) {
    ::meta::comms::logger::logErrorToScuba(
        message, static_cast<int>(code), ncclCodeToString(code), stack);
  }
}

#endif
