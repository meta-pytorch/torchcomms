// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fcntl.h>
#include <fmt/core.h>
#include <folly/Singleton.h>
#include <folly/Synchronized.h>
#include <folly/init/Init.h>
#include <chrono>
#include <thread>

#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/backends/ib/IbvWrap.h"
#include "comms/ctran/backends/ib/ibutils.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

using namespace ctran::ibvwrap;

namespace {
static folly::Singleton<VerbsUtils> kVerbsUtils;

static inline commResult_t addToStackTrace(const std::string errorMessage) {
  CLOGF(ERR, "{}", errorMessage.c_str());
  ErrorStackTraceUtil::logErrorMessage(errorMessage);
  return commSystemError;
}
} // namespace

/* Set ibv_get_async_event to poll mode intead of default blocking mode.
 * Allows checking for link down.  One instance per port.
 */
commResult_t IbUtils::pollForAsyncEvent(
    struct ibverbx::ibv_context* ibvContext,
    IVerbsWrapper* verbsPtr) {
  struct pollfd fdSet;
  fdSet.fd = ibvContext->async_fd;
  fdSet.events = POLLIN;
  fdSet.revents = 0;

  // set ibv_get_async_event to non-blocking mode
  int flags = fcntl(fdSet.fd, F_GETFL);

  if (flags == -1) {
    return addToStackTrace("fcntl flags failure");
  }
  auto ret = fcntl(fdSet.fd, F_SETFL, flags | O_NONBLOCK);
  if (ret == -1) {
    return addToStackTrace("fcntl set NONBLOCK failure");
  }

  bool stopRequested = false;
  bool linkDown;
  do {
    // poll file descriptor; if there is a return value, it means
    // there is an async event read further down
    ret = verbsPtr->ibv_poll_async_fd(
        &fdSet, 1, NCCL_CTRAN_IB_ASYNC_EVENT_POLL_INTERVAL_MS);
    if (NCCL_IB_ASYNC_EVENT_LOOP == NCCL_IB_ASYNC_EVENT_LOOP::ctran) {
      auto singleton = CtranIbSingleton::getInstance();
      CHECK_VALID_IB_SINGLETON(singleton);
      stopRequested = singleton->stopIbAsyncEventHandlerFlag;
    }
    linkDown = linkDownTimeout();
  } while (ret == 0 && !stopRequested && !linkDown);

  if (stopRequested) {
    // stop requested during cleanup; hence exiting this loop.
    CLOGF_SUBSYS(
        INFO, NET, "CTRAN-IB: Exiting ibAsyncEventHandler (requested)");
    return commInProgress;
  }
  if (linkDown) {
    CLOGF_SUBSYS(INFO, NET, "NET/IB : Exiting ibAsyncEventHandler (link down)");
    joinTimeoutThread();
    return commSystemError;
  }
  if (ret < 0) {
    return addToStackTrace(
        fmt::format(
            "NET/IB : poll for IB async events failed with error code:{}",
            ret));
  }
  if ((fdSet.revents & POLLIN) != POLLIN) {
    return addToStackTrace(
        fmt::format(
            "NET/IB : poll returned unexpected POLLERR or POLLHUP for dev={} (probably a bad device)",
            ibvContext->device->name));
  }
  return commSuccess;
}

/* static - used by baseline only */
IVerbsWrapper* VerbsUtils::getVerbsPtr() {
  return kVerbsUtils.try_get()->verbsPtr.get();
}

/* static - used by baseline only */
std::shared_ptr<VerbsUtils> VerbsUtils::getInstance() {
  return kVerbsUtils.try_get();
}

/* static, run in one thread per IB Async Event handler instance */
void IbUtils::timeoutHandler(
    IbUtils* ibutils,
    std::chrono::milliseconds duration,
    const std::string devName,
    const int port) {
  auto locked = ibutils->linkUpEvent_.lock();
  if (!ibutils->linkUpSignal_.wait_for(
          locked.as_lock(), duration, [&] { return *locked; })) {
    std::string errorMessage = fmt::format(
        "Link down timeout reached for {} ({} ms).", devName, duration.count());
    addToStackTrace(errorMessage);
    ProcessGlobalErrorsUtil::setNic(devName, port, errorMessage);
    XLOGF(
        WARN,
        "ctranCommSetAsyncError: error comm NULL sets state %d",
        commSystemError);

    ibutils->setLinkFlapState(
        ctran::ibutils::LinkFlapState::IB_ASYNC_LINK_TIMEOUT);
  } else {
    CLOGF_SUBSYS(INFO, NET, "NET/IB : timeoutHandler: link back up.");
  }
}

/* This function called within the IB Async Event handler thread */
void IbUtils::linkDownSetTimeout(const std::string& devName, const int port) {
  linkFlapState_ = ctran::ibutils::LinkFlapState::IB_ASYNC_LINK_DOWN;
  const int timeoutMs = NCCL_IB_LINK_DOWN_TIMEOUT * 1000;
  if (timeoutMs <= 0) {
    return;
  }
  *linkUpEvent_.lock() = false;
  int device;
  FB_CUDACHECKIGNORE(cudaGetDevice(&device));
  // capture by value rather than reference to handle out-of-scope
  timeoutThread_ = std::thread{[=, this]() {
    // Set cuda device for the thread so that logging can correctly
    // identify the local rank of the thread.
    cudaSetDevice(device);
    commNamedThreadStart("IBtimeoutHandler");
    timeoutHandler(this, std::chrono::milliseconds(timeoutMs), devName, port);
  }};
}

namespace ctran::ibutils {} // namespace ctran::ibutils

void IbUtils::joinTimeoutThread() {
  if (timeoutThread_.joinable()) {
    timeoutThread_.join();
  }
}

void IbUtils::sendLinkUpEvent() {
  {
    auto locked = linkUpEvent_.lock();
    *locked = true;
  }
  linkFlapState_ = ctran::ibutils::LinkFlapState::IB_ASYNC_LINK_UP;
  linkUpSignal_.notify_one();
  joinTimeoutThread();
}

bool IbUtils::linkDownTimeout() {
  return linkFlapState_.load() ==
      ctran::ibutils::LinkFlapState::IB_ASYNC_LINK_TIMEOUT;
}

void IbUtils::setLinkFlapState(ctran::ibutils::LinkFlapState state) {
  linkFlapState_ = state;
}

commResult_t IbUtils::triageIbAsyncEvents(
    ibverbx::ibv_event_type eventType,
    const std::string& devName,
    const int port) {
  // TODO: Rebase functionality for this cvar and remove this local variable
  bool NCCL_IB_ENABLE_REPORT_TO_PROCESS_GLOBAL_ERRORS = true;
  if (!NCCL_IB_ENABLE_REPORT_TO_PROCESS_GLOBAL_ERRORS) {
    CLOGF_SUBSYS(INFO, NET, "IB report to process global errors is disabled");
    return commSuccess;
  }

  char* asyncErrorMsg = nullptr;
  if (commSuccess != wrap_ibv_event_type_str(&asyncErrorMsg, eventType)) {
    CLOGF(WARN, "ibv_event_type_str not parsable - eventType={}", eventType);
  }
  auto msg = fmt::format(
      "NET/IB: {}:{} Got async event: {} ({})",
      devName,
      port,
      asyncErrorMsg,
      eventType);

  commResult_t ret = commSuccess;
  switch (eventType) {
    // non-fatal events
    case ibverbx::IBV_EVENT_COMM_EST:
      break;

    case ibverbx::IBV_EVENT_SQ_DRAINED:
      [[fallthrough]];
    case ibverbx::IBV_EVENT_PATH_MIG:
    case ibverbx::IBV_EVENT_QP_LAST_WQE_REACHED:
    case ibverbx::IBV_EVENT_SRQ_LIMIT_REACHED:
    case ibverbx::IBV_EVENT_LID_CHANGE:
    case ibverbx::IBV_EVENT_PKEY_CHANGE:
    case ibverbx::IBV_EVENT_GID_CHANGE:
    case ibverbx::IBV_EVENT_SM_CHANGE:
    case ibverbx::IBV_EVENT_CLIENT_REREGISTER:
      CLOGF_SUBSYS(INFO, NET, "{}", msg.c_str());
      break;

    case ibverbx::IBV_EVENT_PORT_ACTIVE:
      CLOGF_SUBSYS(INFO, NET, "{}", msg.c_str());
      sendLinkUpEvent();
      break;

    case ibverbx::IBV_EVENT_PORT_ERR:
      // a link down event is not considered an error unless it is down
      // for a period longer than the configured timeout
      // so do not return commSystemError here
      addToStackTrace(msg);
      // for now, only ctran supports timer
      if (NCCL_IB_ASYNC_EVENT_LOOP == NCCL_IB_ASYNC_EVENT_LOOP::ctran) {
        // set timer; if timer expires it will trigger an error
        linkDownSetTimeout(devName, port);
      }
      break;

    // fatal events
    case ibverbx::IBV_EVENT_QP_FATAL:
      [[fallthrough]];
    case ibverbx::IBV_EVENT_QP_REQ_ERR:
    case ibverbx::IBV_EVENT_QP_ACCESS_ERR:
    case ibverbx::IBV_EVENT_PATH_MIG_ERR:
    case ibverbx::IBV_EVENT_CQ_ERR:
    case ibverbx::IBV_EVENT_SRQ_ERR:
    case ibverbx::IBV_EVENT_DEVICE_FATAL:
      ret = commSystemError;
      addToStackTrace(msg);
      ProcessGlobalErrorsUtil::setNic(devName, port, msg);
      // preserve baseline functionality by not sending async error
      if (NCCL_IB_ASYNC_EVENT_LOOP == NCCL_IB_ASYNC_EVENT_LOOP::ctran) {
        XLOGF(
            WARN,
            "ctranCommSetAsyncError: error comm NULL sets state %d",
            commSystemError);
      }
      break;

    default:
      CLOGF_SUBSYS(INFO, NET, "{}", msg.c_str());
      break;
  }
  return ret;
}
