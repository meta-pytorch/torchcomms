// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <poll.h>

#include <atomic>
#include <condition_variable>
#include <thread>

#include <folly/Synchronized.h>

#include "comms/ctran/backends/ib/IbvWrap.h"

namespace ctran::ibutils {

enum class AsyncEventsSeverity {
  IB_ASYNC_EVENT_FATAL = 1,
  IB_ASYNC_EVENT_WARNING,
  IB_ASYNC_EVENT_INFO,
};

enum class LinkFlapState {
  IB_ASYNC_LINK_UP = 1,
  IB_ASYNC_LINK_DOWN,
  IB_ASYNC_LINK_TIMEOUT,
};

} // namespace ctran::ibutils

/*
 * Implementation of verbs wrappers; allows mocking.
 */
class IVerbsWrapper {
 public:
  virtual ~IVerbsWrapper() = default;
  virtual int
  ibv_poll_async_fd(struct pollfd* fds, nfds_t nfds, int timeout) = 0;
  virtual commResult_t ibv_get_async_event(
      ibverbx::ibv_context* context,
      ibverbx::ibv_async_event* event) = 0;
  virtual commResult_t ibv_ack_async_event(
      struct ibverbx::ibv_async_event* event) = 0;
};

class VerbsWrapper : public IVerbsWrapper {
 public:
  int ibv_poll_async_fd(struct pollfd* fds, nfds_t nfds, int timeout) {
    return poll(fds, nfds, timeout);
  }
  commResult_t ibv_get_async_event(
      struct ibverbx::ibv_context* context,
      struct ibverbx::ibv_async_event* event) {
    return ctran::ibvwrap::wrap_ibv_get_async_event(context, event);
  }
  commResult_t ibv_ack_async_event(struct ibverbx::ibv_async_event* event) {
    return ctran::ibvwrap::wrap_ibv_ack_async_event(event);
  }
};

/* Singleton class to allow verbs mocking for testing */
class VerbsUtils {
 public:
  // By default, verbs_ is a pointer to VerbsWrapper but it can be set
  // to a derived class for testing.
  template <typename T>
  T* setVerbs() {
    static_assert(
        std::is_base_of<IVerbsWrapper, T>::value,
        "T must derive from IVerbsWrapper");
    verbsPtr = std::make_unique<T>();
    return dynamic_cast<T*>(verbsPtr.get());
  }

  static IVerbsWrapper* getVerbsPtr();
  static std::shared_ptr<VerbsUtils> getInstance();

  std::unique_ptr<IVerbsWrapper> verbsPtr = std::make_unique<VerbsWrapper>();
};

/* Class to support link failure detection. One instance of this
   class per event handler thread.  In CTran, there is only one
   instance of the event handler thread.  In Baseline, there is one
   instance per port. */
class IbUtils {
 public:
  void joinTimeoutThread();
  bool linkDownTimeout();
  void linkDownSetTimeout(const std::string& devName, const int port);
  void sendLinkUpEvent();
  void setLinkFlapState(::ctran::ibutils::LinkFlapState state);
  commResult_t pollForAsyncEvent(
      struct ibverbx::ibv_context* ibvContext,
      IVerbsWrapper* verbsPtr);

  commResult_t triageIbAsyncEvents(
      ibverbx::ibv_event_type eventType,
      const std::string& devName,
      const int port);

  /* run in a thread */
  static void timeoutHandler(
      IbUtils* ibutils,
      std::chrono::milliseconds duration,
      const std::string devName,
      const int port);

  folly::Synchronized<bool, std::mutex> linkUpEvent_;
  std::condition_variable linkUpSignal_;

 private:
  std::thread timeoutThread_;
  std::atomic<::ctran::ibutils::LinkFlapState> linkFlapState_{
      ::ctran::ibutils::LinkFlapState::IB_ASYNC_LINK_UP};
};
