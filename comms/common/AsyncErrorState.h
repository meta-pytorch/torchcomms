// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <utility>

#include <folly/Synchronized.h>

#include "comms/utils/commSpecs.h"

namespace comms {

struct AsyncErrorSnapshot {
  commResult_t code{commSuccess};
  std::string message;
};

// Host-only, communicator-scoped asynchronous error state. Writes replace the
// complete snapshot so readers cannot observe a code and message from
// different errors.
class AsyncErrorState final {
 public:
  void set(AsyncErrorSnapshot error) {
    state_ = std::move(error);
  }

  AsyncErrorSnapshot get() const {
    return state_.copy();
  }

  commResult_t result() const {
    return state_.rlock()->code;
  }

 private:
  folly::Synchronized<AsyncErrorSnapshot> state_;
};

} // namespace comms
