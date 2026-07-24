// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/Abort.h"

#include <memory>

namespace comms::fault_tolerance {

std::shared_ptr<Abort> createAbort(bool enabled) {
  if (enabled) {
    return std::make_shared<Abort>(/*enabled=*/true);
  } else {
    static const std::shared_ptr<Abort> disabled =
        std::make_shared<Abort>(/*enabled=*/false);
    return disabled;
  }
}

} // namespace comms::fault_tolerance
