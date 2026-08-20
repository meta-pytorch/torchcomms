// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/common/fault_tolerance/TestAbort.h"

#include <memory>

#include "comms/common/fault_tolerance/Abort.h"

namespace comms::fault_tolerance::testing {

AbortDevice testAbortDevice() {
  // Leaked deliberately: the mapped pinned state would otherwise be freed
  // during static destruction, after the CUDA runtime has already torn down.
  static auto* abort = new std::shared_ptr<Abort>([] {
    auto created = createAbort(/*enabled=*/true, AbortBehavior::TRAP);
    created->setDefaultTimeout(kTestAbortTimeout);
    return created;
  }());
  return (*abort)->getDeviceHandle();
}

} // namespace comms::fault_tolerance::testing
