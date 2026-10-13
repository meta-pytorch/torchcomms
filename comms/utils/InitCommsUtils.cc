// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/InitCommsUtils.h"

#include <folly/synchronization/CallOnce.h>

#include "comms/utils/InitFolly.h"

namespace meta::comms {

void initCommsUtils() {
  static folly::once_flag once;
  folly::call_once(once, [] { initFolly(); });
}

} // namespace meta::comms
