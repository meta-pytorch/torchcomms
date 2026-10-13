// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace meta::comms {

// One-time initialization of libcommsutils. Each shared library links its own
// folly, so a consumer initializes its own copy with initFolly() and this one
// with initCommsUtils(); both are idempotent.
void initCommsUtils();

} // namespace meta::comms
