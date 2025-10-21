// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "bootstrap.h" // @manual
#include "comms/common/tests/TestBaselineBootstrap.h"

namespace meta::comms {

folly::SemiFuture<int> TestBaselineBootstrap::allGather(
    void* buf,
    int len,
    int /* rank */,
    int /* nranks */) {
  auto res = bootstrapAllGather(comm_->bootstrap, buf, len);
  return folly::makeSemiFuture<int>(static_cast<int>(res));
}

} // namespace meta::comms
