// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// The contract that a NAMED provider is never silently substituted.
//
// This is deliberately its own binary rather than a case in IbverbxTest.cc.
// buildIbvSymbols keeps its dlopen handles in function-local statics, and its
// failure path closes them, so a failing call placed after a successful
// ibvInit() in the same process would drop a reference the global ibvSymbols is
// still using. A binary that never opens a real provider cannot do that.
//
// Nothing here touches a NIC or a GPU: a path that does not exist fails to
// dlopen on every host.

#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

// Without this, the only symptom of a regression is that every injection run
// comes back green having injected nothing -- a typo'd IBVERBX_IBVERBS_SO, or
// an fbpkg that failed to mount, would fall back to the host's real libibverbs
// and report success. That outcome is indistinguishable from a healthy run,
// which is why the fallback had to go and why it needs a test holding it gone.
TEST(IbverbxProviderPathTest, NamedProviderThatCannotLoadIsAnError) {
#ifdef IBVERBX_BUILD_RDMA_CORE
  GTEST_SKIP() << "the rdma-core variant links its provider, so there is no "
                  "path to honor and buildIbvSymbols never reads ibv_path";
#else
  // A local IbvSymbols, never the global: this must not disturb any ibvSymbols
  // a later test depends on.
  IbvSymbols symbols{};
  EXPECT_NE(
      buildIbvSymbols(symbols, "/nonexistent/ib_injection/libibverbs.so"), 0)
      << "a requested provider that cannot be loaded must fail, not fall back "
         "to the host's real libibverbs";
#endif
}

} // namespace ibverbx
