// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// What injection::available() is allowed to mean.
//
// Every hardware fixture guards on it and skips when it is false, so the answer
// has to be "the control calls below will work", not "something opened". Those
// are different questions because IBVERBX_IBVERBS_SO deliberately names ANY
// provider -- the real libibverbs, a mock, the shim -- so a load check alone is
// true for all three while only one of them can be driven.
//
// Run against the stub libibverbs, which is exactly the interesting shape: a
// perfectly loadable provider with no injection control surface. A guard that
// only checked loadability would say yes here, and the fixture would then throw
// out of its first reset() rather than skipping -- turning an opt-out into a
// failure.
//
// Hardware-free by construction: nothing here opens a device.

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>

#include "comms/ctran/ibverbx/ib_injection/IbInjectionControl.h"

namespace ibverbx::injection::testing {
namespace {

TEST(IbInjectionControlGuardTest, NotAvailableWhenTheProviderIsNotTheShim) {
  // Asserted rather than assumed: with the variable unset, available() would be
  // false for the wrong reason and the test would pass vacuously.
  const char* provider = getenv("IBVERBX_IBVERBS_SO");
  ASSERT_NE(provider, nullptr)
      << "IBVERBX_IBVERBS_SO must point at the stub provider; the BUCK target "
         "sets it via $(location ...)";

  EXPECT_FALSE(available())
      << provider
      << " is loadable but exports no injection control surface, so "
         "available() must be false";
}

// The two have to agree. A guard that says yes where the calls throw is worse
// than no guard: the fixture that meant to skip reports a failure instead.
TEST(IbInjectionControlGuardTest, AnUnavailableShimIsAlsoUncallable) {
  ASSERT_FALSE(available());
  EXPECT_THROW(reset(), std::runtime_error);
}

} // namespace
} // namespace ibverbx::injection::testing
