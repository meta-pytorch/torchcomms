// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Delegation test for the ib_injection shim: proves the forwarders actually
// call through to the real provider, with the caller's arguments.
//
// Separate from IbInjectionDlopenTest, which checks the symbol TABLE. Those
// checks and the forwarders themselves are both generated from
// IbverbxSymbols.def, so a wrong entry yields a wrong forwarder and a matching
// wrong expectation -- and every symbol test stays green. This test resolves
// three verbs by hand against a hand-written stub libibverbs, so nothing here
// agrees with the table by construction.
//
// The stub is IbInjectionStubLibibverbs.cc, which is NOT the FakeProvider in
// IbInjectionEngineTest.cc -- that one is a linked in-process class supplying
// the objects the engine registers, on the other side of the shim entirely.
//
// One verb per lookup style the shim implements, because each is resolved by a
// different code path and a break in one is invisible to the others:
//
//   ibv_query_device     realSym      -> dlvsym on the ibv handle
//   mlx5dv_init_obj      realMlx5Sym  -> dlvsym on the mlx5 handle
//   mlx5dv_query_device  realMlx5Sym  -> plain dlsym on the mlx5 handle
//
// The provider paths come from the BUCK target's env rather than setenv(), so
// they are set before the process starts: the shim latches its real-provider
// handles in function-local statics on first use, and a setenv() racing that
// first use would be resolved-once and silently wrong.

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/tests/IbInjectionStubLibibverbs.h"

using ibverbx::ibv_context;
using ibverbx::ibv_device_attr;
using ibverbx::mlx5dv_context;
using ibverbx::mlx5dv_obj;

namespace {

class IbInjectionDelegationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* shim = getenv("IB_INJECTION_SO");
    ASSERT_NE(shim, nullptr)
        << "IB_INJECTION_SO must point at the shim; the BUCK target sets it";

    // Asserted rather than assumed: if the target stopped setting these the
    // shim would quietly delegate to the host's real libibverbs and this test
    // would be checking that instead of the stub.
    const char* realIbv = getenv("IB_INJECTION_REAL_IBVERBS_SO");
    const char* realMlx5 = getenv("IB_INJECTION_REAL_MLX5_SO");
    ASSERT_NE(realIbv, nullptr) << "IB_INJECTION_REAL_IBVERBS_SO unset";
    ASSERT_NE(realMlx5, nullptr) << "IB_INJECTION_REAL_MLX5_SO unset";

    handle_ = dlopen(shim, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle_, nullptr) << "dlopen(" << shim << "): " << dlerror();
  }

  void TearDown() override {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  void* handle_{nullptr};
};

// A forwarder reached the stub AND carried its arguments. Checking the
// out-params matters as much as the return: a forwarder that dropped or
// reordered arguments would still return the sentinel.
TEST_F(IbInjectionDelegationTest, VersionedIbvVerbReachesTheProvider) {
  auto* queryDevice = reinterpret_cast<int (*)(ibv_context*, ibv_device_attr*)>(
      dlvsym(handle_, "ibv_query_device", "IBVERBS_1.1"));
  ASSERT_NE(queryDevice, nullptr) << "dlvsym: " << dlerror();

  // A non-null context the stub can distinguish from nullptr; the shim passes
  // it straight through and never dereferences it.
  ibv_context stubCtx{};
  ibv_device_attr attr{};
  EXPECT_EQ(queryDevice(&stubCtx, &attr), kStubQueryDeviceRet)
      << "ibv_query_device did not reach the stub libibverbs";
  EXPECT_EQ(attr.max_qp, kStubQueryDeviceMaxQp)
      << "the out-param the stub wrote did not come back to the caller";
  EXPECT_EQ(attr.vendor_id, kStubVendorId)
      << "the context argument did not survive the forwarder";
}

TEST_F(IbInjectionDelegationTest, VersionedMlx5VerbReachesTheProvider) {
  auto* initObj = reinterpret_cast<int (*)(mlx5dv_obj*, uint64_t)>(
      dlvsym(handle_, "mlx5dv_init_obj", "MLX5_1.2"));
  ASSERT_NE(initObj, nullptr) << "dlvsym: " << dlerror();

  mlx5dv_obj obj{};
  // Echoed back through the return, so this pins the scalar argument rather
  // than just the fact of the call.
  constexpr uint64_t kObjType = 9;
  EXPECT_EQ(initObj(&obj, kObjType), kStubInitObjRetBase + kObjType)
      << "mlx5dv_init_obj did not reach the stub, or lost its obj_type";
}

// The lookup style that a version-node-only export would not prove: ibverbx
// resolves this one with plain dlsym, so the shim has to be findable that way
// and has to forward to a provider found the same way.
TEST_F(IbInjectionDelegationTest, UnversionedMlx5VerbReachesTheProvider) {
  auto* queryMlx5 = reinterpret_cast<int (*)(ibv_context*, mlx5dv_context*)>(
      dlsym(handle_, "mlx5dv_query_device"));
  ASSERT_NE(queryMlx5, nullptr) << "dlsym: " << dlerror();

  ibv_context stubCtx{};
  mlx5dv_context attrs{};
  EXPECT_EQ(queryMlx5(&stubCtx, &attrs), kStubQueryMlx5DeviceRet)
      << "mlx5dv_query_device did not reach the stub libibverbs";
  EXPECT_EQ(attrs.comp_mask, kStubCompMask)
      << "the out-param the stub wrote did not come back to the caller";
}

} // namespace
