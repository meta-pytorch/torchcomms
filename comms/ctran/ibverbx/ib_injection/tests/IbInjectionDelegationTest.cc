// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Delegation test for the ib_injection shim: proves the forwarders actually
// call through to the real provider, with the caller's arguments, and that the
// hand-written lifecycle verbs only update the registries once the provider
// agrees.
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
// One pass-through verb per lookup style the shim implements, because each is
// resolved by a different code path and a break in one is invisible to the
// others:
//
//   ibv_query_device     realSym      -> dlvsym on the ibv handle
//   mlx5dv_init_obj      realMlx5Sym  -> dlvsym on the mlx5 handle
//   mlx5dv_query_device  realMlx5Sym  -> plain dlsym on the mlx5 handle
//
// Plus the object lifecycle, which is here because a stub provider is the only
// way to make a destroy verb FAIL: that is what pins the shim's
// deregister-only- on-success ordering, and the engine tests cannot reach it at
// all.
//
// The provider paths come from the BUCK target's env rather than setenv(), so
// they are set before the process starts: the shim latches its real-provider
// handles in function-local statics on first use, and a setenv() racing that
// first use would be resolved-once and silently wrong.

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <string>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionApi.h"
#include "comms/ctran/ibverbx/ib_injection/tests/IbInjectionStubLibibverbs.h"

using ibverbx::ibv_comp_channel;
using ibverbx::ibv_context;
using ibverbx::ibv_cq;
using ibverbx::ibv_device;
using ibverbx::ibv_device_attr;
using ibverbx::ibv_qp;
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

// The exported destroy verbs deregister only once the provider has returned 0.
//
// Only reachable here. The engine tests drive Engine directly and never enter
// ibv_destroy_cq or ibv_close_device at all, so nothing there pins the ordering
// and a refactor could put the bookkeeping back in front of the real verb
// silently. Forcing that needs a provider whose destroy fails on demand, which
// is why the stub carries the two knobs.
//
// Both directions are asserted: a test that only checked the failed destroy
// would pass just as well against a shim that never deregistered anything.
TEST_F(IbInjectionDelegationTest, DestroyVerbsDeregisterOnlyOnProviderSuccess) {
  auto* openDevice = reinterpret_cast<ibv_context* (*)(ibv_device*)>(
      dlvsym(handle_, "ibv_open_device", "IBVERBS_1.1"));
  auto* closeDevice = reinterpret_cast<int (*)(ibv_context*)>(
      dlvsym(handle_, "ibv_close_device", "IBVERBS_1.1"));
  auto* createCq = reinterpret_cast<
      ibv_cq* (*)(ibv_context*, int, void*, ibv_comp_channel*, int)>(
      dlvsym(handle_, "ibv_create_cq", "IBVERBS_1.1"));
  auto* destroyCq = reinterpret_cast<int (*)(ibv_cq*)>(
      dlvsym(handle_, "ibv_destroy_cq", "IBVERBS_1.1"));
  auto* getState = reinterpret_cast<IbInjectionStatus (*)(IbInjectionState*)>(
      dlsym(handle_, "ibInjectionGetState"));
  ASSERT_NE(openDevice, nullptr) << "dlvsym: " << dlerror();
  ASSERT_NE(closeDevice, nullptr) << "dlvsym: " << dlerror();
  ASSERT_NE(createCq, nullptr) << "dlvsym: " << dlerror();
  ASSERT_NE(destroyCq, nullptr) << "dlvsym: " << dlerror();
  ASSERT_NE(getState, nullptr) << "dlsym: " << dlerror();

  // The knobs live in the stub, not the shim. dlopen on the path the shim
  // forwards into returns that same instance, so this arms the state the
  // forwarders will actually read rather than a second copy's.
  void* stub =
      dlopen(getenv("IB_INJECTION_REAL_IBVERBS_SO"), RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(stub, nullptr) << "dlopen(stub): " << dlerror();
  auto* setDestroyCqRet =
      reinterpret_cast<void (*)(int)>(dlsym(stub, "stubSetDestroyCqRet"));
  auto* setCloseDeviceRet =
      reinterpret_cast<void (*)(int)>(dlsym(stub, "stubSetCloseDeviceRet"));
  ASSERT_NE(setDestroyCqRet, nullptr) << "dlsym: " << dlerror();
  ASSERT_NE(setCloseDeviceRet, nullptr) << "dlsym: " << dlerror();

  IbInjectionDeviceState devices[8]{};
  IbInjectionRuleState rules[8]{};
  auto snapshot = [&]() {
    IbInjectionState state{};
    state.numDevices = 8;
    state.numRules = 8;
    state.devices = devices;
    state.rules = rules;
    EXPECT_EQ(getState(&state), IB_INJECTION_OK);
    return state;
  };

  // No other case in this binary opens a device or creates a CQ, so these are
  // this test's own counts.
  ASSERT_EQ(snapshot().patchedContexts, 0u);

  ibv_context* ctx = openDevice(nullptr);
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(snapshot().patchedContexts, 1u);

  ibv_cq* cq = createCq(ctx, /*cqe=*/16, nullptr, nullptr, /*comp_vector=*/0);
  ASSERT_NE(cq, nullptr);
  ASSERT_EQ(snapshot().numDevices, 1u);

  // A refused destroy has to leave the CQ registered. Dropping it would free
  // this CQ's device id while the CQ is still live, so the next CQ would take
  // the same id and every rule written against it would land on the wrong
  // device.
  setDestroyCqRet(EBUSY);
  EXPECT_EQ(destroyCq(cq), EBUSY);
  EXPECT_EQ(snapshot().numDevices, 1u)
      << "a failed ibv_destroy_cq deregistered the CQ anyway";

  setDestroyCqRet(0);
  EXPECT_EQ(destroyCq(cq), 0);
  EXPECT_EQ(snapshot().numDevices, 0u)
      << "a successful ibv_destroy_cq left the CQ registered";

  // Same contract one level up. patchedContexts is what a run reports as proof
  // the seam was live, and a failed close also has to keep the context's
  // SavedOps: without them every later shimPollCq answers -EINVAL and every
  // post ENODEV, so the object would be bricked by its own failed teardown.
  setCloseDeviceRet(EBUSY);
  EXPECT_EQ(closeDevice(ctx), EBUSY);
  EXPECT_EQ(snapshot().patchedContexts, 1u)
      << "a failed ibv_close_device deregistered the context anyway";

  setCloseDeviceRet(0);
  EXPECT_EQ(closeDevice(ctx), 0);
  EXPECT_EQ(snapshot().patchedContexts, 0u)
      << "a successful ibv_close_device left the context registered";
}

// An optional verb the stub does not export, which is what a host without
// libmlx5 looks like. The errno matters: ibverbx maps EOPNOTSUPP to "not
// supported" and carries on, so reporting anything else here turns an absent
// feature into a hard error. `mlx5dv_create_qp` is the one at risk, because it
// became a hand-written wrapper to reach the QP registry and so no longer
// inherits the generated forwarder's missing-symbol handling.
TEST_F(IbInjectionDelegationTest, MissingOptionalVerbReportsTheSharedErrno) {
  auto* createQp = reinterpret_cast<ibv_qp* (*)(ibv_context*, void*, void*)>(
      dlsym(handle_, "mlx5dv_create_qp"));
  ASSERT_NE(createQp, nullptr) << "dlsym: " << dlerror();

  errno = 0;
  EXPECT_EQ(createQp(nullptr, nullptr, nullptr), nullptr)
      << "the stub exports no mlx5dv_create_qp, so this must fail";
  EXPECT_EQ(errno, EOPNOTSUPP)
      << "missing optional verbs report EOPNOTSUPP through missingVerbResult; "
         "got errno "
      << errno;
}

} // namespace
