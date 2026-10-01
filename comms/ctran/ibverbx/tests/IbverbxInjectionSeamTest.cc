// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// That the injection shim is actually reached through ibverbx's own vtable
// reads.
//
// This lives here, not in a consumer's suite, because the code under test is
// ibverbx's: `IbvQp::postSend` reads `qp_->context->ops.post_send` (IbvQp.h)
// and `IbvCq::pollCq` reads `cq_->context->ops.poll_cq` (IbvCq.h). Every
// consumer -- ctran, ctranx, anything else linking ibverbx -- goes through
// those two lines, so proving the patch holds once here covers all of them. A
// copy of this check in each consumer's tests would be the same assertion
// re-paid per caller, and would still be testing ibverbx.
//
// What is genuinely per-consumer is only which ibverbx variant a binary links:
// the rdma-core variant statically links its provider, so there is no dlopen to
// intercept and no seam to check. That is a one-line precondition in the
// consumer's fixture, not a suite.
//
// Neither call has to SUCCEED. A QP still in RESET rejects a post and an empty
// CQ yields no completions, and both are fine: the claim is that the shim saw
// the call, which its counters record before it delegates. Asserting on
// transfer would need a connected peer and would test the NIC instead of the
// seam.

#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionControl.h"
#include "comms/ctran/ibverbx/tests/IbverbxTestFixture.h"

namespace ibverbx {
namespace {

class IbverbxInjectionSeamTest : public IbverbxTestFixture {
 protected:
  void SetUp() override {
    // Fail rather than skip: this target always sets IBVERBX_IBVERBS_SO, so an
    // unavailable shim means IB_INJECTION_SO mis-resolved.
    ASSERT_TRUE(injection::testing::available())
        << "IBVERBX_IBVERBS_SO does not point at the injection shim, but this "
           "target sets env = {\"IBVERBX_IBVERBS_SO\": IB_INJECTION_SO}";
    IbverbxTestFixture::SetUp();
    // Drops rules and zeroes counters while keeping the object registries, so
    // the assertions below measure only this case's traffic rather than
    // whatever ibvInit() did.
    injection::testing::reset();
  }
};

TEST_F(IbverbxInjectionSeamTest, IbverbxVerbReadsReachTheShim) {
  auto devices = IbvDevice::ibvGetDeviceList();
  ASSERT_TRUE(devices);
  ASSERT_FALSE(devices->empty());
  auto& device = devices->at(0);

  auto cq = device.createCq(/*cqe=*/16, nullptr, nullptr, 0);
  ASSERT_TRUE(cq);
  auto pd = device.allocPd();
  ASSERT_TRUE(pd);
  auto initAttr = makeIbvQpInitAttr(cq->cq());
  auto qp = pd->createQp(&initAttr);
  ASSERT_TRUE(qp);

  // ASSERT, not EXPECT: with no context patched every check below fails for the
  // same reason, and three consequential failures read worse than one cause.
  ASSERT_GT(injection::testing::getState().patchedContexts, 0u)
      << "the shim patched no ibv_context, so no verb below can reach it";

  // Return values deliberately ignored: a RESET-state QP refuses the post and
  // the CQ is empty. The shim counts the call either way, which is the claim.
  ibv_send_wr wr{};
  ibv_send_wr* badWr = nullptr;
  (void)qp->postSend(&wr, &badWr);
  (void)cq->pollCq(/*numEntries=*/1);

  const auto state = injection::testing::getState();
  EXPECT_GT(state.total(&IbInjectionDeviceState::postSendCalls), 0u)
      << "IbvQp::postSend read ops.post_send but the shim never saw it";
  EXPECT_GT(state.total(&IbInjectionDeviceState::pollCqCalls), 0u)
      << "IbvCq::pollCq read ops.poll_cq but the shim never saw it";
}

} // namespace
} // namespace ibverbx
