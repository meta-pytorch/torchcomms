// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// End-to-end proof that the injection shim reaches ctran's real code paths, and
// the failure-injection cases that proof makes possible.
//
// Everything before this exercised the shim in isolation: the engine driven by
// a fake provider, and the .so's symbol table read with dlvsym/dlsym. Neither
// shows that ctran's OWN poll_cq and post_send reach a shim -- ctran resolves
// those through cq_->context->ops.poll_cq, a pointer the shim overwrites after
// ibv_open_device returns. If that patch does not stick, every rule is inert
// and the whole approach is worthless. SeamIsLive checks exactly that, and
// gates the rest.
//
// Failure injection comes before skew because it is simpler to make
// deterministic: no ordering, no timing, just "this verb returns an error, and
// ctran does the right thing with it".

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/ibverbx/IbvDevice.h"
#include "comms/ctran/ibverbx/IbvQpUtils.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionControl.h"
#include "comms/ctran/utils/CtranLogger.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace injection = ibverbx::injection::testing;

namespace {

constexpr int kNumDevices = 2;
constexpr size_t kBufBytes = 4096;

// Enough progress() calls for a device to drain; each call polls one CQE per
// device. Generous because the loop exits as soon as its condition holds.
constexpr int kMaxPumps = 10000;

class CtranIbInjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // For NCCL_CTRAN_IB_DEVICES_PER_RANK below: cvars stay at their defaults
    // until ncclCvarInit() copies the environment in. Not for
    // IBVERBX_IBVERBS_SO, which ibvInit() reads with plain getenv.
    ncclCvarInit();

    if (!injection::available()) {
      GTEST_SKIP() << "IBVERBX_IBVERBS_SO does not point at a loadable shim; "
                      "this target sets it via $(exe_target ...)";
    }
    // Two NICs is the minimum that can express cross-device skew, and pinning
    // it keeps device ids stable regardless of host NIC count.
    NCCL_CTRAN_IB_DEVICES_PER_RANK = kNumDevices;

    // Host memory, not device memory: the flush is an RDMA READ whose target
    // only has to be a registered local buffer, and dma-buf registration of
    // device memory needs the NIC to be PCIe-close to the GPU -- an affinity
    // this test has no reason to depend on, since the seam it checks is
    // identical either way.
    if (cudaSetDevice(0) != cudaSuccess) {
      GTEST_SKIP() << "no usable CUDA device on this host";
    }
    if (cudaHostAlloc(&dbuf_, kBufBytes, cudaHostAllocDefault) != cudaSuccess) {
      GTEST_SKIP() << "cudaHostAlloc unavailable; the sandbox restricts "
                      "pinned-host allocation";
    }
    memset(dbuf_, 0, kBufBytes);

    const auto regResult =
        CtranIb::regMem(dbuf_, kBufBytes, /*cudaDev=*/0, &regElem_);
    if (regResult != commSuccess) {
      GTEST_SKIP() << "IB registration failed (" << regResult
                   << "); this host cannot register a local flush buffer";
    }

    ctranIb_ = makeCtranIb();

    // Reset AFTER construction: init opens devices, creates CQs and QPs, and
    // the loopback flush QPs handshake. Counting that setup traffic would make
    // every assertion below depend on it. Registrations survive a reset.
    injection::reset();
  }

  void TearDown() override {
    // Disarm before anything else. Rules live in the shim for the life of the
    // process, so one left armed here is still live while the NEXT case
    // constructs its CtranIb -- and SetUp's reset lands after that
    // construction, too late to help. An unbounded rule on a qp_num the
    // provider recycles can abort the next case's flush setup.
    if (injection::available()) {
      injection::reset();
    }
    ctranIb_.reset();
    if (regElem_ != nullptr) {
      CtranIb::deregMem(regElem_);
    }
    if (dbuf_ != nullptr) {
      // Checked, not discarded: hipHostFree is [[nodiscard]] under the AMD
      // hipify build, and a teardown that cannot free its pinned buffer is
      // worth reporting rather than swallowing.
      EXPECT_EQ(cudaFreeHost(dbuf_), cudaSuccess);
    }
  }

  std::unique_ptr<CtranIb> makeCtranIb() {
    // enableLocalFlush is set explicitly rather than left to
    // shouldEnableLocalFlushByDefault, which returns false for both H100 and
    // GB200 -- the flush path is what this test drives, so it must be on.
    // maxNumNic pins the NIC count so device ids stay stable regardless of how
    // many HCAs the host has.
    CtranIbConfig ibConfig;
    ibConfig.enableLocalFlush = true;
    ibConfig.maxNumNic = kNumDevices;

    return std::make_unique<CtranIb>(
        /*rank=*/0,
        /*cudaDev=*/0,
        /*commHash=*/0x1234,
        /*commDesc=*/"ib_injection_test",
        ibConfig,
        CtranIb::BootstrapMode::kDefaultServer,
        /*qpServerAddr=*/std::nullopt,
        ::comms::fault_tolerance::createAbort(/*enabled=*/true),
        /*socketFactory=*/nullptr);
  }

  commResult_t issueFlush(CtranIbRequest* req) {
    CtranIbEpochRAII epoch(ctranIb_.get());
    return ctranIb_->iflush(dbuf_, regElem_, req);
  }

  // Pump until `done` holds, or give up. Returns false on give-up so the caller
  // reports a meaningful assertion rather than hanging.
  template <typename Fn>
  bool pumpUntil(Fn done) {
    for (int i = 0; i < kMaxPumps && !done(); i++) {
      CtranIbEpochRAII epoch(ctranIb_.get());
      if (ctranIb_->progress() != commSuccess) {
        return false;
      }
    }
    return done();
  }

  // Pump until progress() reports an error. Separate from pumpUntil because
  // here the error IS the expected outcome, which that one treats as give-up.
  bool pumpUntilProgressError() {
    for (int i = 0; i < kMaxPumps; i++) {
      CtranIbEpochRAII epoch(ctranIb_.get());
      if (ctranIb_->progress() != commSuccess) {
        return true;
      }
    }
    return false;
  }

  std::unique_ptr<CtranIb> ctranIb_;
  void* dbuf_{nullptr};
  void* regElem_{nullptr};
};

// THE GATE. If this fails, nothing downstream is worth debugging: it means
// ctran is not reaching the shim at all. Likely causes are the
// ibverbx-rdma-core build variant (statically linked, no dlopen to intercept),
// the ops being replaced after ibv_open_device returns, or an Ibvcore.h ABI
// mismatch.
TEST_F(CtranIbInjectionTest, SeamIsLive) {
  const auto before = injection::getState();
  // ASSERT, not EXPECT: if no context was patched, every check below fails for
  // the same reason, and reading three consequential failures is worse than
  // reading the one cause.
  ASSERT_GE(before.patchedContexts, static_cast<uint32_t>(kNumDevices))
      << "the shim patched no ibv_context, so nothing else here can pass";

  CtranIbRequest req;
  ASSERT_EQ(issueFlush(&req), commSuccess);

  // post_send went through the shim: ctran called
  // qp_->context->ops.post_send, not the exported symbol.
  const uint64_t totalPosts =
      injection::getState().total(&IbInjectionDeviceState::postSendCalls);
  EXPECT_EQ(totalPosts, static_cast<uint64_t>(kNumDevices))
      << "iflush posts one RDMA READ per NIC; the shim saw " << totalPosts;

  ASSERT_TRUE(pumpUntil([&] { return req.isComplete(); }))
      << "flush never completed through the shim";

  // poll_cq went through the shim too, and the completions it released are the
  // ones that retired the request.
  const auto after = injection::getState();
  EXPECT_GT(after.total(&IbInjectionDeviceState::pollCqCalls), 0u);
  EXPECT_EQ(
      after.total(&IbInjectionDeviceState::cqesReleased),
      static_cast<uint64_t>(kNumDevices));
}

// --- failure injection ---

// A failing post must surface as an error rather than a crash or a silently
// dropped WR. CtranIbLocalVc::iflush aborts on a PARTIAL post (device > 0
// already posted), so failing the FIRST post is the recoverable path it is
// written to handle.
TEST_F(CtranIbInjectionTest, FlushReportsAPostSendFailure) {
  injection::addPostError(
      IB_INJECTION_VERB_POST_SEND,
      IB_INJECTION_ANY_DEVICE,
      IB_INJECTION_ANY_QP,
      ENOMEM,
      /*firstMatch=*/1,
      /*count=*/1);

  CtranIbRequest req;
  EXPECT_EQ(issueFlush(&req), commSystemError);

  const uint64_t injected =
      injection::getState().total(&IbInjectionDeviceState::apiErrorsInjected);
  EXPECT_EQ(injected, 1u) << "the rule never fired, so this proves nothing";
}

// A completion whose status is an error must reach ctran's error path. Distinct
// from the post-failure case above: that WR never reached the NIC, this one did
// and came back bad.
//
// CQE_ERROR_CHECK (CtranIbImpl.h:22) turns a non-SUCCESS status into
// commRemoteError out of progressInternal, so the observable behavior is that
// progress() itself fails -- not that a request completes with an error.
TEST_F(CtranIbInjectionTest, ErrorCompletionStatusFailsProgress) {
  injection::addWcStatus(
      /*deviceId=*/IB_INJECTION_ANY_DEVICE,
      ibverbx::IBV_WC_RDMA_READ,
      ibverbx::IBV_WC_RETRY_EXC_ERR,
      /*count=*/1);

  CtranIbRequest req;
  ASSERT_EQ(issueFlush(&req), commSuccess);

  EXPECT_TRUE(pumpUntilProgressError())
      << "a rewritten IBV_WC_RETRY_EXC_ERR did not fail progress()";

  const uint64_t mutated =
      injection::getState().total(&IbInjectionDeviceState::cqesStatusMutated);
  EXPECT_EQ(mutated, 1u) << "the rule never fired, so this proves nothing";
}

// Every rule above uses ANY_DEVICE, which means nothing here has yet shown that
// a device-scoped rule reaches the device it names -- a selector that ignored
// deviceId would pass all of them.
//
// This is the E2E half of WcStatusOnOneDeviceSparesTheOther: that one proves
// the engine's matcher honors deviceId against a fake provider, this one proves
// the ids the shim assigns line up with the NICs ctran actually posts on.
//
// A completion rule rather than a post error, because iflush has no recoverable
// exit for a device past the first: FB_CHECKABORT(device == 0)
// (CtranIbLocalVc.cc:151) kills the process on a partial post, so a device-1
// post error could only ever be a death test -- and forking a process holding a
// CUDA context and open IB devices is its own source of failures. A bad
// completion needs no such exemption: iflush posts one RDMA READ per NIC and
// both complete, so a device-1 rule has a CQE of its own to land on. The
// device-scoped *post* error becomes reachable in the distributed suite, where
// iput's post failure is recoverable.
TEST_F(CtranIbInjectionTest, WcStatusScopedToOneDeviceHitsOnlyThatDevice) {
  constexpr int32_t kTargetDevice = 1;
  injection::addWcStatus(
      kTargetDevice,
      ibverbx::IBV_WC_RDMA_READ,
      ibverbx::IBV_WC_RETRY_EXC_ERR,
      /*count=*/1);

  CtranIbRequest req;
  ASSERT_EQ(issueFlush(&req), commSuccess);

  EXPECT_TRUE(pumpUntilProgressError())
      << "a rewritten status on device " << kTargetDevice
      << " did not fail progress()";

  const auto s = injection::getState();
  EXPECT_EQ(s.device(kTargetDevice).cqesStatusMutated, 1u)
      << "the rule did not fire on device " << kTargetDevice;
  EXPECT_EQ(s.device(0).cqesStatusMutated, 0u)
      << "a device-" << kTargetDevice << " rule also mutated device 0";
  ASSERT_EQ(s.rules.size(), 1u);
  EXPECT_EQ(s.rules[0].firings, 1u);
}

// The device-scoped test above leaves the QP axis unproven, and hwQpNum is the
// selector field most likely to be passed wrongly: IbInjectionApi.h warns that
// a qpIdx here matches nothing and fails silently, because real qp_nums are
// large. Nothing catches that unless a rule is armed with a genuine
// provider-assigned number and shown to discriminate.
//
// Two QPs on one CQ, created through ibverbx so the shim registers both, and
// the rule names exactly one. Driven at the ibverbx layer rather than through
// iflush because ctran's flush QPs are not individually addressable from a test
// -- and post_send here still goes through qp_->context->ops.post_send
// (IbvQp.h:97), which is the patched vtable slot, so the seam under test is the
// same one.
TEST_F(CtranIbInjectionTest, PostErrorScopedToOneQpSparesItsSibling) {
  auto devices = ibverbx::IbvDevice::ibvGetDeviceList(
      /*hcaList=*/{}, /*hcaPrefix=*/"", /*port=*/-1, /*dataDirect=*/false);
  ASSERT_TRUE(devices.hasValue()) << "could not enumerate devices";
  ASSERT_FALSE(devices->empty());

  auto maybePd = (*devices)[0].allocPd();
  ASSERT_TRUE(maybePd.hasValue());
  auto maybeCq = (*devices)[0].createCq(64, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeCq.hasValue());

  auto maybeTarget = ibverbx::createRcQp(&*maybePd, maybeCq->cq(), 8, 8);
  ASSERT_TRUE(maybeTarget.hasValue());
  auto maybeOther = ibverbx::createRcQp(&*maybePd, maybeCq->cq(), 8, 8);
  ASSERT_TRUE(maybeOther.hasValue());

  const uint32_t targetQpNum = maybeTarget->getQpNum();
  const uint32_t otherQpNum = maybeOther->getQpNum();
  ASSERT_NE(targetQpNum, otherQpNum);
  // Both are provider-assigned `qp_num`s, which is the point of the hwQpNum
  // warning: a rule built from a qpIdx would be comparing against a different
  // namespace and could never match. No assertion on their magnitude -- the
  // provider is free to hand out a low qp_num, so a threshold here would only
  // add a way for this to flake.

  injection::addPostError(
      IB_INJECTION_VERB_POST_SEND,
      IB_INJECTION_ANY_DEVICE,
      targetQpNum,
      ENOMEM,
      /*firstMatch=*/1,
      /*count=*/0); // unbounded, so the sibling's miss is a real miss

  // A zero-length send: it never has to reach the wire for the shim to decide,
  // and the QP is unconnected so anything that did reach the NIC would fail
  // regardless. What matters is which QP the rule claims.
  ibverbx::ibv_send_wr wr{};
  ibverbx::ibv_send_wr* bad = nullptr;
  wr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
  wr.num_sge = 0;
  wr.sg_list = nullptr;

  const auto targetResult = maybeTarget->postSend(&wr, &bad);
  // ASSERT, not EXPECT: error() on a value-holding Expected is undefined, so if
  // the rule did not fire this has to stop here rather than read a union member
  // that was never set.
  ASSERT_FALSE(targetResult.hasValue())
      << "the rule did not fail post_send on qp_num " << targetQpNum;
  EXPECT_EQ(targetResult.error().errNum, ENOMEM);

  const auto s = injection::getState();
  ASSERT_EQ(s.rules.size(), 1u);
  EXPECT_EQ(s.rules[0].firings, 1u)
      << "expected exactly one firing, on qp_num " << targetQpNum;

  // The sibling must not match. Its post may still fail for provider reasons on
  // an unconnected QP, so assert on the rule's firing count rather than on the
  // return: a second firing is the only unambiguous sign the selector leaked.
  (void)maybeOther->postSend(&wr, &bad);
  const auto after = injection::getState();
  ASSERT_EQ(after.rules.size(), 1u);
  EXPECT_EQ(after.rules[0].firings, 1u)
      << "a qp_num-" << targetQpNum << " rule also fired on qp_num "
      << otherQpNum;
}

// Setup-verb injection needs no handshake, because it fires before any QP
// exists. This is the class that reaches ctran's construction and cleanup
// paths.
//
// Driven through ibverbx directly rather than a second CtranIb:
// CtranIbSingleton is a folly::Singleton, so devices, CQs and QPs are created
// ONCE per process and a second CtranIb reuses them without calling create_qp
// at all. Testing the construction path properly needs a fresh process, which
// is what the distributed suite gives us later; here the assertion is that the
// rule reaches the verb.
TEST_F(CtranIbInjectionTest, CreateQpFailureReachesTheVerb) {
  // Fail the very next QP creation. firstMatch=3 would be the more interesting
  // mid-VC-setup case, but nothing in this single-rank process creates three
  // more QPs after init.
  injection::addSetupError(
      IB_INJECTION_VERB_CREATE_QP, ENOMEM, /*firstMatch=*/1, /*count=*/1);

  auto pd = ibverbx::IbvDevice::ibvGetDeviceList(
      /*hcaList=*/{}, /*hcaPrefix=*/"", /*port=*/-1, /*dataDirect=*/false);
  ASSERT_TRUE(pd.hasValue()) << "could not enumerate devices";
  ASSERT_FALSE(pd->empty());

  auto maybePd = (*pd)[0].allocPd();
  ASSERT_TRUE(maybePd.hasValue());

  auto maybeCq = (*pd)[0].createCq(64, nullptr, nullptr, 0);
  ASSERT_TRUE(maybeCq.hasValue());

  // The rule turns this into a failure, which is the whole point.
  auto maybeQp = ibverbx::createRcQp(&*maybePd, maybeCq->cq(), 8, 8);
  EXPECT_FALSE(maybeQp.hasValue())
      << "create_qp succeeded despite an ENOMEM rule";

  const auto s = injection::getState();
  ASSERT_EQ(s.rules.size(), 1u);
  EXPECT_EQ(s.rules[0].firings, 1u) << "the create_qp rule never fired";
}

// Same reasoning: regMem goes through ibv_reg_mr, and the rule must make it
// fail. Armed here rather than in SetUp so the fixture's own registration
// succeeds.
TEST_F(CtranIbInjectionTest, RegMrFailureIsReportedAsAnError) {
  injection::addSetupError(
      IB_INJECTION_VERB_REG_MR, ENOMEM, /*firstMatch=*/1, /*count=*/1);

  void* otherElem = nullptr;
  EXPECT_NE(
      CtranIb::regMem(dbuf_, kBufBytes, /*cudaDev=*/0, &otherElem),
      commSuccess);
  if (otherElem != nullptr) {
    CtranIb::deregMem(otherElem);
  }

  const auto s = injection::getState();
  ASSERT_EQ(s.rules.size(), 1u);
  EXPECT_EQ(s.rules[0].firings, 1u) << "the reg_mr rule never fired";
}

} // namespace
