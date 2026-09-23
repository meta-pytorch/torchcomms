// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Unit tests for the injection rule engine, driven directly rather than through
// dlopen. No NIC, no ibverbx, no real libibverbs.
//
// These pin the semantics every later test depends on: what a selector matches,
// when a repeat schedule fires, what releases a held CQE, and that per-CQ order
// survives a hold. A bug here would surface as "the rule didn't fire" in some
// distant distributed test.

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "comms/ctran/ibverbx/ib_injection/InjectionEngine.h"

namespace ibverbx::injection {
namespace {

// Stands in for the real libibverbs the shim delegates to. poll_cq hands out
// queued completions one at a time, which is what a real provider does and what
// the engine's drain loop expects.
//
// Per-CQ state hangs off cq_context so several providers can coexist -- ctran
// opens one context per NIC, so a cross-device test needs exactly that.
class FakeProvider {
 public:
  FakeProvider() {
    context_.ops.poll_cq = &pollCq;
    context_.ops.post_send = &postSend;
    context_.ops.post_recv = &postRecv;
    device_.name[0] = 'f';
    device_.name[1] = '\0';
    context_.device = &device_;
  }

  ibv_context* context() {
    return &context_;
  }

  ibv_cq* makeCq() {
    auto queue = std::make_unique<std::vector<ibv_wc>>();
    auto cq = std::make_unique<ibv_cq>();
    cq->context = &context_;
    cq->cq_context = queue.get();
    queues_.push_back(std::move(queue));
    cqs_.push_back(std::move(cq));
    return cqs_.back().get();
  }

  // qpNum is assigned by the caller here; a real provider assigns it, and the
  // engine reads it back off the returned struct exactly the same way.
  ibv_qp* makeQp(ibv_cq* sendCq, ibv_cq* recvCq, uint32_t qpNum) {
    auto qp = std::make_unique<ibv_qp>();
    qp->context = &context_;
    qp->send_cq = sendCq;
    qp->recv_cq = recvCq;
    qp->qp_num = qpNum;
    qps_.push_back(std::move(qp));
    return qps_.back().get();
  }

  static void push(ibv_cq* cq, ibv_wc_opcode opcode, uint32_t qpNum) {
    ibv_wc wc{};
    wc.opcode = opcode;
    wc.qp_num = qpNum;
    wc.status = IBV_WC_SUCCESS;
    static_cast<std::vector<ibv_wc>*>(cq->cq_context)->push_back(wc);
  }

  // Make this CQ's poll_cq report a provider error, as a real one does with a
  // negative return. Cleared by the next SetUp, since it is keyed per CQ.
  static void setPollError(ibv_cq* cq, int rc) {
    pollErrors()[cq] = rc;
  }

  static void clearPollErrors() {
    pollErrors().clear();
  }

  static int postSendCalls;
  static int postRecvCalls;

 private:
  static std::unordered_map<ibv_cq*, int>& pollErrors() {
    static std::unordered_map<ibv_cq*, int> m;
    return m;
  }

  static int pollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) {
    auto errIt = pollErrors().find(cq);
    if (errIt != pollErrors().end()) {
      return errIt->second;
    }
    auto& q = *static_cast<std::vector<ibv_wc>*>(cq->cq_context);
    if (q.empty() || numEntries < 1) {
      return 0;
    }
    wc[0] = q.front();
    q.erase(q.begin());
    return 1;
  }

  static int postSend(ibv_qp*, ibv_send_wr*, ibv_send_wr**) {
    postSendCalls++;
    return 0;
  }

  static int postRecv(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) {
    postRecvCalls++;
    return 0;
  }

  ibv_context context_{};
  ibv_device device_{};
  std::vector<std::unique_ptr<ibv_cq>> cqs_;
  std::vector<std::unique_ptr<ibv_qp>> qps_;
  std::vector<std::unique_ptr<std::vector<ibv_wc>>> queues_;
};

int FakeProvider::postSendCalls = 0;
int FakeProvider::postRecvCalls = 0;

// Builds a rule with the fields every action needs, so each test sets only what
// it is actually about.
IbInjectionRule baseRule(IbInjectionVerb verb, IbInjectionAction action) {
  IbInjectionRule r{};
  r.verb = verb;
  r.action = action;
  r.selector.deviceId = IB_INJECTION_ANY_DEVICE;
  r.selector.hwQpNum = IB_INJECTION_ANY_QP;
  r.selector.opcode = IB_INJECTION_ANY_OPCODE;
  r.selector.opcodeDomain = IB_INJECTION_OPCODE_WC;
  r.repeat.firstMatch = 1;
  r.repeat.everyNth = 1;
  r.repeat.unbounded = 1;
  return r;
}

class InjectionEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    engine_ = &Engine::get();
    // The Engine is a process-wide singleton, and reset() deliberately keeps
    // the object registries -- correct in production, wrong here: a previous
    // test's entries outlive the FakeProvider they point at and keep holding
    // device ids, so lowestUnusedDeviceId() would hand this test 2,3 instead of
    // 0,1. Without this the suite only passes in declaration order.
    engine_->forgetAllObjectsForTest();
    engine_->reset();
    FakeProvider::postSendCalls = 0;
    FakeProvider::postRecvCalls = 0;
    FakeProvider::clearPollErrors();
  }

  void setupTwoDevices() {
    engine_->registerContext(provider_.context());
    cq0_ = provider_.makeCq();
    cq1_ = provider_.makeCq();
    engine_->registerCq(cq0_);
    engine_->registerCq(cq1_);
  }

  int poll(ibv_cq* cq) {
    ibv_wc wc{};
    int32_t injectErrno = 0;
    return engine_->pollCq(cq, 1, &wc, &injectErrno);
  }

  int pollInto(ibv_cq* cq, ibv_wc* wc) {
    int32_t injectErrno = 0;
    return engine_->pollCq(cq, 1, wc, &injectErrno);
  }

  uint32_t addRule(const IbInjectionRule& r) {
    uint32_t id = 0;
    EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_OK)
        << engine_->lastError();
    return id;
  }

  IbInjectionDeviceState deviceState(int32_t deviceId) {
    IbInjectionDeviceState devices[8]{};
    IbInjectionRuleState rules[8]{};
    IbInjectionState state{};
    state.numDevices = 8;
    state.numRules = 8;
    state.devices = devices;
    state.rules = rules;
    EXPECT_EQ(engine_->getState(&state), IB_INJECTION_OK);
    for (uint32_t i = 0; i < state.numDevices; i++) {
      if (devices[i].deviceId == deviceId) {
        return devices[i];
      }
    }
    return IbInjectionDeviceState{};
  }

  IbInjectionRuleState ruleState(uint32_t ruleId) {
    IbInjectionDeviceState devices[8]{};
    IbInjectionRuleState rules[8]{};
    IbInjectionState state{};
    state.numDevices = 8;
    state.numRules = 8;
    state.devices = devices;
    state.rules = rules;
    EXPECT_EQ(engine_->getState(&state), IB_INJECTION_OK);
    for (uint32_t i = 0; i < state.numRules; i++) {
      if (rules[i].ruleId == ruleId) {
        return rules[i];
      }
    }
    return IbInjectionRuleState{};
  }

  Engine* engine_{nullptr};
  FakeProvider provider_;
  ibv_cq* cq0_{nullptr};
  ibv_cq* cq1_{nullptr};
};

// --- registration ---

TEST_F(InjectionEngineTest, CqsGetDistinctDeviceIdsInCreationOrder) {
  setupTwoDevices();
  EXPECT_EQ(engine_->deviceOfCq(cq0_), 0);
  EXPECT_EQ(engine_->deviceOfCq(cq1_), 1);
}

// ctran opens one ibv_context PER NIC, so the real cross-device layout is two
// contexts with one CQ each -- not one context with two CQs. Ids must still
// come out distinct, or every CQ collapses onto device 0 and a cross-device
// rule can never fire.
TEST_F(InjectionEngineTest, CqsOnSeparateContextsGetDistinctDeviceIds) {
  FakeProvider providerA;
  FakeProvider providerB;
  engine_->registerContext(providerA.context());
  engine_->registerContext(providerB.context());
  ibv_cq* cqA = providerA.makeCq();
  ibv_cq* cqB = providerB.makeCq();
  engine_->registerCq(cqA);
  engine_->registerCq(cqB);

  EXPECT_EQ(engine_->deviceOfCq(cqA), 0);
  EXPECT_EQ(engine_->deviceOfCq(cqB), 1);
}

// A process that builds a second CtranIb after destroying the first must see
// the same ids, or every rule written against id 0 silently matches nothing.
TEST_F(InjectionEngineTest, DeviceIdsAreReusedAfterCqDestruction) {
  setupTwoDevices();
  ASSERT_EQ(engine_->deviceOfCq(cq0_), 0);
  engine_->forgetCq(cq0_);
  engine_->forgetCq(cq1_);

  FakeProvider second;
  engine_->registerContext(second.context());
  ibv_cq* reopened = second.makeCq();
  engine_->registerCq(reopened);

  EXPECT_EQ(engine_->deviceOfCq(reopened), 0);
}

// A context is registered once, at ibv_open_device, BEFORE its ops are patched.
// Re-registering later must not overwrite the saved originals: by then ctx->ops
// holds the shims, so a shim delegating to them would recurse forever.
TEST_F(InjectionEngineTest, RegisterContextDoesNotOverwriteSavedOps) {
  engine_->registerContext(provider_.context());

  SavedOps original;
  ASSERT_TRUE(engine_->lookupSavedOps(provider_.context(), &original));
  ASSERT_NE(original.pollCq, nullptr);

  // Stand in for the patch ibv_open_device applies, then re-register.
  provider_.context()->ops.poll_cq = nullptr;
  engine_->registerContext(provider_.context());

  SavedOps after;
  ASSERT_TRUE(engine_->lookupSavedOps(provider_.context(), &after));
  EXPECT_EQ(after.pollCq, original.pollCq)
      << "re-registration captured the patched ops";
}

TEST_F(InjectionEngineTest, LookupSavedOpsFailsForAnUnknownContext) {
  ibv_context stray{};
  SavedOps out;
  EXPECT_FALSE(engine_->lookupSavedOps(&stray, &out));
}

// Closing a device must take its CQs and QPs with it. Leaving them behind would
// keep records whose `context` points at memory the provider has freed, and
// would hold device ids that lowestUnusedDeviceId() then refuses to reuse.
TEST_F(InjectionEngineTest, ForgetContextAlsoForgetsItsCqsAndQps) {
  setupTwoDevices();
  ibv_qp* qp = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0x1234);
  engine_->registerQp(qp);
  ASSERT_EQ(engine_->deviceOfCq(cq0_), 0);
  ASSERT_EQ(engine_->deviceOfQpSend(qp), 0);

  engine_->forgetContext(provider_.context());

  EXPECT_EQ(engine_->deviceOfCq(cq0_), -1);
  EXPECT_EQ(engine_->deviceOfCq(cq1_), -1);
  EXPECT_EQ(engine_->deviceOfQpSend(qp), -1);

  // Ids are free again, so a second CtranIb in this process numbers from 0.
  FakeProvider second;
  engine_->registerContext(second.context());
  ibv_cq* reopened = second.makeCq();
  engine_->registerCq(reopened);
  EXPECT_EQ(engine_->deviceOfCq(reopened), 0);
}

// Deregistration now runs AFTER the provider's destroy verb returns, so by the
// time forgetContext looks for a context's QPs the ibv_qp memory may already be
// gone. It must therefore identify them from the context recorded at
// registration, never by reading it back off the QP.
TEST_F(InjectionEngineTest, ForgetContextFindsQpsWithoutReadingThem) {
  setupTwoDevices();
  ibv_qp* qp = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0x1234);
  engine_->registerQp(qp);
  ASSERT_EQ(engine_->deviceOfQpSend(qp), 0);

  // What a freed QP looks like to a reader: the provider has torn the struct
  // down and its context back-pointer no longer names the owner.
  qp->context = nullptr;

  engine_->forgetContext(provider_.context());

  EXPECT_EQ(engine_->deviceOfQpSend(qp), -1)
      << "forgetContext missed a QP once its context back-pointer was cleared";
}

// An unregistered QP has no device, and -1 must not become a counter key: it
// would surface from getState() alongside real devices.
TEST_F(InjectionEngineTest, PostOnUnregisteredQpCreatesNoPhantomDevice) {
  setupTwoDevices();
  ibv_qp* stray = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0x999);
  // Deliberately NOT registered.
  ASSERT_EQ(engine_->deviceOfQpSend(stray), -1);

  engine_->decidePost(IB_INJECTION_VERB_POST_SEND, stray, IBV_WR_RDMA_WRITE);

  IbInjectionDeviceState devices[8]{};
  IbInjectionRuleState rules[8]{};
  IbInjectionState state{};
  state.numDevices = 8;
  state.numRules = 8;
  state.devices = devices;
  state.rules = rules;
  ASSERT_EQ(engine_->getState(&state), IB_INJECTION_OK);
  for (uint32_t i = 0; i < state.numDevices; i++) {
    EXPECT_GE(devices[i].deviceId, 0) << "phantom device entry in getState()";
  }
}

// Device ids are REUSED, not monotonic, so a stale counters_ entry does double
// damage: getState() reports a device that no longer exists, and the next CQ to
// take that id inherits the dead device's counts.
TEST_F(InjectionEngineTest, DestroyedDeviceLeavesNoCountersBehind) {
  setupTwoDevices();
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, 1);
  ASSERT_EQ(poll(cq0_), 1);
  ASSERT_EQ(deviceState(0).cqesObserved, 1u);

  engine_->forgetCq(cq0_);

  // Device 0 is gone from the snapshot entirely, not merely zeroed.
  IbInjectionDeviceState devices[8]{};
  IbInjectionRuleState rules[8]{};
  IbInjectionState state{};
  state.numDevices = 8;
  state.numRules = 8;
  state.devices = devices;
  state.rules = rules;
  ASSERT_EQ(engine_->getState(&state), IB_INJECTION_OK);
  for (uint32_t i = 0; i < state.numDevices; i++) {
    EXPECT_NE(devices[i].deviceId, 0) << "destroyed device still in getState()";
  }

  // A new CQ takes id 0 again and must start from zero.
  ibv_cq* reopened = provider_.makeCq();
  engine_->registerCq(reopened);
  ASSERT_EQ(engine_->deviceOfCq(reopened), 0);
  EXPECT_EQ(deviceState(0).cqesObserved, 0u)
      << "reused device id inherited the previous device's counters";
}

// Registering the same QP twice must not re-derive its device attribution: the
// second call happens after the QP is already live, and silently replacing
// sendDeviceId would repoint every rule written against it.
TEST_F(InjectionEngineTest, RegisterQpIsIdempotent) {
  setupTwoDevices();
  ibv_qp* qp = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0x77);
  engine_->registerQp(qp);
  ASSERT_EQ(engine_->deviceOfQpSend(qp), 0);

  // Re-register after mutating the struct, as an unexpected path might.
  qp->send_cq = cq1_;
  engine_->registerQp(qp);

  EXPECT_EQ(engine_->deviceOfQpSend(qp), 0)
      << "re-registration re-derived the device attribution";
}

TEST_F(InjectionEngineTest, QpInheritsItsSendCqDevice) {
  setupTwoDevices();
  ibv_qp* qp = provider_.makeQp(cq1_, cq1_, /*qpNum=*/0x1234);
  engine_->registerQp(qp);

  EXPECT_EQ(engine_->deviceOfQpSend(qp), 1);
  EXPECT_EQ(engine_->deviceOfQpRecv(qp), 1);
}

TEST_F(InjectionEngineTest, UnregisteredCqReportsNoDevice) {
  ibv_cq stray{};
  EXPECT_EQ(engine_->deviceOfCq(&stray), -1);
}

// --- baseline: no rules ---

TEST_F(InjectionEngineTest, PollWithoutRulesPassesEverythingThrough) {
  setupTwoDevices();
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, /*qpNum=*/1);

  EXPECT_EQ(poll(cq0_), 1);
  EXPECT_EQ(poll(cq0_), 0);
}

// --- API_ERROR ---

TEST_F(InjectionEngineTest, ApiErrorOnSetupVerbFiresOnce) {
  auto r = baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = ENOMEM;
  r.repeat.unbounded = 0;
  r.repeat.count = 1;
  addRule(r);

  auto first = engine_->decideSetupCall(IB_INJECTION_VERB_CREATE_QP);
  EXPECT_TRUE(first.injectError);
  EXPECT_EQ(first.errnoValue, ENOMEM);

  // count=1, so the second creation succeeds.
  EXPECT_FALSE(
      engine_->decideSetupCall(IB_INJECTION_VERB_CREATE_QP).injectError);
}

// firstMatch is the whole point of setup-verb injection: failing the 3rd QP
// leaves two already live, which is the partial-construction path.
TEST_F(InjectionEngineTest, ApiErrorRespectsFirstMatch) {
  auto r = baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = ENOMEM;
  r.repeat.firstMatch = 3;
  addRule(r);

  EXPECT_FALSE(
      engine_->decideSetupCall(IB_INJECTION_VERB_CREATE_QP).injectError);
  EXPECT_FALSE(
      engine_->decideSetupCall(IB_INJECTION_VERB_CREATE_QP).injectError);
  EXPECT_TRUE(
      engine_->decideSetupCall(IB_INJECTION_VERB_CREATE_QP).injectError);
}

TEST_F(InjectionEngineTest, ApiErrorOnOneVerbLeavesOthersAlone) {
  auto r = baseRule(IB_INJECTION_VERB_REG_MR, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = ENOMEM;
  addRule(r);

  EXPECT_TRUE(engine_->decideSetupCall(IB_INJECTION_VERB_REG_MR).injectError);
  EXPECT_FALSE(
      engine_->decideSetupCall(IB_INJECTION_VERB_ALLOC_PD).injectError);
}

TEST_F(InjectionEngineTest, ApiErrorOnPostSendTargetsOneQp) {
  setupTwoDevices();
  ibv_qp* target = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0xAAA);
  ibv_qp* other = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0xBBB);
  engine_->registerQp(target);
  engine_->registerQp(other);

  auto r = baseRule(IB_INJECTION_VERB_POST_SEND, IB_INJECTION_ACTION_API_ERROR);
  r.selector.hwQpNum = 0xAAA;
  r.selector.opcodeDomain = IB_INJECTION_OPCODE_WR;
  r.errnoValue = EINVAL;
  addRule(r);

  EXPECT_TRUE(
      engine_
          ->decidePost(IB_INJECTION_VERB_POST_SEND, target, IBV_WR_RDMA_WRITE)
          .injectError);
  EXPECT_FALSE(
      engine_->decidePost(IB_INJECTION_VERB_POST_SEND, other, IBV_WR_RDMA_WRITE)
          .injectError);
}

TEST_F(InjectionEngineTest, ApiErrorOnPollCqReportsErrnoAndSkipsTheQueue) {
  setupTwoDevices();
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, 1);

  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_API_ERROR);
  r.selector.deviceId = 0;
  r.errnoValue = EIO;
  addRule(r);

  ibv_wc wc{};
  int32_t injectErrno = 0;
  EXPECT_EQ(engine_->pollCq(cq0_, 1, &wc, &injectErrno), -2);
  EXPECT_EQ(injectErrno, EIO);
  // The completion is still queued in the provider: an API error is the call
  // failing, not the completion being consumed.
  EXPECT_EQ(deviceState(0).cqesObserved, 0u);
}

// decideSetupCall is only ever reached with create_qp, reg_mr and alloc_pd
// above, but IbInjectionVerb declares six setup verbs and IbInjectionDso.cc
// wires all six. A verb the enum declares and the engine quietly refuses to
// match is the worst failure this tool can have: addRule returns OK, a rule id
// comes back, the rule sits there, and nothing ever fires -- which reads
// exactly like "the code under test handled the error fine". So assert every
// declared setup verb fires, and that each is matched by its OWN rule rather
// than by any setup rule.
TEST_F(InjectionEngineTest, ApiErrorHonorsEveryDeclaredSetupVerb) {
  const IbInjectionVerb setupVerbs[] = {
      IB_INJECTION_VERB_OPEN_DEVICE,
      IB_INJECTION_VERB_ALLOC_PD,
      IB_INJECTION_VERB_REG_MR,
      IB_INJECTION_VERB_CREATE_CQ,
      IB_INJECTION_VERB_CREATE_QP,
      IB_INJECTION_VERB_MODIFY_QP,
  };

  for (IbInjectionVerb armed : setupVerbs) {
    engine_->reset();
    auto r = baseRule(armed, IB_INJECTION_ACTION_API_ERROR);
    r.errnoValue = ENOMEM;
    r.repeat.unbounded = 0;
    r.repeat.count = 1;
    const auto id = addRule(r);

    for (IbInjectionVerb called : setupVerbs) {
      const auto decision = engine_->decideSetupCall(called);
      if (called == armed) {
        EXPECT_TRUE(decision.injectError)
            << "a rule on setup verb " << static_cast<int>(armed)
            << " never fired on that verb";
        EXPECT_EQ(decision.errnoValue, ENOMEM);
      } else {
        EXPECT_FALSE(decision.injectError)
            << "a rule on setup verb " << static_cast<int>(armed)
            << " also fired on verb " << static_cast<int>(called);
      }
    }
    EXPECT_EQ(ruleState(id).firings, 1u);
  }
}

// post_recv is declared in IbInjectionVerb and shimmed in IbInjectionDso.cc,
// but no rule has ever targeted it. It needs its own case because it is
// selected differently: ibv_recv_wr carries no opcode, so the shim passes
// IB_INJECTION_ANY_OPCODE and a recv rule can only narrow by device and QP. A
// rule that leaked across the two post verbs would be easy to miss in
// production, where ctran posts receives on the same control QP it sends on.
TEST_F(InjectionEngineTest, ApiErrorOnPostRecvFiresAndSparesPostSend) {
  setupTwoDevices();
  ibv_qp* qp = provider_.makeQp(cq0_, cq0_, /*qpNum=*/0xCCC);
  engine_->registerQp(qp);

  auto r = baseRule(IB_INJECTION_VERB_POST_RECV, IB_INJECTION_ACTION_API_ERROR);
  r.selector.opcodeDomain = IB_INJECTION_OPCODE_WR;
  r.errnoValue = EAGAIN;
  const auto id = addRule(r);

  const auto recvDecision = engine_->decidePost(
      IB_INJECTION_VERB_POST_RECV, qp, IB_INJECTION_ANY_OPCODE);
  EXPECT_TRUE(recvDecision.injectError);
  EXPECT_EQ(recvDecision.errnoValue, EAGAIN);

  // Unbounded, so post_send's miss is a real miss rather than a rule that had
  // already been spent.
  EXPECT_FALSE(
      engine_->decidePost(IB_INJECTION_VERB_POST_SEND, qp, IBV_WR_RDMA_WRITE)
          .injectError);

  EXPECT_EQ(ruleState(id).firings, 1u);
  // Both calls are still counted against the QP's device, injected or not.
  EXPECT_EQ(deviceState(0).postRecvCalls, 1u);
  EXPECT_EQ(deviceState(0).postSendCalls, 1u);
}

// everyNth is the one repeat field nothing has ever set to anything but 1, so a
// rule engine that dropped it entirely would pass every test above. "Fail every
// other post" is how an intermittently bad link gets modelled, which makes this
// the field a skew or flaky-NIC scenario leans on hardest.
TEST_F(InjectionEngineTest, EveryNthFiresOnAlternatingMatches) {
  auto r = baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = ENOMEM;
  r.repeat.everyNth = 2;
  const auto id = addRule(r);

  // firstMatch defaults to 1, so by the formula in IbInjectionApi.h matches
  // 1, 3 and 5 fire while 2 and 4 pass through.
  const bool expected[] = {true, false, true, false, true};
  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(
        engine_->decideSetupCall(IB_INJECTION_VERB_CREATE_QP).injectError,
        expected[i])
        << "match " << (i + 1) << " of an everyNth=2 rule";
  }

  EXPECT_EQ(ruleState(id).matches, 5u);
  EXPECT_EQ(ruleState(id).firings, 3u);
}

// --- WC_STATUS ---

TEST_F(InjectionEngineTest, WcStatusRewritesStatusButStillDelivers) {
  setupTwoDevices();
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, 1);

  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.selector.deviceId = 0;
  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  addRule(r);

  ibv_wc wc{};
  ASSERT_EQ(pollInto(cq0_, &wc), 1);
  EXPECT_EQ(wc.status, IBV_WC_RETRY_EXC_ERR);
  EXPECT_EQ(deviceState(0).cqesStatusMutated, 1u);
}

// The positive case above proves a device-0 rule fires on device 0. This proves
// the other half: it must leave device 1 alone. Without this, a selector that
// ignored deviceId entirely would still pass every test above -- and a
// cross-device skew rule that silently hits both NICs is indistinguishable from
// one that works.
TEST_F(InjectionEngineTest, WcStatusOnOneDeviceSparesTheOther) {
  setupTwoDevices();
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, 1);
  FakeProvider::push(cq1_, IBV_WC_RDMA_READ, 1);

  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.selector.deviceId = 0;
  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  addRule(r);

  ibv_wc wc0{};
  ASSERT_EQ(pollInto(cq0_, &wc0), 1);
  EXPECT_EQ(wc0.status, IBV_WC_RETRY_EXC_ERR);

  ibv_wc wc1{};
  ASSERT_EQ(pollInto(cq1_, &wc1), 1);
  EXPECT_EQ(wc1.status, IBV_WC_SUCCESS) << "a device-0 rule mutated device 1";

  EXPECT_EQ(deviceState(0).cqesStatusMutated, 1u);
  EXPECT_EQ(deviceState(1).cqesStatusMutated, 0u);
}

TEST_F(InjectionEngineTest, SelectorIgnoresNonMatchingOpcode) {
  setupTwoDevices();
  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.selector.deviceId = 0;
  r.selector.opcode = IBV_WC_RDMA_READ;
  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  addRule(r);

  // A WRITE completion, so the READ-only rule must leave it alone.
  FakeProvider::push(cq0_, IBV_WC_RDMA_WRITE, 1);
  ibv_wc wc{};
  ASSERT_EQ(pollInto(cq0_, &wc), 1);
  EXPECT_EQ(wc.status, IBV_WC_SUCCESS);
  EXPECT_EQ(deviceState(0).cqesStatusMutated, 0u);
}

// --- validation: reject rules that could never fire ---

TEST_F(InjectionEngineTest, RejectsActionVerbMismatches) {
  uint32_t id = 0;

  auto wcOnPost =
      baseRule(IB_INJECTION_VERB_POST_SEND, IB_INJECTION_ACTION_WC_STATUS);
  EXPECT_EQ(engine_->addRule(&wcOnPost, &id), IB_INJECTION_ERR_ARG);

  auto wcOnSetup =
      baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_WC_STATUS);
  EXPECT_EQ(engine_->addRule(&wcOnSetup, &id), IB_INJECTION_ERR_ARG);
}

// An API_ERROR rule carrying errno 0 would fire and still look like success at
// every call site: shimPollCq returns -errno (0 reads as "no completions") and
// the post shims return errno directly (0 reads as success).
TEST_F(InjectionEngineTest, RejectsApiErrorWithZeroErrno) {
  uint32_t id = 0;
  auto r = baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = 0;
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_ERR_ARG);

  r.errnoValue = ENOMEM;
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_OK);
}

// The WC_STATUS counterpart of the zero-errno rejection above, and the more
// dangerous of the two because IBV_WC_SUCCESS is 0: a caller that never
// initialized wcStatus lands on it by default. Such a rule fires and moves
// cqesStatusMutated while leaving the completion good, so the counters claim an
// injection that did not happen -- the one outcome this tool must never
// produce.
TEST_F(InjectionEngineTest, RejectsWcStatusOfSuccess) {
  uint32_t id = 0;
  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);

  // Left at its default rather than assigned, which is exactly the mistake.
  ASSERT_EQ(r.wcStatus, IBV_WC_SUCCESS)
      << "baseRule no longer defaults wcStatus to SUCCESS, so this case is not "
         "testing the uninitialized path it was written for";
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_ERR_ARG);
  // The reason has to name the status, since "rule rejected" alone does not
  // tell a caller which field to look at.
  EXPECT_NE(
      std::string(engine_->lastError()).find("IBV_WC_SUCCESS"),
      std::string::npos);

  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_OK) << engine_->lastError();
}

// Re-registering the same CQ must not mint a second device id while the first
// is still held -- that would orphan the counters attached to the original id.
TEST_F(InjectionEngineTest, RegisterCqIsIdempotent) {
  setupTwoDevices();
  ASSERT_EQ(engine_->deviceOfCq(cq0_), 0);

  engine_->registerCq(cq0_);

  EXPECT_EQ(engine_->deviceOfCq(cq0_), 0)
      << "re-registration renumbered the CQ";
  // cq1_ still owns id 1, so a fresh CQ must take 2 rather than a hole.
  ibv_cq* third = provider_.makeCq();
  engine_->registerCq(third);
  EXPECT_EQ(engine_->deviceOfCq(third), 2);
}

// A rule whose selector the decision path cannot honor must be REJECTED, not
// accepted and left inert. Setup verbs fire before any device or QP exists, and
// a POLL_CQ API_ERROR fails the call before a completion is read, so those
// fields have nothing to compare against.
TEST_F(InjectionEngineTest, RejectsSelectorsTheVerbCannotHonor) {
  uint32_t id = 0;

  auto setupWithDevice =
      baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  setupWithDevice.errnoValue = ENOMEM;
  setupWithDevice.selector.deviceId = 0;
  EXPECT_EQ(engine_->addRule(&setupWithDevice, &id), IB_INJECTION_ERR_ARG);

  auto pollErrWithQp =
      baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_API_ERROR);
  pollErrWithQp.errnoValue = EIO;
  pollErrWithQp.selector.hwQpNum = 0x1234;
  EXPECT_EQ(engine_->addRule(&pollErrWithQp, &id), IB_INJECTION_ERR_ARG);

  // deviceId IS honored for a poll_cq API_ERROR, so this one must be accepted.
  auto pollErrWithDevice =
      baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_API_ERROR);
  pollErrWithDevice.errnoValue = EIO;
  pollErrWithDevice.selector.deviceId = 0;
  EXPECT_EQ(engine_->addRule(&pollErrWithDevice, &id), IB_INJECTION_OK);

  // ibv_recv_wr has no opcode field, so shimPostRecv can only ever decide with
  // ANY_OPCODE. A post_recv rule naming one would be stored, handed a rule id,
  // and never fire.
  auto recvWithOpcode =
      baseRule(IB_INJECTION_VERB_POST_RECV, IB_INJECTION_ACTION_API_ERROR);
  recvWithOpcode.errnoValue = EIO;
  recvWithOpcode.selector.opcode = IBV_WR_RDMA_WRITE;
  recvWithOpcode.selector.opcodeDomain = IB_INJECTION_OPCODE_WR;
  EXPECT_EQ(engine_->addRule(&recvWithOpcode, &id), IB_INJECTION_ERR_ARG);

  // The same rule without the opcode is the useful one, and still accepted.
  recvWithOpcode.selector.opcode = IB_INJECTION_ANY_OPCODE;
  EXPECT_EQ(engine_->addRule(&recvWithOpcode, &id), IB_INJECTION_OK);

  // post_send keeps its opcode selector; only post_recv lacks the field.
  auto sendWithOpcode =
      baseRule(IB_INJECTION_VERB_POST_SEND, IB_INJECTION_ACTION_API_ERROR);
  sendWithOpcode.errnoValue = EIO;
  sendWithOpcode.selector.opcode = IBV_WR_RDMA_WRITE;
  sendWithOpcode.selector.opcodeDomain = IB_INJECTION_OPCODE_WR;
  EXPECT_EQ(engine_->addRule(&sendWithOpcode, &id), IB_INJECTION_OK);
}

// IBV_WR_RDMA_READ is 4 and IBV_WC_RDMA_READ is 2, so an opcode from the wrong
// namespace matches the wrong traffic rather than nothing -- worse than a miss.
TEST_F(InjectionEngineTest, RejectsOpcodeFromTheWrongNamespace) {
  uint32_t id = 0;
  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  r.selector.opcode = IBV_WC_RDMA_READ;
  r.selector.opcodeDomain = IB_INJECTION_OPCODE_WR; // wrong for poll_cq
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_ERR_ARG);

  r.selector.opcodeDomain = IB_INJECTION_OPCODE_WC;
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_OK);
}

// A negative errno is worse than zero: shimPollCq returns -errno, so it would
// come back POSITIVE and ctran would read it as "that many completions
// written".
TEST_F(InjectionEngineTest, RejectsNegativeErrno) {
  uint32_t id = 0;
  auto r = baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = -ENOMEM;
  EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_ERR_ARG);
}

// A provider error mid-drain must not be reported as an empty queue: a caller
// spinning on poll_cq would loop forever on a dead CQ.
TEST_F(InjectionEngineTest, ProviderErrorIsReportedNotSwallowed) {
  setupTwoDevices();
  FakeProvider::setPollError(cq0_, -EIO);

  ibv_wc wc{};
  int32_t injectErrno = 0;
  EXPECT_EQ(engine_->pollCq(cq0_, 1, &wc, &injectErrno), -2);
  EXPECT_EQ(injectErrno, EIO);
}

// A verb value outside the enum can only ever sit inert: no shim queries with
// one, so decideSetupCall never sees it. The numeric range check this replaced
// accepted anything >= IB_INJECTION_VERB_OPEN_DEVICE, so addRule returned OK
// and handed back a rule id for a rule that could never fire --
// indistinguishable from the code under test handling the injected error
// correctly.
TEST_F(InjectionEngineTest, RejectsVerbValuesOutsideTheEnum) {
  uint32_t id = 0;
  for (const int32_t verb : {16, 100, std::numeric_limits<int32_t>::max()}) {
    auto r =
        baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
    r.verb = verb;
    // Set so the only thing left to reject is the verb; a rule missing this
    // would be turned away for its errno and the case would pass vacuously.
    r.errnoValue = ENOMEM;
    EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_ERR_ARG)
        << "verb " << verb << " was accepted";
  }

  // The six declared setup verbs must still go in, or the tightening went too
  // far and every setup rule silently stops being installable.
  for (const auto verb :
       {IB_INJECTION_VERB_OPEN_DEVICE,
        IB_INJECTION_VERB_ALLOC_PD,
        IB_INJECTION_VERB_REG_MR,
        IB_INJECTION_VERB_CREATE_CQ,
        IB_INJECTION_VERB_CREATE_QP,
        IB_INJECTION_VERB_MODIFY_QP}) {
    auto r = baseRule(verb, IB_INJECTION_ACTION_API_ERROR);
    r.errnoValue = ENOMEM;
    EXPECT_EQ(engine_->addRule(&r, &id), IB_INJECTION_OK)
        << "verb " << verb << ": " << engine_->lastError();
  }
}

TEST_F(InjectionEngineTest, RejectsMalformedRepeat) {
  uint32_t id = 0;

  auto zeroFirst =
      baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  zeroFirst.repeat.firstMatch = 0;
  EXPECT_EQ(engine_->addRule(&zeroFirst, &id), IB_INJECTION_ERR_ARG);

  auto boundedZero =
      baseRule(IB_INJECTION_VERB_CREATE_QP, IB_INJECTION_ACTION_API_ERROR);
  boundedZero.repeat.unbounded = 0;
  boundedZero.repeat.count = 0;
  EXPECT_EQ(engine_->addRule(&boundedZero, &id), IB_INJECTION_ERR_ARG);
}

// --- readback ---

TEST_F(InjectionEngineTest, GetStateReportsRequiredSizesOnCapacityError) {
  setupTwoDevices();
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, 1);
  ASSERT_EQ(poll(cq0_), 1);

  IbInjectionState state{};
  state.numDevices = 0;
  state.numRules = 0;
  EXPECT_EQ(engine_->getState(&state), IB_INJECTION_ERR_CAPACITY);
  EXPECT_EQ(state.requiredDevices, 2u);
}

TEST_F(InjectionEngineTest, ResetClearsRulesAndCounters) {
  setupTwoDevices();
  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.selector.deviceId = 0;
  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  addRule(r);
  FakeProvider::push(cq0_, IBV_WC_RDMA_READ, 1);
  ASSERT_EQ(poll(cq0_), 1);
  ASSERT_EQ(deviceState(0).cqesStatusMutated, 1u);

  ASSERT_EQ(engine_->reset(), IB_INJECTION_OK);

  EXPECT_EQ(deviceState(0).cqesStatusMutated, 0u);
  EXPECT_EQ(deviceState(0).cqesObserved, 0u);
  // Registrations survive a reset, so a test can reset after ctran init.
  EXPECT_EQ(engine_->deviceOfCq(cq0_), 0);
}

// A ruleId is a handle the caller keeps. Restarting the counter at reset()
// would hand the same value to a different rule, so a caller reading back a
// stale id would get someone else's counters instead of nothing -- the silent
// mismatch addRule exists to prevent everywhere else.
TEST_F(InjectionEngineTest, ResetDoesNotReuseRuleIds) {
  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.wcStatus = IBV_WC_RETRY_EXC_ERR;
  const uint32_t stale = addRule(r);

  ASSERT_EQ(engine_->reset(), IB_INJECTION_OK);
  const uint32_t fresh = addRule(r);

  EXPECT_NE(fresh, stale);
  // Ids start at 1, so a zero ruleId from the lookup means "no such rule" --
  // which is what a stale handle has to resolve to.
  EXPECT_EQ(ruleState(stale).ruleId, 0u);
  EXPECT_EQ(ruleState(fresh).ruleId, fresh);
}

} // namespace
} // namespace ibverbx::injection
