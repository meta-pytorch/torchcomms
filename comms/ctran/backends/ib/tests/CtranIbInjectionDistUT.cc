// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Distributed failure-injection tests for the IB backend.
//
// Separate from CtranIbDistUT.cc, and separately targeted: this file is the
// only one built into ctran_ib_injection_dist_ut, the only target that sets
// ib_injection = True. Keeping them apart means the ordinary distributed suite
// never runs under the shim, and the cases here never have to reason about
// whether a rule they armed is still live for somebody else's test.
//
// What needs two ranks: the single-rank CtranIbInjectionTest can only drive the
// loopback flush QPs -- with no peer there is no data QP, no iput and no
// control traffic at all, so every failure case there is a flush.
//
// PutSucceedsWithNoRuleArmed comes first on purpose. With the shim loaded but
// nothing armed, ctran must behave exactly as it does without it; if that
// regresses, every assertion below is measuring the wrong thing.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cerrno>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionControl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/utils/CtranLogger.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace injection = ibverbx::injection::testing;

namespace {

// Enough progress() calls for the traffic these tests issue to drain. Generous
// because every loop exits as soon as its condition holds.
constexpr int kMaxPumps = 10000;

class CtranIbInjectionDistTest : public ctran::CtranDistTestFixture {
 protected:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
    comm_ = makeCtranComm();
    comm = comm_.get();
  }

  void TearDown() override {
    // Rules live in the shim for the life of the process, so a rule left armed
    // would leak into the next test in this binary.
    if (injection::available()) {
      injection::reset();
    }
    ctranIb.reset();
    comm_.reset();
    CtranDistTestFixture::TearDown();
    ASSERT_EQ(getIbRegCount(), 0);
  }

  size_t getIbRegCount() {
    auto s = CtranIbSingleton::getInstance();
    CHECK_VALID_IB_SINGLETON(s);
    return s->getActiveRegCount();
  }

  bool dmaBufSupported() {
    auto s = CtranIbSingleton::getInstance();
    CHECK_VALID_IB_SINGLETON(s);
    return s->getDevToDmaBufSupport(this->localRank);
  }

  // Skips unless the shim is loaded, then builds the CtranIb every test needs.
  // GTEST_SKIP is only valid in a void function, and it marks the test without
  // returning from the caller, so every caller follows this with
  // `if (IsSkipped()) return;`.
  void setUpInjectedIb() {
    if (!injection::available()) {
      GTEST_SKIP()
          << "IBVERBX_IBVERBS_SO does not point at a loadable shim; only "
             "ctran_ib_injection_dist_ut sets it";
    }
    try {
      ctranIb = std::make_unique<CtranIb>(comm);
    } catch (const std::bad_alloc&) {
      GTEST_SKIP() << "IB backend not enabled. Skip test";
    }
  }

  void printTestDesc(const std::string& name, const std::string& desc) {
    CTRAN_LOG_STREAM_IF(WARN, this->globalRank == 0)
        << name << " numRanks " << this->numRanks << ". Description: " << desc;
  }

  commResult_t waitIbReq(CtranIbRequest& req) {
    do {
      COMMCHECK_TEST(ctranIb->progress());
    } while (!req.isComplete());
    return commSuccess;
  }

  constexpr static int kSockSyncLen = 16;
  void sockSend(int peerRank) {
    char buf[kSockSyncLen] = "ping";
    auto res = comm->bootstrap_->send(buf, sizeof(buf), peerRank, 0);
    ASSERT_EQ(static_cast<commResult_t>(std::move(res).get()), commSuccess);
  }
  void sockRecv(int peerRank) {
    char buf[kSockSyncLen];
    auto res = comm->bootstrap_->recv(buf, sizeof(buf), peerRank, 0);
    ASSERT_EQ(static_cast<commResult_t>(std::move(res).get()), commSuccess);
  }

  // Outcome of a scaffolded put. `ran` is false when the scaffold could not get
  // far enough to arm anything, in which case nothing else here is meaningful.
  struct PutOutcome {
    bool ran{false};
    commResult_t iputResult{commSuccess};
    commResult_t progressResult{commSuccess};
    bool completed{false};
    injection::Snapshot state;
  };

  // Register a buffer, connect the VC, exchange the rkey, prove one put
  // succeeds with NO rule armed, then let `arm` install rules against the
  // connected VC and issue a second put.
  //
  // The REGULAR put, not the fast one: that is the production route. Where an
  // injected errno surfaces is capacity-dependent and no test here pins it --
  // iputImpl drains queued WQEs inline while the QP has room (queueWriteOnQp,
  // "Post next data immediately") and leaves them for progress when it does
  // not.
  //
  // The message stays at or below the QP scaling threshold so it is a single
  // WQE on a single QP; a larger one fans out and the per-QP firing counts stop
  // adding to one.
  //
  // notify=false, so the receiver needs no IB progress and the ranks rendezvous
  // on the bootstrap socket -- which is what keeps the receiver from
  // deregistering the target buffer while a write is still in flight. The
  // rendezvous happens on every path, so a failure never leaves the peer
  // waiting.
  void runPut(
      const std::function<void(CtranIbVirtualConn&)>& arm,
      PutOutcome* out) {
    const size_t len = NCCL_CTRAN_IB_QP_SCALING_THRESHOLD;
    void* buf = nullptr;
    void* handle = nullptr;
    ControlMsg msg;
    CtranIbRequest ctrlReq;
    CtranIbEpochRAII epoch(ctranIb.get());

    CUDACHECK_TEST(cudaMalloc(&buf, len));
    COMMCHECK_TEST(CtranIb::regMem(buf, len, this->localRank, &handle));
    COMMCHECK_TEST(CtranIb::exportMem(buf, handle, msg));

    if (this->globalRank == recvRank) {
      COMMCHECK_TEST(ctranIb->preConnect({sendRank}));
      COMMCHECK_TEST(ctranIb->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), sendRank, ctrlReq));
    } else if (this->globalRank == sendRank) {
      COMMCHECK_TEST(ctranIb->preConnect({recvRank}));
      COMMCHECK_TEST(
          ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, ctrlReq));
    } else {
      COMMCHECK_TEST(ctrlReq.complete());
    }
    COMMCHECK_TEST(waitIbReq(ctrlReq));

    if (this->globalRank == sendRank) {
      void* remoteBuf = reinterpret_cast<void*>(msg.ibDesc.remoteAddr);
      CtranIbRemoteAccessKey key{};
      for (int i = 0; i < msg.ibDesc.nKeys; i++) {
        key.rkeys[i] = msg.ibDesc.rkeys[i];
      }
      auto put = [&](CtranIbRequest* req) {
        return ctranIb->iput(
            buf,
            remoteBuf,
            len,
            recvRank,
            handle,
            key,
            /*notify*/ false,
            nullptr,
            req);
      };

      CtranIbRequest cleanReq;
      const auto cleanResult = put(&cleanReq);
      EXPECT_EQ(cleanResult, commSuccess)
          << "the scaffold could not complete an un-injected put, so nothing "
             "below would mean anything";

      // Only drain a request the put accepted. A rejected one never completes,
      // so waiting on it spins to the harness timeout and buries the assertion.
      if (cleanResult == commSuccess) {
        COMMCHECK_TEST(waitIbReq(cleanReq));

        auto vc = ctranIb->getVc(recvRank);
        EXPECT_NE(vc, nullptr) << "preConnect left no VC for the peer";
        if (vc != nullptr) {
          // Reset after the clean put, so counters and every rule's firstMatch
          // start from the traffic under test rather than from VC setup and the
          // rkey exchange.
          injection::reset();
          arm(*vc);

          CtranIbRequest injectedReq;
          out->iputResult = put(&injectedReq);
          if (out->iputResult == commSuccess) {
            for (int i = 0; i < kMaxPumps; i++) {
              const auto res = ctranIb->progress();
              if (res != commSuccess) {
                out->progressResult = res;
                break;
              }
              if (injectedReq.isComplete()) {
                out->completed = true;
                break;
              }
            }
          }
          out->state = injection::getState();
          out->ran = true;
          injection::reset();
        }
      }
      sockSend(recvRank);
    } else if (this->globalRank == recvRank) {
      sockRecv(sendRank);
    }

    COMMCHECK_TEST(CtranIb::deregMem(handle));
    CUDACHECK_TEST(cudaFree(buf));
  }

  std::unique_ptr<CtranComm> comm_{nullptr};
  CtranComm* comm{nullptr};
  std::unique_ptr<CtranIb> ctranIb{nullptr};
  const int sendRank{0}, recvRank{1};
};

// The control. Everything else asserts that an injected failure is reported, so
// nothing else would notice the shim breaking traffic it was told to leave
// alone
// -- a shim that failed every post would pass all of them.
TEST_F(CtranIbInjectionDistTest, PutSucceedsWithNoRuleArmed) {
  setUpInjectedIb();
  if (IsSkipped()) {
    return;
  }
  printTestDesc(
      "PutSucceedsWithNoRuleArmed",
      "Expect a put to succeed with the shim loaded and no rule armed, so the "
      "shim is a faithful pass-through and not merely an error source.");

  PutOutcome outcome;
  runPut([](CtranIbVirtualConn&) {}, &outcome);

  if (this->globalRank != sendRank) {
    return;
  }
  ASSERT_TRUE(outcome.ran);
  EXPECT_EQ(outcome.iputResult, commSuccess);
  EXPECT_EQ(outcome.progressResult, commSuccess);
  EXPECT_TRUE(outcome.completed) << "the put never retired through the shim";
  EXPECT_TRUE(outcome.state.rules.empty());
}

// The iput counterpart of the flush case, and the contrast is the point. iflush
// kills the process when a post fails on any device past the first --
// FB_CHECKABORT(device == 0), CtranIbLocalVc.cc:151 -- because the completions
// of the devices that already posted would arrive untracked. iput has no such
// exemption: FOLLY_EXPECTED_CHECK turns the failed post into commSystemError.
TEST_F(CtranIbInjectionDistTest, IputReportsAPostSendFailure) {
  setUpInjectedIb();
  if (IsSkipped()) {
    return;
  }
  printTestDesc(
      "IputReportsAPostSendFailure",
      "Expect a put whose post_send is injected with ENOMEM to report the error "
      "to its caller rather than aborting or dropping the WQE.");

  PutOutcome outcome;
  uint32_t rule = 0;
  runPut(
      [&](CtranIbVirtualConn&) {
        rule = injection::addPostError(
            IB_INJECTION_VERB_POST_SEND,
            IB_INJECTION_ANY_DEVICE,
            IB_INJECTION_ANY_QP,
            ENOMEM,
            /*firstMatch*/ 1,
            /*count*/ 1);
      },
      &outcome);

  if (this->globalRank != sendRank) {
    return;
  }
  ASSERT_TRUE(outcome.ran);
  EXPECT_TRUE(
      outcome.iputResult != commSuccess ||
      outcome.progressResult != commSuccess)
      << "neither iput nor progress reported the injected post error, so it "
         "was swallowed";

  const auto* fired = outcome.state.rule(rule);
  ASSERT_NE(fired, nullptr);
  EXPECT_EQ(fired->firings, 1u)
      << "the rule never fired, so the return values prove nothing";
}

// The QP axis, against ctran's own QPs. The single-rank per-QP test builds both
// QPs by hand, so it shows the selector honors a hardware qp_num but not that
// the numbers reachable for a real VC are the ones the shim sees.
// getVc(peer)->getDataQpNums() and getControlQpNum() are that reachable
// surface.
//
// Rather than predict which data QP the round-robin picks, arm every one and
// require the firings to sum to one: true at any QP count, dependent on no
// internal. The control QP is the negative half and a real miss rather than a
// contrived one -- it carried the rkey exchange moments earlier, so a selector
// that ignored qp_num would have fired on it.
TEST_F(CtranIbInjectionDistTest, PostErrorIsScopedToADataQpFromGetDataQpNums) {
  setUpInjectedIb();
  if (IsSkipped()) {
    return;
  }
  printTestDesc(
      "PostErrorIsScopedToADataQpFromGetDataQpNums",
      "Expect post_send rules armed with the qp_nums from getDataQpNums() to "
      "fail the put on exactly one data QP and spare the control QP.");

  PutOutcome outcome;
  std::vector<uint32_t> dataRules;
  std::vector<uint32_t> dataQpNums;
  uint32_t ctrlRule = 0, ctrlQpNum = 0;

  runPut(
      [&](CtranIbVirtualConn& vc) {
        dataQpNums = vc.getDataQpNums();
        if (dataQpNums.empty()) {
          ADD_FAILURE() << "a connected VC reported no data QPs";
          return;
        }
        ctrlQpNum = vc.getControlQpNum();
        // count=0 is unbounded everywhere, so the control QP's miss is a real
        // miss rather than a rule that had already been spent.
        for (const auto qpNum : dataQpNums) {
          dataRules.push_back(
              injection::addPostError(
                  IB_INJECTION_VERB_POST_SEND,
                  IB_INJECTION_ANY_DEVICE,
                  qpNum,
                  ENOMEM,
                  /*firstMatch*/ 1,
                  /*count*/ 0));
        }
        ctrlRule = injection::addPostError(
            IB_INJECTION_VERB_POST_SEND,
            IB_INJECTION_ANY_DEVICE,
            ctrlQpNum,
            ENOMEM,
            /*firstMatch*/ 1,
            /*count*/ 0);
      },
      &outcome);

  if (this->globalRank != sendRank) {
    return;
  }
  ASSERT_TRUE(outcome.ran);
  EXPECT_TRUE(
      outcome.iputResult != commSuccess ||
      outcome.progressResult != commSuccess)
      << "neither iput nor progress reported the injected post error";

  // IB_INJECTION_ANY_QP *is* 0, so a data QP reported as 0 would quietly widen
  // its rule into match-anything and the control QP's miss would stop meaning
  // anything. "Real qp_nums are large" is an observation about providers rather
  // than a contract, so it is not asserted.
  for (const auto qpNum : dataQpNums) {
    EXPECT_NE(qpNum, static_cast<uint32_t>(IB_INJECTION_ANY_QP));
    EXPECT_NE(qpNum, ctrlQpNum);
  }

  uint64_t dataFirings = 0;
  for (const auto id : dataRules) {
    const auto* r = outcome.state.rule(id);
    ASSERT_NE(r, nullptr);
    dataFirings += r->firings;
  }
  EXPECT_EQ(dataFirings, 1u)
      << "a single-WQE put should fire exactly one data-QP rule; "
      << dataQpNums.size() << " data QPs were armed";

  const auto* ctrl = outcome.state.rule(ctrlRule);
  ASSERT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->firings, 0u)
      << "a data-QP rule set also fired on control qp_num " << ctrlQpNum;
}

// post_recv, which is not reachable the way the post_send cases are: ctran
// pre-posts its control receives while building the VC, then RE-posts one each
// time a control message is consumed (postRecvCtrlMsg from the IBV_WC_RECV arm
// of processCqeImpl). Arming after the VC is up targets the re-post, which
// keeps this clear of VC setup -- where a failure would instead hang preConnect
// (CtranIb.cc:1059).
//
// The rule goes on sendRank because that is the rank calling irecvCtrlMsg, so
// it is the one whose progress() consumes an IBV_WC_RECV and re-posts.
TEST_F(CtranIbInjectionDistTest, CtrlMsgPostRecvFailureIsReported) {
  setUpInjectedIb();
  if (IsSkipped()) {
    return;
  }
  printTestDesc(
      "CtrlMsgPostRecvFailureIsReported",
      "Expect a failed post_recv, injected while re-arming the control QP after "
      "a control message is consumed, to surface as an error from progress.");
  CtranIbEpochRAII epoch(ctranIb.get());

  const int peer = this->globalRank == sendRank ? recvRank : sendRank;
  COMMCHECK_TEST(ctranIb->preConnect({peer}));

  ControlMsg msg;
  CtranIbRequest firstReq;
  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(
        ctranIb->isendCtrlMsg(msg.type, &msg, sizeof(msg), sendRank, firstReq));
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(
        ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, firstReq));
  } else {
    COMMCHECK_TEST(firstReq.complete());
  }
  COMMCHECK_TEST(waitIbReq(firstReq));

  // Armed after the first exchange so setup's pre-posted receives are not what
  // the rule lands on. The barrier is not decoration: without it the peer's
  // second message can arrive while the first exchange is still draining, so
  // its CQE -- and the re-post that CQE triggers -- happen before the rule
  // exists, and the later irecvCtrlMsg is then satisfied from the
  // unexpected-message buffer without posting a receive at all.
  uint32_t rule = 0;
  if (this->globalRank == sendRank) {
    injection::reset();
    rule = injection::addPostError(
        IB_INJECTION_VERB_POST_RECV,
        IB_INJECTION_ANY_DEVICE,
        IB_INJECTION_ANY_QP,
        ENOMEM,
        /*firstMatch*/ 1,
        /*count*/ 1);
    sockSend(recvRank);
  } else if (this->globalRank == recvRank) {
    sockRecv(sendRank);
  }

  CtranIbRequest secondReq;
  commResult_t progressResult = commSuccess;
  if (this->globalRank == recvRank) {
    COMMCHECK_TEST(ctranIb->isendCtrlMsg(
        msg.type, &msg, sizeof(msg), sendRank, secondReq));
    COMMCHECK_TEST(waitIbReq(secondReq));
    sockRecv(sendRank);
  } else if (this->globalRank == sendRank) {
    COMMCHECK_TEST(
        ctranIb->irecvCtrlMsg(&msg, sizeof(msg), recvRank, secondReq));
    for (int i = 0; i < kMaxPumps; i++) {
      const auto res = ctranIb->progress();
      if (res != commSuccess) {
        progressResult = res;
        break;
      }
      if (secondReq.isComplete()) {
        break;
      }
    }
    const auto state = injection::getState();
    injection::reset();
    sockSend(recvRank);

    EXPECT_NE(progressResult, commSuccess)
        << "a failed control-QP post_recv was not reported";
    const auto* fired = state.rule(rule);
    ASSERT_NE(fired, nullptr);
    EXPECT_EQ(fired->firings, 1u)
        << "the rule never fired, so this proves nothing about post_recv";
  }
}

// reg_mr against DEVICE memory, which takes a different verb: regMem routes
// CUDA device memory through ibv_reg_dmabuf_mr (or mlx5dv_reg_dmabuf_mr with
// data-direct) whenever the device supports it -- useDmaBuf, CtranIb.cc:855.
// Those verbs were pure forwarders until the shim grew REG_MR hooks for them,
// so a rule armed against a device buffer was stored, handed an id, and inert.
// This is the case that would have caught that.
TEST_F(CtranIbInjectionDistTest, RegMrFailureReachesDmaBufRegistration) {
  setUpInjectedIb();
  if (IsSkipped()) {
    return;
  }
  if (!dmaBufSupported()) {
    GTEST_SKIP() << "device does not support dma-buf registration, so regMem "
                    "would take the ibv_reg_mr path this test is not about";
  }
  printTestDesc(
      "RegMrFailureReachesDmaBufRegistration",
      "Expect a REG_MR rule to fail a device-memory registration, which ctran "
      "performs through the dma-buf verbs rather than ibv_reg_mr.");

  constexpr size_t kLen = 8192;
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, kLen));

  injection::reset();
  const auto rule = injection::addSetupError(
      IB_INJECTION_VERB_REG_MR, ENOMEM, /*firstMatch*/ 1, /*count*/ 1);

  void* handle = nullptr;
  const auto regResult = CtranIb::regMem(buf, kLen, this->localRank, &handle);
  const auto state = injection::getState();
  injection::reset();

  EXPECT_NE(regResult, commSuccess)
      << "device-memory registration succeeded despite a REG_MR rule";
  const auto* fired = state.rule(rule);
  ASSERT_NE(fired, nullptr);
  EXPECT_EQ(fired->firings, 1u)
      << "the rule never fired, so the dma-buf verbs are not covered by REG_MR";

  if (regResult == commSuccess) {
    COMMCHECK_TEST(CtranIb::deregMem(handle));
  }
  CUDACHECK_TEST(cudaFree(buf));
}

// Rules live in the shim, and the shim is loaded once per process -- so two
// ranks must be able to hold different rules. Every other case here arms on one
// rank only, so a shared or accidentally-global rule set would break all of
// them while still passing them.
//
// reg_mr rather than a put: a rank whose put fails leaves its peer waiting for
// data that never arrives, and with preConnect already known to spin unbounded
// (CtranIb.cc:1059) an asymmetric failure is not worth expressing that way.
// regMem is local, so each rank observes its own outcome and neither can hang.
// Host memory keeps it on ibv_reg_mr and independent of dma-buf availability.
TEST_F(CtranIbInjectionDistTest, RulesAreScopedToOneRank) {
  setUpInjectedIb();
  if (IsSkipped()) {
    return;
  }
  printTestDesc(
      "RulesAreScopedToOneRank",
      "Expect a reg_mr rule armed only on rank 0 to fail rank 0's registration "
      "and leave rank 1's untouched.");

  constexpr size_t kLen = 8192;
  void* buf = nullptr;
  if (cudaHostAlloc(&buf, kLen, cudaHostAllocDefault) != cudaSuccess) {
    GTEST_SKIP() << "cudaHostAlloc unavailable on this host";
  }

  injection::reset();
  uint32_t rule = 0;
  const bool armed = this->globalRank == sendRank;
  if (armed) {
    rule = injection::addSetupError(
        IB_INJECTION_VERB_REG_MR, ENOMEM, /*firstMatch*/ 1, /*count*/ 1);
  }

  void* handle = nullptr;
  const auto regResult = CtranIb::regMem(buf, kLen, this->localRank, &handle);
  const auto state = injection::getState();
  injection::reset();

  if (armed) {
    EXPECT_NE(regResult, commSuccess)
        << "rank " << this->globalRank
        << " armed a reg_mr rule but registered successfully";
    const auto* fired = state.rule(rule);
    ASSERT_NE(fired, nullptr);
    EXPECT_EQ(fired->firings, 1u);
  } else {
    EXPECT_EQ(regResult, commSuccess)
        << "rank " << this->globalRank
        << " armed nothing, so another rank's rule reached this process";
    EXPECT_TRUE(state.rules.empty())
        << "another rank's rule is visible in this process's shim";
  }

  if (regResult == commSuccess) {
    COMMCHECK_TEST(CtranIb::deregMem(handle));
  }
  EXPECT_EQ(cudaFreeHost(buf), cudaSuccess);
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
