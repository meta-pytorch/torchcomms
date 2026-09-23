// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// Internal state for the ib_injection shim. Not part of the control ABI
// (IbInjectionApi.h is); this header is private to the shim.
//
// Threading: a single mutex guards everything. ctran polls CQs from the GPE
// thread and posts from the same, but nothing guarantees one CQ is touched by
// only one thread, and progressInternal is reachable from any caller holding
// the epoch lock. A mutex on the shim path costs far less than the dlopen'd
// verbs call it wraps.

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionApi.h"

namespace ibverbx::injection {

// The ops the shim intercepts. Everything else on ibv_context_ops keeps
// pointing at the real provider's implementation -- including req_notify_cq,
// which is deliberately not patched.
struct SavedOps {
  int (*pollCq)(ibv_cq*, int, ibv_wc*){nullptr};
  int (*postSend)(ibv_qp*, ibv_send_wr*, ibv_send_wr**){nullptr};
  int (*postRecv)(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**){nullptr};
};

struct ContextRecord {
  SavedOps saved;
};

// A CQ's logical device index. ctran creates exactly one CQ per NIC, so the Nth
// CQ created maps 1:1 to NIC N.
struct CqRecord {
  int32_t deviceId{-1};
  ibv_context* context{nullptr};
};

struct QpRecord {
  uint32_t qpNum{0};
  int32_t sendDeviceId{-1};
  int32_t recvDeviceId{-1};
  // Recorded at registration so forgetContext can find this QP's owner without
  // dereferencing the ibv_qp*. Deregistration runs AFTER the provider's verb
  // returns, so by then the pointer may already be freed memory.
  ibv_context* context{nullptr};
};

struct Rule {
  uint32_t id{0};
  IbInjectionVerb verb{IB_INJECTION_VERB_POLL_CQ};
  IbInjectionAction action{IB_INJECTION_ACTION_API_ERROR};
  IbInjectionSelector selector{};
  IbInjectionRepeat repeat{};

  int32_t errnoValue{0};
  int32_t wcStatus{0};

  // Bookkeeping. `matches` counts selector hits; `firings` counts the subset
  // the repeat schedule actually acted on.
  uint64_t matches{0};
  uint64_t firings{0};
};

struct DeviceCounters {
  uint64_t pollCqCalls{0};
  uint64_t postSendCalls{0};
  uint64_t postRecvCalls{0};
  uint64_t cqesObserved{0};
  uint64_t cqesReleased{0};
  uint64_t cqesStatusMutated{0};
  uint64_t apiErrorsInjected{0};
};

// What a shim should do with a call it is about to delegate.
struct CallDecision {
  bool injectError{false};
  int32_t errnoValue{0};
};

// Process-wide injection state. One instance, created on first use.
class Engine {
 public:
  static Engine& get();

  std::mutex& mutex() {
    return mutex_;
  }

  // --- registration, called from the exported verb wrappers ---

  // Records the context's original ops. Idempotent per context, which matters:
  // by the time a shim runs, ctx->ops already points at the shims, so a second
  // "registration" would save a shim as the original and recurse.
  void registerContext(ibv_context* ctx);

  // Copy out the ops recorded at registration. Returns false for a context this
  // engine never saw. The shims copy rather than hold a reference, because they
  // release the mutex before delegating and a concurrent forgetContext() would
  // invalidate a reference into contexts_.
  bool lookupSavedOps(ibv_context* ctx, SavedOps* out) const;

  // Forgets the context and every CQ/QP registered on it -- leaving those
  // behind would keep records pointing at a freed context.
  void forgetContext(ibv_context* ctx);

  void registerCq(ibv_cq* cq);
  void forgetCq(ibv_cq* cq);
  void registerQp(ibv_qp* qp);
  void forgetQp(ibv_qp* qp);

  // --- shim path ---

  // Resolve the logical device for a CQ/QP. -1 when unregistered, which means
  // some path created the object without going through our wrapper; the shim
  // then delegates without injecting rather than guessing.
  int32_t deviceOfCq(ibv_cq* cq) const;
  int32_t deviceOfQpSend(ibv_qp* qp) const;
  int32_t deviceOfQpRecv(ibv_qp* qp) const;

  // Should this setup-verb call fail? Setup verbs have no device or QP, so only
  // the repeat schedule and the verb itself select.
  CallDecision decideSetupCall(IbInjectionVerb verb);

  // Should this post fail. Counts the call against the QP's device, or not at
  // all when the QP is unregistered.
  CallDecision decidePost(IbInjectionVerb verb, ibv_qp* qp, int32_t wrOpcode);

  // Poll the provider and apply any WC_STATUS rewrite on the way out. Returns
  // the number written to wc, -1 when the CQ is unregistered and the caller
  // must delegate, or -2 with *injectErrno set when a rule fails the call
  // itself.
  int pollCq(ibv_cq* cq, int numEntries, ibv_wc* wc, int32_t* injectErrno);

  // --- control ABI backing ---

  IbInjectionStatus reset();
  IbInjectionStatus addRule(const IbInjectionRule* rule, uint32_t* ruleId);
  IbInjectionStatus getState(IbInjectionState* state);

  void setLastError(std::string msg);
  const char* lastError() const;

  // Test seam. reset() deliberately KEEPS the object registries, because in
  // production a test resets rules after ctran init and must not lose the
  // contexts/CQs/QPs ctran already created. But the Engine is a process-wide
  // singleton, so a unit-test fixture needs the opposite: entries from an
  // earlier test outlive the fake objects they point at and keep holding device
  // ids, which makes the suite order-dependent. Fixtures call this in SetUp.
  void forgetAllObjectsForTest();

 private:
  Engine() = default;

  // True when the rule's selector matches, AND its repeat schedule says this
  // match should act. Bumps matches/firings. Evaluated ONCE per event.
  bool shouldFire(Rule& rule, int32_t deviceId, uint32_t qpNum, int32_t opcode);
  // Selector match only, without touching the repeat schedule.
  bool selectorMatches(
      const Rule& rule,
      int32_t deviceId,
      uint32_t qpNum,
      int32_t opcode) const;

  int32_t lowestUnusedDeviceId() const;
  // Erase a device's counters once no CQ holds its id. Ids are reused, so a
  // stale entry both reports a dead device from getState() and leaks its counts
  // into the next CQ that takes the id.
  void dropDeviceIfUnreferenced(int32_t deviceId);

  mutable std::mutex mutex_;

  std::unordered_map<ibv_context*, ContextRecord> contexts_;
  std::unordered_map<ibv_cq*, CqRecord> cqs_;
  std::unordered_map<ibv_qp*, QpRecord> qps_;

  std::vector<Rule> rules_;
  uint32_t nextRuleId_{1};

  std::unordered_map<int32_t, DeviceCounters> counters_;

  std::string lastError_;
};

} // namespace ibverbx::injection
