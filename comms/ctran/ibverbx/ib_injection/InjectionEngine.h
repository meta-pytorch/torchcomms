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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionApi.h"

namespace ibverbx::injection {

// --- IB_INJECTION_SPEC: the environment front end ---
//
// The surfaces that need injection most -- collperf, MAST and farm jobs -- have
// no C++ hook point and cannot reach the control ABI at all. What they do have
// is job-wide environment, so IB_INJECTION_SPEC carries a declarative rule
// list, parsed at load into the very rules addRule() builds. One engine, two
// front ends: nothing here re-implements matching or validation.
//
//   IB_INJECTION_SPEC = rule ( ';' rule )*
//   rule              = field ( ',' field )*
//   field             = key '=' value
//
// Keys are `fn action errno status rank dev qp opcode first every count`. The
// per-key value grammar and defaults live in
// comms/ctran/ibverbx/ib_injection/README.md, where the table renders as
// markdown instead of being reflowed into nonsense by clang-format.
//
// `opcode` takes the SHORT name and the namespace is derived from `fn`. That is
// deliberate: IBV_WR_RDMA_READ is 4 and IBV_WC_RDMA_READ is 2, so naming the
// namespace is the one thing a hand-written spec must not be allowed to get
// wrong -- it would match nothing, or the wrong traffic, and say so either way.
//
// `rank` is matched against $RANK, so one job-wide string arms selected ranks.
// That exists because the surfaces this is for -- collperf, farm jobs -- have
// no per-rank environment channel: their env is set once for the whole task
// group.
//
// Everything the current engine cannot act on is REJECTED, not ignored: the
// planned `qp=data:N` role form and the `cq_gate` / `call_delay` skew actions
// name mechanisms that do not exist yet, and a spec that quietly did nothing is
// the exact failure this whole mechanism exists to remove.
struct SpecParseResult {
  // Rules whose `rank` selector matched this rank, in spec order.
  std::vector<IbInjectionRule> rules;
  // Rules addressed to a different rank. Reported so a banner can distinguish
  // "this rank was not a target" from "the spec armed nothing at all". These
  // were fully validated -- only their arming was skipped.
  uint32_t skippedForRank{0};
  // Empty on success. Otherwise names the offending token.
  std::string error;
};

// Pure: no env, no engine, no dlopen. The Engine constructor is the only
// production caller; tests drive this directly.
//
// `rank` is this process's rank, or nullopt when the environment does not name
// one. A spec that uses `rank=` with nullopt is an ERROR, not a guess: assuming
// rank 0 would make `rank=1` arm nothing and `rank=0` arm every process, both
// silently, which is the one outcome an injector must never produce.
//
// Every rule is validated whatever rank it names. Only arming is filtered, so a
// typo in a rule addressed to another rank still fails here -- otherwise a spec
// whose rules all target ranks outside this job would be validated by nobody
// and arm nothing anywhere, reporting success.
SpecParseResult parseInjectionSpec(
    const std::string& spec,
    std::optional<int32_t> rank);

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
  Engine();

  // Parse IB_INJECTION_SPEC (against $RANK) and install the rules it names.
  // Aborts on a malformed spec or a rule addRule rejects, naming the offending
  // token: a spec that failed to parse and ran anyway is indistinguishable from
  // an unmounted fbpkg, and both look like a green uninjected pass.
  //
  // Called from the constructor, which every shim path reaches through
  // Engine::get() before it delegates -- including decideSetupCall() for an
  // open_device rule, so it is early enough for every verb.
  //
  // reset() drops these along with every other rule. That is right for a C++
  // test, which resets after ctran init to measure only its own traffic, and
  // irrelevant for the env-driven surfaces, which never call it.
  void applyEnvSpec();

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
