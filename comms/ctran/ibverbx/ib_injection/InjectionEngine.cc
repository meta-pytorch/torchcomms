// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/ib_injection/InjectionEngine.h"

#include <algorithm>
#include <vector>

namespace ibverbx::injection {

namespace {

bool wildcardOrEqual(int32_t want, int32_t got, int32_t any) {
  return want == any || want == got;
}

bool isPostVerb(IbInjectionVerb verb) {
  return verb == IB_INJECTION_VERB_POST_SEND ||
      verb == IB_INJECTION_VERB_POST_RECV;
}

// Membership in the six declared setup verbs, not a numeric range. addRule
// validates with this, so a `>=` comparison accepts a verb value no shim ever
// passes to decideSetupCall -- stored, handed a rule id, and inert, which is
// the one outcome the rest of addRule exists to reject. The switch also means a
// seventh setup verb cannot be declared without the compiler pointing here.
bool isSetupVerb(IbInjectionVerb verb) {
  switch (verb) {
    case IB_INJECTION_VERB_OPEN_DEVICE:
    case IB_INJECTION_VERB_ALLOC_PD:
    case IB_INJECTION_VERB_REG_MR:
    case IB_INJECTION_VERB_CREATE_CQ:
    case IB_INJECTION_VERB_CREATE_QP:
    case IB_INJECTION_VERB_MODIFY_QP:
      return true;
    case IB_INJECTION_VERB_POLL_CQ:
    case IB_INJECTION_VERB_POST_SEND:
    case IB_INJECTION_VERB_POST_RECV:
      return false;
  }
  // A value outside the enum, cast in across the C ABI. Not a setup verb: no
  // shim will ever query with it, so a rule naming it could only sit inert.
  return false;
}

} // namespace

Engine& Engine::get() {
  // Function-local static: the shim is dlopen(3)ed, so a namespace-scope object
  // would race the host's own static init order.
  static Engine* instance = new Engine();
  return *instance;
}

void Engine::registerContext(ibv_context* ctx) {
  auto [it, inserted] = contexts_.try_emplace(ctx);
  if (!inserted) {
    // Already recorded. Overwriting would capture the CURRENT ops, which are
    // the shims by now, and delegating to those recurses forever.
    return;
  }
  it->second.saved.pollCq = ctx->ops.poll_cq;
  it->second.saved.postSend = ctx->ops.post_send;
  it->second.saved.postRecv = ctx->ops.post_recv;
}

void Engine::forgetAllObjectsForTest() {
  contexts_.clear();
  cqs_.clear();
  qps_.clear();
  counters_.clear();
}

bool Engine::lookupSavedOps(ibv_context* ctx, SavedOps* out) const {
  auto it = contexts_.find(ctx);
  if (it == contexts_.end()) {
    return false;
  }
  *out = it->second.saved;
  return true;
}

void Engine::forgetContext(ibv_context* ctx) {
  // Drop the CQs and QPs on this context too. Keeping them would leave records
  // whose `context` points at memory the provider has freed, and would hold
  // device ids that lowestUnusedDeviceId() then refuses to reuse.
  std::vector<int32_t> orphaned;
  for (auto it = cqs_.begin(); it != cqs_.end();) {
    if (it->second.context == ctx) {
      orphaned.push_back(it->second.deviceId);
      it = cqs_.erase(it);
    } else {
      it = std::next(it);
    }
  }
  for (const int32_t deviceId : orphaned) {
    dropDeviceIfUnreferenced(deviceId);
  }
  for (auto it = qps_.begin(); it != qps_.end();) {
    it = it->second.context == ctx ? qps_.erase(it) : std::next(it);
  }
  contexts_.erase(ctx);
}

int32_t Engine::lowestUnusedDeviceId() const {
  for (int32_t candidate = 0;; candidate++) {
    bool taken = false;
    for (const auto& [cq, rec] : cqs_) {
      if (rec.deviceId == candidate) {
        taken = true;
        break;
      }
    }
    if (!taken) {
      return candidate;
    }
  }
}

void Engine::registerCq(ibv_cq* cq) {
  if (cq == nullptr || cq->context == nullptr) {
    return;
  }
  // Device ids are handed out per CQ in creation order, NOT per context: ctran
  // opens one ibv_context per NIC and creates one CQ on each, so a per-context
  // counter would hand every CQ id 0 and collapse every device onto one -- a
  // cross-device rule could then never fire.
  //
  // The id is the lowest currently-unused value rather than a monotonic
  // counter, so a process that builds a second CtranIb after destroying the
  // first sees the same ids again. A monotonic counter would renumber NIC 0 to
  // id 2 in that second object and every rule written against id 0 would
  // silently match nothing.
  // Idempotent, like registerContext: re-registering the same CQ would compute
  // a second id while the first is still held, orphaning the original's
  // counters.
  auto [it, inserted] = cqs_.try_emplace(cq);
  if (!inserted) {
    return;
  }
  it->second.deviceId = lowestUnusedDeviceId();
  it->second.context = cq->context;
  counters_.try_emplace(it->second.deviceId);
}

void Engine::forgetCq(ibv_cq* cq) {
  auto it = cqs_.find(cq);
  if (it == cqs_.end()) {
    return;
  }
  const int32_t deviceId = it->second.deviceId;
  cqs_.erase(it);
  dropDeviceIfUnreferenced(deviceId);
}

// getState() derives its device list from counters_, and ids are REUSED rather
// than monotonic, so a stale entry does double damage: it reports a device that
// no longer exists, and the next CQ to take that id inherits the dead device's
// counts.
void Engine::dropDeviceIfUnreferenced(int32_t deviceId) {
  for (const auto& [cq, rec] : cqs_) {
    (void)cq;
    if (rec.deviceId == deviceId) {
      return; // another CQ still holds this id
    }
  }
  counters_.erase(deviceId);
}

void Engine::registerQp(ibv_qp* qp) {
  if (qp == nullptr) {
    return;
  }
  // Idempotent, like registerContext and registerCq: a re-registration would
  // silently replace the device attribution recorded when the QP was created.
  auto [it, inserted] = qps_.try_emplace(qp);
  if (!inserted) {
    return;
  }
  it->second.qpNum = qp->qp_num;
  it->second.sendDeviceId = deviceOfCq(qp->send_cq);
  it->second.recvDeviceId = deviceOfCq(qp->recv_cq);
  it->second.context = qp->context;
}

void Engine::forgetQp(ibv_qp* qp) {
  qps_.erase(qp);
}

int32_t Engine::deviceOfCq(ibv_cq* cq) const {
  auto it = cqs_.find(cq);
  return it == cqs_.end() ? -1 : it->second.deviceId;
}

int32_t Engine::deviceOfQpSend(ibv_qp* qp) const {
  auto it = qps_.find(qp);
  return it == qps_.end() ? -1 : it->second.sendDeviceId;
}

int32_t Engine::deviceOfQpRecv(ibv_qp* qp) const {
  auto it = qps_.find(qp);
  return it == qps_.end() ? -1 : it->second.recvDeviceId;
}

bool Engine::selectorMatches(
    const Rule& rule,
    int32_t deviceId,
    uint32_t qpNum,
    int32_t opcode) const {
  const auto& sel = rule.selector;
  if (!wildcardOrEqual(sel.deviceId, deviceId, IB_INJECTION_ANY_DEVICE)) {
    return false;
  }
  if (sel.hwQpNum != IB_INJECTION_ANY_QP && sel.hwQpNum != qpNum) {
    return false;
  }
  if (!wildcardOrEqual(sel.opcode, opcode, IB_INJECTION_ANY_OPCODE)) {
    return false;
  }
  return true;
}

bool Engine::shouldFire(
    Rule& rule,
    int32_t deviceId,
    uint32_t qpNum,
    int32_t opcode) {
  if (!selectorMatches(rule, deviceId, qpNum, opcode)) {
    return false;
  }
  const uint64_t ordinal = ++rule.matches;
  const auto& rep = rule.repeat;
  if (ordinal < rep.firstMatch) {
    return false;
  }
  if ((ordinal - rep.firstMatch) % rep.everyNth != 0) {
    return false;
  }
  if (!rep.unbounded && rule.firings >= rep.count) {
    return false;
  }
  rule.firings++;
  return true;
}

CallDecision Engine::decideSetupCall(IbInjectionVerb verb) {
  CallDecision out;
  for (auto& rule : rules_) {
    if (rule.verb != verb || rule.action != IB_INJECTION_ACTION_API_ERROR) {
      continue;
    }
    // Setup verbs carry no device or QP, so only the repeat schedule selects.
    if (!shouldFire(
            rule,
            IB_INJECTION_ANY_DEVICE,
            IB_INJECTION_ANY_QP,
            IB_INJECTION_ANY_OPCODE)) {
      continue;
    }
    out.injectError = true;
    out.errnoValue = rule.errnoValue;
    return out;
  }
  return out;
}

CallDecision
Engine::decidePost(IbInjectionVerb verb, ibv_qp* qp, int32_t wrOpcode) {
  const int32_t deviceId = verb == IB_INJECTION_VERB_POST_SEND
      ? deviceOfQpSend(qp)
      : deviceOfQpRecv(qp);
  auto qpIt = qps_.find(qp);
  const uint32_t qpNum = qpIt == qps_.end() ? 0u : qpIt->second.qpNum;

  // Only count against a device we actually know. operator[] on deviceId -1
  // would mint a phantom entry that then shows up in getState() alongside real
  // devices, contradicting what -1 means.
  if (deviceId >= 0) {
    auto& counters = counters_[deviceId];
    if (verb == IB_INJECTION_VERB_POST_SEND) {
      counters.postSendCalls++;
    } else {
      counters.postRecvCalls++;
    }
  }

  CallDecision out;
  for (auto& rule : rules_) {
    if (rule.verb != verb || rule.action != IB_INJECTION_ACTION_API_ERROR) {
      continue;
    }
    if (!shouldFire(rule, deviceId, qpNum, wrOpcode)) {
      continue;
    }
    out.injectError = true;
    out.errnoValue = rule.errnoValue;
    if (deviceId >= 0) {
      counters_[deviceId].apiErrorsInjected++;
    }
    return out;
  }
  return out;
}

int Engine::pollCq(
    ibv_cq* cq,
    int numEntries,
    ibv_wc* wc,
    int32_t* injectErrno) {
  auto cqIt = cqs_.find(cq);
  if (cqIt == cqs_.end()) {
    return -1; // caller must delegate; unregistered CQ
  }
  auto& rec = cqIt->second;
  const int32_t deviceId = rec.deviceId;

  auto ctxIt = contexts_.find(rec.context);
  if (ctxIt == contexts_.end() || ctxIt->second.saved.pollCq == nullptr) {
    return -1; // caller must delegate; nothing recorded to delegate through
  }

  // Counted after validation, so a delegated call is not recorded as a poll we
  // serviced. Resolved once: the drain loop below would otherwise hash per CQE.
  auto& counters = counters_[deviceId];
  counters.pollCqCalls++;

  // API_ERROR is evaluated before the queue is touched: it models the call
  // failing, not a completion being bad.
  for (auto& rule : rules_) {
    if (rule.verb != IB_INJECTION_VERB_POLL_CQ ||
        rule.action != IB_INJECTION_ACTION_API_ERROR) {
      continue;
    }
    if (!shouldFire(
            rule, deviceId, IB_INJECTION_ANY_QP, IB_INJECTION_ANY_OPCODE)) {
      continue;
    }
    counters.apiErrorsInjected++;
    *injectErrno = rule.errnoValue;
    return -2; // sentinel: shim returns the negative errno
  }

  // Pass the provider's completions straight through, rewriting status where a
  // WC_STATUS rule matches. Nothing is withheld here: holding a CQE is skew,
  // not failure, and lands with the CQE_DELAY action in its own diff.
  //
  // Drained one at a time rather than in a batch. A rule selects on qp_num and
  // opcode, which are per-completion, so a batch would have to be walked
  // element-by-element anyway; asking for one keeps the "which CQE did this
  // rule see" bookkeeping obvious. The cost is an extra provider call per
  // completion, on a path ctran already drives from a single progress thread.
  int written = 0;
  while (written < numEntries) {
    ibv_wc scratch;
    const int got = ctxIt->second.saved.pollCq(cq, 1, &scratch);
    if (got == 0) {
      break; // queue drained
    }
    if (got < 0) {
      // A provider error, not an empty queue. Surfacing it as "no completions"
      // would let a caller spinning on poll_cq loop forever on a dead CQ, so
      // report it -- but only once anything already gathered has been handed
      // back, since those completions really did happen.
      if (written > 0) {
        break;
      }
      *injectErrno = -got;
      return -2;
    }
    counters.cqesObserved++;

    const int32_t opcode = static_cast<int32_t>(scratch.opcode);
    const uint32_t qpNum = scratch.qp_num;
    for (auto& rule : rules_) {
      if (rule.verb != IB_INJECTION_VERB_POLL_CQ ||
          rule.action != IB_INJECTION_ACTION_WC_STATUS) {
        continue;
      }
      if (!shouldFire(rule, deviceId, qpNum, opcode)) {
        continue;
      }
      scratch.status = static_cast<ibv_wc_status>(rule.wcStatus);
      counters.cqesStatusMutated++;
      break;
    }

    wc[written++] = scratch;
    counters.cqesReleased++;
  }
  return written;
}

IbInjectionStatus Engine::reset() {
  rules_.clear();

  // nextRuleId_ deliberately keeps climbing. Restarting it would let a ruleId
  // the caller still holds from before the reset name a DIFFERENT rule in
  // getState() -- the same silent-mismatch class this file rejects for errno 0,
  // IBV_WC_SUCCESS and the wildcard sentinels. A stale handle must miss.
  for (auto& [deviceId, c] : counters_) {
    c = DeviceCounters{};
  }
  lastError_.clear();
  return IB_INJECTION_OK;
}

IbInjectionStatus Engine::addRule(
    const IbInjectionRule* rule,
    uint32_t* ruleId) {
  if (rule == nullptr || ruleId == nullptr) {
    setLastError("addRule: null rule or ruleId");
    return IB_INJECTION_ERR_ARG;
  }

  const auto verb = static_cast<IbInjectionVerb>(rule->verb);
  const auto action = static_cast<IbInjectionAction>(rule->action);

  // The action/verb matrix is deliberately sparse; reject the gaps rather than
  // accept a rule that can never fire.
  switch (action) {
    case IB_INJECTION_ACTION_WC_STATUS:
      if (verb != IB_INJECTION_VERB_POLL_CQ) {
        setLastError("addRule: WC_STATUS applies to poll_cq only");
        return IB_INJECTION_ERR_ARG;
      }
      // IBV_WC_SUCCESS is the one status that changes nothing: the rule fires,
      // the counter moves, and the completion still reads as good, so the run
      // looks injected and behaves exactly as it would have anyway. Same reason
      // API_ERROR rejects errno 0 below.
      if (rule->wcStatus == IBV_WC_SUCCESS) {
        setLastError(
            "addRule: WC_STATUS needs a failure status; IBV_WC_SUCCESS fires "
            "but leaves the completion indistinguishable from an uninjected one");
        return IB_INJECTION_ERR_ARG;
      }
      break;
    case IB_INJECTION_ACTION_API_ERROR:
      if (!isPostVerb(verb) && !isSetupVerb(verb) &&
          verb != IB_INJECTION_VERB_POLL_CQ) {
        setLastError("addRule: unknown verb for API_ERROR");
        return IB_INJECTION_ERR_ARG;
      }
      // Must be a positive errno. 0 would fire and still look like success at
      // every call site -- shimPollCq returns -errno (0 reads as "no
      // completions") and the post shims return errno directly (0 reads as
      // success). A NEGATIVE value is worse: shimPollCq's -errno becomes
      // positive, which ctran reads as "that many completions were written",
      // handing it uninitialized wc[] entries. Both are silent, which is the
      // worst outcome for an injector.
      if (rule->errnoValue <= 0) {
        setLastError("addRule: API_ERROR needs a positive errnoValue");
        return IB_INJECTION_ERR_ARG;
      }
      break;
    default:
      setLastError("addRule: unknown action");
      return IB_INJECTION_ERR_ARG;
  }

  if (rule->repeat.firstMatch == 0 || rule->repeat.everyNth == 0) {
    setLastError("addRule: repeat firstMatch and everyNth must be >= 1");
    return IB_INJECTION_ERR_ARG;
  }
  if (rule->repeat.unbounded == 0 && rule->repeat.count == 0) {
    setLastError("addRule: bounded repeat needs count >= 1");
    return IB_INJECTION_ERR_ARG;
  }

  // Reject selector fields the decision path for this verb cannot honor. Setup
  // verbs fire before any device or QP exists, a POLL_CQ API_ERROR decision is
  // made before the queue is touched so it has no QP or opcode to match, and
  // post_recv has no opcode to match at all. Accepting these would leave a rule
  // that silently never fires.
  if (isSetupVerb(verb)) {
    if (rule->selector.deviceId != IB_INJECTION_ANY_DEVICE ||
        rule->selector.hwQpNum != IB_INJECTION_ANY_QP ||
        rule->selector.opcode != IB_INJECTION_ANY_OPCODE) {
      setLastError(
          "addRule: setup verbs fire before any device or QP exists, so "
          "selector.deviceId/hwQpNum/opcode must be the wildcards");
      return IB_INJECTION_ERR_ARG;
    }
  } else if (
      verb == IB_INJECTION_VERB_POLL_CQ &&
      action == IB_INJECTION_ACTION_API_ERROR) {
    if (rule->selector.hwQpNum != IB_INJECTION_ANY_QP ||
        rule->selector.opcode != IB_INJECTION_ANY_OPCODE) {
      setLastError(
          "addRule: API_ERROR on poll_cq fails the call before any completion "
          "is read, so selector.hwQpNum/opcode must be the wildcards");
      return IB_INJECTION_ERR_ARG;
    }
  } else if (verb == IB_INJECTION_VERB_POST_RECV) {
    // ibv_recv_wr has no opcode field, so shimPostRecv always decides with
    // ANY_OPCODE and a rule naming one could never match.
    if (rule->selector.opcode != IB_INJECTION_ANY_OPCODE) {
      setLastError(
          "addRule: ibv_recv_wr carries no opcode, so selector.opcode must be "
          "the wildcard for post_recv");
      return IB_INJECTION_ERR_ARG;
    }
  }

  // opcodeDomain must match the namespace the verb reports in. IBV_WR_RDMA_READ
  // is 4 and IBV_WC_RDMA_READ is 2, so a cross-namespace opcode compares
  // against the wrong numbers and matches either nothing or the wrong traffic.
  if (rule->selector.opcode != IB_INJECTION_ANY_OPCODE) {
    const int32_t wantDomain = verb == IB_INJECTION_VERB_POLL_CQ
        ? IB_INJECTION_OPCODE_WC
        : IB_INJECTION_OPCODE_WR;
    if (rule->selector.opcodeDomain != wantDomain) {
      setLastError(
          "addRule: selector.opcodeDomain does not match the verb's namespace "
          "(poll_cq reports ibv_wc_opcode, post_* take ibv_wr_opcode)");
      return IB_INJECTION_ERR_ARG;
    }
  }

  Rule r;
  r.id = nextRuleId_++;
  r.verb = verb;
  r.action = action;
  r.selector = rule->selector;
  r.repeat = rule->repeat;
  r.errnoValue = rule->errnoValue;
  r.wcStatus = rule->wcStatus;
  rules_.push_back(r);
  *ruleId = r.id;
  return IB_INJECTION_OK;
}

IbInjectionStatus Engine::getState(IbInjectionState* state) {
  if (state == nullptr) {
    setLastError("getState: null state");
    return IB_INJECTION_ERR_ARG;
  }

  const uint32_t haveDevices = static_cast<uint32_t>(counters_.size());
  const uint32_t haveRules = static_cast<uint32_t>(rules_.size());
  state->patchedContexts = static_cast<uint32_t>(contexts_.size());
  state->requiredDevices = haveDevices;
  state->requiredRules = haveRules;

  if (state->numDevices < haveDevices || state->numRules < haveRules ||
      (haveDevices > 0 && state->devices == nullptr) ||
      (haveRules > 0 && state->rules == nullptr)) {
    state->numDevices = 0;
    state->numRules = 0;
    setLastError("getState: caller buffers too small");
    return IB_INJECTION_ERR_CAPACITY;
  }

  // Sorted by device id so a test can index positionally.
  std::vector<int32_t> deviceIds;
  deviceIds.reserve(haveDevices);
  for (const auto& [deviceId, unused] : counters_) {
    deviceIds.push_back(deviceId);
  }
  std::sort(deviceIds.begin(), deviceIds.end());

  uint32_t written = 0;
  for (const int32_t deviceId : deviceIds) {
    const auto& c = counters_.at(deviceId);
    auto& out = state->devices[written++];
    out.deviceId = deviceId;
    out.pollCqCalls = c.pollCqCalls;
    out.postSendCalls = c.postSendCalls;
    out.postRecvCalls = c.postRecvCalls;
    out.cqesObserved = c.cqesObserved;
    out.cqesReleased = c.cqesReleased;
    out.cqesStatusMutated = c.cqesStatusMutated;
    out.apiErrorsInjected = c.apiErrorsInjected;
  }
  state->numDevices = written;

  written = 0;
  for (const auto& rule : rules_) {
    auto& out = state->rules[written++];
    out.ruleId = rule.id;
    out.matches = rule.matches;
    out.firings = rule.firings;
  }
  state->numRules = written;

  return IB_INJECTION_OK;
}

void Engine::setLastError(std::string msg) {
  lastError_ = std::move(msg);
}

const char* Engine::lastError() const {
  return lastError_.c_str();
}

} // namespace ibverbx::injection
