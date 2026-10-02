// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/ib_injection/InjectionEngine.h"

#include <fmt/core.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
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

// --- spec parsing ---

constexpr const char* kSpecEnv = "IB_INJECTION_SPEC";
constexpr const char* kRankEnv = "RANK";

struct NamedValue {
  std::string_view name;
  int32_t value;
};

constexpr NamedValue kVerbs[] = {
    {"poll_cq", IB_INJECTION_VERB_POLL_CQ},
    {"post_send", IB_INJECTION_VERB_POST_SEND},
    {"post_recv", IB_INJECTION_VERB_POST_RECV},
    {"open_device", IB_INJECTION_VERB_OPEN_DEVICE},
    {"alloc_pd", IB_INJECTION_VERB_ALLOC_PD},
    {"reg_mr", IB_INJECTION_VERB_REG_MR},
    {"create_cq", IB_INJECTION_VERB_CREATE_CQ},
    {"create_qp", IB_INJECTION_VERB_CREATE_QP},
    {"modify_qp", IB_INJECTION_VERB_MODIFY_QP},
};

constexpr NamedValue kActions[] = {
    {"api_error", IB_INJECTION_ACTION_API_ERROR},
    {"wc_status", IB_INJECTION_ACTION_WC_STATUS},
};

// The errnos a verb failure plausibly reports. Deliberately short: an unlisted
// name is rejected rather than guessed, and a bare integer always works.
constexpr NamedValue kErrnos[] = {
    {"EPERM", EPERM},
    {"EIO", EIO},
    {"EAGAIN", EAGAIN},
    {"ENOMEM", ENOMEM},
    {"EFAULT", EFAULT},
    {"EBUSY", EBUSY},
    {"ENODEV", ENODEV},
    {"EINVAL", EINVAL},
    {"ENOSPC", ENOSPC},
    {"EOPNOTSUPP", EOPNOTSUPP},
    {"ECONNRESET", ECONNRESET},
    {"ETIMEDOUT", ETIMEDOUT},
};

// Full names here, unlike opcode's short forms: ibv_wc_status has one
// namespace, so there is nothing to disambiguate, and the raw numbers are
// meaningless to a reader of a job's env.
constexpr NamedValue kWcStatuses[] = {
    {"IBV_WC_LOC_LEN_ERR", IBV_WC_LOC_LEN_ERR},
    {"IBV_WC_LOC_QP_OP_ERR", IBV_WC_LOC_QP_OP_ERR},
    {"IBV_WC_LOC_PROT_ERR", IBV_WC_LOC_PROT_ERR},
    {"IBV_WC_WR_FLUSH_ERR", IBV_WC_WR_FLUSH_ERR},
    {"IBV_WC_BAD_RESP_ERR", IBV_WC_BAD_RESP_ERR},
    {"IBV_WC_LOC_ACCESS_ERR", IBV_WC_LOC_ACCESS_ERR},
    {"IBV_WC_REM_INV_REQ_ERR", IBV_WC_REM_INV_REQ_ERR},
    {"IBV_WC_REM_ACCESS_ERR", IBV_WC_REM_ACCESS_ERR},
    {"IBV_WC_REM_OP_ERR", IBV_WC_REM_OP_ERR},
    {"IBV_WC_RETRY_EXC_ERR", IBV_WC_RETRY_EXC_ERR},
    {"IBV_WC_RNR_RETRY_EXC_ERR", IBV_WC_RNR_RETRY_EXC_ERR},
    {"IBV_WC_REM_ABORT_ERR", IBV_WC_REM_ABORT_ERR},
    {"IBV_WC_FATAL_ERR", IBV_WC_FATAL_ERR},
    {"IBV_WC_RESP_TIMEOUT_ERR", IBV_WC_RESP_TIMEOUT_ERR},
    {"IBV_WC_GENERAL_ERR", IBV_WC_GENERAL_ERR},
};

// Short opcode names, resolved against the namespace the verb reports in. The
// two tables share names on purpose -- that is the whole point of deriving the
// domain from `fn` rather than asking a spec author to name it.
constexpr NamedValue kWcOpcodes[] = {
    {"SEND", IBV_WC_SEND},
    {"RDMA_WRITE", IBV_WC_RDMA_WRITE},
    {"RDMA_READ", IBV_WC_RDMA_READ},
    {"COMP_SWAP", IBV_WC_COMP_SWAP},
    {"FETCH_ADD", IBV_WC_FETCH_ADD},
    {"RECV", IBV_WC_RECV},
    {"RECV_RDMA_WITH_IMM", IBV_WC_RECV_RDMA_WITH_IMM},
};

constexpr NamedValue kWrOpcodes[] = {
    {"RDMA_WRITE", IBV_WR_RDMA_WRITE},
    {"RDMA_WRITE_WITH_IMM", IBV_WR_RDMA_WRITE_WITH_IMM},
    {"SEND", IBV_WR_SEND},
    {"SEND_WITH_IMM", IBV_WR_SEND_WITH_IMM},
    {"RDMA_READ", IBV_WR_RDMA_READ},
    {"ATOMIC_CMP_AND_SWP", IBV_WR_ATOMIC_CMP_AND_SWP},
    {"ATOMIC_FETCH_AND_ADD", IBV_WR_ATOMIC_FETCH_AND_ADD},
};

template <size_t N>
bool lookupName(
    const NamedValue (&table)[N],
    std::string_view name,
    int32_t* out) {
  for (const auto& entry : table) {
    if (entry.name == name) {
      *out = entry.value;
      return true;
    }
  }
  return false;
}

template <size_t N>
bool tableHasValue(const NamedValue (&table)[N], int32_t value) {
  for (const auto& entry : table) {
    if (entry.value == value) {
      return true;
    }
  }
  return false;
}

template <size_t N>
std::string nameList(const NamedValue (&table)[N]) {
  std::string out;
  for (const auto& entry : table) {
    if (!out.empty()) {
      out += " ";
    }
    out.append(entry.name);
  }
  return out;
}

std::string_view trim(std::string_view s) {
  const auto notSpace = [](char c) {
    return c != ' ' && c != '\t' && c != '\n';
  };
  while (!s.empty() && !notSpace(s.front())) {
    s.remove_prefix(1);
  }
  while (!s.empty() && !notSpace(s.back())) {
    s.remove_suffix(1);
  }
  return s;
}

std::vector<std::string_view> split(std::string_view s, char sep) {
  std::vector<std::string_view> parts;
  size_t start = 0;
  while (true) {
    const size_t at = s.find(sep, start);
    if (at == std::string_view::npos) {
      parts.push_back(trim(s.substr(start)));
      return parts;
    }
    parts.push_back(trim(s.substr(start, at - start)));
    start = at + 1;
  }
}

bool parseU32(std::string_view s, uint32_t* out) {
  if (s.empty()) {
    return false;
  }
  const auto res = std::from_chars(s.data(), s.data() + s.size(), *out);
  return res.ec == std::errc{} && res.ptr == s.data() + s.size();
}

bool parseI32(std::string_view s, int32_t* out) {
  if (s.empty()) {
    return false;
  }
  const auto res = std::from_chars(s.data(), s.data() + s.size(), *out);
  return res.ec == std::errc{} && res.ptr == s.data() + s.size();
}

// One rule's worth of spec, before the verb-dependent fields are resolved.
struct RuleFields {
  std::string_view verb;
  std::string_view action;
  std::string_view errnoValue;
  std::string_view status;
  std::string_view rank;
  std::string_view dev;
  std::string_view qp;
  std::string_view opcode;
  std::string_view first;
  std::string_view every;
  std::string_view count;
};

// The one place a spec key is named. Both the lookup in assignField and the
// "unknown key" message read this table, so adding a key cannot leave the two
// disagreeing.
struct FieldSlot {
  std::string_view name;
  std::string_view RuleFields::* slot;
};

constexpr FieldSlot kFieldSlots[] = {
    {"fn", &RuleFields::verb},
    {"action", &RuleFields::action},
    {"errno", &RuleFields::errnoValue},
    {"status", &RuleFields::status},
    {"rank", &RuleFields::rank},
    {"dev", &RuleFields::dev},
    {"qp", &RuleFields::qp},
    {"opcode", &RuleFields::opcode},
    {"first", &RuleFields::first},
    {"every", &RuleFields::every},
    {"count", &RuleFields::count},
};

std::string fieldSlotNames() {
  std::string out;
  for (const auto& entry : kFieldSlots) {
    if (!out.empty()) {
      out += " ";
    }
    out.append(entry.name);
  }
  return out;
}

// `field` is the whole "key=value" token, quoted verbatim in any error so the
// message points at what the operator actually typed.
bool assignField(RuleFields& f, std::string_view field, std::string* error) {
  const size_t eq = field.find('=');
  if (eq == std::string_view::npos) {
    *error = fmt::format("field '{}' is not key=value", field);
    return false;
  }
  const std::string_view key = trim(field.substr(0, eq));
  const std::string_view value = trim(field.substr(eq + 1));
  if (value.empty()) {
    *error = fmt::format("field '{}' has an empty value", field);
    return false;
  }

  std::string_view* slot = nullptr;
  for (const auto& entry : kFieldSlots) {
    if (key == entry.name) {
      slot = &(f.*entry.slot);
      break;
    }
  }
  if (slot == nullptr) {
    // Name the two planned-but-absent mechanisms explicitly. Both appear in the
    // README's planned shape, so an operator who read it will reach for them,
    // and "unknown key" alone would read like a typo rather than a feature that
    // has not landed.
    if (key == "after_dev" || key == "after_count" || key == "manual") {
      *error = fmt::format(
          "key '{}' belongs to skew injection (cq_gate/call_delay), which the "
          "engine does not implement yet",
          key);
    } else {
      *error = fmt::format(
          "unknown key '{}' in field '{}'; expected one of {}",
          key,
          field,
          fieldSlotNames());
    }
    return false;
  }

  if (!slot->empty()) {
    *error = fmt::format("key '{}' appears twice in one rule", key);
    return false;
  }
  *slot = value;
  return true;
}

// Resolve one rule's fields into an IbInjectionRule. Only shape is checked
// here; the sparse verb/action matrix, the positive-errno requirement and the
// wildcard-only selector constraints all belong to addRule, which every front
// end shares.
bool buildRule(const RuleFields& f, IbInjectionRule* rule, std::string* error) {
  if (f.verb.empty()) {
    *error = "rule has no fn=; expected one of " + nameList(kVerbs);
    return false;
  }
  if (f.action.empty()) {
    *error = "rule has no action=; expected one of " + nameList(kActions);
    return false;
  }

  int32_t verb = 0;
  if (!lookupName(kVerbs, f.verb, &verb)) {
    *error = fmt::format(
        "unknown fn '{}'; expected one of {}", f.verb, nameList(kVerbs));
    return false;
  }
  int32_t action = 0;
  if (!lookupName(kActions, f.action, &action)) {
    // cq_gate and call_delay are the planned skew actions; say so rather than
    // letting them read as a misspelling.
    if (f.action == "cq_gate" || f.action == "call_delay") {
      *error = fmt::format(
          "action '{}' is skew injection, which the engine does not implement "
          "yet; today's actions are {}",
          f.action,
          nameList(kActions));
    } else {
      *error = fmt::format(
          "unknown action '{}'; expected one of {}",
          f.action,
          nameList(kActions));
    }
    return false;
  }

  *rule = {};
  rule->verb = verb;
  rule->action = action;
  rule->selector.deviceId = IB_INJECTION_ANY_DEVICE;
  rule->selector.hwQpNum = IB_INJECTION_ANY_QP;
  rule->selector.opcode = IB_INJECTION_ANY_OPCODE;
  rule->repeat.firstMatch = 1;
  rule->repeat.everyNth = 1;
  rule->repeat.count = 1;

  if (action == IB_INJECTION_ACTION_API_ERROR) {
    if (f.errnoValue.empty()) {
      *error = "action=api_error needs errno=";
      return false;
    }
    if (!lookupName(kErrnos, f.errnoValue, &rule->errnoValue) &&
        !parseI32(f.errnoValue, &rule->errnoValue)) {
      *error = fmt::format(
          "errno '{}' is neither an integer nor one of {}",
          f.errnoValue,
          nameList(kErrnos));
      return false;
    }
    if (!f.status.empty()) {
      *error = "status= applies to action=wc_status, not api_error";
      return false;
    }
  } else {
    if (f.status.empty()) {
      *error = "action=wc_status needs status=";
      return false;
    }
    if (!lookupName(kWcStatuses, f.status, &rule->wcStatus) &&
        !parseI32(f.status, &rule->wcStatus)) {
      *error = fmt::format(
          "status '{}' is neither an integer nor one of {}",
          f.status,
          nameList(kWcStatuses));
      return false;
    }
    if (!f.errnoValue.empty()) {
      *error = "errno= applies to action=api_error, not wc_status";
      return false;
    }
  }

  if (!f.dev.empty() && f.dev != "*") {
    if (!parseI32(f.dev, &rule->selector.deviceId)) {
      *error = fmt::format("dev '{}' is not an integer or *", f.dev);
      return false;
    }
    // A value equal to the wildcard sentinel would be indistinguishable from
    // having named no device at all, so a spec asking for one specific thing
    // would silently match everything. Say so rather than widen the rule.
    if (rule->selector.deviceId == IB_INJECTION_ANY_DEVICE) {
      *error = fmt::format(
          "dev {} is the any-device wildcard, not a device; omit dev or pass "
          "dev=* to match every device",
          f.dev);
      return false;
    }
    // Device ids are handed out from 0 upward as CQs appear, so nothing
    // negative can ever be assigned one.
    if (rule->selector.deviceId < 0) {
      *error = fmt::format(
          "dev {} is negative; device ids start at 0, so this rule could never "
          "fire",
          f.dev);
      return false;
    }
  }

  if (!f.qp.empty() && f.qp != "*") {
    // The role form the README plans -- qp=data:3 -- needs the engine to tag
    // QPs by role at registration, which it does not do. Reject it by name so
    // an operator who tries it learns that, rather than seeing "not a number".
    if (f.qp.find(':') != std::string_view::npos) {
      *error = fmt::format(
          "qp '{}' uses the role form, which is not implemented; pass a "
          "hardware qp_num, which a spec generally cannot know -- prefer a "
          "C++ test for QP-specific rules",
          f.qp);
      return false;
    }
    if (!parseU32(f.qp, &rule->selector.hwQpNum)) {
      *error = fmt::format("qp '{}' is not a qp_num or *", f.qp);
      return false;
    }
    // Same trap as dev above: qp_num 0 is the any-QP wildcard (and never a real
    // RC QP), so accepting it would turn a QP-specific rule into a global one.
    if (rule->selector.hwQpNum == IB_INJECTION_ANY_QP) {
      *error = fmt::format(
          "qp {} is the any-QP wildcard and never a real RC qp_num; omit qp or "
          "pass qp=* to match every QP",
          f.qp);
      return false;
    }
  }

  if (!f.opcode.empty() && f.opcode != "*") {
    // Derived, never spelled: the verb already says which namespace it reports
    // in, and addRule rejects a mismatch, so resolving here means a spec cannot
    // express the cross-namespace comparison at all.
    const bool wc = verb == IB_INJECTION_VERB_POLL_CQ;
    rule->selector.opcodeDomain =
        wc ? IB_INJECTION_OPCODE_WC : IB_INJECTION_OPCODE_WR;
    const bool found = wc
        ? lookupName(kWcOpcodes, f.opcode, &rule->selector.opcode)
        : lookupName(kWrOpcodes, f.opcode, &rule->selector.opcode);
    // The numeric form has to land on the same set the names do. These tables
    // list only the opcodes ctran and ctranx actually issue, so a number
    // outside them -- opcode=99, or the any-opcode sentinel -- describes
    // traffic that never appears and yields a rule that could never fire.
    if (!found) {
      const bool numeric = parseI32(f.opcode, &rule->selector.opcode) &&
          (wc ? tableHasValue(kWcOpcodes, rule->selector.opcode)
              : tableHasValue(kWrOpcodes, rule->selector.opcode));
      if (!numeric) {
        *error = fmt::format(
            "unknown opcode '{}' for fn={}; expected one of {}",
            f.opcode,
            f.verb,
            wc ? nameList(kWcOpcodes) : nameList(kWrOpcodes));
        return false;
      }
    }
  }

  if (!f.first.empty() && !parseU32(f.first, &rule->repeat.firstMatch)) {
    *error = fmt::format("first '{}' is not a positive integer", f.first);
    return false;
  }
  if (!f.every.empty() && !parseU32(f.every, &rule->repeat.everyNth)) {
    *error = fmt::format("every '{}' is not a positive integer", f.every);
    return false;
  }
  if (!f.count.empty()) {
    if (f.count == "inf") {
      rule->repeat.unbounded = 1;
      rule->repeat.count = 0;
    } else if (!parseU32(f.count, &rule->repeat.count)) {
      *error =
          fmt::format("count '{}' is not a positive integer or inf", f.count);
      return false;
    }
  }
  return true;
}

} // namespace

SpecParseResult parseInjectionSpec(
    const std::string& spec,
    std::optional<int32_t> rank) {
  SpecParseResult result;
  uint32_t ordinal = 0;
  for (const std::string_view chunk : split(spec, ';')) {
    if (chunk.empty()) {
      continue; // tolerate a trailing or doubled ';'
    }
    ++ordinal;

    RuleFields fields;
    std::string error;
    for (const std::string_view field : split(chunk, ',')) {
      if (field.empty()) {
        continue;
      }
      if (!assignField(fields, field, &error)) {
        result.error = fmt::format("rule {}: {}", ordinal, error);
        return result;
      }
    }

    bool armsThisRank = true;
    if (!fields.rank.empty() && fields.rank != "*") {
      int32_t want = 0;
      if (!parseI32(fields.rank, &want)) {
        result.error = fmt::format(
            "rule {}: rank '{}' is not an integer or *", ordinal, fields.rank);
        return result;
      }
      if (!rank.has_value()) {
        // Refuse to guess. Assuming rank 0 would make rank=1 arm nothing and
        // rank=0 arm every process, both silently -- the one outcome an
        // injector must never produce.
        result.error = fmt::format(
            "rule {}: rank={} needs $RANK, which is unset or not an integer; "
            "refusing to assume a rank, since guessing arms either nothing or "
            "every process with no way to tell",
            ordinal,
            want);
        return result;
      }
      armsThisRank = want == *rank;
    }

    // Validated whatever rank it names, so a typo in a rule addressed elsewhere
    // still fails here. Skipping validation with the arming would let a spec
    // whose rules all target ranks outside this job be checked by nobody: it
    // would arm nothing, anywhere, and report success on every rank.
    IbInjectionRule rule;
    if (!buildRule(fields, &rule, &error)) {
      result.error = fmt::format("rule {}: {}", ordinal, error);
      return result;
    }

    if (armsThisRank) {
      result.rules.push_back(rule);
    } else {
      ++result.skippedForRank;
    }
  }
  return result;
}

Engine::Engine() {
  applyEnvSpec();
}

void Engine::applyEnvSpec() {
  const char* spec = getenv(kSpecEnv);
  if (spec == nullptr || *spec == '\0') {
    return;
  }

  // Left empty when the environment does not usably name a rank, including an
  // unparseable value: only a rule that actually says rank= needs one, and
  // parseInjectionSpec is where that is known. Failing here instead would take
  // down a process over a variable its spec never reads.
  std::optional<int32_t> rank;
  if (const char* r = getenv(kRankEnv); r != nullptr && *r != '\0') {
    int32_t value = 0;
    if (parseI32(r, &value)) {
      rank = value;
    }
  }

  const SpecParseResult parsed = parseInjectionSpec(spec, rank);
  if (!parsed.error.empty()) {
    fmt::print(
        stderr,
        "ib_injection: {} is malformed: {}\n  spec: {}\n",
        kSpecEnv,
        parsed.error,
        spec);
    abort();
  }

  // A spec that was set but yielded nothing armed nothing, which is the silent
  // no-op this whole mechanism exists to remove: a stray ';' or an
  // all-whitespace value would otherwise run completely uninjected and green.
  // Rules addressed to other ranks are a different case and counted separately.
  if (parsed.rules.empty() && parsed.skippedForRank == 0) {
    fmt::print(
        stderr,
        "ib_injection: {} is set but describes no rules, so nothing would be "
        "injected\n  spec: {}\n",
        kSpecEnv,
        spec);
    abort();
  }

  for (const IbInjectionRule& rule : parsed.rules) {
    uint32_t ruleId = 0;
    if (addRule(&rule, &ruleId) != IB_INJECTION_OK) {
      fmt::print(
          stderr,
          "ib_injection: {} rule rejected: {}\n  spec: {}\n",
          kSpecEnv,
          lastError(),
          spec);
      abort();
    }
  }

  // Always printed, because on the env-driven surfaces there is no assertion
  // and this line plus the per-rule FIRED lines are the entire record of what
  // was armed.
  fmt::print(
      stderr,
      "ib_injection: rank {} armed {} rule(s) from {} ({} addressed to other ranks)\n",
      rank.has_value() ? std::to_string(*rank)
                       : fmt::format("<no {}>", kRankEnv),
      parsed.rules.size(),
      kSpecEnv,
      parsed.skippedForRank);
}

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
  // The whole record that a rule acted, on the surfaces with no assertion:
  // their only signal is the job's own output. Once per rule, since this sits
  // on the hot path; a C++ test wanting exact counts reads getState() instead.
  if (rule.firings == 1) {
    fmt::print(
        stderr,
        "ib_injection: rule {} FIRED (verb {} action {} dev {} qp {} opcode {})\n",
        rule.id,
        static_cast<int>(rule.verb),
        static_cast<int>(rule.action),
        deviceId,
        qpNum,
        opcode);
  }
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
