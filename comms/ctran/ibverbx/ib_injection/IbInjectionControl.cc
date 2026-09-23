// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/ib_injection/IbInjectionControl.h"

#include <dlfcn.h>
#include <fmt/format.h>
#include <cstdlib>
#include <stdexcept>

namespace ibverbx::injection::testing {

namespace {

// Resolved on first use and kept for the process lifetime: the shim must not be
// unloaded while ibvSymbols and the patched contexts still hold pointers into
// it.
void* shimHandle() {
  static void* handle = []() -> void* {
    const char* path = getenv("IBVERBX_IBVERBS_SO");
    if (path == nullptr || *path == '\0') {
      return nullptr;
    }
    // RTLD_NOLOAD first: ibverbx may already have dlopen'd this exact path, and
    // we want a reference to the SAME instance. A second copy would carry its
    // own Engine, so every assertion would read zeros while the real one did
    // the work.
    void* h = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (h == nullptr) {
      // Not loaded yet (no ibvInit() so far). Loading it here is correct:
      // ibverbx's later dlopen of the same path returns this handle.
      h = dlopen(path, RTLD_LAZY);
    }
    return h;
  }();
  return handle;
}

template <typename Fn>
Fn sym(const char* name) {
  void* handle = shimHandle();
  if (handle == nullptr) {
    throw std::runtime_error(
        "ib_injection: IBVERBX_IBVERBS_SO is unset or not loadable");
  }
  auto fn = reinterpret_cast<Fn>(dlsym(handle, name));
  if (fn == nullptr) {
    throw std::runtime_error(
        fmt::format(
            "ib_injection: {} not found in IBVERBX_IBVERBS_SO -- is it pointing at "
            "the real libibverbs rather than the injection shim?",
            name));
  }
  return fn;
}

const char* lastError() {
  auto fn = sym<const char* (*)()>("ibInjectionLastError");
  return fn();
}

void check(IbInjectionStatus status, const char* what) {
  if (status == IB_INJECTION_OK) {
    return;
  }
  throw std::runtime_error(
      fmt::format(
          "ib_injection: {} failed ({}): {}",
          what,
          static_cast<int>(status),
          lastError()));
}

// Every rule shares these defaults; each helper overrides only what it is
// about.
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

void setRepeat(IbInjectionRule& r, uint32_t firstMatch, uint32_t count) {
  r.repeat.firstMatch = firstMatch;
  r.repeat.everyNth = 1;
  if (count == 0) {
    r.repeat.unbounded = 1;
  } else {
    r.repeat.unbounded = 0;
    r.repeat.count = count;
  }
}

} // namespace

IbInjectionDeviceState Snapshot::device(int32_t deviceId) const {
  for (const auto& d : devices) {
    if (d.deviceId == deviceId) {
      return d;
    }
  }
  IbInjectionDeviceState empty{};
  empty.deviceId = deviceId;
  return empty;
}

const IbInjectionRuleState* Snapshot::rule(uint32_t ruleId) const {
  for (const auto& r : rules) {
    if (r.ruleId == ruleId) {
      return &r;
    }
  }
  return nullptr;
}

uint64_t Snapshot::total(uint64_t IbInjectionDeviceState::* counter) const {
  uint64_t sum = 0;
  for (const auto& d : devices) {
    sum += d.*counter;
  }
  return sum;
}

bool available() {
  void* handle = shimHandle();
  // A loadable path is not enough to answer this. IBVERBX_IBVERBS_SO is
  // deliberately allowed to name any provider -- the real libibverbs, a mock --
  // so a handle that opened says nothing about whether the shim is behind it.
  // Probing for a control symbol is what makes "available" mean the same thing
  // as "the calls below will resolve"; without it a guarded fixture clears the
  // guard and then throws out of its first control call instead of skipping.
  return handle != nullptr && dlsym(handle, "ibInjectionReset") != nullptr;
}

void reset() {
  auto fn = sym<IbInjectionStatus (*)()>("ibInjectionReset");
  check(fn(), "reset");
}

uint32_t addRule(const IbInjectionRule& rule) {
  uint32_t ruleId = 0;
  auto fn = sym<IbInjectionStatus (*)(const IbInjectionRule*, uint32_t*)>(
      "ibInjectionAddRule");
  check(fn(&rule, &ruleId), "addRule");
  return ruleId;
}

uint32_t addSetupError(
    IbInjectionVerb verb,
    int32_t errnoValue,
    uint32_t firstMatch,
    uint32_t count) {
  auto r = baseRule(verb, IB_INJECTION_ACTION_API_ERROR);
  r.errnoValue = errnoValue;
  setRepeat(r, firstMatch, count);
  return addRule(r);
}

uint32_t addPostError(
    IbInjectionVerb verb,
    int32_t deviceId,
    uint32_t hwQpNum,
    int32_t errnoValue,
    uint32_t firstMatch,
    uint32_t count) {
  auto r = baseRule(verb, IB_INJECTION_ACTION_API_ERROR);
  r.selector.deviceId = deviceId;
  r.selector.hwQpNum = hwQpNum;
  r.selector.opcodeDomain = IB_INJECTION_OPCODE_WR;
  r.errnoValue = errnoValue;
  setRepeat(r, firstMatch, count);
  return addRule(r);
}

uint32_t addWcStatus(
    int32_t deviceId,
    int32_t wcOpcode,
    int32_t wcStatus,
    uint32_t count) {
  auto r = baseRule(IB_INJECTION_VERB_POLL_CQ, IB_INJECTION_ACTION_WC_STATUS);
  r.selector.deviceId = deviceId;
  r.selector.opcode = wcOpcode;
  r.wcStatus = wcStatus;
  setRepeat(r, /*firstMatch=*/1, count);
  return addRule(r);
}
Snapshot getState() {
  auto fn =
      sym<IbInjectionStatus (*)(IbInjectionState*)>("ibInjectionGetState");

  // Probe for the sizes, then fetch. Retried rather than done once, because
  // ctran can create a device or a rule between the two calls -- the fetch
  // would then come back ERR_CAPACITY, and treating that as fatal would make a
  // concurrent registration look like a test failure. A handful of attempts is
  // plenty: the count only grows during setup.
  Snapshot out;
  IbInjectionState state{};
  constexpr int kMaxAttempts = 8;
  for (int attempt = 0; attempt < kMaxAttempts; attempt++) {
    IbInjectionState probe{};
    const IbInjectionStatus probeStatus = fn(&probe);
    if (probeStatus != IB_INJECTION_OK &&
        probeStatus != IB_INJECTION_ERR_CAPACITY) {
      check(probeStatus, "getState (probe)");
    }

    out.devices.assign(probe.requiredDevices, IbInjectionDeviceState{});
    out.rules.assign(probe.requiredRules, IbInjectionRuleState{});

    state = IbInjectionState{};
    state.numDevices = probe.requiredDevices;
    state.numRules = probe.requiredRules;
    state.devices = out.devices.empty() ? nullptr : out.devices.data();
    state.rules = out.rules.empty() ? nullptr : out.rules.data();

    const IbInjectionStatus status = fn(&state);
    if (status == IB_INJECTION_OK) {
      break;
    }
    if (status != IB_INJECTION_ERR_CAPACITY) {
      check(status, "getState");
    }
    if (attempt + 1 == kMaxAttempts) {
      check(status, "getState (capacity kept growing)");
    }
  }

  out.devices.resize(state.numDevices);
  out.rules.resize(state.numRules);
  out.patchedContexts = state.patchedContexts;
  return out;
}

} // namespace ibverbx::injection::testing
