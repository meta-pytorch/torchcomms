// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// C++ bridge to the ib_injection shim.
//
// The shim is loaded by ibverbx via dlopen, not linked, so a test reaches its
// control surface the same way: dlopen the path in IBVERBX_IBVERBS_SO and dlsym
// the exported entry points.
//
// Every call throws std::runtime_error on failure, carrying the shim's own
// lastError() text. A test that mis-specifies a rule must fail loudly rather
// than run un-injected and report a green pass -- the whole value of the tool
// is that a rule which never fires is a failed run, not a passing one.

#include <cstdint>
#include <vector>

#include "comms/ctran/ibverbx/ib_injection/IbInjectionApi.h"

namespace ibverbx::injection::testing {

struct Snapshot {
  uint32_t patchedContexts{0};
  std::vector<IbInjectionDeviceState> devices;
  std::vector<IbInjectionRuleState> rules;

  // Counters for one device, or a zeroed entry if the device never appeared.
  // Absence is not an error: a test asserting "device 1 released nothing" is
  // satisfied by a device that has no completions yet.
  IbInjectionDeviceState device(int32_t deviceId) const;
  const IbInjectionRuleState* rule(uint32_t ruleId) const;

  // One counter summed over every device, for the common assertion that cares
  // only whether something happened at all rather than on which NIC.
  uint64_t total(uint64_t IbInjectionDeviceState::* counter) const;
};

// True when IBVERBX_IBVERBS_SO is loadable AND the library behind it exports
// the injection control surface. Both halves matter: that variable legitimately
// names any provider, so a plain load check would pass on the real libibverbs
// and every call below would then throw. Tests should skip rather than fail
// when this is false -- injection is opt-in per target.
bool available();

// Drop all rules and zero all counters, keeping object registrations. Call
// after ctran init so a test measures only its own traffic.
void reset();

// The general entry point. The helpers below cover the common shapes; reach for
// this when a rule needs a field they do not expose.
uint32_t addRule(const IbInjectionRule& rule);

// --- failure injection ---

// Fail a setup verb. firstMatch=3 fails the third call, which for create_qp is
// mid-VC-setup with two QPs already live -- the partial-construction path.
// count is how many times to fire; 0 means unbounded.
uint32_t addSetupError(
    IbInjectionVerb verb,
    int32_t errnoValue,
    uint32_t firstMatch = 1,
    uint32_t count = 1);

// Fail a post. hwQpNum is the HARDWARE qp_num from ctran's accessors
// (getDataQpNums(), getControlQpNum(), ...), not a qpIdx. count=0 is unbounded.
uint32_t addPostError(
    IbInjectionVerb verb,
    int32_t deviceId,
    uint32_t hwQpNum,
    int32_t errnoValue,
    uint32_t firstMatch = 1,
    uint32_t count = 1);

// Deliver a completion with a bad status, exercising ctran's processCqe error
// path rather than its post-failure path. count=0 is unbounded.
uint32_t addWcStatus(
    int32_t deviceId,
    int32_t wcOpcode,
    int32_t wcStatus,
    uint32_t count = 1);

Snapshot getState();

} // namespace ibverbx::injection::testing
