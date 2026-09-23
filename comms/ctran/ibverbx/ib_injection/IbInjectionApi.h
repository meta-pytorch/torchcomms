// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// Control ABI for the ib_injection libibverbs shim.
//
// This header is the ONLY declaration source shared by the shim and the C++
// bridge. It crosses a dlopen boundary, so: C linkage, fixed-width types, no
// STL, no exceptions.
//
// No version handshake, deliberately. The boundary is a dlopen, but it is not a
// distribution boundary: the only caller is ib-injection-control, and Buck
// builds it and the .so from this one header in the same graph, so the two
// cannot disagree about a layout. The surfaces that DO load a packaged .so
// (collperf, farm) have no C++ hook point and never call these entry points.
// Version checks for a skew that cannot happen only cost a field in every
// struct and an error path no test can reach.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wildcards for the selector fields.
#define IB_INJECTION_ANY_DEVICE (-1)
#define IB_INJECTION_ANY_OPCODE (-1)
#define IB_INJECTION_ANY_QP (0u) // qp_num 0 is never a real RC QP

// Result of every control entry point. Negative values are errors; the shim
// never aborts the host process from a control call.
typedef enum IbInjectionStatus {
  IB_INJECTION_OK = 0,
  IB_INJECTION_ERR_ARG = -2, // malformed rule or unknown id
  IB_INJECTION_ERR_CAPACITY = -3, // caller buffer too small; see required*
} IbInjectionStatus;

// Which verb a rule applies to. A rule matches at most one.
typedef enum IbInjectionVerb {
  IB_INJECTION_VERB_POLL_CQ = 0,
  IB_INJECTION_VERB_POST_SEND = 1,
  IB_INJECTION_VERB_POST_RECV = 2,
  // Setup verbs, named individually because a rule targets exactly one and the
  // failure convention differs by return shape (pointer vs int).
  IB_INJECTION_VERB_OPEN_DEVICE = 10,
  IB_INJECTION_VERB_ALLOC_PD = 11,
  IB_INJECTION_VERB_REG_MR = 12,
  IB_INJECTION_VERB_CREATE_CQ = 13,
  IB_INJECTION_VERB_CREATE_QP = 14,
  IB_INJECTION_VERB_MODIFY_QP = 15,
} IbInjectionVerb;

// Failure injection only. Skew -- holding a completion, deferring a post -- is
// a different mechanism (it shifts timing with no error at all) and lands in
// its own diff, which adds CQE_DELAY and CALL_DELAY here.
typedef enum IbInjectionAction {
  // Fail the call. Pointer-returning verbs get nullptr + errno; int-returning
  // verbs return the errno. On post_* the WR list is left untouched.
  IB_INJECTION_ACTION_API_ERROR = 0,
  // Deliver a real CQE with ibv_wc.status overwritten. poll_cq only.
  IB_INJECTION_ACTION_WC_STATUS = 1,
} IbInjectionAction;

// Which opcode namespace `opcode` is drawn from. Not redundant with the verb:
// IBV_WR_RDMA_READ is 4 and IBV_WC_RDMA_READ is 2, so comparing across
// namespaces matches nothing at all.
typedef enum IbInjectionOpcodeDomain {
  IB_INJECTION_OPCODE_WR = 0, // post_send / post_recv
  IB_INJECTION_OPCODE_WC = 1, // poll_cq
} IbInjectionOpcodeDomain;

// Which traffic a rule applies to. Every field is ANDed; wildcards widen.
typedef struct IbInjectionSelector {
  int32_t deviceId; // or IB_INJECTION_ANY_DEVICE
  // HARDWARE qp_num, not a qpIdx. ctran's getDataQpNums() returns
  // provider-assigned numbers; an index is a different thing that happens to
  // share uint32_t, and passing one here would match nothing (real qp_nums are
  // large) and fail silently. Hence the name.
  uint32_t hwQpNum; // or IB_INJECTION_ANY_QP
  int32_t opcode; // or IB_INJECTION_ANY_OPCODE
  int32_t opcodeDomain; // IbInjectionOpcodeDomain
} IbInjectionSelector;

// When a matching event fires, across repeated matches.
//
//   fires iff ordinal >= firstMatch
//        && (ordinal - firstMatch) % everyNth == 0
//        && (unbounded || fired < count)
//
// firstMatch is what makes setup-verb rules useful: create_qp with firstMatch=3
// fails the third QP creation -- mid-VC-setup with two QPs already live, where
// cleanup bugs live. firstMatch=1 only tests the trivial early-exit.
typedef struct IbInjectionRepeat {
  uint32_t firstMatch; // 1 = the first match. 0 is rejected.
  uint32_t everyNth; // 1 = every match. 0 is rejected.
  uint32_t count; // ignored when unbounded
  uint8_t unbounded;
} IbInjectionRepeat;

typedef struct IbInjectionRule {
  int32_t verb; // IbInjectionVerb
  int32_t action; // IbInjectionAction
  IbInjectionSelector selector;
  IbInjectionRepeat repeat;

  // API_ERROR
  int32_t errnoValue;

  // WC_STATUS: the ibv_wc_status to write.
  int32_t wcStatus;
} IbInjectionRule;

// Per-device counters. "released" means handed to the caller of poll_cq, which
// is the only count a test can correlate with host-side state; "observed" is
// what the real provider produced.
typedef struct IbInjectionDeviceState {
  int32_t deviceId;
  uint64_t pollCqCalls;
  uint64_t postSendCalls;
  uint64_t postRecvCalls;
  uint64_t cqesObserved;
  uint64_t cqesReleased;
  uint64_t cqesStatusMutated;
  uint64_t apiErrorsInjected;
} IbInjectionDeviceState;

typedef struct IbInjectionRuleState {
  uint32_t ruleId;
  uint64_t matches;
  uint64_t firings;
} IbInjectionRuleState;

typedef struct IbInjectionState {
  // Number of live contexts whose ops vtable the shim patched. Zero after a run
  // that was supposed to inject is the signature of a vacuous run.
  uint32_t patchedContexts;
  uint32_t numDevices; // entries written to devices[]
  uint32_t numRules; // entries written to rules[]
  uint32_t requiredDevices; // entries available, for the retry
  uint32_t requiredRules;
  IbInjectionDeviceState* devices; // caller-owned
  IbInjectionRuleState* rules; // caller-owned
} IbInjectionState;

// --- Control entry points, all exported under IB_INJECTION_1.0 ---

// Drop all rules and zero all counters. Object registrations survive, so a test
// can reset after init and measure only its own traffic. Rule ids are never
// reused across a reset, so a ruleId held from before one misses in
// ibInjectionGetState() rather than naming whatever rule took its place.
IbInjectionStatus ibInjectionReset(void);

// Install a rule. On success *ruleId identifies it for release/readback.
IbInjectionStatus ibInjectionAddRule(
    const IbInjectionRule* rule,
    uint32_t* ruleId);

// Snapshot counters. Set devices/rules and numDevices/numRules to the caller's
// buffer capacity; on IB_INJECTION_ERR_CAPACITY, required* report the sizes
// needed. numDevices/numRules are overwritten with the counts actually written.
IbInjectionStatus ibInjectionGetState(IbInjectionState* state);

// Human-readable reason for the last failing control call in this process.
// Never null. The returned pointer is per-thread, but the message it carries is
// not: a thread can read a reason another thread's failing call recorded.
const char* ibInjectionLastError(void);

#ifdef __cplusplus
} // extern "C"
#endif
