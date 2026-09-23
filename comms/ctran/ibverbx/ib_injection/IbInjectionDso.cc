// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// The ib_injection libibverbs shim: soname libibverbs.so, selected by
// IBVERBX_IBVERBS_SO.
//
// It DELEGATES. Every verb forwards to a real provider
// (IB_INJECTION_REAL_IBVERBS_SO, default libibverbs.so.1), so a run against a
// real NIC behaves identically except where a rule fires. Nothing is emulated.
//
// It serves BOTH verb sets. ibverbx points its ibvhandle and mlx5dvhandle
// dlopen calls at the same file, and nothing calls an mlx5dv_* function
// directly -- every call site goes through ibvSymbols.mlx5dv_internal_*. So one
// library covers ibv_* and mlx5dv_*, with exactly one Engine.
//
// Two routes reach it, needing opposite things:
//
//   Setup verbs are resolved BY NAME, with dlvsym on this library's handle.
//   They must be exported at the right version node -- IbverbxSymbols.def
//   supplies the body, version.script the tag, and a missing entry fails
//   ibvInit().
//
//   The hot path is reached BY POINTER: ibverbx reads cq_->context->ops.poll_cq
//   (IbvCq.h) and qp_->context->ops.post_send (IbvQp.h), never calling
//   ibv_poll_cq by name. So ibv_open_device delegates, then overwrites those
//   ops in the returned context. Those shims are deliberately not exported.
//
// req_notify_cq is deliberately NOT patched -- see the note on shimPollCq.
//
// Because every verb is exported unconditionally, ibverbx can no longer detect
// a verb the real provider lacks -- the shim's symbol is always there to find.
// So IbverbxSymbols.def marks each verb required or optional, mirroring how
// ibverbx loads it, and the forwarder answers on its behalf: abort for a
// required verb, a caller-visible failure for an optional one.
//
// Aborting is a DELIBERATE divergence from ibverbx, which reports a missing
// required symbol as a recoverable error out of ibvInit(). This library is only
// ever loaded because someone pointed IBVERBX_IBVERBS_SO at it, so a broken
// real provider means the run cannot do what it was asked to do. Degrading
// quietly would produce a green run that injected nothing, which is the one
// outcome an injection tool must never produce.

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <cstdio>
#include <mutex>
#include <string>
#include <type_traits>

#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/ib_injection/IbInjectionApi.h"
#include "comms/ctran/ibverbx/ib_injection/InjectionEngine.h"

// Pulled in after the core headers: the table names ibv_* and mlx5dv_* types.
#include "comms/ctran/ibverbx/ib_injection/IbverbxSymbols.def"

using ibverbx::ibv_async_event;
using ibverbx::ibv_comp_channel;
using ibverbx::ibv_context;
using ibverbx::ibv_cq;
using ibverbx::ibv_device;
using ibverbx::ibv_device_attr;
using ibverbx::ibv_ece;
using ibverbx::ibv_event_type;
using ibverbx::ibv_fork_status;
using ibverbx::ibv_gid;
using ibverbx::ibv_mr;
using ibverbx::ibv_parent_domain_init_attr;
using ibverbx::ibv_pd;
using ibverbx::ibv_port_attr;
using ibverbx::ibv_qp;
using ibverbx::ibv_qp_attr;
using ibverbx::ibv_qp_init_attr;
using ibverbx::ibv_recv_wr;
using ibverbx::ibv_send_wr;
using ibverbx::ibv_wc;
using ibverbx::mlx5dv_context;
using ibverbx::mlx5dv_obj;
using ibverbx::injection::CallDecision;
using ibverbx::injection::Engine;
using ibverbx::injection::SavedOps;

namespace {

// Whether ibverbx treats the verb as load-bearing (LOAD_IBVERBS_SYM, which
// fails ibvInit() without it) or optional (LOAD_IBVERBS_SYM_WARN_ONLY, which
// leaves the pointer null and lets callers degrade). The shim exports every
// verb unconditionally, so it cannot signal "absent" by being absent; it has to
// reproduce each outcome explicitly. IbverbxSymbols.def carries the column.
enum VerbRequirement { kRequired, kOptional };

// What an optional verb returns when the real provider does not have it.
//
// A plain (ret)0 is wrong for the int-returning verbs: 0 is success, so a
// caller like Mlx5dv::initObj would treat an untouched output struct as
// populated and walk it. Every case below is therefore chosen to be a value the
// caller already treats as failure -- except void, which has no channel to
// report one. The only void verb here is ibv_ack_cq_events, where skipping an
// ack for an event the provider never delivered is harmless; a future void verb
// needs its own audit.
//
// EOPNOTSUPP rather than ENOSYS because it is what ibverbx's own null-symbol
// paths report (Mlx5dv::queryDevice, IbvWrap's dmabuf probe), and because
// IBV_INT_CHECK_RET_ERRNO_OPTIONAL maps it to "not supported" without an error
// -- the same graceful outcome a null pointer would have produced.
template <typename Ret>
Ret missingVerbResult() {
  errno = EOPNOTSUPP;
  if constexpr (std::is_void_v<Ret>) {
    return;
  } else if constexpr (std::is_pointer_v<Ret>) {
    return nullptr;
  } else if constexpr (std::is_same_v<Ret, bool>) {
    return false;
  } else if constexpr (std::is_same_v<Ret, ibv_fork_status>) {
    // The enum has no error state; claim the weaker of the two.
    return ibverbx::IBV_FORK_DISABLED;
  } else {
    // errno-style int: these verbs return the error directly, not -1.
    return static_cast<Ret>(EOPNOTSUPP);
  }
}

// dlerror() is allowed to return null, and passing null to %s is undefined.
// Every caller here is already on an error path, so substitute a placeholder
// rather than risk it.
const char* dlErrorText() {
  const char* err = dlerror();
  return err == nullptr ? "no dlerror message" : err;
}

// The real provider. Deliberately a SEPARATE variable from IBVERBX_IBVERBS_SO:
// resolving through the same one, or through RTLD_NEXT, would find this library
// again and recurse.
constexpr const char* kRealLibEnv = "IB_INJECTION_REAL_IBVERBS_SO";
constexpr const char* kDefaultRealLib = "libibverbs.so.1";

// The real mlx5 provider. ibverbx points both its handles at this shim, so the
// shim has to reach the genuine libmlx5 itself for the mlx5dv_* forwarders.
constexpr const char* kRealMlx5Env = "IB_INJECTION_REAL_MLX5_SO";

// The real provider, opened once. A failure here aborts rather than returning
// null: every forwarder delegates through this handle, so without it the shim
// can serve nothing at all -- it would answer every verb with a failure and
// look like a machine with no working RDMA stack. Aborting names the path and
// the dlerror instead, which is the difference between
// "IB_INJECTION_REAL_IBVERBS_SO points at a file that is not there" and a run
// that fails much later for no visible reason.
void* realHandle() {
  static void* handle = [] {
    const char* path = getenv(kRealLibEnv);
    if (path == nullptr || *path == '\0') {
      path = kDefaultRealLib;
    }
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
      fprintf(
          stderr,
          "ib_injection: cannot dlopen real provider '%s': %s\n",
          path,
          dlErrorText());
      abort();
    }
    return h;
  }();
  return handle;
}

// libmlx5 is optional: a host without it simply has no mlx5 features, exactly
// as ibverbx already tolerates (its own mlx5 load path is warn-only). So a null
// handle here is not fatal -- the forwarders below report EOPNOTSUPP instead.
//
// Naming one explicitly is different. Searching the sonames is a search, but
// substituting a different library for the one an operator asked for is a
// silent lie: mlx5 features would come from somewhere other than the path
// under test, and the run would still look clean. Same reason ibvInit() refuses
// to fall back on a set-but-unloadable IBVERBX_IBVERBS_SO.
void* realMlx5Handle() {
  static void* handle = [] {
    const char* path = getenv(kRealMlx5Env);
    if (path != nullptr && *path != '\0') {
      void* named = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (named == nullptr) {
        fprintf(
            stderr,
            "ib_injection: %s names '%s', which cannot be loaded: %s\n",
            kRealMlx5Env,
            path,
            dlErrorText());
        abort();
      }
      return named;
    }
    void* h = dlopen("libmlx5.so", RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
      h = dlopen("libmlx5.so.1", RTLD_NOW | RTLD_LOCAL);
    }
    return h;
  }();
  return handle;
}

// Resolved once per verb. dlvsym rather than dlsym: libibverbs versions its
// symbols, and an unversioned lookup can bind a different implementation than
// the one ibverbx asked for.
//
// A missing REQUIRED verb aborts. ibverbx itself would report this as a
// recoverable error from ibvInit(), so this is a deliberate divergence, and it
// is specific to what this library is for: it loads only when
// IBVERBX_IBVERBS_SO names it, so a real provider missing a load-bearing verb
// means the injection run is void. Aborting here names the verb and version;
// degrading instead would hand back a shim that silently injects nothing and
// reports success. An optional verb returns null, and the forwarder degrades
// the way a null ibvSymbols entry does.
void* realSym(const char* name, const char* version, VerbRequirement req) {
  void* sym = dlvsym(realHandle(), name, version);
  if (sym == nullptr && req == kRequired) {
    fprintf(
        stderr,
        "ib_injection: real provider has no %s@%s: %s\n",
        name,
        version,
        dlErrorText());
    abort();
  }
  return sym;
}

// mlx5 lookup. Returns null rather than aborting: ibverbx's own mlx5 loads are
// warn-only, so a provider missing one of these is a supported state.
//
// A null version selects the unversioned lookup, which is what the symbols
// fbcode's vendored libmlx5 tags at non-upstream versions need -- ibverbx
// resolves those with plain dlsym, so the shim must be findable the same way.
void* realMlx5Sym(const char* name, const char* version) {
  void* h = realMlx5Handle();
  if (h == nullptr) {
    return nullptr;
  }
  return version == nullptr ? dlsym(h, name) : dlvsym(h, name, version);
}

// --- the three hot-path shims ---

int shimPollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) {
  auto& engine = Engine::get();
  // The lock is held ACROSS the delegated poll, unlike the post shims which
  // copy and release. That asymmetry is deliberate: poll_cq consumes
  // completions, so the drain and the rule decisions have to be atomic together
  // -- releasing the lock mid-drain would let a concurrent poll on the same CQ
  // interleave and reorder what each caller sees. ctran polls from one progress
  // thread per communicator, so the contention this costs is not on a real hot
  // path, and a shim that reordered completions would be a worse provider than
  // a slow one.
  std::lock_guard<std::mutex> lock(engine.mutex());
  int32_t injectErrno = 0;
  const int rc = engine.pollCq(cq, numEntries, wc, &injectErrno);
  if (rc >= 0) {
    return rc;
  }
  if (rc == -2) {
    // Injected failure. Real poll_cq reports errors as a negative return, so
    // that is what ctran's wrapper checks.
    return -injectErrno;
  }
  // Unregistered CQ, so no attribution is possible. Delegate untouched rather
  // than guess a device -- but only via ops the engine recorded BEFORE the
  // patch. Re-registering here would save shimPollCq itself as the "original"
  // and recurse forever.
  SavedOps saved;
  if (!engine.lookupSavedOps(cq->context, &saved) || saved.pollCq == nullptr) {
    // We are installed in this context's vtable yet have no record of it, so
    // there is no real poll_cq to call. Report an error rather than 0: a caller
    // spinning on `while (poll_cq(...) == 0)` would hang forever on "no
    // completions", turning a mis-registration into a silent deadlock instead
    // of a visible failure.
    return -EINVAL;
  }
  return saved.pollCq(cq, numEntries, wc);
}

int shimPostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) {
  auto& engine = Engine::get();
  // Copied by value, not held by reference: the mutex is released before
  // delegating (so the real provider call is not serialized), and a concurrent
  // ibv_close_device would erase the ContextRecord a reference pointed into.
  SavedOps saved;
  bool known = false;
  CallDecision decision;
  {
    std::lock_guard<std::mutex> lock(engine.mutex());
    // The opcode of the first WR selects for the whole call: a rule targets the
    // post, not an individual WR in a batched list.
    const int32_t opcode = wr != nullptr ? static_cast<int32_t>(wr->opcode)
                                         : IB_INJECTION_ANY_OPCODE;
    decision = engine.decidePost(IB_INJECTION_VERB_POST_SEND, qp, opcode);
    known = engine.lookupSavedOps(qp->context, &saved);
  }
  if (decision.injectError) {
    // A real post_send names the offending WR here, and ctran reads it.
    if (badWr != nullptr) {
      *badWr = wr;
    }
    return decision.errnoValue;
  }
  if (!known || saved.postSend == nullptr) {
    return ENODEV; // context was never patched by us
  }
  return saved.postSend(qp, wr, badWr);
}

int shimPostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** badWr) {
  auto& engine = Engine::get();
  SavedOps saved;
  bool known = false;
  CallDecision decision;
  {
    std::lock_guard<std::mutex> lock(engine.mutex());
    // ibv_recv_wr has no opcode field, so recv rules select on device and QP.
    decision = engine.decidePost(
        IB_INJECTION_VERB_POST_RECV, qp, IB_INJECTION_ANY_OPCODE);
    known = engine.lookupSavedOps(qp->context, &saved);
  }
  if (decision.injectError) {
    if (badWr != nullptr) {
      *badWr = wr;
    }
    return decision.errnoValue;
  }
  if (!known || saved.postRecv == nullptr) {
    return ENODEV; // context was never patched by us
  }
  return saved.postRecv(qp, wr, badWr);
}

// Shared by every setup-verb wrapper: ask the engine, and report the failure in
// whichever shape this verb uses. Pointer-returning verbs set errno and return
// null; int-returning verbs return the errno directly.
bool injectSetupFailure(IbInjectionVerb verb, int* errnoOut) {
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  const CallDecision decision = engine.decideSetupCall(verb);
  if (!decision.injectError) {
    return false;
  }
  *errnoOut = decision.errnoValue;
  return true;
}

// req_notify_cq is deliberately left pointing at the real provider. It has no
// callers in ctran, ctranx or ncclx, and ctran passes a null completion channel
// at every CQ it creates, so there is nothing to model. Leaving the real
// pointer in place is strictly safer than a shim that could be wrong.
//
// The one interaction worth recording: holding a CQE and arming a completion
// channel are incompatible, because a withheld completion would still wake the
// channel. That only matters if something starts using channels.

} // namespace

// --- exported verbs ---
//
// Signatures come from ibverbx/Ibvcore.h and Mlx5core.h, the SAME headers ctran
// compiles against -- so the shim cannot drift from the layout ctran reads.
// That is the whole reason this library lives under ibverbx/.

#define IBVERBS_1_1 "IBVERBS_1.1"

extern "C" {

// One body for all three tables below; they differ only in how the real symbol
// is resolved.
#define IB_INJECTION_FORWARD(ret, name, params, args, resolver)   \
  ret name params {                                               \
    static auto real = reinterpret_cast<ret(*) params>(resolver); \
    if (real == nullptr) {                                        \
      return missingVerbResult<ret>();                            \
    }                                                             \
    return real args;                                             \
  }

// Pure forwarders for every verb the shim does not need to observe. These have
// to exist because buildIbvSymbols dlvsym's each one on THIS library's handle
// (the real provider is opened RTLD_LOCAL, so it is not in that scope), and
// LOAD_IBVERBS_SYM fails ibvInit() on a miss.
//
// Exporting them all unconditionally is what makes the requirement column
// necessary: ibverbx can no longer tell a verb the provider lacks from one it
// has, so the shim answers on its behalf -- abort for a required verb, a
// caller-visible failure for an optional one.
#define X(name, version, req, ret, params, args) \
  IB_INJECTION_FORWARD(ret, name, params, args, realSym(#name, version, req))
IB_INJECTION_FORWARDED_VERBS_VERSIONED
#undef X

// mlx5 forwarders. All optional: a host without libmlx5 simply has no mlx5
// features, which is a state ibverbx already tolerates (its mlx5 loads are
// warn-only and its callers null-guard).
#define X(name, version, ret, params, args) \
  IB_INJECTION_FORWARD(ret, name, params, args, realMlx5Sym(#name, version))
IB_INJECTION_FORWARDED_VERBS_MLX5_VER
#undef X

// The two mlx5 symbols ibverbx resolves with plain dlsym; a null version picks
// that lookup.
#define X(name, ret, params, args) \
  IB_INJECTION_FORWARD(ret, name, params, args, realMlx5Sym(#name, nullptr))
IB_INJECTION_FORWARDED_VERBS_MLX5_UNVER
#undef X

#undef IB_INJECTION_FORWARD

ibv_context* ibv_open_device(ibv_device* device) {
  static auto real = reinterpret_cast<ibv_context* (*)(ibv_device*)>(
      realSym("ibv_open_device", IBVERBS_1_1, kRequired));
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_OPEN_DEVICE, &injected)) {
    errno = injected;
    return nullptr;
  }
  ibv_context* ctx = real(device);
  if (ctx == nullptr) {
    return nullptr;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  // Save the originals BEFORE overwriting, so the shims have something to
  // delegate to. req_notify_cq is left alone on purpose; see above.
  engine.registerContext(ctx);
  ctx->ops.poll_cq = &shimPollCq;
  ctx->ops.post_send = &shimPostSend;
  ctx->ops.post_recv = &shimPostRecv;
  return ctx;
}

// Hand-written rather than generated so an API_ERROR rule can target it. Each
// reports failure in the shape its return type demands: pointer verbs set errno
// and return null, int verbs return the errno.

ibv_pd* ibv_alloc_pd(ibv_context* context) {
  static auto real = reinterpret_cast<ibv_pd* (*)(ibv_context*)>(
      realSym("ibv_alloc_pd", IBVERBS_1_1, kRequired));
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_ALLOC_PD, &injected)) {
    errno = injected;
    return nullptr;
  }
  return real(context);
}

ibv_mr* ibv_reg_mr(ibv_pd* pd, void* addr, size_t length, int access) {
  static auto real = reinterpret_cast<ibv_mr* (*)(ibv_pd*, void*, size_t, int)>(
      realSym("ibv_reg_mr", IBVERBS_1_1, kRequired));
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_REG_MR, &injected)) {
    errno = injected;
    return nullptr;
  }
  return real(pd, addr, length, access);
}

// Hand-written rather than generated, because a REG_MR rule has to reach it.
// ctran registers CUDA device memory through the dma-buf path whenever the
// device supports it (useDmaBuf, CtranIb.cc:855), so a rule that only covered
// ibv_reg_mr would silently never fire for a device buffer: stored, handed a
// rule id, and inert -- indistinguishable from the code under test handling the
// error.
//
// It shares IB_INJECTION_VERB_REG_MR with ibv_reg_mr deliberately, the same way
// ibv_create_qp and mlx5dv_create_qp share CREATE_QP: a rule says "fail a
// registration" and should not have to name the entry point that performs it.
//
// Optional, so the missing-symbol case is checked FIRST and does not consume
// the rule -- firing on a call that was never going to reach a provider would
// spend a firstMatch position on nothing and shift which registration a later
// rule lands on.
ibv_mr* ibv_reg_dmabuf_mr(
    ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access) {
  static auto real = reinterpret_cast<
      ibv_mr* (*)(ibv_pd*, uint64_t, size_t, uint64_t, int, int)>(
      realSym("ibv_reg_dmabuf_mr", "IBVERBS_1.12", kOptional));
  if (real == nullptr) {
    return missingVerbResult<ibv_mr*>();
  }
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_REG_MR, &injected)) {
    errno = injected;
    return nullptr;
  }
  return real(pd, offset, length, iova, fd, access);
}

// The mlx5 half of the same story: ctran reaches this one when it registers
// with data-direct placement, and it must honor a REG_MR rule for the same
// reason.
ibv_mr* mlx5dv_reg_dmabuf_mr(
    ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access,
    int mlx5_access) {
  static auto real = reinterpret_cast<
      ibv_mr* (*)(ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)>(
      realMlx5Sym("mlx5dv_reg_dmabuf_mr", "MLX5_1.25"));
  if (real == nullptr) {
    return missingVerbResult<ibv_mr*>();
  }
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_REG_MR, &injected)) {
    errno = injected;
    return nullptr;
  }
  return real(pd, offset, length, iova, fd, access, mlx5_access);
}

int ibv_modify_qp(ibv_qp* qp, ibv_qp_attr* attr, int mask) {
  static auto real = reinterpret_cast<int (*)(ibv_qp*, ibv_qp_attr*, int)>(
      realSym("ibv_modify_qp", IBVERBS_1_1, kRequired));
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_MODIFY_QP, &injected)) {
    return injected;
  }
  return real(qp, attr, mask);
}

int ibv_close_device(ibv_context* context) {
  static auto real = reinterpret_cast<int (*)(ibv_context*)>(
      realSym("ibv_close_device", IBVERBS_1_1, kRequired));
  // Deregister only once the provider has actually let go. A failed close (say
  // EBUSY) leaves the context live with the shims still installed in its ops,
  // and without a SavedOps entry every later shimPollCq answers -EINVAL and
  // every post ENODEV -- the object would be bricked by its own failed
  // teardown.
  const int ret = real(context);
  if (ret != 0) {
    return ret;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  engine.forgetContext(context);
  return ret;
}

ibv_cq* ibv_create_cq(
    ibv_context* context,
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) {
  static auto real = reinterpret_cast<
      ibv_cq* (*)(ibv_context*, int, void*, ibv_comp_channel*, int)>(
      realSym("ibv_create_cq", IBVERBS_1_1, kRequired));
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_CREATE_CQ, &injected)) {
    errno = injected;
    return nullptr;
  }
  ibv_cq* cq = real(context, cqe, cq_context, channel, comp_vector);
  if (cq == nullptr) {
    return nullptr;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  // cq_context stays the caller's cookie -- ctran passes nullptr today, but
  // other callers do not, and it is theirs to read back.
  engine.registerCq(cq);
  return cq;
}

int ibv_destroy_cq(ibv_cq* cq) {
  static auto real = reinterpret_cast<int (*)(ibv_cq*)>(
      realSym("ibv_destroy_cq", IBVERBS_1_1, kRequired));
  // As in ibv_close_device: a failed destroy would otherwise free this CQ's
  // device id while the CQ is still live, so the next CQ takes the same id and
  // every rule written against it lands on the wrong device.
  const int ret = real(cq);
  if (ret != 0) {
    return ret;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  engine.forgetCq(cq);
  return ret;
}

ibv_qp* ibv_create_qp(ibv_pd* pd, ibv_qp_init_attr* qp_init_attr) {
  static auto real = reinterpret_cast<ibv_qp* (*)(ibv_pd*, ibv_qp_init_attr*)>(
      realSym("ibv_create_qp", IBVERBS_1_1, kRequired));
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_CREATE_QP, &injected)) {
    errno = injected;
    return nullptr;
  }
  ibv_qp* qp = real(pd, qp_init_attr);
  if (qp == nullptr) {
    return nullptr;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  engine.registerQp(qp);
  return qp;
}

int ibv_destroy_qp(ibv_qp* qp) {
  static auto real = reinterpret_cast<int (*)(ibv_qp*)>(
      realSym("ibv_destroy_qp", IBVERBS_1_1, kRequired));
  const int ret = real(qp);
  if (ret != 0) {
    return ret;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  engine.forgetQp(qp);
  return ret;
}

// Hand-written rather than a forwarder, because the QP it returns must enter
// the registry. With NCCL_CTRAN_IB_ENABLE_OOO_RQ set, createRcQpWithOooDp
// routes EVERY data QP through here instead of ibv_create_qp -- so without this
// those QPs would be invisible and any rule naming one would silently match
// nothing.
//
// Resolved unversioned to match how ibverbx resolves it (fbcode's vendored
// libmlx5 tags it at a non-upstream version). Returning null on a missing
// symbol is the same signal ibverbx's own null-guard produces (IbvPd.cc:
// ENOTSUP).
ibv_qp* mlx5dv_create_qp(
    ibv_context* context,
    ibverbx::ibv_qp_init_attr_ex* qp_attr,
    ibverbx::mlx5dv_qp_init_attr* mlx5_qp_attr) {
  static auto real =
      reinterpret_cast<ibv_qp* (*)(ibv_context*,
                                   ibverbx::ibv_qp_init_attr_ex*,
                                   ibverbx::mlx5dv_qp_init_attr*)>(
          realMlx5Sym("mlx5dv_create_qp", /*version=*/nullptr));
  // No libmlx5 on this host, so report what every other missing-symbol path in
  // this file reports -- missingVerbResult, i.e. EOPNOTSUPP, which is the
  // ENOTSUP above spelled the way ibverbx's own null-guards spell it. Setting a
  // different errno here would change what ibverbx observes for this verb
  // purely because the verb became hand-written. A CREATE_QP rule is
  // deliberately NOT consumed: nothing can be created, so firing would spend a
  // repeat position on a call that was never going to reach a provider, and
  // shift which creation a firstMatch=3 rule lands on.
  if (real == nullptr) {
    return missingVerbResult<ibv_qp*>();
  }
  // A CREATE_QP rule covers this path too: with OOO_DP the data QPs come from
  // here, and a rule targeting "the third QP creation" must not care which
  // entry point produced it.
  int injected = 0;
  if (injectSetupFailure(IB_INJECTION_VERB_CREATE_QP, &injected)) {
    errno = injected;
    return nullptr;
  }
  ibv_qp* qp = real(context, qp_attr, mlx5_qp_attr);
  if (qp == nullptr) {
    return nullptr;
  }
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  engine.registerQp(qp);
  return qp;
}

// --- control ABI ---

IbInjectionStatus ibInjectionReset(void) {
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  return engine.reset();
}

IbInjectionStatus ibInjectionAddRule(
    const IbInjectionRule* rule,
    uint32_t* ruleId) {
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  return engine.addRule(rule, ruleId);
}

IbInjectionStatus ibInjectionGetState(IbInjectionState* state) {
  auto& engine = Engine::get();
  std::lock_guard<std::mutex> lock(engine.mutex());
  return engine.getState(state);
}

const char* ibInjectionLastError(void) {
  // Copied into thread-local storage under the lock. Returning
  // engine.lastError() directly would hand back a pointer into a std::string
  // that another thread's setLastError can reallocate the moment the lock
  // drops, so the caller could be reading freed memory. Thread-local keeps each
  // caller's copy alive until its next call without holding the engine lock.
  static thread_local std::string copy;
  {
    auto& engine = Engine::get();
    std::lock_guard<std::mutex> lock(engine.mutex());
    copy = engine.lastError();
  }
  return copy.c_str();
}

} // extern "C"
