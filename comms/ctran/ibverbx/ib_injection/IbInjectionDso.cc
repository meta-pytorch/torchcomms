// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// The ib_injection libibverbs shim: soname libibverbs.so, selected by
// IBVERBX_IBVERBS_SO.
//
// THIS DIFF IS A PURE PASS-THROUGH. Every verb forwards to a real provider and
// nothing else -- no interception, no state, no behavior change. Loading it is
// observably identical to loading the real libibverbs, which is exactly the
// property that makes it reviewable: the only questions are whether the symbols
// resolve and whether delegation is correct. The vtable patch, the object
// registries and the rule engine land in the next diff.
//
// It serves BOTH verb sets. ibverbx points its ibvhandle and mlx5dvhandle
// dlopen calls at the same file, and nothing calls an mlx5dv_* function
// directly -- every call site goes through ibvSymbols.mlx5dv_internal_*. So one
// library covers ibv_* and mlx5dv_*.
//
// Setup verbs are resolved BY NAME, with dlvsym (or plain dlsym for the two
// mlx5 symbols fbcode's vendored libmlx5 tags at non-upstream versions) on this
// library's handle. They must therefore be exported at the right version node:
// IbverbxSymbols.def supplies the body, version.script the tag, and a missing
// entry fails ibvInit(). That coupling is what ib_injection_dlopen_test pins.
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
#include <type_traits>

#include "comms/ctran/ibverbx/Ibvcore.h"

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
using ibverbx::ibv_qp_init_attr_ex;
using ibverbx::mlx5dv_context;
using ibverbx::mlx5dv_obj;
using ibverbx::mlx5dv_qp_init_attr;

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
// caller already treats as failure.
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

// The real provider. Deliberately a SEPARATE variable from IBVERBX_IBVERBS_SO:
// resolving through the same one, or through RTLD_NEXT, would find this library
// again and recurse.
constexpr const char* kRealLibEnv = "IB_INJECTION_REAL_IBVERBS_SO";
constexpr const char* kDefaultRealLib = "libibverbs.so.1";

// The real mlx5 provider. ibverbx points both its handles at this shim, so the
// shim has to reach the genuine libmlx5 itself.
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
          dlerror());
      abort();
    }
    return h;
  }();
  return handle;
}

// libmlx5 is optional: a host without it simply has no mlx5 features, exactly
// as ibverbx already tolerates (its own mlx5 load path is warn-only). So a null
// handle here is not fatal -- the forwarders report EOPNOTSUPP instead.
void* realMlx5Handle() {
  static void* handle = [] {
    const char* path = getenv(kRealMlx5Env);
    void* h = (path != nullptr && *path != '\0')
        ? dlopen(path, RTLD_NOW | RTLD_LOCAL)
        : nullptr;
    if (h == nullptr) {
      h = dlopen("libmlx5.so", RTLD_NOW | RTLD_LOCAL);
    }
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
        dlerror());
    abort();
  }
  return sym;
}

// Versioned mlx5 lookup. Returns null rather than aborting: ibverbx's own mlx5
// loads are warn-only, so a provider missing one of these is a supported state.
void* realMlx5Sym(const char* name, const char* version) {
  void* h = realMlx5Handle();
  return h == nullptr ? nullptr : dlvsym(h, name, version);
}

// Unversioned mlx5 lookup, for the symbols fbcode's vendored libmlx5 tags at
// non-upstream versions -- ibverbx resolves those with plain dlsym, so the shim
// must be findable the same way.
void* realMlx5SymUnversioned(const char* name) {
  void* h = realMlx5Handle();
  return h == nullptr ? nullptr : dlsym(h, name);
}

} // namespace

// --- exported verbs ---
//
// Signatures come from ibverbx/Ibvcore.h and Mlx5core.h, the SAME headers ctran
// compiles against -- so the shim cannot drift from the layout ctran reads.
// That is the whole reason this library lives under ibverbx/.

extern "C" {

// Forwarders for every verb ibverbx resolves. These have to exist because
// buildIbvSymbols dlvsym's each one on THIS library's handle (the real provider
// is opened RTLD_LOCAL, so it is not in that scope), and LOAD_IBVERBS_SYM
// fails ibvInit() on a miss.
//
// Exporting them all unconditionally is what makes the requirement column
// necessary: ibverbx can no longer tell a verb the provider lacks from one it
// has, so the shim answers on its behalf -- abort for a required verb, a
// caller-visible failure for an optional one.
#define X(name, version, req, ret, params, args)                       \
  ret name params {                                                    \
    static auto real =                                                 \
        reinterpret_cast<ret(*) params>(realSym(#name, version, req)); \
    if (real == nullptr) {                                             \
      return missingVerbResult<ret>();                                 \
    }                                                                  \
    return real args;                                                  \
  }
IB_INJECTION_FORWARDED_VERBS_VERSIONED
#undef X

// mlx5 forwarders. All optional: a host without libmlx5 simply has no mlx5
// features, which is a state ibverbx already tolerates (its mlx5 loads are
// warn-only and its callers null-guard).
#define X(name, version, ret, params, args)                           \
  ret name params {                                                   \
    static auto real =                                                \
        reinterpret_cast<ret(*) params>(realMlx5Sym(#name, version)); \
    if (real == nullptr) {                                            \
      return missingVerbResult<ret>();                                \
    }                                                                 \
    return real args;                                                 \
  }
IB_INJECTION_FORWARDED_VERBS_MLX5_VER
#undef X

#define X(name, ret, params, args)                                      \
  ret name params {                                                     \
    static auto real =                                                  \
        reinterpret_cast<ret(*) params>(realMlx5SymUnversioned(#name)); \
    if (real == nullptr) {                                              \
      return missingVerbResult<ret>();                                  \
    }                                                                   \
    return real args;                                                   \
  }
IB_INJECTION_FORWARDED_VERBS_MLX5_UNVER
#undef X

} // extern "C"
