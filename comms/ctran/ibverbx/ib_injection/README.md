# IB injection: a libibverbs shim for fault and skew injection

A drop-in `libibverbs.so` that injects **verbs failures** and **completion skew**
into ctran without touching production code. Selected at runtime by the
`IBVERBX_IBVERBS_SO` environment variable.

Nothing is emulated. The shim opens a real `libibverbs.so.1` itself and delegates
every verb to it, so a run behaves identically except where a rule fires.

## Why this exists

Some ctran bugs are races or hardware-failure paths that no test can reach: they
need a completion order, a transfer delay, or a verb failure that real hardware
produces only rarely and never on demand. Nothing could control ibverbs behavior
from a test, so those stayed unguarded. Two landed examples:

- **D118906933** — a completion-ordering race. `iflush` retired per-NIC loopback
  READ completions from one positional FIFO, so flush 2's device-0 CQE popped
  flush 1's slot while flush 1's device-1 READ was still in flight. IB orders
  completions *within* a device, never across devices, so the interleaving that
  exposes it is up to the fabric — reproducing it needs cross-device skew on
  demand.
- **D118961602** — a transfer-timing race. AGP released the CUDA stream before its
  outgoing puts drained, so the next collective could overwrite a buffer a put was
  still reading. An idle box never opens the window; the diff had to hand-edit
  production code to sleep the worker, and its followup named the missing piece
  directly: ctran had no ibverbx injection layer to shift a completion with.

## How the interception works

The shim is a `libibverbs.so` that ctran loads instead of the real one:
`ibvInit()` `dlopen`s whatever `IBVERBX_IBVERBS_SO` names and resolves the verbs
from that handle with `dlvsym`. The shim then opens the real provider named by
`IB_INJECTION_REAL_IBVERBS_SO` (default `libibverbs.so.1`) and forwards to it.

The two variables are deliberately separate, and named after **who reads them**:
`IBVERBX_IBVERBS_SO` is ibverbx's provider selector — a plain path, whose
production meaning is unchanged and whose other users point it at libraries that
inject nothing. `IB_INJECTION_REAL_IBVERBS_SO` is read by the shim itself.
Resolving both through one variable, or through `RTLD_NEXT`, would find this
library again and recurse.

Nothing is overridden or interposed. ctran calls through function-pointer
fields it was handed at load; the shim is simply what those fields point at.

### It reaches both verb sets, by two different routes

```
A · SETUP VERBS — resolved by name        B · HOT PATH — reached by pointer
  ctran: ibvSymbols.ibv_internal_*          ctran: cq_->context->ops.poll_cq
                                                   qp_->context->ops.post_send

  dlvsym(h,"ibv_create_qp",                 ibv_open_device()
         "IBVERBS_1.1")                       ctx = real(device);
    → the shim's exported body                save ctx->ops
      (IbverbxSymbols.def + version.script)   ctx->ops.{3} = shims
```

The two need opposite things from the shim:

| | A · setup verbs | B · hot path |
|---|---|---|
| Verbs | `open_device`, `alloc_pd`, `reg_mr`, `create_cq`, `create_qp`, `modify_qp`, `query_*` … | `poll_cq`, `post_send`, `post_recv` |
| Resolved | once at load, by `dlvsym` | never — the pointer is read on every call |
| Must be exported | **yes** — `.def` supplies the body, `version.script` the version tag; a missing entry aborts `ibvInit()` | **no** — the shims are `local: *`, reached by address only |
| Interception | already in place: ctran calls the shim's own function | needs the vtable patch, since the pointer belongs to the real provider |

`dlvsym` matches a name **and** a version node, which is why set A needs both
files: compiling a verb is not enough, an untagged symbol is invisible to
`dlvsym` and a wrong node fails exactly like a missing one. Set B needs neither
— `ibv_open_device` is the one function that hands over the `ibv_context`, so
overwriting three of its `ops` entries on the way out is the whole seam. Neither
route requires a production change.

`req_notify_cq` is deliberately **left pointing at the real provider**. It has no
callers in ctran, ctranx or ncclx, and ctran passes a null completion channel at
every CQ it creates, so there is nothing to model — and leaving the real pointer
in place is safer than a shim that could be wrong. The one interaction worth
recording is that holding a CQE and arming a completion channel are incompatible,
since a withheld completion would still wake the channel. That only matters if
something starts using channels.

### Which enables which kind of injection

- **Failure injection** — a verb returns an error instead of succeeding. Works on
  both sets, because both run through shim-owned code. On setup verbs it reaches
  ctran's error and cleanup paths (`CtranIbSingleton` ctor throw → the
  `NCCL_CTRAN_BACKENDS=nvl,socket` fallback; mid-VC-setup partial construction).
  On the hot path it covers post failure and CQE error status.
- **Skew injection** — timing is shifted, with no error at all. Needs set B, and
  comes in two forms:
  - **Hold a completion** (`poll_cq`). Changes *when ctran observes* a transfer
    that already finished; this is what a race like D118906933 requires.
  - **Defer a post** (`post_send`). Returns success immediately but hands the WR
    to the NIC `n` ms later, so the DMA itself starts late while the caller runs
    on.

Holding is *withholding* a real CQE — no fabricated completions, no error codes,
and per-CQ order preserved exactly as a provider would.


## Rules

A rule is `(verb, action, selector, repeat)` plus action-specific fields. Four
actions:

**Failure injection only.** Skew — holding a completion, deferring a post — shifts
timing with no error at all, so it is a separate mechanism and lands in its own
diff, which adds `CQE_DELAY` and `CALL_DELAY` to this table.

| Action | Verbs | Effect |
|---|---|---|
| `API_ERROR` | setup verbs, `post_send`, `post_recv`, `poll_cq` | fail the call. Pointer-returning verbs get `nullptr` + `errno`; int-returning verbs return the `errno` |
| `WC_STATUS` | `poll_cq` | deliver a real CQE with `ibv_wc.status` overwritten |

The two are deliberately separate rather than one "inject an error" action,
because they reach different ctran paths: a `post_send` that fails never produces
a CQE at all, whereas a bad `ibv_wc.status` flows through `processCqe`. Collapsing
them would leave one of those untested.

The matrix is sparse, and `addRule` **rejects** every gap rather than accepting a
rule that could never fire — `WC_STATUS` on a post verb has no `ibv_wc` to write.

What each combination can actually select, all enforced in `addRule` so both front
ends are held to one contract:

| Verb | Action | `deviceId` | `hwQpNum` | `opcode` | Payload |
|---|---|---|---|---|---|
| setup verbs (`open_device`, `alloc_pd`, `reg_mr`, `create_cq`, `create_qp`, `modify_qp`) | `API_ERROR` | wildcard only | wildcard only | wildcard only | `errnoValue` > 0 |
| `post_send` | `API_ERROR` | yes | yes | yes (`WR` domain) | `errnoValue` > 0 |
| `post_recv` | `API_ERROR` | yes | yes | **wildcard only** | `errnoValue` > 0 |
| `poll_cq` | `API_ERROR` | yes | wildcard only | wildcard only | `errnoValue` > 0 |
| `poll_cq` | `WC_STATUS` | yes | yes | yes (`WC` domain) | `wcStatus` ≠ `IBV_WC_SUCCESS` |

Every "wildcard only" is a constraint, not a default: the decision is made before
the thing exists or before it is read. Setup verbs fire before any device or QP
exists. `ibv_recv_wr` carries no opcode field, so `shimPostRecv` can only ever
decide with the wildcard. And an `API_ERROR` on `poll_cq` fails the call before a
completion is read, so there is no QP or opcode to match yet. A rule naming one
anyway would be stored, handed back a rule id, and sit inert — indistinguishable
from the code under test handling the error correctly, which is why it is rejected
instead.

The two payload rules exist for one reason: `errnoValue` 0 and `IBV_WC_SUCCESS`
both fire and move the counters while leaving the call, or the completion, reading
exactly as it would have anyway. A run that reports an injection it did not
perform is the one result this tool must never produce.

### Selector

```
deviceId  | ANY      hwQpNum | ANY      opcode | ANY   + opcodeDomain(WR|WC)
```

`hwQpNum` is the **hardware** `qp_num`, not a `qpIdx`. ctran's `getDataQpNums()`
returns provider-assigned numbers (`IbvQp::getQpNum()` → `qp_->qp_num`); an index
is a different thing that happens to share `uint32_t`. Passing an index would
match nothing — real `qp_num`s are large — and fail *silently*, which is the worst
outcome for an injector. Hence the `hw` in the name.

`opcodeDomain` is not redundant with the verb: `IBV_WR_RDMA_READ` is 4 and
`IBV_WC_RDMA_READ` is 2, so comparing across namespaces matches nothing.

### Repeat

```
fires iff ordinal >= firstMatch
     && (ordinal - firstMatch) % everyNth == 0
     && (unbounded || fired < count)
```

`firstMatch` is what makes setup-verb injection useful: `create_qp` with
`firstMatch=3` fails the **third** QP creation — mid-VC-setup with two QPs already
live, which is the partial-construction path where cleanup bugs live.
`firstMatch=1` only tests the trivial early-exit.

Matching is evaluated **once** per event, so a rule's schedule advances only when
new traffic arrives.

## Usage

Two surfaces today. They differ only in **how a rule is armed** — the shim and the
seam are identical.

### C++ CI test

The only surface that can call the control API, so the only one that gets
assertions. Point the target at the shim and drive it through the bridge
(`IbInjectionControl.h`):

```python
env = {
    "IBVERBX_IBVERBS_SO": "$(exe_target //comms/ctran/ibverbx/ib_injection:libibverbs.so)",
}
deps = ["//comms/ctran/ibverbx/ib_injection:ib-injection-control"]
```

```cpp
injection::reset();  // after ctran init, so only your traffic is measured

// Fail the third QP creation: mid-VC-setup, two QPs already live.
injection::addSetupError(
    IB_INJECTION_VERB_CREATE_QP, ENOMEM, /*firstMatch=*/3);

// ... drive ctran ...
const auto s = injection::getState();
EXPECT_EQ(s.rules[0].firings, 1u);  // a rule that never fired proves nothing
```

Setup-verb rules need no handshake at all, because they fire before any QP exists.
A data-path rule that must name one specific QP has to be armed after connections
are up, since `qp_num` is assigned by the provider — read it back from ctran's
public accessors (`getDataQpNums()`) and pass it to `addPostError`.

`injection::*` is the C++ bridge; each call is a thin wrapper over the exported C
entry points (`ibInjectionAddRule`, `ibInjectionGetState`, …), which is what
crosses the `dlopen` boundary.

### collperf, ctranx MAST runs and e2e farm jobs

No C++ hook point — collperf is a Python harness, a farm job is a training run.
All they pass is job-wide env, so both halves travel that way: one variable
selects the shim, another arms the rules.

```bash
IBVERBX_IBVERBS_SO=/packages/ib_injection/lib/libibverbs.so
IB_INJECTION_SPEC="rank=1,fn=create_qp,action=api_error,errno=ENOMEM,first=3"
```

`IB_INJECTION_SPEC` is parsed at load into the very rules `addRule()` builds, so
nothing about matching or validation is duplicated — one engine, two front ends.

```
IB_INJECTION_SPEC = rule ( ';' rule )*
rule              = field ( ',' field )*
```

| key | values | default |
|---|---|---|
| `fn` | `poll_cq` `post_send` `post_recv` `open_device` `alloc_pd` `reg_mr` `create_cq` `create_qp` `modify_qp` | **required** |
| `action` | `api_error` `wc_status` | **required** |
| `errno` | positive int, or a name (`ENOMEM`, `EINVAL`, …) | required for `api_error` |
| `status` | an `IBV_WC_*` name, or an int | required for `wc_status` |
| `rank` | int, or `*` | `*` |
| `dev` | int, or `*` | `*` |
| `qp` | hardware `qp_num`, or `*` | `*` |
| `opcode` | `SEND` `RDMA_WRITE` `RDMA_READ` `RECV` …, or `*` | `*` |
| `first` `every` `count` | the repeat triple; `count=inf` is unbounded | `1` `1` `1` |

`opcode` takes the **short** name and the namespace is derived from `fn`. That is
the one thing a hand-written spec must not be allowed to get wrong: `RDMA_READ` is
4 in the WR namespace and 2 in the WC namespace, so naming the namespace by hand
buys nothing and silently matches the wrong traffic when it is wrong.

`rank` is matched against `$RANK`, so one job-wide string arms selected ranks. It
exists because these surfaces have **no per-rank environment channel** — collperf
and farm jobs set env once for the whole task group, and torchrun then hands every
process the same value plus its own `RANK`. A launcher that *can* vary env per
rank (`ctranx_dist_launcher` builds it per worker) does not need `rank=` at all.

Two things follow from a job-wide string reaching processes that will never act on
it, and both are deliberate:

- **A missing `$RANK` is an error, not rank 0.** Defaulting would make `rank=1`
  arm nothing and `rank=0` arm every process, with nothing to distinguish either
  from a working run. A spec that never mentions `rank` needs no `$RANK` and works
  anywhere.
- **Every rule is validated whatever rank it names**; only *arming* is filtered.
  Validating just the local rank's rules would let a spec whose rules all target
  ranks outside the job be checked by nobody — it would arm nothing, anywhere, and
  report success everywhere.

**Everything the engine cannot act on is rejected, not ignored.** A malformed
spec, an unknown key, an `action=cq_gate` or a `qp=data:3` role form aborts at
load, naming the offending token — the two skew actions and the role form are
still unimplemented, and a spec that quietly did nothing is the exact outcome this
mechanism exists to prevent. Rules that need a runtime `qp_num` remain a poor fit
here in any case: no QP exists at load, so prefer a C++ test for those.

The spec exists **for these surfaces only** — a C++ test has the control API and
needs nothing from it. It matters most where there is **no assertion**: the signal
is the job's own output, a busBw delta against an uninjected baseline or whether
the rank survived. So the shim reports itself:

```
ib_injection: loaded, delegating to 'libibverbs.so.1'
ib_injection: rank 1 armed 1 rule(s) from IB_INJECTION_SPEC (0 addressed to other ranks)
ib_injection: rule 1 FIRED (verb 0 action 1 dev 0 qp 114173 opcode 0)
```

**No `FIRED` line means no injection happened, and that is a failed run rather
than a pass.** It is printed once per rule, since it sits on the hot path; a C++
test wanting exact counts, `matches`, or the per-device counters reads them from
`getState()`, which is assertable and therefore strictly better than any log line.

What this deliberately does not tell you is whether traffic reached ctran-IB at
all, so a rule that never fired is ambiguous between "the selector is wrong" and
"nothing ever went over IB". Check the job's own transport logs for that — on
ctranx, `CTRANX IB: connected peer=...` and the CQ/QP setup lines.

The `.so` ships in its own fbpkg, `//comms/ctran/ibverbx/ib_injection:ib_injection`,
built for x86_64 and aarch64 because H100 is x86 and GB200/GB300 are Grace. MAST
mounts it at `/packages/ib_injection`, so the path above is a plain runtime string
— a Buck `$(location ...)` macro does not survive to a worker.

```bash
fbpkg build //comms/ctran/ibverbx/ib_injection:ib_injection --build-remote
```

Build it from a revision compatible with whatever carries ibverbx on the target
(the conda `libnccl.so` for collperf, the launcher PAR for ctranx). The boundary is
the libibverbs C ABI — `Ibvcore.h` struct layouts and the `version.script` nodes —
so a large skew can break the `ops` patch or fail a `dlvsym` lookup.

## Scope and limits

**Reaches ctran and ctranx, not NCCL core.** `IBVERBX_IBVERBS_SO` is read by
`ibverbx::ibvInit()`. NCCL core's IB transport is a separate `dlopen` on
`NCCL_IBVERBS_LIB` with its own symbol table. That is deliberate: an
init-failure rule would otherwise kill a job during NCCL bootstrap, long before
reaching the ctran code under test.

**Reaches every ibverbx consumer, because it is a plain env var and not a cvar.**
A cvar is empty until `ncclCvarInit()` runs, which would restrict selection to
the binaries that call it, in the right order: ctranx has its own cvar system and
never calls it, prims deliberately never calls it, and
`uniflow-light/RdmaTransportFactory.cpp` calls it *after* `ibvInit()`. Reading
the variable with `getenv` at the point of use covers all of them with one name.
For a path handed straight to `dlopen` a cvar adds nothing anyway — there is
nothing to type-check or range-check, and `/etc/nccl.conf` layering is actively
unwanted for an injection shim.

Every surface selects the shim the same way, through this variable — a C++ test
sets it per target (`env = {...}` in BUCK), collperf and farm jobs set it job-wide.
`ibvInit()` takes no argument, so there is no in-process way to name a provider.
And it is `folly::call_once`, so the first caller in a process latches the path
for every later one: the variable must be set before the process starts, and two
different shims cannot be selected within one run.

**An injection is inert unless traffic reaches ctran-IB.** A C++ test should assert
`patchedContexts` and the per-device counters from `getState()`; on the env-driven
surfaces the equivalent check is that a `rule N FIRED` line appeared at all. A run
that was supposed to inject and shows neither is a failed run, not a passing one.

**A bad path fails loud.** If `dlopen` of the requested library fails,
`buildIbvSymbols` returns an error and `ibvInit()` fails, naming the path and the
`dlerror`. It does *not* fall back to `libibverbs.so.1`: a fallback would load a
different provider than the caller named and still report success, so a typo'd
path or an unmounted fbpkg would yield a green, completely uninjected run — the
same signature as "IB was never on the path". Strictness applies only when the
variable is set; unset still means `libibverbs.so.1`.

That closes the *load* half. It cannot tell you a rule never matched, so the
counter check above is still not optional.

**Requires the default dlopen build.** `ibverbx-rdma-core` statically links
rdma-core and never `dlopen`s, so there is nothing to intercept there.

**The shim can only move software timing, which decides what corruption it can
produce.** Two questions: is the racing pair local, and does the race have a
software event the shim can move?

| Race | Reachable | Why |
|---|---|---|
| Local DMA vs a CUDA write to the same buffer — D118961602's put still reading `recvbuff` when `copyToSelf` overwrites it | **yes** | Both sides are local, and the shim controls when the DMA starts. |
| Host-side mis-attribution — D118906933's flush 1 retired against flush 2's completion | **yes** | Purely a bookkeeping decision driven by CQE arrival order, which the gate controls. |
| **Anything about whether the bytes are actually there** — is an inbound write DMA'd and coherently visible when its CQE appears, does a peer's payload arrive torn, does a flush actually fence anything | **no** | These races live below the verbs API, in the NIC and PCIe path, with no software event on either boundary. The shim's levers do not reach: it can delay a CQE, which only makes ctran's view *more* conservative than reality, and it cannot fabricate one — corruption needs a completion visible *earlier* than the data landed. Nor can it touch a peer's writes, posted on another rank. |

## Layout

| File | Role |
|---|---|
| `IbInjectionApi.h` | control ABI; the only declaration shared across the dlopen boundary |
| `IbInjectionDso.cc` | exported verbs, the 3 hot-path shims, real-provider delegation, load banner |
| `IbverbxSymbols.def` | table of verbs to forward, and how each is resolved |
| `InjectionEngine.{h,cc}` | rule matching, repeat scheduling, counters, and the `IB_INJECTION_SPEC` parser |
| `version.script` | export list; a missing entry aborts `ibvInit()` |
| `IbInjectionControl.{h,cc}` | C++ bridge: dlsym the control surface, named rule helpers |
| `BUCK` `:ib_injection` | the fbpkg, x86_64 + aarch64, mounted at `/packages/ib_injection` |

The spec parser lives with the engine rather than in the DSO so
`ib_injection_engine_test` covers it with no `dlopen` and no NIC, and so both front
ends provably build the same rules — one test asserts a spec rule and a hand-built
rule are indistinguishable to the engine.

`IbverbxSymbols.def` and `version.script` must stay a superset of what
`buildIbvSymbols` resolves: it uses `dlvsym` (or plain `dlsym` for two mlx5
symbols) on this library's handle, and the real provider is opened `RTLD_LOCAL`, so
a missing export is simply absent. `ib_injection_dlopen_test` pins that, in both
lookup styles.

`ib_injection_engine_test` covers the rule semantics with no NIC and no dlopen: the
engine is linked directly and driven against a fake provider.

The module sits outside `tests/` on purpose: the `.so` ships in an fbpkg to
collperf and farm jobs, so it is a shipped artifact, not test-only code. Its own
tests live in `ib_injection/tests/`.

## Appendix: mlx5

The mlx5 **data path** is covered by the vtable patch: the shim overwrites
`ibv_context::ops` and every QP on a context shares it, so `poll_cq` /
`post_send` / `post_recv` land in a shim no matter how the QP was created.

`mlx5dv_*` is the vendor escape hatch for features with no vendor-neutral verb.
It is exported from libmlx5 and resolved through a **second** handle that ibverbx
opens separately — so it needs its own coverage, which this shim now has.

**One library serves both handles.** Version nodes are just named buckets in a
version script, so one `.so` exports `IBVERBS_1.1` and `MLX5_1.8` side by side,
exactly as `IB_INJECTION_1.0` already coexists with the `IBVERBS_*` nodes.
ibverbx passes its `ibv_path` to the libmlx5 `dlopen` as well, and `dlopen` is
refcounted by resolved path, so both handles land on the same instance — and
therefore the same `Engine`, with no cross-library state to reconcile.

This works only because **nothing calls an `mlx5dv_*` function directly**: every
call site goes through `ibvSymbols.mlx5dv_internal_*`, so which handle a symbol
was resolved from is invisible once loaded.

Two details the export list has to get right, because both fail exactly like a
missing symbol:

- **`mlx5dv_query_device` and `mlx5dv_create_qp` are resolved with plain `dlsym`,
  not `dlvsym`** — fbcode's vendored libmlx5 tags them at non-upstream versions,
  so ibverbx deliberately does an unversioned lookup. Exporting them on a node
  still makes them the default version, which is what `dlsym` finds;
  `ib_injection_dlopen_test` asserts both lookup styles.
- **The mlx5 forwarders tolerate a missing real symbol**, returning `ENOSYS`
  rather than aborting. ibverbx's own mlx5 loads are warn-only and its callers
  null-guard (`IbvPd.cc` returns `ENOTSUP`), so a host without libmlx5 is a
  supported state.

`mlx5dv_create_qp` is hand-written rather than generated, so the QP it returns
enters the registry. That is what makes `NCCL_CTRAN_IB_ENABLE_OOO_RQ` safe: with
it set, `createRcQpWithOooDp` routes **every** data QP through that call, and
without the wrapper a `qp=data:N` rule would match nothing and report zero fires —
indistinguishable from "IB was never on the path". The cvar defaults to `false`
but `GB300_ENVS` sets it to `1`, so this matters on GB300.
