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


## Usage

Two surfaces today. They differ only in **how a rule is armed** — the shim and the
seam are identical.

This file lands ahead of the code, so the API below is the design: the shim starts
as a pure forwarder, and the rule engine, control ABI and `injection::*` bridge
arrive in the diffs after it.

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

// Hold device 1's flush READs until device 0 has released both of its own.
const uint32_t rule = injection::addCqeDelayAfterReleases(
    /*deviceId=*/1, ibverbx::IBV_WC_RDMA_READ, /*afterDevice=*/0,
    /*afterCount=*/2);

// ... drive ctran ...
const auto s = injection::getState();
EXPECT_EQ(s.device(0).cqesReleased, 2u);
EXPECT_EQ(s.device(1).cqesReleased, 0u);
injection::releaseRule(rule);
```

Failure injection needs no handshake, because it fires before any QP exists:

```cpp
// Fail the 3rd QP creation: mid-VC-setup, two QPs already live.
injection::addSetupError(
    IB_INJECTION_VERB_CREATE_QP, ENOMEM, /*firstMatch=*/3);
```

`injection::*` is the C++ bridge; each call is a thin wrapper over the exported C
entry points (`ibInjectionAddRule`, `ibInjectionGetState`, …), which is what
crosses the `dlopen` boundary.

### collperf and e2e farm jobs

No C++ hook point — collperf is a Python harness, and a farm job is a training
run. Both pass job-wide env, which is enough to select the shim:

```bash
# collperf (genai/msl/comms/benchmarks/collperf/launcher.py)
--mast_envs IBVERBX_IBVERBS_SO=/packages/<pkg>/lib/libibverbs.so

# farm: same variable through the launcher's mast_env list
```

Selecting the shim is not the same as arming a rule, and these two surfaces
cannot call the control API at all. The planned mechanism is `IB_INJECTION_SPEC`,
carrying a **declarative spec** parsed at load into the same rule list
`addRule()` builds — one engine, two front ends:

```bash
# not implemented yet -- planned shape
IB_INJECTION_SPEC="rank=1,dev=1,fn=poll_cq,opcode=RDMA_READ,\
                   action=cq_gate,after_dev=0,after_count=2,manual"
```

`rank` is matched against `$RANK` inside the shim, so one job-wide string arms
selected ranks. A spec that fails to parse must abort at load rather than run
un-injected. Rules needing a runtime `qp_num` cannot be expressed this way (no QP
exists at load); a role form such as `qp=data:3` is planned for that.

The spec exists **for these surfaces only** — a C++ test needs nothing from it,
including for setup verbs. It matters most where there is **no assertion**: the
signal is the job's own output, a busBw delta against an uninjected baseline or
whether the rank survived, so check the per-device counters and treat a rule that
never fired as a **failed** run rather than a pass.

Shipping the `.so` is the other remaining gap. The path must be a plain runtime
string resolvable on the worker (`/packages/<pkg>/...`), not a Buck
`$(location ...)` macro, so the shim needs to ride in an fbpkg — collperf and
farm both mount them under `/packages/<name>`.

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

A C++ test needs no environment at all: `ibvInit(path)` takes the library
explicitly, and the env var is only how the surfaces with no C++ hook point
(collperf, farm jobs) can select anything. Note that `ibvInit()` is
`folly::call_once`, so the first caller in a process latches the path for every
later one.

**An injection is inert unless traffic reaches ctran-IB.** Check
`patchedContexts` and the per-device counters — a run that was supposed to inject
and shows zero is a failed run, not a passing one.

**A bad path fails open, not loud.** If `dlopen` of the requested library fails,
`buildIbvSymbols` falls back to `libibverbs.so.1` and initialization *succeeds*
against the real provider, so a typo'd path or an unmounted fbpkg yields a green,
completely uninjected run. This is the same failure signature as "IB was never on
the path", which is why the counter check above is not optional.

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
| `IbInjectionDso.cc` | exported verbs, real-provider delegation |
| `IbverbxSymbols.def` | table of verbs to forward, and how each is resolved |
| `version.script` | export list; a missing entry aborts `ibvInit()` |

`IbverbxSymbols.def` and `version.script` must stay a superset of what
`buildIbvSymbols` resolves: it uses `dlvsym` (or plain `dlsym` for two mlx5
symbols) on this library's handle, and the real provider is opened `RTLD_LOCAL`, so
a missing export is simply absent. `ib_injection_dlopen_test` pins that, in both
lookup styles.

The module sits outside `tests/` on purpose: the `.so` ships in an fbpkg to
collperf and farm jobs, so it is a shipped artifact, not test-only code. Its own
tests live in `ib_injection/tests/`.

## Appendix: mlx5

The mlx5 **data path** needs nothing special: the vtable patch overwrites
`ibv_context::ops`, and every QP on a context shares it, so `poll_cq` /
`post_send` / `post_recv` route through it no matter how the QP was created.

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
