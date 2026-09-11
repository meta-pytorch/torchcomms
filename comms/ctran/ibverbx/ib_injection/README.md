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
      (IbverbxSymbols.def + version.script)   ctx->ops.{4} = shims
```

The two need opposite things from the shim:

| | A · setup verbs | B · hot path |
|---|---|---|
| Verbs | `open_device`, `alloc_pd`, `reg_mr`, `create_cq`, `create_qp`, `modify_qp`, `query_*` … | `poll_cq`, `post_send`, `post_recv`, `req_notify_cq` |
| Resolved | once at load, by `dlvsym` | never — the pointer is read on every call |
| Must be exported | **yes** — `.def` supplies the body, `version.script` the version tag; a missing entry aborts `ibvInit()` | **no** — the shims are `local: *`, reached by address only |
| Interception | already in place: ctran calls the shim's own function | needs the vtable patch, since the pointer belongs to the real provider |

`dlvsym` matches a name **and** a version node, which is why set A needs both
files: compiling a verb is not enough, an untagged symbol is invisible to
`dlvsym` and a wrong node fails exactly like a missing one. Set B needs neither
— `ibv_open_device` is the one function that hands over the `ibv_context`, so
overwriting four of its `ops` entries on the way out is the whole seam. Neither
route requires a production change.

### Which enables which kind of injection

- **Failure injection** — a verb returns an error instead of succeeding. Works on
  both sets, because both run through shim-owned code. On setup verbs it reaches
  ctran's error and cleanup paths (`CtranIbSingleton` ctor throw → the
  `NCCL_CTRAN_BACKENDS=nvl,socket` fallback; mid-VC-setup partial construction).
  On the hot path it covers post failure and CQE error status.
- **Skew injection** — timing is shifted, with no error at all. Needs set B, and
  comes in two forms:
  - **Hold a completion** (`poll_cq`). Changes *when ctran observes* a transfer
    that already finished. Implemented; this is what a race like D118906933
    requires. See details in following diagram.
  - **Defer a post** (`post_send`). Returns success immediately but hands the WR
    to the NIC `n` ms later, so the DMA itself starts late while the caller runs
    on. Not implemented yet.

Holding is *withholding* a real CQE — no fabricated completions, no error codes,
and per-CQ order preserved exactly as a provider would: once a CQ's head is held,
nothing behind it passes.

```
SETUP                                      ARM
─────                                      ───
IBVERBX_IBVERBS_SO=/path/libibverbs.so     injection::addGate(
  → ibverbx::ibvInit()                       gatedDevice=1, afterDevice=0,
      dlopen(shim) ──┐                       opcode=RDMA_READ, afterCount=2,
                     │                       MANUAL)
                     ▼                         │
        ┌───────────────────────────────────────────────────┐
        │  ib_injection libibverbs.so                       │
        │                                                   │
        │  at load:  dlopen(real provider)                  │
        │                                                   │
        │  ibv_open_device → delegate to real, then         │
        │      overwrite ctx->ops.{poll_cq, post_send,      │
        │      post_recv, req_notify_cq} with shims         │
        │      + record ContextRecord(saved original ops)   │
        │                                                   │
        │  ibv_create_cq / _qp → delegate, then record      │
        │      CqRecord(deviceId) / QpRecord(qp_num) so     │
        │      rules can name a device or QP                │
        └───────────────────────────────────────────────────┘

RUN                                    the seam: ctran chases the vtable
───                                    pointer, not the exported symbol
CtranIb::progressInternal
  → devices[d].ibvCq->pollCq(1)
      → cq_->context->ops.poll_cq  ─────►  injection shim
                                             │ 1. deviceId ← CqRecord
                                             │ 2. drain real CQ into held queue
                                             │ 3. match rules on (dev, opcode)
                                             │ 4. gate closed for dev 1?
                                             │      ── yes → return 0  (CQ looks
                                             │                empty; ctran skips it)
                                             │      ── no  → release 1 held CQE
                                             │ 5. bump dev's cqesReleased
                                             ▼
                                        dev 0 releases F1,F2 · dev 1 releases none
                                        ⇒ the order the old FIFO mis-retires on

ASSERT
──────
poll getState() until device(0).cqesReleased==2 && device(1).cqesReleased==0
  EXPECT_FALSE(req1->isComplete())   ← fails on the pre-fix shared FIFO
releaseGate(); pump; EXPECT_TRUE(both complete)
```

Device ids are assigned per CQ in creation order, **not** per context: ctran opens
one `ibv_context` per NIC (`IbvDevice::ibvGetDeviceList`) and creates one CQ on
each (`CtranIbSingleton`), so id N is NIC N. Read them back from `getState()`
rather than assuming, since a consumer with a different CQ layout gets different
ids.


## Usage

Three surfaces. They differ only in **how a rule is armed** — the shim and the
seam are identical.

### C++ CI test

The only surface that can call the control API, so the only one that gets
assertions. Point the target at the shim and drive it through the bridge
(`IbInjectionControl.h`):

```python
env = {
    "IBVERBX_IBVERBS_SO": "$(location //comms/ctran/ibverbx/ib_injection:libibverbs.so)",
}
deps = ["//comms/ctran/ibverbx/ib_injection:ib-injection-control"]
```

```cpp
injection::reset();  // after ctran init, so only your traffic is measured
const uint32_t gate = injection::addGate(
    /*gatedDevice=*/1, /*afterDevice=*/0, IBV_WC_RDMA_READ, /*afterCount=*/2);
// ... drive ctran ...
const auto s = injection::getState();
EXPECT_EQ(s.device(0).cqesReleased, 2u);
EXPECT_EQ(s.device(1).cqesReleased, 0u);
injection::releaseGate(gate);
```

`injection::*` is the C++ bridge; each call is a thin wrapper over the exported C
entry point of the same name (`ibInjectionAddGate`, `ibInjectionGetState`, …),
which is what crosses the `dlopen` boundary.

`MANUAL` is the default gate policy on purpose: `progressInternal` drains every
CQ in a `while(1)` loop, so an auto-opening gate releases the held CQEs inside
the same `progress()` call that satisfied it — erasing the intermediate state the
assertion needs.

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
`addGate()` builds — one engine, two front ends:

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
| `IbInjectionApi.h` | control ABI; the only declaration shared across the dlopen boundary |
| `IbInjectionDso.cc` | exported verbs, the 4 shims, real-provider delegation |
| `IbverbxSymbols.def` | table of verbs to forward, with the symbol version each is resolved at |
| `InjectionEngine.{h,cc}` | rule matching, held-CQE queues, per-device counters |
| `version.script` | export list; a missing entry aborts `ibvInit()` |
| `IbInjectionControl.{h,cc}` | test-side dlsym bridge |

`IbverbxSymbols.def` and `version.script` must stay a superset of what
`buildIbvSymbols` resolves: it uses `dlvsym` on this library's handle, and the
real provider is opened `RTLD_LOCAL`, so a missing export is simply absent.
`ib_injection_dlopen_test` pins that.

The module sits outside `tests/` on purpose: the `.so` ships in an fbpkg to
collperf and farm jobs, so it is a shipped artifact, not test-only code. Its own
tests live in `ib_injection/tests/`.

## Appendix: mlx5

The mlx5 **data path** is already covered, because the shim patches
`ibv_context::ops` and every QP on a context shares it — `poll_cq` / `post_send` /
`post_recv` land in a shim no matter how the QP was created.

What sits outside that is `mlx5dv_*`, the vendor escape hatch for features with no
vendor-neutral verb. It is exported from libmlx5 and called by name through a
second handle ibverbx opens with a hardcoded soname, which no path override
reaches.

The one that matters is `mlx5dv_create_qp`. With `NCCL_CTRAN_IB_ENABLE_OOO_RQ`
set, every data QP comes from it (`createRcQpWithOooDp`) rather than from the
shim's `ibv_create_qp` wrapper, so no `QpRecord` is created for it. Ctrl / notify
/ atomic QPs still register, so the registry is not empty — it holds exactly 3
QPs, and a `qp=data:N` rule matches nothing and reports zero fires instead of
failing loudly. That is indistinguishable from "IB was never on the path", so
until the coverage below lands the shim aborts when it is set. It defaults
to `false` but `GB300_ENVS` sets it to `1`, so a GB300 run needs an explicit `=0`.

**Planned coverage: serve both handles from this one `.so`.** Version nodes are
just named buckets in a version script; one library can export `IBVERBS_1.1` and
`MLX5_1.3` side by side, exactly as `CTRAN_IB_INJECTION_1.0` already coexists with
the `IBVERBS_*` nodes here. `dlopen` is refcounted by resolved path, so both of
ibverbx's handles land on the same instance and the same engine, and wrapping
`mlx5dv_create_qp` then registers OOO_DP data QPs exactly like `ibv_create_qp`
does.
