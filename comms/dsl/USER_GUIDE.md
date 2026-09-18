# comms/dsl — User Guide

Build a **customized, autotunable collective** by writing only the part that differs from a plain
transfer (a per-tile hook), while the framework owns the ~95% plumbing (schedule, multi-peer
addressing, staging, signal/wait) and turns performance into an autotuner run.

This guide is task-oriented and runnable. It covers the **three layers of control** (call a
collective → add your hooks → compose your own collective from send/recv primitives) and the
**auto-tuning** workflow end to end.

---

## 0. Mental model (30 seconds)

| Piece | What it is | You touch it? |
|---|---|---|
| **Transport** | Cross-GPU staging + signaling, created once via `nvl_rendezvous` over PyTorch symmetric memory. | Create it (one line). |
| **Hook + `Ctx`** | The 5% you write: a per-tile transform. Two tiers — **value** (`produce`/`consume`, per-thread) and **block-tile / layout** (`layout_hook`, CTA-cooperative over a SMEM tile). | Write it (optional). |
| **Collective** | A shipped schedule (`all_to_all`) that runs your hook over the fused `peer × block` transfer. | Call it. |
| **`Config` + `Key`** | `Config` = launch knobs the tuner sweeps; `Key` = how a tuned config is looked up at runtime. | Autotuner fills these. |
| **Adapter** | Plugs a collective into the comms-owned tuner (`comm_tuning`). | Not needed for a2a (shipped). |

Import surface:

```python
from comms.dsl import nvl_rendezvous                      # transport
from comms.dsl.cute import all_to_all, all_to_all_zc      # collectives
from comms.dsl.collectives import A2A                     # collective object (callable + .autotune)
from comms.dsl.cute.send_recv import pipelined_sendrecv   # send/recv primitive (Layer 3)
import cutlass.cute as cute                               # only if you WRITE a hook
```

### Shared test harness (used by every example below)

Every example is a `run(rank, ws, group, device)` function. Drive it locally on an 8-GPU host with:

```python
import os, torch, torch.distributed as dist, torch.multiprocessing as mp

def _spawn(run):
    def _worker(rank, ws, port):
        os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=ws)
        run(rank, ws, dist.group.WORLD, torch.device(f"cuda:{rank}"))
        dist.barrier(); dist.destroy_process_group()
    ws = min(torch.cuda.device_count(), 8)
    mp.spawn(_worker, args=(ws, 29500), nprocs=ws, join=True)
```

Run a file with `buck2 run @fbcode//mode/opt //your/target -- ...` (single-host correctness/CI).
Production shapes (GB300, multi-node) run on MAST — see §5.

---

## Layer 1 — Call a shipped collective (simplest)

Plain all-to-all. No hook, no tuning needed: `config=None` looks up the tuned table for this
(shape, hardware) and falls back to a safe analytic default on a miss.

```python
from comms.dsl import nvl_rendezvous
from comms.dsl.cute import all_to_all

def run(rank, ws, group, device):
    CHUNK = 1 << 20                                  # per-peer elements; numel = ws * CHUNK
    inp = torch.randn(ws * CHUNK, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    t = nvl_rendezvous(group, device, per_peer_bytes=CHUNK * inp.element_size())
    all_to_all(t, out, inp)                          # config=None -> tuned lookup / analytic default
    # `out` now holds the equal-split all-to-all result (bit-exact vs dist.all_to_all_single).
```

The built-in **non-contiguous transpose** all-to-all is one argument away — `rows=R` makes each
received `[rows, cols]` chunk arrive transposed to `[cols, rows]` (this selects the block-tile
transpose hook under the hood; see Layer 2b):

```python
    all_to_all(t, out, inp, rows=R)                  # R must divide the per-peer chunk; R, cols multiples of 32
```

---

## Layer 2 — Add your own hooks

Write the 5% (the transform); the framework runs it inside the fused schedule.

### 2a. Value hook (`produce` / `consume`) — per-thread, e.g. scale / quantize / accumulate

A value hook works in registers on a per-thread tile fragment. `produce` runs on the **send** leg
(input → fragment), `consume` on the **recv** leg (fragment → output). Bodies are constexpr-traced,
so constants are baked in.

```python
import cutlass.cute as cute
from comms.dsl import nvl_rendezvous
from comms.dsl.cute import all_to_all

def scale_produce(ctx):                       # send: load input tile, multiply by 2
    frag = cute.make_fragment_like(ctx.part)  # ctx.part = this thread's tile; ctx.atom = copy atom
    cute.copy(ctx.atom, ctx.part, frag)
    frag.store(frag.load() * 2.0)             # <-- your transform (register-only)
    return frag

def bias_consume(ctx, frag):                  # recv: add 1, store to output tile
    frag.store(frag.load() + 1.0)
    cute.copy(ctx.atom, frag, ctx.part)

def run(rank, ws, group, device):
    CHUNK = 1 << 20
    inp = torch.randn(ws * CHUNK, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    t = nvl_rendezvous(group, device, per_peer_bytes=CHUNK * inp.element_size())
    all_to_all(t, out, inp, produce=scale_produce, consume=bias_consume, variant="scalebias")
    # out == (all_to_all(inp * 2)) + 1, fused into the transfer leg (no extra HBM pass).
```

`Ctx` also exposes read-only **facts** for position/peer-dependent transforms: `ctx.coord` (tile
index), `ctx.peer` (dest/source rank), `ctx.rank` / `ctx.world_size`, `ctx.rows` / `ctx.cols` /
`ctx.chunk`. A per-peer dequant scale, positional mask, etc. read these without touching machinery.

> Pass `variant="..."` whenever you use a non-default hook — it tags this hook's tuned configs so
> they don't collide with the default copy hook's (required by `A2A`; see §4).

### 2b. Block-tile / layout hook (`layout_hook`) — CTA-cooperative, e.g. transpose / permute

Coalescing-critical layout changes need the whole CTA to cooperate on a 2D SMEM tile (a per-thread
gather would be uncoalesced). The framework coalesced-loads a `[tile, tile]` input tile into padded
SMEM and barriers; **your hook does the in-SMEM transform + coalesced store**. Contract (`BlockCtx`):
`bctx.sA` (loaded SMEM tile), `bctx.dst` (destination tensor), `bctx.tile`, `bctx.block_rows`,
`bctx.tx`/`bctx.ty` (this thread's lane/row), `bctx.br`/`bctx.bc` (2D tile coords).

The **shipped block-tile hook is the transpose**, reached via `rows=R` (Layer 1). Its hook body is
the template for writing your own:

```python
def transpose_tile(bctx):                     # the shipped reference (comms.dsl.cute.hooks)
    tile = bctx.tile
    for r in range(0, tile, bctx.block_rows):
        bctx.dst[(bctx.bc * tile + bctx.ty + r, bctx.br * tile + bctx.tx)] = \
            bctx.sA[(bctx.tx, bctx.ty + r)]   # swapped indices = transpose; +1 SMEM pad => no bank conflict
```

Runnable today (transpose is wired into the public entry):

```python
def run(rank, ws, group, device):
    R, COLS = 512, 512                        # per-peer chunk = R*COLS; both multiples of 32
    inp = torch.randn(ws * R * COLS, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    t = nvl_rendezvous(group, device, per_peer_bytes=R * COLS * inp.element_size())
    all_to_all(t, out, inp, rows=R)           # each [R,COLS] chunk arrives transposed to [COLS,R]
```

> **Scope note (be honest with users):** the block-tile **contract** (`BlockCtx` + `layout_hook`)
> is general, but the public `all_to_all(rows=R)` currently wires the shipped `transpose_tile`
> reference. Plumbing an *arbitrary* user block-tile hook (e.g. a permute) through the public entry
> is a near-term extension; the value-hook tier (2a) is the fully general path today.

---

## Layer 3 — Compose your own collective from send/recv primitives

When you need a schedule the framework doesn't ship (a ring, a custom pattern), drop to the
**send/recv primitive** and keep the framework's staging + signal/wait + slot pipeline. Your hooks
still apply.

`pipelined_sendrecv(transport, send_buf, recv_buf, send_peer, recv_peer, *, num_blocks,
mode="bidir"|"send"|"recv", produce=copy_produce, consume=copy_consume)` — a graph-safe, slot-
pipelined whole-buffer transfer for one peer pair. Size the transport `per_peer_bytes >= numel*elem`.

Minimal: a neighbor exchange (send to `rank+1`, recv from `rank-1`), optionally with a value hook.

```python
from comms.dsl import nvl_rendezvous
from comms.dsl.cute.send_recv import pipelined_sendrecv

def run(rank, ws, group, device):
    N = 1 << 20
    send_buf = torch.full((N,), float(rank), dtype=torch.bfloat16, device=device)
    recv_buf = torch.empty_like(send_buf)
    t = nvl_rendezvous(group, device, per_peer_bytes=N * send_buf.element_size())
    pipelined_sendrecv(
        t, send_buf, recv_buf,
        send_peer=(rank + 1) % ws, recv_peer=(rank - 1) % ws,
        num_blocks=8, mode="bidir",
        # produce=my_produce, consume=my_consume,   # optional: transform in-flight
    )
    # recv_buf now holds the previous rank's shard (== (rank-1) % ws).
```

Compose a **ring all-gather** by looping it `ws-1` steps, rotating the chunk each step (each step is
one `pipelined_sendrecv`; the framework owns the per-step staging/signal). For the lowest level of
control, the transport's device ops — `put` / `get` / `signal` / `wait` (`comms.dsl.cute.nvl_ops`) —
are callable directly inside your own `@cute.kernel`, but that is rarely needed.

---

## 4. Auto-tuning — how it works, and how to run it

The launch parameters that win on one shape/hardware lose on others, so performance is an
**autotuner run**, not hand-tuning. You do not write a tuner; you drive the comms-owned
`comm_tuning` engine through a one-line entry.

### What happens under the hood

1. **`A2A(...).autotune()`** hands the engine an adapter that knows your hook, `variant`, candidate
   search, and a correctness `reference`.
2. The engine, **per input shape** (a size ladder or your `shapes=`):
   - builds a **`Key`** = `(world_size, dtype, numel, rows, transport_kind, device, backend, variant)`
     — the *same* key the runtime rebuilds, so lookups match (no drift);
   - enumerates **candidate `Config`s** — the shipped expert size-banded grid by default, or your
     `{field: [values]}` dict (cartesian) or `(spec, key) -> [Config]` callable;
   - runs each candidate, times it, and **correctness-gates it bit-exact** (`assert_close(atol=0,
     rtol=0)`) against the `reference` (default = `dist.all_to_all_single`; a transforming hook needs
     its own reference);
   - keeps the lowest-latency **correct** candidate for that key.
3. It writes a generated `{Key: Config}` table (`cute/a2a/generated/a2a_tuned_configs.py`). Commit it.
4. At runtime, `all_to_all(..., config=None)` rebuilds the key and looks it up; a miss falls back to
   the safe analytic default. **User code never changes between tuned and untuned.**

The **Config knobs** the tuner sweeps (each defaults to `0` = "use the analytic pick", so the default
config reproduces the analytic defaults): `num_blocks` (grid = `ws × num_blocks`), `num_threads`,
`num_slots`, `unroll`, `cluster` / `cluster_y` (CGA), `tma_drain_warps`, and `primitive`
(`copy` staging / `tma` / `direct` zero-copy / `ce` copy-engine).

### Tuning a shipped collective (no code)

```python
# tune_my_a2a.py  — the body of the tuning-job entrypoint
from comms.dsl.collectives import A2A
A2A(backend="cute").autotune()          # default copy hook = plain a2a
```

### Tuning YOUR hook (no adapter file)

```python
from comms.dsl.collectives import A2A
A2A(backend="cute",
    produce=scale_produce, consume=bias_consume, variant="scalebias",
    reference=my_reference,             # (inputs, group)->Tensor capturing your hook's semantics
    # candidates=<dict | callable>,     # optional; default = expert size-banded grid
).autotune(shapes=PROD_SHAPES)          # PROD_SHAPES = list of per-rank byte sizes or specs
```

`my_reference` for the scale/bias hook above:

```python
def my_reference(inputs, group):
    ref = torch.empty_like(inputs.input)
    dist.all_to_all_single(ref, inputs.input * 2.0, group=group)
    return ref + 1.0
```

### Where and how to launch it (MAST — concrete)

Tuning runs on **MAST with conda delivery** (the DSL JIT-compiles in a conda env; production shapes
only exist on cluster GPUs). `buck` only builds/launches the launcher; MAST runs the sweep.

**As a master user, you provide your own tenant via `--entitlement`** (the `rmAttribution` leaf that
holds your GB300 quota). Full command (quota path):

```bash
buck2 run @fbcode//mode/opt //comms/dsl/tests:mast_launch -- \
    --tune \                                  # run parent -> select -> emit-table in one job
    --delivery conda \                        # required for tuning (JIT + source overlay)
    --nnode 2 --ppn 4 --nvl-hosts 2 \         # 2 GB300 hosts x 4 GPU = world_size 8, one NVL72 domain
    --hw gb300_dsf \                          # the registered MastProdCluster sub-type (quota path)
    --cluster MastProdCluster \
    --region uco \                            # locality (ignored if --flex-pool-id is set)
    --entitlement <YOUR_TENANT_LEAF> \        # rmAttribution: YOUR tenant with GB300 quota  <-- master user sets this
    --identity <YOUR_DATA_PROJECT_ACL> \      # hpcIdentity / data-project ACL (default: networkai_comms_tools)
    --oncall <YOUR_ONCALL> \                  # default: hpc_comms_lib
    --flex-pool-id '' \                       # '' = use the quota path; omit to use the default NetAI burn-in pool
    --tune-output-dir /tmp/a2a_tune \         # remote dir: parent results + selected table
    --submit                                  # ACTUALLY schedule (default is dry-run — no capacity used)
```

Notes for the runner:
- **`--entitlement` / `--rm-attribution`** is the tenant hook. Point it at your team's leaf that has
  GB300 quota. `--identity` (hpcIdentity) is the data-project/ACL your job runs under.
- **`--flex-pool-id ''`** selects the quota path. Leaving it default routes to a dedicated NetAI
  burn-in pool (no quota / no preempt) if you have access; the quota path is the portable choice.
- **`--hw gb300_dsf`** is the sub-type MastProdCluster registers; a bare `gb300` is rejected.
- Omit `--submit` first to **dry-run** (validates the request, uses no capacity).
- Smoke a tiny sweep before a full run: add `--bench-args "--max-sizes 2 --max-candidates 2"`.
- The job runs parent → select → prints the generated `TUNED_A2A_CONFIGS` table; copy it into
  `comms/dsl/cute/a2a/generated/a2a_tuned_configs.py` and commit. **Re-tune = re-run this; no kernel
  edits.**

### Consuming the tuned table (runtime)

Nothing to change — the same call now hits the tuned config:

```python
all_to_all(t, out, inp, rows=R)             # config=None -> get_a2a_config(key) -> tuned or default
```

---

## 5. Gotchas

- **Custom hook ⇒ `variant=` + `reference=`.** `variant` keeps your tuned entries from colliding
  with the default copy hook (`A2A` raises without it). `reference` is required for a *transforming*
  hook — the default bit-exact gate compares against a plain `all_to_all_single`, which won't match a
  transformed output.
- **Hooks are constexpr-traced.** Bake constants into the body (or use module constants / distinct
  variants); they are not runtime parameters.
- **Zero-copy returns a view.** `all_to_all_zc` (and the internal transpose zero-copy path) returns a
  view into the transport's symmetric-memory output buffer. Consume or `.clone()` it **before**
  reusing/freeing the transport, or before a graph replay overwrites it. The staging `all_to_all`
  writes your owned `output`, so it has no such caveat.
- **Transport reuse across geometries.** Switching `num_blocks` / `primitive` on a *reused* transport
  is unsafe without a drain (not implemented); a runtime guard raises. Use a fresh transport per
  geometry, or set `COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1` if your calls are device-sync-separated.
- **Transpose floor.** `rows>0` uses a 32×32 SMEM tile, so its practical floor is 2KB/peer; plain
  (`rows=0`) a2a goes down to 32B.
- **Sizing.** `all_to_all`: `per_peer_bytes = (numel // ws) * elem`. `pipelined_sendrecv` (whole
  buffer): `per_peer_bytes >= numel * elem`. `numel % world_size == 0`.
