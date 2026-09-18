# comms/dsl: customized collectives without the plumbing - a hook in, an autotuned kernel out

`comms/dsl` framework lets a kernel developer (user) build a customized, autotunable collective in a DSL (CuTe today; the façade is backend-parametrized so Triton can be re-added) by writing only the small part that differentiates their kernel: per-tile hooks and optionally transports, while the framework owns the generic ~95% (schedule, multi-peer addressing, signal/wait protocol) and automates the performance tuning.

## Why

Researchers prototype new ideas as kernels in DSLs like CuTe/Triton instead of CUDA/C++ for the fast iteration loop. Scaling the idea needs communication, and that usually leads to repetitive work: days-to-weeks of race-prone stage/signal/wait/fence logic, re-derived per kernel, with bugs that could pass tests and fail at scale. Notably, the part that differentiates the kernel is typically small: a transpose, a quantize, or an accumulate, while a lot around it is plumbing.

For instance, the `all_to_all_single_non_contig` kernel is ~800 lines, of which roughly 95% is generic machinery, including the signaling protocol, the pipeline, symmetric-memory staging, multi-peer addressing, and tile-size math. The remaining ~5% is layout transform, which is what really makes this kernel unique.

Furthermore, reaching correctness is only half the journey. Each kernel is then tuned against production shapes maintained per hardware tier, because the launch parameters that win on one shape/hardware may lose on the others.

comms/dsl breaks both loops. It provides the customizable boilerplate, and it turns performance into an autotuner run.

## The pieces

| Piece | What it is |
|---|---|
| **Transport** | A fabric binding (NVLink today, IB later): cross-GPU memory + signaling state, created once via a `rendezvous` over PyTorch symmetric memory. The framework ships defaults and the user either sets one up or could provide their own. |
| **Endpoint (`PeerEndpoint`)** | How the transport hands a schedule per-peer addresses - `send_dst` (where to write into a peer), `recv_src` (where to read its data), and the signal slots - so no schedule touches raw pointers. Two forms: a host-resolved `PeerEndpoint` for a single peer (used by `send_tiles`/`recv_tiles`), and a device-side table of all peers that a fused collective indexes by peer in-kernel (e.g. `all_to_all`). |
| **Ops** | The transport's four device functions that act on those addresses: `put` (remote write), `get` (local read), `signal`/`wait` (the data-ready handshake). The only fabric-specific (NVLink vs IB) device code; hooks and schedules call them and stay transport-agnostic. |
| **Hook + `Ctx`** | The 5% the user writes, in **two tiers**. *Per-element / value* — `produce(ctx) -> tile` / `consume(ctx, tile)`: a per-thread transform on the tile *value* (scale / quantize / accumulate), over the per-tile `Ctx` (CuTe exposes `part`/`atom` + read-only facts `coord`/`peer`). *Block-tile / layout* — `layout_hook(bctx)`: a CTA-cooperative transform over a SMEM-staged 2D tile (`BlockCtx` exposes the loaded tile `sA` + tile coords), for coalescing-critical layout changes (transpose / permute / reshape) where a per-element gather would go uncoalesced. The framework owns the coalesced SMEM load + barriers; the hook does the in-SMEM transform. The portable contract is the hook *role*; the field set is per-DSL, so a hook body is written against its backend's `Ctx`. |
| **Collective** | The shipped schedule (e.g. `all_to_all`) that runs the hook over the fused `peer x block` transfer, owning all addressing and signal/wait. |
| **`Config` + `Key`** | `Config` = the launch tunables the autotuner sweeps; `Key` = how a tuned config is looked up at runtime. Both are defined by the user (see principles). |
| **Adapter** | A thin class that plugs a collective into the comms-owned tuner engine (`comm_tuning`). |

Flow: a `Transport` gives the kernel per-peer addresses; the `Collective` loops tiles, calling the user's `Hook` (compute) and the `ops` (move + signal); a `Config`, looked up by `Key`, sets the launch parameters.

## Design principles

- Own the 95%, expose the 5%. The framework owns the schedule and the race-prone protocol; the user writes a hook — a per-element `produce`/`consume` value transform (quantize, scale, accumulate) or a block-tile `layout_hook` (transpose, permute, reshape) — and supplies a transport. No fences, no wait loops, no deadlock reasoning in user code.
- Performance is autotuned, and the tuner is generic. The user declares two things: a `Key` (which input properties should map to one tuned config - size, dtype, world size, layout, whatever matters for that kernel) and a `Config` (the launch knobs to sweep). A shared engine sweeps configs offline and emits a `{Key: Config}` map; at runtime the collective rebuilds the same key and looks it up (safe default if absent). Changing shapes means re-running the tuner, not rewriting the kernel.
- Spectrum of control. Plug-and-play collective (write a hook) -> `send_tiles`/`recv_tiles` (keep your own schedule) -> raw `put`/`get`/`signal`/`wait` ops (full control). Climb only as high as you need.
- Backend-parametrized, code per-DSL. The transport contract, hook roles, lookup `Key` shape, and the `comms.dsl.collectives` façade are DSL-agnostic; the `Config` launch knobs and the device kernels are per-DSL (partitions, DSL-specific tunables), since device code is not portable across DSLs. CuTe is the shipped backend; the façade keeps a `backend=` seam so a second DSL (Triton) re-registers without touching callers.

## Example: a custom collective (a2a non-contig), end to end

The whole workflow for a non-contiguous all-to-all - write it, then autotune it.

### 1. Write it (the ~5%)

A transport, one hook, one call:

```python
from comms.dsl import nvl_rendezvous
from comms.dsl.cute import all_to_all

t = nvl_rendezvous(group, dev, per_peer_bytes=chunk_bytes)   # transport (staging + signaling)
all_to_all(t, out, inp, rows=R)                              # config=None -> tuned lookup
```

`nvl_rendezvous` allocates the per-peer staging buffer + signal pad once; `rows` describes the 2D tile layout the hook reads; with the default identity copy hook this is a plain all-to-all.

The hook is the 5%, in two tiers. A **per-element** `produce`/`consume` transforms the tile *value* (scale / quantize / accumulate). The `rows > 0` transpose is the flagship **block-tile** hook (`transpose_tile`): the framework coalesced-loads each `[tile, tile]` chunk into padded SMEM and barriers (via the shared substrate leaf `_block_tile_u`, the block-tile twin of the value leaf `_send_u`), and the hook coalesced-stores it transposed — both gmem legs stay coalesced, unlike a per-element gather (the CuTe twin of a Triton on-chip `tl.trans`). The transpose SMEM tile is `32x32`, so `rows > 0` has a `2KB` floor (a sub-tile transpose is degenerate); plain (`rows=0`) a2a still goes down to `32B`.

The framework runs the fused `peer x block` schedule (multi-peer addressing, signal/wait) in one launch, validated bit-exact against `dist.all_to_all_single`.

> **Backend note (honest scope).** The per-element `produce`/`consume` value hooks (scale / quantize / accumulate) run on the CuTe copy-staging schedule, and are **proven reusable across collectives** — the same value leaf (`_send_u`/`_send_slot`) is composed by both `all_to_all` and the standalone `send_tiles`/`recv_tiles`. The block-tile `rows > 0` transpose (the non-contiguous headline) ships on the **zero-copy transfer** (`all_to_all(rows=R)` auto-selects an orchestrated mid-band / fused large-band kernel, both composing the shared `_block_tile_u` leaf), on-par-or-exceeding the Triton baseline across `0.5MB–128MB` on 8×GB300 (bit-exact, SM-matched). The block-tile tier ships today as the `BlockCtx` contract + `_block_tile_u` leaf + this **a2a reference**; reuse across other collectives is a documented backlog item (unlike the value tier). A fused copy-staged block-tile variant was evaluated and removed as send-leg-bound at the mid band (archived; see the CuTe backend notes). The TMA bulk-copy and raw zero-copy (`direct`/`ce`) paths move raw bytes and apply no hook.

### 2. Autotune it

No adapter to write - declare the hook on a collective *object* and call `.autotune()`. The object is both the callable collective and the thing that tunes itself, because it already knows its hook, `variant`, candidate search, and correctness reference:

```python
from comms.dsl.collectives import A2A

# The non-contig transpose: rows>0 selects the block-tile transpose hook, and its own tuned
# table falls out of the `rows` field in the lookup key (no produce= -- the transpose is a
# block-tile LAYOUT hook, not a per-element value hook).
a2a = A2A(
    backend="cute",
    reference=transpose_ref,           # (inputs, group)->Tensor; default = nccl identity
    # candidates=<dict | callable>     # optional; default = shipped expert size-banded grid
)

a2a(t, out, inp, rows=R)               # callable: config=None -> per-shape tuned lookup
a2a.autotune(shapes=PROD_SHAPES)       # sweep -> select -> generate (one tuning job)

# A per-element VALUE hook (quantize / scale / accumulate) instead plugs in as produce=/consume=
# and MUST carry an explicit `variant` so its tuned configs don't collide with the default copy:
#   A2A(backend="cute", produce=quantize_produce, consume=dequant_consume, variant="fp8")
```

`.autotune()` drives the comms-owned `comm_tuning` engine and writes the generated table; `__call__` rebuilds the **same** key (including `rows` and `variant`, so the transpose / a custom value hook never reuses the default copy hook's config) and looks it up. The candidate search is pluggable: omit it for the expert default grid, pass a flat `{CuteA2AConfig_field: [values]}` dict (cartesian), or a `(spec, key) -> [CuteA2AConfig]` callable.

```python
# comms/dsl/cute/a2a/generated/a2a_tuned_configs.py  (generated; do not hand-edit)
from comms.dsl.cute.a2a.tuning import CuteA2AConfig, CuteA2AKey

TUNED_A2A_CONFIGS = {
    CuteA2AKey(world_size=8, dtype="bfloat16", numel=8192, rows=0,
               transport_kind="NvlTransport", device="GB300", backend="cute", variant=""):
        CuteA2AConfig(num_blocks=8, num_threads=512, primitive="copy"),
    # ... one row per tuned key (per shape x hardware x hook) ...
}
```

Re-tune = re-run `.autotune()`, regenerate the table - no kernel edits.

**Where it runs.** Perf tuning runs on **MAST/conda**, not a local buck par: the DSL JIT-compiles in a conda env, and production shapes (GB300, multi-node, TP=8) only exist on cluster GPUs. So `.autotune()` is the *body of the tuning-job entrypoint*; you launch it with the MAST launcher (`buck2 run //comms/dsl/tests:mast_launch -- --tune ...` -- buck only builds/launches the launcher; MAST runs the sweep). The framework stays buck-importable for single-host correctness tests / CI.

**Reuse contract (M1).** Each `(shape)` should get its own transport; reusing one transport across configs of differing *geometry* (`num_blocks`/`primitive`) is unsafe without a drain, which is not implemented here yet. A runtime guard raises on a geometry switch on a reused transport (no runtime drain, so reinterpreting in-flight bytes would corrupt staging); `COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1` downgrades it to a silent advance for callers whose successive launches are device-sync-separated (a benchmark / tuner sweeping configs at one size). A fully user-declared `Key` (arbitrary fields) and a numel shape-bucket are planned extensions. See `USER_GUIDE.md` for runnable, task-oriented examples of all three usage layers + the autotuner workflow.

## What the same pattern enables next

Concrete doables on the shipped transport + ops + Config/Key + tuner:

- reduce-scatter: an accumulate-on-`consume` hook (sum the received tile into the output) over the same schedule family.
- all-gather: the gather schedule with the default copy hook.
- quantized all-to-all: a `produce` hook that casts/scales to fp8 and a `consume` that dequantizes, fusing the conversion into the transfer leg (no extra HBM pass).
- variable-split a2a (MoE dispatch / combine): the same collective with per-rank split sizes added to the `Key` and layout.

Each is a new hook and/or a sibling schedule on the same substrate - not a new comm stack.

Further out, and a bigger milestone than the items above: comm-compute overlap - hiding a compute kernel (e.g. a GEMM or MoE expert) under the collective via pipelining to cover its latency.
