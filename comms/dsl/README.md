# comms/dsl: customized collectives without the plumbing - a hook in, an autotuned kernel out

`comms/dsl` framework lets a kernel developer (user) build a customized, autotunable collective in DSLs (Triton or CuTe) by writing only the small part that differentiates their kernel: per-tile hooks and optionally transports, while the framework owns the generic ~95% (schedule, multi-peer addressing, signal/wait protocol) and automates the performance tuning.

## Why

Researchers prototype new ideas as kernels in DSLs like Triton/CuTe instead of CUDA/C++ for the fast iteration loop. Scaling the idea needs communication, and that usually leads to repetitive work: days-to-weeks of race-prone stage/signal/wait/fence logic, re-derived per kernel, with bugs that could pass tests and fail at scale. Notably, the part that differentiates the kernel is typically small: a transpose, a quantize, or an accumulate, while a lot around it is plumbing.

For instance, the `all_to_all_single_non_contig` kernel is ~800 lines, of which roughly 95% is generic machinery, including the signaling protocol, the pipeline, symmetric-memory staging, multi-peer addressing, and tile-size math. The remaining ~5% is layout transform, which is what really makes this kernel unique.

Furthermore, reaching correctness is only half the journey. Each kernel is then tuned against production shapes maintained per hardware tier, because the launch parameters that win on one shape/hardware may lose on the others.

comms/dsl breaks both loops. It provides the customizable boilerplate, and it turns performance into an autotuner run.

## The pieces

| Piece | What it is |
|---|---|
| **Transport** | A fabric binding (NVLink today, IB later): cross-GPU memory + signaling state, created once via a `rendezvous` over PyTorch symmetric memory. The framework ships defaults and the user either sets one up or could provide their own. |
| **Endpoint (`PeerEndpoint`)** | How the transport hands a schedule per-peer addresses - `send_dst` (where to write into a peer), `recv_src` (where to read its data), and the signal slots - so no schedule touches raw pointers. Two forms: a host-resolved `PeerEndpoint` for a single peer (used by `send_tile`/`recv_tile`), and a device-side table of all peers that a fused collective indexes by peer in-kernel (e.g. `all_to_all`). |
| **Ops** | The transport's four device functions that act on those addresses: `put` (remote write), `get` (local read), `signal`/`wait` (the data-ready handshake). The only fabric-specific (NVLink vs IB) device code; hooks and schedules call them and stay transport-agnostic. |
| **Hook + `Ctx`** | The 5% the user writes: `produce(ctx) -> tile` and `consume(ctx, tile)`. `Ctx` is the per-tile view the framework hands the hook - input/output pointer, flat index, mask, and within-chunk position. This is where a transpose / gather / quantize / accumulate goes. |
| **Collective** | The shipped schedule (e.g. `all_to_all`) that runs the hook over the fused `peer x block` transfer, owning all addressing and signal/wait. |
| **`Config` + `Key`** | `Config` = the launch tunables the autotuner sweeps; `Key` = how a tuned config is looked up at runtime. Both are defined by the user (see principles). |
| **Adapter** | A thin class that plugs a collective into the comms-owned tuner engine (`comm_tuning`). |

Flow: a `Transport` gives the kernel per-peer addresses; the `Collective` loops tiles, calling the user's `Hook` (compute) and the `ops` (move + signal); a `Config`, looked up by `Key`, sets the launch parameters.

## Design principles

- Own the 95%, expose the 5%. The framework owns the schedule and the race-prone protocol; the user writes a per-tile `produce`/`consume` hook (transpose, gather, quantize, accumulate) and supplies a transport. No fences, no wait loops, no deadlock reasoning in user code.
- Performance is autotuned, and the tuner is generic. The user declares two things: a `Key` (which input properties should map to one tuned config - size, dtype, world size, layout, whatever matters for that kernel) and a `Config` (the launch knobs to sweep). A shared engine sweeps configs offline and emits a `{Key: Config}` map; at runtime the collective rebuilds the same key and looks it up (safe default if absent). Changing shapes means re-running the tuner, not rewriting the kernel.
- Spectrum of control. Plug-and-play collective (write a hook) -> `send_tile`/`recv_tile` (keep your own schedule) -> raw `put`/`get`/`signal`/`wait` ops (full control). Climb only as high as you need.
- Contracts shared, code per-DSL. Triton and CuTe share the same op names, hook roles, and Config/Key shapes; the device kernels are written per DSL (flat pointers vs partitions), since device code is not portable across DSLs.

## Example: a custom collective (a2a non-contig), end to end

The whole workflow for a non-contiguous all-to-all - write it, then autotune it.

### 1. Write it (the ~5%)

A transport, one hook, one call:

```python
from comms.dsl import nvl_rendezvous
from comms.dsl.triton import all_to_all
from comms.dsl.triton.hooks import transpose_produce   # the 5%: the layout transform

t = nvl_rendezvous(group, dev, per_peer_bytes=chunk_bytes)   # transport (staging + signaling)
all_to_all(t, out, inp, produce=transpose_produce, rows=R)   # config=None -> tuned lookup
```

`nvl_rendezvous` allocates the per-peer staging buffer + signal pad once; `rows` describes the 2D tile layout the hook reads; with no `produce` hook this is a plain all-to-all.

The hook is the 5%. `ctx` is the per-tile view (input pointer, flat index, mask, within-chunk position `pos` with `rows`/`cols`); it returns the tile to send - here, the transposed source position, fused into the transfer leg with no extra HBM pass:

```python
@triton.jit
def transpose_produce(ctx):
    r = ctx.pos % ctx.rows           # position in the transposed [cols, rows] layout
    c = ctx.pos // ctx.rows
    src = r * ctx.cols + c           # source position in the [rows, cols] layout
    base = ctx.idx - ctx.pos         # chunk base in the input
    return tl.load(ctx.ptr + base + src, mask=ctx.mask)
```

The framework runs the fused `peer x block` schedule (multi-peer addressing, signal/wait) in one launch, validated bit-exact against `dist.all_to_all_single` on 4 ranks.

### 2. Autotune it

Write one thin adapter - the engine is generic, so you only declare your `Key` and the `Config`s to sweep:

```python
class A2ATuningAdapter(CommKernelTuningAdapter):
    def enumerate_input_specs(self, world_size): ...      # the shapes to tune
    def make_key(self, spec, world_size): ...             # YOUR key (must match runtime)
    def enumerate_candidate_configs(self, spec, key): ... # YOUR configs to sweep
    def make_inputs(self, spec, *, rank, world_size, device): ...   # tensors + nvl_rendezvous
    def run_candidate(self, inputs, config, group):       # the collective with the tuned config
        all_to_all(inputs.transport, inputs.output, inputs.input, rows=inputs.rows, config=config)
        return inputs.output
    def run_baseline(self, inputs, baseline, group): ...  # e.g. NCCL, the speed baseline
    def check_correctness(self, candidate, reference): ...
```

Parent mode sweeps every candidate, benchmarks it against your baseline, and checks correctness; select mode picks the winner per key and writes a generated table:

```python
# comms/dsl/triton/generated/a2a_tuned_configs.py  (generated; do not hand-edit)
from comms.dsl.triton.collectives_tuning import A2AConfig, A2AKey

TUNED_A2A_CONFIGS = {
    A2AKey(world_size=8, dtype="bfloat16", numel=8192, rows=0, transport_kind="NvlTransport"):
        A2AConfig(num_blocks=16, block=2048, num_warps=8),
    # ... one row per tuned key ...
}
```

Then nothing else changes: the Step 1 call `all_to_all(..., config=None)` rebuilds the key and looks it up, so it now runs the tuned config automatically. When shapes change, re-run the tuner and regenerate the table - no kernel edits.

## What the same pattern enables next

Concrete doables on the shipped transport + ops + Config/Key + tuner:

- reduce-scatter: an accumulate-on-`consume` hook (sum the received tile into the output) over the same schedule family.
- all-gather: the gather schedule with the default copy hook.
- quantized all-to-all: a `produce` hook that casts/scales to fp8 and a `consume` that dequantizes, fusing the conversion into the transfer leg (no extra HBM pass).
- variable-split a2a (MoE dispatch / combine): the same collective with per-rank split sizes added to the `Key` and layout.

Each is a new hook and/or a sibling schedule on the same substrate - not a new comm stack.

Further out, and a bigger milestone than the items above: comm-compute overlap - hiding a compute kernel (e.g. a GEMM or MoE expert) under the collective via pipelining to cover its latency.
