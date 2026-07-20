# CuTe NVLink collectives

`comms/dsl` provides copy-based GPU communication over PyTorch symmetric memory and
NVLink. The public API currently includes:

- equal-split `all_to_all` schedules written in the CuTe DSL;
- local, MAST, AnyBench, and Nsight Compute entry points.

The data path supports `torch.float32` and `torch.bfloat16`.

## Public API

```python
from comms.dsl import NvlTransport, nvl_rendezvous
from comms.dsl.cute import all_to_all
from comms.dsl.cute.a2a.tuning import CuteA2AConfig
```

| API | Purpose |
|---|---|
| `nvl_rendezvous` | Collectively allocate a user-owned NVLink transport for a process group. |
| `NvlTransport` | Own staging memory, signal storage, persistent counters, and resolved launch metadata. |
| `all_to_all` | Execute an equal-split all-to-all into a caller-owned output tensor. |
| `CuteA2AConfig` | Select the schedule and its launch geometry explicitly. |

Advanced schedule authors can also import `PeerTable`, `check_transfer`, and `nvl_ops`.
Application code normally uses the higher-level APIs above; `all_to_all` invokes the
required validation and transport operations internally.

## Basic all-to-all

All ranks must call `nvl_rendezvous` and `all_to_all` in the same collective order.

```python
import torch
import torch.distributed as dist

from comms.dsl import nvl_rendezvous
from comms.dsl.cute import all_to_all

rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", rank % torch.cuda.device_count())
torch.cuda.set_device(device)

input = torch.randn(1 << 20, dtype=torch.bfloat16, device=device)
output = torch.empty_like(input)

# Equal split: every destination receives input.numel() / world_size elements.
chunk_elements = input.numel() // world_size
chunk_bytes = chunk_elements * input.element_size()
transport = nvl_rendezvous(
    dist.group.WORLD,
    device,
    per_peer_bytes=chunk_bytes,
)

# config=None uses the fixed DEFAULT_A2A_CONFIG. Its zero-valued tuning fields are
# resolved analytically for this input; its primitive remains classic copy.
all_to_all(transport, output, input)
```

The result has the same equal-split layout as `torch.distributed.all_to_all_single`.

## Choosing a schedule

`CuteA2AConfig` exposes the following fields:

| Field | Default | Meaning |
|---|---:|---|
| `num_blocks` | `8` | Blocks per peer for `copy`; contributes to the total logical-channel grid for channel schedules. |
| `num_threads` | `0` | Threads per CTA. `0` selects an analytic value. |
| `num_slots` | `0` | Classic pipeline slots or bounded-ring slots. `0` selects the schedule default. |
| `unroll` | `0` | Per-thread copy unroll. `0` selects the analytic value. |
| `primitive` | `"copy"` | Schedule family. |
| `cluster` | `0` | Classic CTA cluster size. Channel schedules require clustering off. |
| `cluster_y` | `1` | Classic peer-axis cluster size. Channel schedules require `1`. |
| `send_threads` | `0` | Threads assigned to the send role in channel schedules. `0` uses half the CTA. |
| `peer_fanout` | `0` | Concurrent peer groups for `copy_channel_full`. `0` means fanout `1`. |

Supported primitives:

- `copy`: classic per-`(peer, block)` staging with a slot pipeline;
- `copy_channel_full`: one CTA range spans all peers, with separate send/receive warp
  groups and `peer_fanout` of `1`, `2`, or `4`;
- `copy_channel_ring`: the same channel ownership with bounded circular staging.

Example full-staging channel configuration:

```python
from comms.dsl import nvl_rendezvous
from comms.dsl.cute import all_to_all
from comms.dsl.cute.a2a.tuning import CuteA2AConfig

full_config = CuteA2AConfig(
    num_blocks=4,
    num_threads=1024,
    primitive="copy_channel_full",
    send_threads=512,
    peer_fanout=4,
    unroll=12,
)
full_transport = nvl_rendezvous(
    dist.group.WORLD,
    device,
    per_peer_bytes=chunk_bytes,
)
all_to_all(full_transport, output, input, config=full_config)
```

Example bounded-ring configuration:

```python
ring_config = CuteA2AConfig(
    num_blocks=4,
    num_threads=1024,
    num_slots=4,
    primitive="copy_channel_ring",
    send_threads=448,
    unroll=3,
)
ring_transport = nvl_rendezvous(
    dist.group.WORLD,
    device,
    # Capacity is intentionally bounded and may be smaller than the logical chunk.
    per_peer_bytes=64 * 1024 * 1024,
)
all_to_all(ring_transport, output, input, config=ring_config)
```

### Size selection is explicit

The runtime does **not** automatically select a schedule from the message size.
`config=None` always starts from `DEFAULT_A2A_CONFIG`, whose primitive is classic `copy`.
The size-band policies used for performance evaluation are not a public runtime API.

A transport is also bound to the resolved geometry of its first call. Different message
sizes or changes to dtype, vector width, slots, primitive, fanout, or ring layout must use
different transports. `check_geometry` raises instead of reinterpreting persistent counters
and staging ownership.

Applications with multiple stable shapes should keep one transport per geometry:

```python
config = CuteA2AConfig(
    num_blocks=4,
    num_threads=1024,
    primitive="copy_channel_full",
    send_threads=512,
    peer_fanout=4,
    unroll=12,
)
chunk_bytes = input.numel() // world_size * input.element_size()
transports: dict[tuple[int, torch.dtype, CuteA2AConfig], NvlTransport] = {}

key = (input.numel(), input.dtype, config)
transport = transports.get(key)
if transport is None:
    transport = nvl_rendezvous(
        dist.group.WORLD,
        device,
        per_peer_bytes=chunk_bytes,
    )
    transports[key] = transport

all_to_all(transport, output, input, config=config)
```

Applications with additional geometry-affecting inputs must include them in the cache key.
All ranks must make the same cache-miss and rendezvous decisions.

`COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1` is only for calls that are already device- and
cross-rank-synchronized. It does not drain an in-flight transport and is not an automatic
schedule-selection mechanism.

## Constraints and common errors

`all_to_all` validates the contract before compiling or launching:

- input and output must be non-empty, contiguous CUDA tensors;
- both tensors must have the same dtype and current CUDA device;
- input and output storage must not overlap;
- input/output pointers and `per_peer_bytes` must be 16-byte aligned;
- `input.numel()` must be divisible by `world_size`;
- the transport must have enough per-peer staging and signal slots;
- every signal-spinning CTA must be co-resident; the host uses a conservative maximum of
  one CTA per SM.

For `copy_channel_full` fanout `2` or `4`, the validated GB300 geometry is world size `8`,
`32` total CTAs, `1024` threads per CTA, and `512` send threads. Channel schedules reject
CTA clustering.

## Benchmarking

### Local

The maintained published benchmark compares the default CuTe all-to-all with NCCL and
checks correctness before timing:

```bash
buck2 run @fbcode//mode/opt fbcode//comms/dsl/tests:benchmark_a2a_cute

A2A_SIZES=8388608,67108864 \
  buck2 run @fbcode//mode/opt fbcode//comms/dsl/tests:benchmark_a2a_cute
```

### GB300 MAST

The launcher is dry-run by default. Supply access values as flags or through
`MAST_HPC_IDENTITY`, `MAST_RM_ATTRIBUTION`, and `MAST_HPC_ONCALL`.

```bash
buck2 run @fbcode//mode/opt --prefer-local \
  fbcode//comms/dsl/tests:mast_launch -- \
  --delivery conda \
  --module comms.dsl.tests.benchmark_a2a_cute \
  --nnode 2 \
  --ppn 4 \
  --nvl-hosts 2 \
  --hw gb300_dsf \
  --identity networkai_comms_tools \
  --rm-attribution infra_projects \
  --oncall hpc_comms_lib \
  --submit
```

Omit `--submit` to inspect the generated specification without consuming capacity.
Structured rows are emitted as `A2A_RESULT_JSON`.

### AnyBench

```bash
fbpkg build fbcode//comms/dsl/tests:comms.dsl.benchmark_a2a.aarch64.gb300 \
  --build-remote --ephemeral --acl-free-uuid-only

buck2 run @fbcode//mode/opt fbcode//hpc_comms/uni_bench:run_uni_bench -- \
  --uni_bench_config comms/dsl/tests/anybench_a2a_cute.json
```

## Nsight Compute

D6 integrates single-pass NCU profiling into the published benchmark.

Local example:

```bash
buck2 run @fbcode//mode/opt fbcode//comms/dsl/tests:benchmark_a2a_cute -- \
  --ncu \
  --ncu-launch-count 1 \
  --ncu-kernel-regex 'regex:.*_a2a_kernel.*'
```

MAST example:

```bash
buck2 run @fbcode//mode/opt --prefer-local \
  fbcode//comms/dsl/tests:mast_launch -- \
  --delivery conda \
  --module comms.dsl.tests.benchmark_a2a_cute \
  --nnode 2 \
  --ppn 4 \
  --nvl-hosts 2 \
  --hw gb300_dsf \
  --identity networkai_comms_tools \
  --rm-attribution infra_projects \
  --oncall hpc_comms_lib \
  --ncu \
  --ncu-launch-count 1 \
  --ncu-kernel-regex 'regex:.*_a2a_kernel.*' \
  --submit
```

Use an exact kernel regex. The default `launch__` metrics require one pass. NCU 2025.3.1
cannot safely replay a distributed signal-spinning communication kernel for arbitrary
multi-pass metrics, so profiling such metrics can deadlock. MAST NCU profiling requires
conda delivery because the launcher swaps to the NCU-capable base image.

## Current GB300 results

The current source was measured on two hosts with four GB300 GPUs each. The main matrix
covers 32 B through 2 GiB per rank. Its 437 structured rows passed bit-exact pre-timing,
timed-replay, post-timing, queued changed-input, and actual-grid checks. A follow-up sweep
added the production 48 MiB/peer and 96 MiB/peer shapes (384 MiB and 768 MiB per rank); all
10 additional rows were bit-exact.

`Ratio` is CuTe/NCCL performance: each run computes NCCL latency divided by CuTe
latency, equivalently CuTe bus bandwidth divided by NCCL bus bandwidth, and the table
reports the median ratio across rotations. `(lat)` marks latency as the primary comparison
metric; `(bw)` marks bus bandwidth. Both forms are greater-is-better. The displayed NCCL
and CuTe latencies are independent medians, so their rounded quotient need not equal the
displayed ratio.
Bandwidth excludes the local diagonal chunk. `NCCL grid` is the measured launch CTA count,
not a measurement of distinct occupied SMs. The main matrix uses the selected size-band
policy; the two follow-up rows use the best of all five large-message configurations. For
adaptive classic rows, the table expands the resolved thread count and unroll as
`classic<threads>u<unroll>` instead of displaying the requested `analytic` sentinel config.

| Size/rank | NCCL grid | CuTe config | NCCL latency | CuTe latency | NCCL busbw | CuTe busbw | Ratio |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32 B | 8 | `classic768u16` | 31.04 us | 9.95 us | 0.000902 GB/s | 0.002814 GB/s | 3.108x (lat) |
| 64 B | 8 | `classic768u16` | 27.58 us | 10.69 us | 0.002031 GB/s | 0.005241 GB/s | 2.827x (lat) |
| 128 B | 8 | `classic768u16` | 31.36 us | 10.29 us | 0.003572 GB/s | 0.010886 GB/s | 3.048x (lat) |
| 256 B | 8 | `classic768u16` | 33.19 us | 10.82 us | 0.006749 GB/s | 0.020693 GB/s | 3.010x (lat) |
| 512 B | 8 | `classic768u16` | 14.11 us | 10.67 us | 0.031744 GB/s | 0.041979 GB/s | 1.293x (lat) |
| 1 KiB | 8 | `classic768u16` | 13.74 us | 11.95 us | 0.065205 GB/s | 0.074969 GB/s | 1.258x (lat) |
| 2 KiB | 8 | `classic768u16` | 31.57 us | 12.00 us | 0.056769 GB/s | 0.149362 GB/s | 2.631x (lat) |
| 4 KiB | 8 | `classic768u16` | 32.59 us | 11.65 us | 0.109989 GB/s | 0.307567 GB/s | 2.695x (lat) |
| 8 KiB | 8 | `classic768u16` | 14.07 us | 11.06 us | 0.509371 GB/s | 0.648231 GB/s | 1.273x (lat) |
| 16 KiB | 8 | `classic1024u6` | 13.92 us | 10.89 us | 1.0 GB/s | 1.3 GB/s | 1.269x (lat) |
| 32 KiB | 8 | `classic1024u6` | 33.99 us | 11.18 us | 0.843661 GB/s | 2.6 GB/s | 3.040x (lat) |
| 64 KiB | 8 | `classic1024u6` | 35.02 us | 11.00 us | 1.6 GB/s | 5.2 GB/s | 3.184x (lat) |
| 128 KiB | 8 | `classic1024u6` | 35.04 us | 11.92 us | 3.3 GB/s | 9.6 GB/s | 3.077x (lat) |
| 256 KiB | 8 | `classic1024u6` | 34.67 us | 12.84 us | 6.6 GB/s | 17.9 GB/s | 2.679x (lat) |
| 512 KiB | 8 | `classic1024u6` | 34.56 us | 13.76 us | 13.3 GB/s | 33.3 GB/s | 2.512x (lat) |
| 1 MiB | 16 | `classic1024u6` | 20.95 us | 14.22 us | 43.8 GB/s | 64.5 GB/s | 1.474x (lat) |
| 2 MiB | 32 | `classic512u8` | 22.47 us | 15.32 us | 81.7 GB/s | 119.8 GB/s | 1.475x (lat) |
| 4 MiB | 32 | `classic1024u8` | 37.73 us | 19.57 us | 97.3 GB/s | 187.6 GB/s | 1.928x (lat) |
| 8 MiB | 32 | `classic1024u8` | 35.78 us | 28.54 us | 205.2 GB/s | 257.2 GB/s | 1.248x (bw) |
| 16 MiB | 32 | `classic1024u8` | 52.54 us | 45.16 us | 279.4 GB/s | 325.1 GB/s | 1.194x (bw) |
| 32 MiB | 32 | `full_f4_u12` | 83.31 us | 78.69 us | 352.4 GB/s | 373.1 GB/s | 1.061x (bw) |
| 48 MiB | 32 | `full_f4_u12` | 107.87 us | 101.76 us | 408.3 GB/s | 432.8 GB/s | 1.058x (bw) |
| 64 MiB | 32 | `full_f2_u3` | 132.57 us | 120.69 us | 442.9 GB/s | 486.5 GB/s | 1.103x (bw) |
| 96 MiB | 32 | `full_f2_u3` | 181.95 us | 174.07 us | 484.1 GB/s | 506.0 GB/s | 1.047x (bw) |
| 128 MiB | 32 | `full_f2_u3` | 230.75 us | 221.51 us | 509.0 GB/s | 530.2 GB/s | 1.042x (bw) |
| 256 MiB | 32 | `full_f1_s480_u4` | 422.18 us | 412.22 us | 556.4 GB/s | 569.8 GB/s | 1.024x (bw) |
| 384 MiB | 32 | `full_f1_s480_u4` | 610.86 us | 593.79 us | 576.8 GB/s | 593.3 GB/s | 1.029x (bw) |
| 512 MiB | 32 | `full_f1_s480_u4` | 802.08 us | 775.74 us | 585.7 GB/s | 605.6 GB/s | 1.034x (bw) |
| 768 MiB | 32 | `ring_s448_u3` | 1146.52 us | 1153.13 us | 614.6 GB/s | 611.1 GB/s | 0.994x (bw) |
| 1 GiB | 32 | `ring_s448_u3` | 1503.62 us | 1496.61 us | 624.8 GB/s | 627.8 GB/s | 1.002x (bw) |
| 2 GiB | 32 | `ring_s448_u3` | 3033.46 us | 2936.06 us | 619.4 GB/s | 640.0 GB/s | 1.033x (bw) |

At 384 MiB/rank, the NCCL result (610.86 us, 659.2 GB/s algorithm bandwidth,
576.8 GB/s bus bandwidth) reproduces the TorchLabs baseline (610.8 us, 659.3 GB/s
algorithm bandwidth, 576.9 GB/s bus bandwidth). At 768 MiB/rank, NCCL reaches
614.6 GB/s bus bandwidth, within 0.6% of the 615-618 GB/s TorchLabs range. This
cross-check found no evidence of a suboptimal NCCL denominator.

The detailed reproducible commands, job names, correctness gates, ranges, and source digest
are recorded in the Test Plans for
[D112061822](https://www.internalfb.com/diff/D112061822) and
[D112073340](https://www.internalfb.com/diff/D112073340).
