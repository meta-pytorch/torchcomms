# Ctran Hierarchical NVLink + IBGDA AllGather Integration Plan

Status: architecture plan only. This document is intended to drive a later implementation pass.

## Phase 1: Architecture Review

### Ctran AllGather Dispatch

Ctran has two AllGather surfaces:

- Eager AllGather under `ctran/algos/AllGather/`.
- Persistent/window AllGatherP under `ctran/algos/AllGatherP/`.

The eager path is selected from `NCCL_ALLGATHER_ALGO` and routed through `ctranAllGather()`. The current implementation maps the generic `ctran` setting to `ctdirect` when multiple local ranks exist and to `ctring` otherwise. The supported eager algorithms are:

- `ctdirect`: local NVLink broadcast plus mapper-driven inter-node puts.
- `ctring`: host/mapper ring path, only supported when `nLocalRanks == 1`.
- `ctrd`: host/mapper recursive doubling path, only supported when `nLocalRanks == 1`.
- `ctbrucks`: host/mapper Brucks fast-forward path.
- `ctgraph*`: CUDA graph-aware wrappers that eventually execute AllGatherP variants.

The persistent path is selected from `NCCL_ALLGATHER_P_ALGO` and routed through `allGatherPExec()` and `allGatherWinExec()`. The current persistent choices are:

- `ctdirect`
- `ctpipeline`
- `ctrdpipeline`

The persistent path already has the right shape for the new work: it has an explicit algorithm enum, a benchmark switch in `AllgatherPBench.cc`, a window path, and CUDA graph-aware entry points.

### Ctran Host-Side and Device-Side Split

The current ctran AllGather implementations are mostly host/GPE orchestrated:

- `AllGatherDirect` builds `KernelElem` work, performs local NVLink broadcast in a ctran kernel, and performs inter-node movement through `CtranMapper` `iput` and notify operations.
- `AllGatherP/PipelineImpl` uses small synchronization kernels (`PipeStart`, `PipeSync`, `PipeEnd`) to coordinate GPE host work. Bulk local movement is done by CUDA copy engine NVLink copies, and inter-node movement is mapper-driven.
- `AllGatherP/RecursiveDoublingImpl` follows the same pattern, with recursive doubling over nodes and copy engine broadcast after each remote exchange.

This means ctran today does not have a single device kernel that directly combines:

- self-copy into final `recvbuff`,
- IBGDA staging,
- IB ring forwarding,
- NVLink local broadcast from the received rank chunk.

That fusion is the core performance difference in D104725448.

### GPE Launch Model

`CtranGpe` can launch a kernel with or without a host-side op group. `DeviceAllToAllvPipes` is the closest existing pattern:

- build a ctran-local `KernArgs`,
- set `KernelConfig.algoArgs`,
- submit an empty `opGroup`,
- launch a GPE-compatible kernel signature,
- ignore `CtranAlgoDeviceState` inside the Pipes-based kernel.

For the hierarchical AllGather path, this is the right integration model. It preserves ctran stream ordering and CUDA graph behavior while avoiding mapper host work in the hot data path.

Block diagram:

```text
Current ctran AllGatherP pipeline

  user stream
      |
      v
  GPE sync kernels
      |
      +--> host/GPE mapper work --> CtranMapper/CtranIb --> remote GPU memory
      |
      +--> CUDA copy engine NVLink copies
      |
      v
  recvbuff


Proposed hierarchical Pipes AllGatherP

  user stream
      |
      v
  Ctran GPE kernel launch with empty opGroup
      |
      v
  one GPU kernel
      |
      +--> CopyOp fused self-copy: sendbuf -> IB staging and recvbuff[self]
      +--> P2pIbgdaTransportDevice ring: inter-node forward/recv
      +--> P2pNvlTransportDevice broadcast: intra-node/NVL fanout
      |
      v
  recvbuff
```

### Ctran Pipes Integration

`CtranPipes.cc` already constructs a `comms::pipes::MultiPeerTransport` when Pipes is enabled. The existing integration:

- configures NVLink buffer settings,
- configures baseline IBGDA fields such as CUDA device, GID/address family, HCA, data buffer size, QP depth, timeout, retry, TC, SL, and RNR settings,
- reuses ctran shared NVLink staging buffers through `setExternalNvlDataBuffers()`,
- calls `exchange()` to build device-side transport handles.

`MultiPeerTransport` already exposes the host accessors needed by the hierarchical AllGather:

- `get_p2p_nvl_transport_device(globalPeerRank)`
- `get_p2p_ibgda_transport_device(globalPeerRank)`
- `nvl_local_rank()`
- `nvl_n_ranks()`
- `nvl_peer_ranks()`
- `global_to_nvl_local(globalRank)`
- `my_rank()`
- `n_ranks()`

The gap is tuning and lifecycle: the current ctran Pipes config does not yet expose the D104 performance knobs for IBGDA send/recv state, multi-QP, block count, signal granularity, or pipeline depth.

### Memory Registration and Buffer Management

Ctran's existing AllGatherP init path registers the user `recvbuff` for mapper-driven remote writes and exchanges access keys. The Pipes hierarchical path does not need remote ranks to RDMA directly into user `recvbuff`; it uses IBGDA transport-owned staging rings and copies from local staging into final `recvbuff` inside the GPU kernel.

Implications:

- The first implementation can live inside AllGatherP and tolerate the existing persistent registration overhead, because the hot path does not need mapper writes.
- No per-exec buffer registration or exchange should occur.
- IBGDA `sendRecv` staging resources must be configured during `CtranPipes.cc` initialization and remain stable for the lifetime of the communicator.
- The path must guard against overlapping hierarchical collectives on the same `MultiPeerTransport` until the send/recv state is explicitly proven channel-safe.

### Copy Model: Fused Staging, Not Zero-Copy

The proposed integration is copy-based and staged. It is GPU-direct and copy-fused, but it is not true end-to-end zero-copy from every sender's `sendbuf` into every receiver's final `recvbuff` slot.

The intended data movement is:

```text
sendbuf[self]
    |
    | fused CopyOp
    v
IBGDA send staging  +  recvbuff[self]
    |
    v
remote IBGDA staging / forward path
    |
    v
recvbuff[remote rank]
    |
    v
NVLink fanout within the local NVL group
```

This still removes CPU bounce buffers and host-posted data movement from the hot path. The performance bet is one GPU kernel, GPU-issued IBGDA, fused self-copy into IB staging plus final recv slot, staged IB ring forwarding, and NVLink local fanout.

True zero-copy final-buffer RDMA is a separate future design. It would require final `recvbuff` registration, remote final-buffer address exchange, more complex correctness for in-place/window execution, and a different flow-control model. It should not be mixed into the first integration unless the staged design cannot meet the 128MB and 1GB targets.

### Topology Awareness

Ctran has local-rank and node information, and Pipes `MultiPeerTransport` discovers NVLink reachability. Today there is no ctran AllGather selector that specifically says "uniform NVLink groups plus inter-node IBGDA ring."

D104 assumes a rectangular rank layout:

```text
global_rank = ib_rank * nvl_size + nvl_rank
```

The ctran integration must either:

- strictly require uniform, contiguous NVLink groups in that layout, or
- add an explicit rank remap in the kernel and launcher args.

The first implementation should choose strict gating. It is safer, easier to benchmark, and prevents silent writes to the wrong allgather slots.

### Pipes D104725448 Hierarchical AllGather

D104725448 adds the Pipes DirectNvl path:

- `pipes/collectives/DirectNvl.cu`
- `pipes/collectives/DirectNvl.cuh`
- `pipes/collectives/DirectNvlTypes.h`
- `pipes/collectives/DirectNvlLauncher.*`

The key device behavior is:

- one CUDA kernel performs hierarchical AllGather,
- each block group participates in an IBGDA ring across NVLink groups,
- each IB ring step receives or forwards a chunk through `P2pIbgdaTransportDevice`,
- local NVLink peers receive chunks through `P2pNvlTransportDevice`,
- the self chunk is copied once into both IB send staging and final `recvbuff[self]` via `CopyOp`.

The key performance mechanisms are:

- `MemcpyAndSelfCopy` uses the `CopyOp` hook to perform the self-copy and IB send-staging copy in one vectorized path.
- `P2pIbgdaTransportDevice::send/recv/forward` owns the staged ring protocol and flow control for DATA_READY, SLOT_FREE, and NIC completion state.
- `nic_qp_for_group()` spreads block groups over the available NIC/QP pool, so block count and QPs per peer per NIC are coupled tuning parameters.
- `pipeline_window(active_blocks)` partitions the staging window across active groups; a block count that helps 1GB can over-fragment the window and hurt 128MB.
- fused self-copy and IB staging through `memcpy_vectorized` with dual destinations,
- GPU-issued IBGDA WQEs,
- per-block-group QP/NIC selection,
- configurable IB send/recv pipeline depth,
- configurable block count and signal granularity,
- one-kernel overlap of self-copy, inter-node IB, and local NVLink broadcast.

### Current Performance-Critical Differences

Current ctran `ctpipeline`:

- uses host/GPE mapper orchestration for IB,
- uses CUDA copy engine local NVLink broadcast,
- copies self separately,
- has multiple phase boundaries,
- has clear CUDA graph support through existing GPE/window paths.

Pipes D104 hierarchical AllGather:

- uses one GPU kernel,
- keeps IBGDA control on device,
- fuses self-copy with IB staging,
- maps block groups across NICs/QPs,
- needs stricter transport state lifecycle and rank geometry guarantees.

The integration should preserve the Pipes hot path and adapt ctran around it, not translate it into mapper operations.

### 128MB Performance Gap Risk

D104's strongest evidence is at 1GB. The 128MB target needs separate analysis because the same tuning can lose bandwidth at smaller large-message sizes.

Likely 128MB failure modes:

- IBGDA pipeline startup and drain overhead becomes a larger fraction of runtime.
- Too many blocks can over-partition each staging slot, reducing per-block contiguous copy size and increasing signal pressure.
- IB signal granularity tuned for 1GB can be too coarse or too fine at 128MB.
- Pipeline depth and data buffer size can waste staging capacity or add unnecessary slot-management overhead.
- NCCL may choose a different protocol/topology at 128MB than at 1GB, so a single ctran threshold is risky.

The benchmark phase must include a targeted 128MB sweep:

- block count: `4, 8, 16, 32`
- IB signal bytes: `512KB, 1MB, 2MB`
- IBGDA pipeline depth: `1, 2, 4`
- data buffer size: `32MB, 64MB, 128MB`
- QPs per peer per NIC: `1, 2, 4, 8`
- NCCL baselines: production default and fair matched configurations

The production selector must be size-aware. Do not auto-select the hierarchical path for 128MB, or for any size bucket, unless that size bucket independently passes the NCCL acceptance criteria on the target topology.

## Phase 2: Integration Approaches

### Approach A: New AllGatherP Algorithm With GPE Wrapper Over Pipes Device Path

Architecture:

- Add an opt-in AllGatherP algorithm such as `cthierarchical_pipes`.
- Dispatch from `allGatherPExec()` and `allGatherWinExec()`.
- Build a ctran-local `KernArgs` from `CommStateX`, `PersistArgs`, and `comm->multiPeerTransport_`.
- Submit a GPE-compatible kernel with an empty `opGroup`, following the `DeviceAllToAllvPipes` pattern.

Reuse vs rewrite:

- Reuse Pipes `P2pIbgdaTransportDevice`, `P2pNvlTransportDevice`, `TiledBuffer`, and `CopyOp` behavior.
- Prefer a small shared device helper in Pipes if required so the Pipes benchmark launcher and ctran wrapper share the same kernel body.
- Keep collective policy, dispatch, support checks, and ctran args under `fbcode/comms/ctran/`.

Transport layer:

- Use `MultiPeerTransport` accessors for previous/next IBGDA peers and local NVLink peers.
- Extend `CtranPipes.cc` config to expose and set `ibgdaConfig.sendRecv`, `numQpsPerPeerPerNic`, `dataBufferSize`, pipeline depth, block count, and signaling bytes.
- Do not route IBGDA through `CtranMapper`.

Kernel strategy:

- Single kernel.
- Preserve `CopyOp` dual-destination fusion.
- Set GPE dynamic shared memory to zero unless the wrapper needs ctran shared state.
- Avoid default-stream launcher behavior; launch on the ctran stream through GPE.

Algorithm selection:

- Add an enum/cvar value under `NCCL_ALLGATHER_P_ALGO`.
- Keep it opt-in until correctness and performance acceptance criteria are met.
- Gate support on `ENABLE_PIPES`, `NCCL_CTRAN_USE_PIPES`, `multiPeerTransport_ != nullptr`, uniform NVLink group size, IBGDA availability for inter-group peers, and valid rank geometry.
- Add production auto-selection only after size/topology buckets have measured wins. The first selector should be explicit opt-in; the later selector should use a benchmark-derived table rather than a single hardcoded large-message threshold.

Performance hypothesis:

- Best chance to match D104 performance because it preserves the one-kernel fused data path.
- Should beat host/mapper ctran paths on large transfers by reducing CPU/GPE phase overhead and redundant memory movement.
- Should be competitive with NCCL on GB200 when QP/NIC/block tuning matches the D104 benchmark.

Risk and complexity:

- Medium to high.
- Requires careful rank geometry validation.
- Requires send/recv state lifecycle rules to avoid overlapping collective corruption.
- Requires transport error handling rules for the shared multi-QP pool. One QP entering an error state can poison subsequent sends to that peer until the communicator is torn down or the transport is rebuilt.
- Requires config work in `CtranPipes.cc`.

Benchmark plan:

- Add `cthierarchical_pipes` to `AllgatherPBench.cc`.
- Benchmark against `ctpipeline`, `ctrdpipeline`, and NCCL in the same process and buffer setup.
- Benchmark NCCL both with production defaults and with fair matched settings. Do not claim NCCL superiority based only on a baseline that disables NCCL NVLS, P2P, or SHM optimizations.
- Include CUDA graph/window execution once correctness is established.

Recommendation: primary approach.

### Approach B: Benchmark-Only Ctran Sidecar First

Architecture:

- Add a benchmark-only mode in `AllgatherPBench.cc` that builds the same hierarchical launch parameters from ctran communicator state and launches the Pipes hierarchical kernel.
- Do not expose it through production ctran dispatch initially.

Reuse vs rewrite:

- Reuse the Pipes launcher or a stream-aware wrapper.
- Reuse ctran benchmark allocation, warmup, iteration, and NCCL comparison logic.

Transport layer:

- Use `comm->multiPeerTransport_`.
- Configure IBGDA send/recv resources through `CtranPipes.cc`; otherwise the benchmark will not reflect the target production path.

Kernel strategy:

- Prefer the same kernel wrapper intended for Approach A, but restrict call sites to benchmarks.

Algorithm selection:

- Benchmark flag only.
- No production selector change.

Performance hypothesis:

- Fastest way to verify whether the D104 win survives ctran initialization, buffers, topology discovery, and launch context.

Risk and complexity:

- Low to medium.
- Can become throwaway code if not built with the same args and config as production.

Benchmark plan:

- Use it as Phase 0 validation before defaulting or auto-selecting anything.

Recommendation: useful as a risk-reduction step, not sufficient as the final integration.

### Approach C: Eager `NCCL_ALLGATHER_ALGO` Variant

Architecture:

- Add a new eager algorithm such as `cthierarchical_pipes`.
- Dispatch from `ctranAllGather()` for non-persistent calls.

Reuse vs rewrite:

- Reuse the same ctran GPE kernel wrapper as Approach A.
- Reuse support checks and topology gating.

Transport layer:

- Same as Approach A.

Kernel strategy:

- Single kernel, but eager calls must build params on every execution.

Algorithm selection:

- Add a value under `NCCL_ALLGATHER_ALGO`.
- Do not auto-map `ctran` to this path until the persistent path passes acceptance criteria.

Performance hypothesis:

- Could provide a faster non-persistent path for large messages once Approach A is validated.

Risk and complexity:

- Higher than Approach A because eager calls interact with dynamic counts, registrations, and existing `ctranAllGatherSupport()` behavior.
- CUDA graph support is better handled first through the AllGatherP/window path.

Benchmark plan:

- Add only after Approach A reaches stable performance.

Recommendation: second-stage production extension.

### Approach D: Hybridize Existing `ctpipeline` or `ctdirect`

Architecture:

- Keep the existing ctran mapper/GPE control flow.
- Replace one part of the data path with Pipes NVLink or IBGDA operations.

Reuse vs rewrite:

- Reuse some ctran orchestration.
- Reuse only fragments of Pipes transport.

Transport layer:

- Mixed `CtranMapper` plus Pipes device transport.

Kernel strategy:

- Multi-phase.
- Likely loses self-copy and IB staging fusion.

Algorithm selection:

- Could remain under `ctpipeline` or a new hybrid name.

Performance hypothesis:

- May improve one bottleneck but is unlikely to reproduce D104 performance.

Risk and complexity:

- Medium complexity with poor performance upside.
- Creates two control planes in one algorithm.

Benchmark plan:

- Only investigate if Approach A fails due to an unavoidable ctran/Pipes lifecycle issue.

Recommendation: fallback experiment only.

### Approach E: New Ctran Mapper Backend or CtranIb Port

Architecture:

- Translate IBGDA into `CtranMapper` or `CtranIb` abstractions.

Reuse vs rewrite:

- Rewrites most of D104 transport semantics.
- Loses direct mapping to `P2pIbgdaTransportDevice::send/recv/forward`.

Transport layer:

- Host-posted verbs or mapper requests instead of GPU-issued IBGDA WQEs.

Kernel strategy:

- Multi-phase host/device orchestration.

Algorithm selection:

- Could be hidden behind existing mapper selection, but would affect more than AllGather.

Performance hypothesis:

- Unlikely to beat NCCL because it discards the main D104 advantages.

Risk and complexity:

- High complexity.
- High abstraction leakage.
- Global transport semantics become harder to reason about.

Benchmark plan:

- Not recommended for this goal.

Recommendation: no-go for the NCCL parity target.

### Approach F: Replace or Default `ctpipeline` Immediately

Architecture:

- Make hierarchical Pipes AllGather the default whenever topology appears to match.

Reuse vs rewrite:

- Could reuse Approach A internally, but without sufficient validation.

Transport layer:

- Same as Approach A.

Kernel strategy:

- Same as Approach A.

Algorithm selection:

- Auto-select or replace `ctpipeline`.

Performance hypothesis:

- Could expose wins quickly, but also risks regressions for sizes and topologies where D104 was not proven.

Risk and complexity:

- High production risk.
- Poor A/B isolation.

Benchmark plan:

- Not valid until Approach A has passed acceptance criteria.

Recommendation: no-go until after opt-in validation.

## Phase 3: Domain Expert Review

### Ranking Matrix

| Approach | GPU Kernel Performance | Network/RDMA Transport | Systems Architecture | Benchmark/Validation |
| --- | --- | --- | --- | --- |
| A. AllGatherP/GPE wrapper over Pipes device path | 1 | 1 | 1 | 1 |
| B. Direct stream-aware Pipes launcher adapter | 2 | 2 | 2 | Not preferred |
| C. Benchmark-only ctran sidecar | Useful, not final | Useful, not final | Useful, not final | 2 |
| D. Hybrid existing ctpipeline/ctdirect | 3 | 3 | No-go if spliced into `ctdirect` | 4 if replacing default |
| E. Ctran-native rewrite or mapper/IB backend | 4 or no-go | No-go | 3-4 depending shape | 3, long-term only |
| F. Replace/default `ctpipeline` immediately | No-go | No-go | No-go | No-go |

### Agent 1: GPU Kernel Performance Engineer

Top pick: Approach A.

Key arguments:

- Preserves D104's fused self-copy and IB staging, so `sendbuf` is read once and written to both IB staging and `recvbuff[self]`.
- Keeps Pipes block-group scheduling and multi-QP/NIC mapping intact.
- Avoids extra shared-memory and register pressure from ctran native kernel state.

Biggest risk:

- `CtranPipes.cc` must configure IBGDA send/recv state, QPs per peer per NIC, data buffer size, pipeline depth, and block count. Reusing the current `multiPeerTransport_` config as-is may under-saturate IB.

No-go:

- Do not adapt generic `DeviceAllToAllvPipes` into an all-to-all style allgather. It destroys the hierarchical traffic pattern.

### Agent 2: Network/RDMA Transport Architect

Top pick: Approach A.

Key arguments:

- Preserves GPU-issued IBGDA WQEs, fused self-copy, and `P2pIbgdaTransportDevice::forward()` ring flow control.
- Matches the existing ctran/Pipes seam through `CtranPipes.cc` and `MultiPeerTransport`.
- Keeps multi-NIC/QP semantics in Pipes instead of translating them into host verbs config.

Biggest risk:

- `sendRecvState` is comm-wide and persistent. Ctran must prevent overlapping hierarchical collectives or incompatible knob changes without quiescence/reset.

No-go:

- Do not port D104 onto `ctran/backends/ib/CtranIb` host verbs.

### Agent 3: Systems Software Architect

Top pick: Approach A.

Key arguments:

- Keeps ownership clean: `CtranPipes.cc` owns transport construction, while `AllGatherP` owns algorithm selection and support checks.
- Keeps API surface narrow with a ctran-local args struct rather than leaking Pipes internals into public ctran APIs.
- Gives the cleanest CUDA graph story because GPE launches on the user stream with existing ordering guards.

Biggest risk:

- Rank geometry. The D104 layout assumption must be validated or remapped before production use.

No-go:

- Do not splice this into `ctdirect` mapper flow or call the D104 benchmark launcher as-is.

### Agent 4: Performance Benchmarking and Validation Engineer

Top pick: Approach A, with Approach C as a pre-production measurement step.

Key arguments:

- `AllGatherP` already has a selector, persistent state, benchmark harness, and NCCL baseline.
- A new opt-in algorithm gives clean A/B comparisons without redefining `ctpipeline`.
- Validation includes real ctran costs rather than only the standalone Pipes benchmark.

Biggest risk:

- D104 proves a tuned standalone path. The reported 1GB win may not survive production NCCL defaults, ctran registration/init behavior, or the 128MB target. Benchmarking against NCCL with NVLS, P2P, or SHM disabled is useful for controlled comparison but is not sufficient to claim production superiority.

No-go:

- Do not replace/default `ctpipeline` based only on standalone Pipes benchmark results.

## Phase 4: Consensus Recommendation

### Consensus

All four reviewers prefer a ctran-owned AllGatherP integration that launches the Pipes hierarchical device path through GPE.

The shared reasoning is:

- keep the D104 one-kernel hot path intact,
- keep ctran algorithm dispatch and support checks in ctran,
- use the existing `CtranPipes.cc` transport construction seam,
- avoid mapper/host verbs for the IBGDA data path,
- preserve CUDA graph behavior through GPE.

### Divergence

The main disagreement is sequencing:

- GPU and RDMA reviewers prioritize preserving the exact D104 device path first.
- Systems review prioritizes clean API ownership and rank-geometry validation.
- Benchmark review recommends a benchmark-only sidecar before production exposure to prove the D104 win survives ctran overhead.

The final implementation order should incorporate all three concerns: first make the transport config benchmark-equivalent, then validate in `AllgatherPBench`, then expose as an opt-in AllGatherP algorithm.

### Additional Review Feedback Incorporated

Follow-up review agreed that the GPE empty-opGroup path is architecturally stronger than bypassing GPE. The main additions from that review are:

- Treat the 128MB target as its own tuning problem, not a scaled-down version of the 1GB result.
- Add explicit size-based dispatch and defer auto-selection until each size/topology bucket proves a win.
- Benchmark against NCCL production defaults in addition to fair matched configurations.
- Track multi-QP error-state propagation as part of the send/recv transport lifecycle risk.
- Keep the implementation copy-based and fused-staged initially; do not expand scope to true zero-copy final-buffer RDMA unless the staged path cannot meet the acceptance criteria.

### Final Recommendation

Primary approach: implement `cthierarchical_pipes` as a new opt-in AllGatherP algorithm launched through GPE, using a ctran-local wrapper over the Pipes hierarchical device body.

Proposed implementation flow:

```text
allGatherPExec / allGatherWinExec
    |
    v
support check
    |
    +-- unsupported topology/config --> existing ctpipeline or NCCL fallback
    |
    v
build HierarchicalAllGatherPipes KernArgs
    |
    +-- rank geometry from CommStateX / MultiPeerTransport
    +-- NVL peer handles from get_p2p_nvl_transport_device()
    +-- IB prev/next handles from get_p2p_ibgda_transport_device()
    +-- tuning from ctran Pipes config/cvars
    |
    v
CtranGpe::submit(empty opGroup, kernel, stream)
    |
    v
GPE-compatible wrapper kernel
    |
    v
shared Pipes hierarchical device body
```

Key modifications:

- Add ctran config/cvars for hierarchical Pipes AllGather tuning:
  - enabled flag,
  - IB block count,
  - IB signal bytes,
  - IBGDA send/recv pipeline depth,
  - IBGDA data buffer size,
  - QPs per peer per NIC,
  - size bucket threshold table or explicit opt-in override,
  - optional HCA/NIC selection override.
- Extend `CtranPipes.cc` to set `ibgdaConfig.sendRecv` and `numQpsPerPeerPerNic`.
- Add strict support checks for uniform NVLink group size, IBGDA availability, rank geometry, message alignment, and max supported rank count.
- Add a per-communicator or per-algo guard against overlapping hierarchical collectives on the same send/recv state.
- Add transport error handling that fails closed on QP error state instead of reusing a poisoned multi-QP pool.
- Add a ctran GPE-compatible kernel wrapper that launches on the ctran stream and does not use the default-stream Pipes launcher.
- Keep any Pipes source changes minimal. If a Pipes refactor is required, limit it to exposing a shared device helper so ctran and the Pipes benchmark use the same implementation.

### Implementation Checklist

- [ ] Phase 0: Freeze assumptions and guardrails.
  - [ ] Document exact rank geometry required by the D104 kernel.
  - [ ] Decide the first supported topology: uniform NVLink groups with contiguous global ranks.
  - [ ] Define fallback behavior for unsupported topology: return not supported and use existing `ctpipeline` or NCCL.

- [ ] Phase 1: Make ctran Pipes transport config performance-equivalent.
  - [ ] Add ctran config/cvars for hierarchical AllGather knobs.
  - [ ] Set `ibgdaConfig.sendRecv.maxGroups` from the hierarchical block count.
  - [ ] Set `ibgdaConfig.sendRecv.pipelineDepth`.
  - [ ] Set `ibgdaConfig.numQpsPerPeerPerNic`.
  - [ ] Set data buffer size and signal granularity defaults matching the best D104 results.
  - [ ] Add a separate 128MB tuning profile candidate instead of assuming the 1GB profile generalizes.
  - [ ] Validate that all ranks use identical transport config.
  - [ ] Define fail-closed behavior for QP error state in the multi-QP pool.

- [ ] Phase 2: Add benchmark-only validation hook.
  - [ ] Add `cthierarchical_pipes` as a benchmark option in `AllgatherPBench.cc`.
  - [ ] Build hierarchical launch args from `comm->multiPeerTransport_`.
  - [ ] Run correctness against NCCL for `nvl_size == 1` and hierarchical cases.
  - [ ] Measure 128MB and 1GB before adding production dispatch.
  - [ ] Run the targeted 128MB sweep for block count, IB signal bytes, pipeline depth, data buffer size, and QPs per peer per NIC.
  - [ ] Compare against NCCL production defaults with NVLS, P2P, and SHM enabled, plus a fair matched NCCL configuration.

- [ ] Phase 3: Add AllGatherP production dispatch.
  - [ ] Add `cthierarchical_pipes` to `NCCL_ALLGATHER_P_ALGO`.
  - [ ] Add support checks in the AllGatherP path.
  - [ ] Add GPE-compatible kernel wrapper and ctran-local args.
  - [ ] Route `allGatherPExec()` to the new implementation.
  - [ ] Route `allGatherWinExec()` to the same implementation for window execution.

- [ ] Phase 4: CUDA graph compatibility.
  - [ ] Verify capture does not allocate, exchange, register, or launch on the default stream.
  - [ ] Add graph capture/replay correctness tests.
  - [ ] Benchmark captured and non-captured execution separately.

- [ ] Phase 5: Topology expansion.
  - [ ] Add support for non-contiguous rank layouts only with explicit rank remap.
  - [ ] Add fallback for IB-only topologies.
  - [ ] Add topology detection for NVSwitch/direct NVLink differences if performance tuning differs.

- [ ] Phase 6: Auto-selection.
  - [ ] Keep opt-in until all acceptance criteria pass.
  - [ ] Add mapper/selector auto-pick only for proven topologies and message sizes.
  - [ ] Use a benchmark-derived size/topology table rather than a single static threshold.
  - [ ] Keep 128MB disabled or routed to the existing path until its own confidence interval clears the NCCL target.
  - [ ] Preserve an immediate cvar fallback to `ctpipeline`.

### Acceptance Criteria

Correctness:

- [ ] Byte-exact match with NCCL for all supported dtypes and tested counts.
- [ ] Correct for in-place and out-of-place modes if both are claimed supported.
- [ ] Correct for `nvl_size == 1`, single-node NVLink-only, and multi-node hierarchical topologies.
- [ ] Correct under CUDA graph capture and replay.

Performance:

- [ ] On GB200, at 128MB total input, `cthierarchical_pipes` beats NCCL allgather with bootstrap 95 percent CI lower bound greater than 1.02x.
- [ ] On GB200, at 1GB total input, `cthierarchical_pipes` beats NCCL allgather with bootstrap 95 percent CI lower bound greater than 1.02x.
- [ ] No production auto-selection if either 128MB or 1GB misses the NCCL target.
- [ ] No production auto-selection for any intermediate size bucket until that bucket independently meets the NCCL target or has an explicit fallback threshold.
- [ ] Existing `ctpipeline` and `ctrdpipeline` performance does not regress when the new path is not selected.

Benchmark methodology:

- [ ] Sweep message sizes from 8KB to 1GB.
- [ ] Include 128MB and 1GB as mandatory release gates.
- [ ] Sweep nodes, local ranks per node, blocks, QPs per peer per NIC, pipeline depth, data buffer size, and HCA selection.
- [ ] Compare against NCCL production defaults with NVLS, P2P, and SHM enabled.
- [ ] Also compare against a fair matched NCCL configuration for diagnosis, but do not use a handicapped NCCL-only comparison as the production acceptance claim.
- [ ] Run a dedicated 128MB sweep across block count `4, 8, 16, 32`, IB signal bytes `512KB, 1MB, 2MB`, pipeline depth `1, 2, 4`, data buffer size `32MB, 64MB, 128MB`, and QPs per peer per NIC `1, 2, 4, 8`.
- [ ] Use at least 5 process restarts per key configuration.
- [ ] Use randomized run order and at least 100 measured iterations after warmup.
- [ ] Report median, P95, coefficient of variation, and bootstrap confidence intervals.
- [ ] Record GPU SKU, driver, CUDA, NCCL, build SHA, host list, HCA list, and full environment.

### Rollback Plan

- Keep `cthierarchical_pipes` opt-in until acceptance criteria pass.
- If correctness fails, disable the enum support check and fall back to existing `ctpipeline` or NCCL.
- If 128MB does not beat NCCL, keep the path benchmark-only or large-message opt-in; do not auto-select.
- If 1GB does not beat NCCL, stop production integration and investigate transport config, QP/NIC mapping, and rank geometry before further rollout.
- If CUDA graph capture fails, gate the path out of `allGatherWinExec()` while preserving non-captured opt-in testing.
- If send/recv state collisions are observed, add stricter serialization or allocate independent transport state before enabling overlap.
- If any QP enters error state during testing, disable auto-selection and require communicator or transport rebuild before retrying the hierarchical path.
