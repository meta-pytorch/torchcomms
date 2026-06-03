# Folding `pipes` into `ctran`

**Date:** 2026-05-29

## 1. Goal & motivation

`comms/pipes` grew up as a standalone project, but it is now effectively a `ctran` subsystem: an ncclx user who creates a `ctran` object already gets pipes (when the pipes buck flags + `NCCL_CTRAN_PIPES_*` cvars are set). This proposal makes that relationship structural:

- Move `comms/pipes` to live **inside** `ctran` as `comms/ctran/pipes/`, the `ctran::pipes` submodule.
- Reorganize it into a clear taxonomy, and relocate the experimental DOCA-GPUNetIO **IBGDA** transport into a `transport/experimental/` folder so its status is obvious to readers.

Non-goals: no behavior changes, no API/cvar renames, no functional decoupling of IBGDA. This is a relocation + reorganization, not a rewrite.

## 2. Current state

The integration seam is already narrow and ctran-owned:

- `ctran/CtranPipes.{cc,h}` translates `NCCL_CTRAN_*` cvars into a `MultiPeerTransportConfig`, constructs `comms::pipes::MultiPeerTransport`, wires ctran's staging buffers in, and calls `exchange()`.
- `CtranComm` holds `multiPeerTransport_` / `pipesTrace_`; the whole integration is gated by the `ENABLE_PIPES` compile define.

**Key enabler:** pipes has **zero dependency on ctran** — no `#include "comms/ctran/..."` and no `//comms/ctran` Buck dep anywhere under `comms/pipes`. pipes is a clean lower layer; ctran → pipes is the only direction. Folding pipes under ctran therefore introduces **no dependency cycle**.

## 3. Target structure

```
comms/ctran/pipes/
  core/          # shared, transport-agnostic primitives
                 #   DeviceSpan, ThreadGroup, Tile, TiledBuffer, Copy{Utils,Op}, MemcpyCopyOp,
                 #   SignalState, BarrierState, ChunkState, Timeout, Device{Macros,Check}, Checks
    memory/      #   GpuMemHandler, CudaDriverLazy
    platform/    #   HipHostCompat, HipDeviceCompat
    trace/       #   PipesTrace, PipesTraceTypes
  topology/      # TopologyDiscovery, NvmlFabricInfo
  transport/     # Transport.cuh (tagged union), MultiPeerDeviceHandle.cuh, MultiPeerTransport.* (facade)
    self/        #   P2pSelfTransportDevice
    nvl/         #   P2pNvlTransportDevice, MultiPeerNvlTransport, ll/, ll128/
    experimental/
      ibgda/     #   MultipeerIbgda*, P2pIbgdaTransportDevice, IbgdaBuffer, Doca*, IbverbsLazy
        rdma/    #     NicDiscovery, IbHcaParser, NicConstants  (ibgda-private)
        amd/     #     DocaCompat, pipes_gda/, nic/             (ibgda AMD backend)
  collectives/   # relocated as-is (internal shape unchanged)
  window/        # HostWindow, DeviceWindow
  bootstrap/     # NvlBootstrapAdapter
  triton/        # python bindings (base_module → comms.ctran.pipes.*)
  tests/  benchmarks/
```

The `transport/{self, nvl, experimental/ibgda}` shape is self-documenting: two stable transports plus the experimental one. Shared sync/state primitives (`Signal/Barrier/ChunkState`) live in `core/`; NVL-specific optimizations (`ll`, `ll128`) live under `nvl/`.

## 4. Migration as diffs

| Diff | Scope | Nature |
|---|---|---|
| **D1 — fold** | Move `comms/pipes` → `comms/ctran/pipes` (flat, current structure) and rename `comms::pipes` → `ctran::pipes`. All consumers updated in place. | Pure codemod: path prefix + namespace. Trivially reviewable ("nothing changed but the prefix/namespace"). |
| **D2 — shape** | Reorganize into the §3 taxonomy and carve `transport/experimental/ibgda`. Buck **target names kept stable**, so only include paths change. | Structural. The considered reorg. |
| **D3 — OSS packaging (deferred)** | Fold the standalone OSS `comms/github/fb/pipes` package into ctran's packaging. | Separate follow-up, decoupled from D1/D2. |

D1 and D2 are a stacked pair. Splitting them keeps each diff single-character — D1 mechanical (move + rename), D2 structural (reorg) — and independently reviewable.

## 5. Key decisions & trade-offs

| Decision | Choice | Rationale / trade-off |
|---|---|---|
| Fold vs. reorg in one diff | **Two diffs** (D1 mechanical, D2 structural) | D1 is a trivially-verifiable prefix+namespace codemod; D2 is the real reorg. Cost: a file's path changes in both diffs — but D2 keeps Buck target names stable, so its churn is include-paths only. |
| Namespace | **Rename `comms::pipes` → `ctran::pipes`** | pipes is a ctran submodule, not standalone. Keep pipes' existing `snake_case` style — harmonizing to ctran's style would be large, unnecessary churn. |
| `experimental/` placement | **Location only**, nested under `transport/` | Signals that IBGDA (DOCA-GPUNetIO) is the experimental, API-unstable transport with heavyweight external deps (DOCA, ibverbs, NIC backends) — **without** implying it is optional or dead. It remains the production inter-node transport and a mandatory build dependency. No interface, no optional-build, no behavior change. |
| Buck layout | **Keep granular per-file targets; pipes is a ctran sub-package** | Buck sub-package boundaries already shield pipes from ctran's source globs (the same way `backends/`, `commstate/` are excluded today). Only the CMake build needs an explicit `pipes` exclude (its `file(GLOB_RECURSE)` has no package concept). Absorbing pipes into ctran's glob targets would break the per-target AMD `select()`s and pull DOCA into core unconditionally. |
| `ll`/`ll128` vs `Signal/Barrier/ChunkState` | `ll`,`ll128` → `nvl/`; `Signal/Barrier/ChunkState` → `core/` | `ll`/`ll128` are NVL-specific optimizations; the sync/state primitives are general per-transport building blocks. |
| OSS packaging | **Re-path now, keep separate; fold later (D3)** | Keeps D1/D2 mechanical and avoids entangling them with an OSS-release restructure. (Note a pre-existing inconsistency to reconcile later: `ENABLE_PIPES` defaults ON under Buck but OFF in the OSS make build.) |
| Cvars / `ENABLE_PIPES` | **Unchanged**; `ENABLE_PIPES` stays exported | All cvars stay `NCCL_CTRAN_*` (renames break operator muscle memory). `ENABLE_PIPES` is consumed by downstream ncclx/torchcomms guards, so it stays an exported define. |
| Opportunistic cleanups | **Deferred** | E.g. the currently-unconditional ctran→pipes NVL dependency (`CtranAlgo` includes `P2pNvlTransportDevice` even with pipes off) and the raw `comms::pipes::Transport*` returned by `getMultiPeerTransportsPtr()`. Real, but orthogonal to the move — better as follow-ups. |

## 6. Validation

- **Build:** pipes-on (`-c nccl.enable_pipes=true`) and pipes-off, plus AMD (`@mode/opt-amd-gpu`, `-c hpc_comms.nic=...`); all with `-c hpc_comms.use_ncclx=stable`.
- **Downstream consumers:** `torchcomms`, `ncclx` (meta/rma), `experiments/afd`.
- **Tests:** existing unit tests around the seam (`MultiPeerTransportTest`, window device tests, `DeviceAllToAllv*`, hierarchical AllGather) + the mandatory MAST device-API iterated tests (H100/GB200 × n1/n2).
- Each diff is self-consistent and correct on its own, so **CI is expected green per-diff**.
