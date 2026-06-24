# CTRAN-IB NIC-interleaved QP round-robin — GB200 sweep

`NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE` distributes a single op's QP-scaling
sub-chunks round-robin across a VC's NICs. Data QPs are stored device-major
(`[NIC0: qp0..K-1, NIC1: qpK..2K-1, ...]`), so the default walk fills one NIC's
K QPs before the next; interleaving remaps the visit order to
`qp0, qpK, qp2K, ..., qp1, qpK+1, ...` so an op fans out across all NICs. Gated
by `NCCL_CTRAN_IB_QP_INTERLEAVE_MIN_WQE_SIZE` (default 64K). Change under test:
D108913561. GB200 (`b200a`, CUDA 12.8, 2 NICs/rank), ppn=2 nolocal, gated fbpkg
`nccl_tests_suite:8e4174e24577b70749b8013f56765893`.

---

## 1. TL;DR

**Microbench (`BM_CtranIb_MultiPut`, see §2).** With N concurrent puts,
interleave gives up to **+70% aggregate BW** in a mid-size window (~128K–2M for
2 puts, ~128K–1M for 4 puts) and lands the first put up to ~50% sooner (2-put
1M: 55 → 29 µs). Outside the window it is flat to slightly negative (≤64K,
latency-bound) — the region the per-WQE gate (default 64K) excludes. The upper
bound is NIC saturation: a put with ≥ `K=8` sub-chunks already spans both NICs
even OFF, which is why toggling the cvar on one big put shows nothing.

**Algo sweeps (see §3), interleave OFF→ON on multi-NIC GB200/nolocal:**

- **AllGather `ctring`** — peak **+56% / +81% / +87% / +74%** (n=2/4/8/16); the
  benefit band moves right with scale (8–16 MiB at n=2 → 64–128 MiB at n=16).
- **AllGather `ctsrd`** — peak **+33% / +38% / +36% / +27%** (4–32 MiB).
- **AllReduce `ctring` (fp32)** — peak **+27% / +36% / +34% / +27%** (8–64 MiB).
- **No regression (gated).** At ≤64K the microbench is flat-to-slightly-negative
  and the AG `ctring` *collective* regresses ~5–11% at per-rank `sendSize=128K`
  without a gate (64K pieces from `NUM_SPLIT=2` + dqplb in-order notify). The
  per-WQE gate (`NCCL_CTRAN_IB_QP_INTERLEAVE_MIN_WQE_SIZE`, default 64K) skips
  interleave for WQEs ≤ 64K, so ON matches OFF there (≈±0.1%) while keeping the
  wins; AllReduce never hit it (WQEs ≥256K). Small (<1 MiB) and very-large
  (≥512 MiB) sizes are neutral.

**Recommendation:** enable for multi-NIC CTRAN collectives
(`NCCL_CTRAN_IB_DEVICES_PER_RANK > 1`); with the gate there is no downside —
worth considering on-by-default when a VC spans multiple NICs. Keep
`NCCL_CTRAN_IB_QP_INTERLEAVE_MIN_WQE_SIZE` (default 64K) to avoid the loss at
small, latency-bound messages where one NIC already suffices and interleave is
not needed.

---

## 2. CtranIB bench result

### What

`BM_CtranIb_MultiPut2` / `BM_CtranIb_MultiPut4` in
`fbcode/comms/ctran/backends/ib/benchmarks/CtranIbBench.cc` measure the effect of
`NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE`, which a single large put cannot
reveal. Each issues N concurrent `CtranIb::iput`s (distinct offsets), drives
`progress()` while polling `checkNotify()`, and timestamps each put's notify
arrival (`notify1_us` … `notifyN_us`) plus aggregate `BW_GBps`. High-level iput
API only, on the existing `kExternal` single-VC setup.

### Results (GB200, `DEVICES_PER_RANK=2`, `MAX_QPS=16`, `qpScalingTh=512K`)

Aggregate BW (GB/s) and per-put notify arrival (µs), interleave OFF vs ON,
devs=2, swept 32K→4M. `1st-put` = `notify1_us` (first of the N puts arrives),
`last-put` = `notifyN_us` (all N complete).

**2 concurrent puts**

| Size | BW OFF | BW ON | ON/OFF | 1st-put OFF→ON | last-put OFF→ON |
|---|---|---|---|---|---|
| 32K | 3.53 | 3.46 | ×0.98 | 14.1 → 14.3 | 15.2 → 15.4 |
| 64K | 7.10 | 6.63 | ×0.93 | 14.9 → 15.2 | 16.2 → 16.2 |
| 128K | 11.68 | 12.88 | ×1.10 ← win starts | 17.8 → 17.4 | 19.1 → 17.9 |
| 256K | 18.82 | 21.20 | ×1.13 | 21.4 → 20.1 | 24.5 → 21.2 |
| 512K | 28.05 | 34.68 | ×1.24 | 28.4 → **25.6** | 35.2 → **26.7** |
| 1M | 34.69 | 52.85 | ×1.52 | 55.2 → **29.3** | 57.1 → **36.2** |
| 2M | 40.43 | **68.62** | **×1.70** ← peak | 94.2 → **54.7** | 100.4 → **57.7** |
| 4M | 77.41 | 79.50 | ×1.03 ← converged | 101.7 → 98.9 | 105.1 → 102.0 |

**4 concurrent puts**

| Size | BW OFF | BW ON | ON/OFF | 1st-put OFF→ON | last-put OFF→ON |
|---|---|---|---|---|---|
| 32K | 6.34 | 6.24 | ×0.98 | 14.2 → 14.4 | 17.6 → 18.0 |
| 64K | 11.68 | 11.95 | ×1.02 | 16.0 → 15.3 | 19.1 → 18.9 |
| 128K | 18.75 | 24.09 | ×1.28 ← win starts | 21.1 → 18.1 | 24.6 → 20.0 |
| 256K | 26.71 | 35.73 | ×1.34 | 25.6 → 22.8 | 36.0 → **25.9** |
| 512K | 34.45 | 52.13 | ×1.51 | 45.8 → **28.8** | 57.5 → **36.7** |
| 1M | 40.10 | **68.44** | **×1.71** ← peak | 92.2 → **43.9** | 101.3 → **57.7** |
| 2M | 77.17 | 78.76 | ×1.02 ← converged | 95.2 → 97.6 | 105.4 → 103.0 |
| 4M | 85.91 | 87.42 | ×1.02 | 118.3 → 115.3 | 192.0 → 188.4 |

In the win band interleave lands the **first** put much sooner — 2-put 1M:
55.2 → 29.3 µs (−47%); 4-put 1M: 92.2 → 43.9 µs (−52%) — the latency benefit
that matters for the pipelined collective. At ≤64K, ON ≈ OFF (within noise,
slightly negative at 2-put 64K) — the latency-bound region the gate excludes.

### Key finding

- **The win window is bounded on both sides.** 2 puts: starts ~128K, peaks at 2M (**+70%**), converged by 4M. 4 puts: starts ~128K, peaks at 1M (**+71%**), converged by 2M.
  - **Lower bound (latency floor):** at ≤64K interleave is flat to slightly negative (2-put 64K ×0.93, 4-put 64K ×1.02) — the transfer is latency-bound, one NIC finishes in ~the same time, and splitting across NICs can cost a little. This is exactly the region the per-WQE gate (default 64K) excludes.
  - **Upper bound (NIC saturation):** once each put has ≥ `K=8` sub-chunks (`qpScalingTh=512K` → 4M for 2 puts, 2M for 4 puts) interleave-OFF *already* spreads across both NICs, so the gap closes. Also why toggling the cvar on a single large put shows nothing.
- **The upper bound halves as the put count doubles** (converges at 4M for 2 puts vs 2M for 4 puts): more concurrent puts saturate both NICs at smaller per-put sizes.
- **Why the min-size gate.** Both signals point to gating small WQEs: the microbench is flat-to-slightly-negative at ≤64K (above), and the AG `ctring` *collective* regresses harder — ~5–11% at per-rank `sendSize=128K`, ungated (n4 @1MiB −8%, n8 @2MiB −9%, n16 @4MiB −11%). The microbench fires *independent* puts, so spreading is mostly harmless; the AG ring is a dependency pipeline with dqplb in-order notify, and `NUM_SPLIT=2` makes each step two 64K puts that interleave splits 1-per-NIC — a cross-NIC straggler / head-of-line stall with no bandwidth to offset it. `NCCL_CTRAN_IB_QP_INTERLEAVE_MIN_WQE_SIZE` (default 64K) skips interleave for WQEs ≤ 64K: removes the collective regression while keeping the win (the §3 AllGather tables are gated, so the dip is gone).

### Reproduce

```bash
# interleave OFF, then re-run with ...ENABLE=1
NCCL_CTRAN_IB_DEVICES_PER_RANK=2 NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE=0 \
  buck2 run @fbcode//mode/opt -c fbcode.arch=aarch64 \
  -c fbcode.platform010-aarch64_clang=17 -c fbcode.nvcc_arch=b200 \
  -c fbcode.platform010_cuda_version=12.8 \
  -m ovr_config//third-party/cuda/constraints:12.8 \
  fbcode//comms/ctran/backends/ib/benchmarks:ctranib_bench -- \
  --benchmark_filter=BM_CtranIb_MultiPut
```

---

## 3. Algo bench — config & result

### Configs

| Sweep | Collective | Algo | Interleave |
|---|---|---|---|
| `ag ctring OFF` | AllGather | `ctring` | off |
| `ag ctring ON`  | AllGather | `ctring` | `NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE=1` |
| `ag ctsrd OFF`  | AllGather | `ctsrd`  | off |
| `ag ctsrd ON`   | AllGather | `ctsrd`  | `NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE=1` |
| `ar ctring OFF` | AllReduce | `ctring` | off |
| `ar ctring ON`  | AllReduce | `ctring` | `NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE=1` |

Interleave ON additionally requires each posted WQE to exceed
`NCCL_CTRAN_IB_QP_INTERLEAVE_MIN_WQE_SIZE` (default 64K); WQEs at or below it
fall back to the sequential QP walk.

### MAST jobs (ppn=2 nolocal, GB200)

AllGather:

| Nodes | GPUs | MAST job |
|---|---|---|
| 2 | 4 | [torchx-nccltests-gb200-ag-n2ppn2-nolocal-815](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ag-n2ppn2-nolocal-815) |
| 4 | 8 | [torchx-nccltests-gb200-ag-n4ppn2-nolocal-527](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ag-n4ppn2-nolocal-527) |
| 8 | 16 | [torchx-nccltests-gb200-ag-n8ppn2-nolocal-542](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ag-n8ppn2-nolocal-542) |
| 16 | 32 | [torchx-nccltests-gb200-ag-n16ppn2-nolocal-557](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ag-n16ppn2-nolocal-557) |

AllReduce:

| Nodes | GPUs | MAST job |
|---|---|---|
| 2 | 4 | [torchx-nccltests-gb200-ar-n2ppn2-nolocal-724](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ar-n2ppn2-nolocal-724) |
| 4 | 8 | [torchx-nccltests-gb200-ar-n4ppn2-nolocal-763](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ar-n4ppn2-nolocal-763) |
| 8 | 16 | [torchx-nccltests-gb200-ar-n8ppn2-nolocal-778](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ar-n8ppn2-nolocal-778) |
| 16 | 32 | [torchx-nccltests-gb200-ar-n16ppn2-nolocal-793](https://www.internalfb.com/mlhub/pipelines/runs/mast/torchx-nccltests-gb200-ar-n16ppn2-nolocal-793) |

### Setup

- **Hardware / topology:** GenAI GB200 cluster (`MastGenAICluster`, `gb200`),
  ppn=2 with `--nolocal` (SHM/P2P disabled), `--bha 2`,
  `NCCL_CTRAN_IB_DEVICES_PER_RANK=2`.
- **Entitlement / locality / DP:** `msl_tbd_iris` / `region;nao` /
  `genai_llm_research-llama`.
- **Build:** `nvcc_arch=b200a`, CUDA 12.8 (`use_ncclx=stable`, local stack incl.
  D108913561 + the per-WQE interleave gate). fbpkg `nccl_tests_suite:8e4174e24577b70749b8013f56765893` (gated; both AG/AR).
- **Interleave gate:** `NCCL_CTRAN_IB_QP_INTERLEAVE_MIN_WQE_SIZE` = 65536 (64K).
- **NCCL test args:** `-b 4 -e 1G -n 500 -w 50 -d <dtype> -c 1 -z 1 -f 2 -R 1 -M 1 -u 1`
  (AllGather: `bfloat16`; AllReduce: `float`/fp32 — `ctring` AllReduce rejects
  bfloat16).
- **Reported column:** out-of-place BusBW (GB/s), via `read_sweep_result.py`.
  Δ = (ON − OFF) / OFF.

### AllGather — per-scale tables (out-of-place BusBW, GB/s; interleave gated)

#### n=2 (4 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring | ctsrd OFF | ctsrd ON | Δ ctsrd |
|---|---|---|---|---|---|---|
| 4 KiB | 0.05 | 0.05 | +0.0% | 0.06 | 0.06 | +0.0% |
| 8 KiB | 0.10 | 0.10 | +0.0% | 0.12 | 0.12 | +0.0% |
| 16 KiB | 0.19 | 0.20 | +5.3% | 0.24 | 0.24 | +0.0% |
| 32 KiB | 0.37 | 0.37 | +0.0% | 0.46 | 0.47 | +2.2% |
| 64 KiB | 0.68 | 0.69 | +1.5% | 0.89 | 0.90 | +1.1% |
| 128 KiB | 1.33 | 1.34 | +0.8% | 1.82 | 1.85 | +1.6% |
| 256 KiB | 2.62 | 2.66 | +1.5% | 3.51 | 3.60 | +2.6% |
| 512 KiB | 5.00 | 5.06 | +1.2% | 6.50 | 6.64 | +2.2% |
| 1 MiB | 8.75 | 9.25 | +5.7% | 11.64 | 11.82 | +1.5% |
| 2 MiB | 15.06 | 16.56 | +10.0% | 19.75 | 19.99 | +1.2% |
| 4 MiB | 25.00 | 28.34 | +13.4% | 28.83 | 37.92 | +31.5% |
| 8 MiB | 31.38 | 49.11 | +56.5% | 39.17 | 52.04 | +32.9% |
| 16 MiB | 38.47 | 61.43 | +59.7% | 61.29 | 68.80 | +12.3% |
| 32 MiB | 71.91 | 76.41 | +6.3% | 78.80 | 80.09 | +1.6% |
| 64 MiB | 87.93 | 87.96 | +0.0% | 87.66 | 88.19 | +0.6% |
| 128 MiB | 91.60 | 91.81 | +0.2% | 91.94 | 92.13 | +0.2% |
| 256 MiB | 93.72 | 93.66 | -0.1% | 94.09 | 94.07 | -0.0% |
| 512 MiB | 95.15 | 95.18 | +0.0% | 95.25 | 95.22 | -0.0% |
| 1 GiB | 95.85 | 95.87 | +0.0% | 95.84 | 95.82 | -0.0% |

#### n=4 (8 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring | ctsrd OFF | ctsrd ON | Δ ctsrd |
|---|---|---|---|---|---|---|
| 4 KiB | 0.04 | 0.04 | +0.0% | 0.06 | 0.06 | +0.0% |
| 8 KiB | 0.07 | 0.07 | +0.0% | 0.11 | 0.11 | +0.0% |
| 16 KiB | 0.14 | 0.14 | +0.0% | 0.21 | 0.21 | +0.0% |
| 32 KiB | 0.26 | 0.26 | +0.0% | 0.42 | 0.42 | +0.0% |
| 64 KiB | 0.50 | 0.50 | +0.0% | 0.80 | 0.80 | +0.0% |
| 128 KiB | 0.94 | 0.94 | +0.0% | 1.56 | 1.57 | +0.6% |
| 256 KiB | 1.68 | 1.69 | +0.6% | 2.87 | 2.85 | -0.7% |
| 512 KiB | 3.31 | 3.30 | -0.3% | 5.70 | 5.68 | -0.4% |
| 1 MiB | 6.49 | 6.45 | -0.6% | 10.53 | 10.55 | +0.2% |
| 2 MiB | 11.31 | 11.42 | +1.0% | 18.65 | 18.35 | -1.6% |
| 4 MiB | 19.85 | 21.56 | +8.6% | 30.52 | 30.67 | +0.5% |
| 8 MiB | 31.14 | 35.22 | +13.1% | 39.28 | 54.09 | +37.7% |
| 16 MiB | 33.96 | 61.35 | +80.7% | 49.88 | 65.64 | +31.6% |
| 32 MiB | 41.45 | 67.79 | +63.5% | 66.11 | 79.75 | +20.6% |
| 64 MiB | 77.56 | 84.21 | +8.6% | 87.08 | 87.73 | +0.7% |
| 128 MiB | 92.71 | 92.79 | +0.1% | 92.80 | 92.91 | +0.1% |
| 256 MiB | 94.26 | 94.35 | +0.1% | 94.78 | 94.83 | +0.1% |
| 512 MiB | 95.42 | 95.42 | +0.0% | 95.85 | 95.78 | -0.1% |
| 1 GiB | 96.12 | 96.15 | +0.0% | 96.40 | 96.37 | -0.0% |

#### n=8 (16 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring | ctsrd OFF | ctsrd ON | Δ ctsrd |
|---|---|---|---|---|---|---|
| 4 KiB | 0.02 | 0.02 | +0.0% | 0.05 | 0.05 | +0.0% |
| 8 KiB | 0.04 | 0.04 | +0.0% | 0.09 | 0.10 | +11.1% |
| 16 KiB | 0.08 | 0.08 | +0.0% | 0.19 | 0.19 | +0.0% |
| 32 KiB | 0.16 | 0.16 | +0.0% | 0.35 | 0.37 | +5.7% |
| 64 KiB | 0.31 | 0.31 | +0.0% | 0.72 | 0.72 | +0.0% |
| 128 KiB | 0.59 | 0.58 | -1.7% | 1.41 | 1.46 | +3.5% |
| 256 KiB | 1.03 | 1.04 | +1.0% | 2.12 | 2.59 | +22.2% |
| 512 KiB | 1.84 | 1.82 | -1.1% | 4.72 | 4.67 | -1.1% |
| 1 MiB | 3.58 | 3.63 | +1.4% | 9.23 | 9.39 | +1.7% |
| 2 MiB | 7.10 | 7.09 | -0.1% | 16.67 | 17.13 | +2.8% |
| 4 MiB | 12.27 | 12.30 | +0.2% | 28.95 | 29.41 | +1.6% |
| 8 MiB | 22.35 | 22.90 | +2.5% | 36.79 | 44.65 | +21.4% |
| 16 MiB | 33.66 | 37.49 | +11.4% | 50.18 | 68.26 | +36.0% |
| 32 MiB | 35.55 | 66.52 | +87.1% | 64.84 | 78.88 | +21.7% |
| 64 MiB | 45.24 | 70.46 | +55.7% | 84.26 | 88.87 | +5.5% |
| 128 MiB | 79.55 | 89.40 | +12.4% | 91.92 | 92.60 | +0.7% |
| 256 MiB | 94.89 | 94.91 | +0.0% | 94.88 | 95.04 | +0.2% |
| 512 MiB | 95.84 | 95.75 | -0.1% | 95.93 | 96.04 | +0.1% |
| 1 GiB | 96.36 | 96.30 | -0.1% | 96.54 | 96.55 | +0.0% |

#### n=16 (32 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring | ctsrd OFF | ctsrd ON | Δ ctsrd |
|---|---|---|---|---|---|---|
| 4 KiB | 0.01 | 0.01 | +0.0% | 0.04 | 0.04 | +0.0% |
| 8 KiB | 0.02 | 0.02 | +0.0% | 0.08 | 0.08 | +0.0% |
| 16 KiB | 0.05 | 0.05 | +0.0% | 0.16 | 0.17 | +6.3% |
| 32 KiB | 0.09 | 0.09 | +0.0% | 0.33 | 0.33 | +0.0% |
| 64 KiB | 0.18 | 0.18 | +0.0% | 0.64 | 0.65 | +1.6% |
| 128 KiB | 0.35 | 0.35 | +0.0% | 1.27 | 1.32 | +3.9% |
| 256 KiB | 0.65 | 0.65 | +0.0% | 2.52 | 2.53 | +0.4% |
| 512 KiB | 1.10 | 1.10 | +0.0% | 4.49 | 4.49 | +0.0% |
| 1 MiB | 1.88 | 1.91 | +1.6% | 8.07 | 8.08 | +0.1% |
| 2 MiB | 3.74 | 3.75 | +0.3% | 15.97 | 15.91 | -0.4% |
| 4 MiB | 7.36 | 7.35 | -0.1% | 28.08 | 28.50 | +1.5% |
| 8 MiB | 12.82 | 12.85 | +0.2% | 40.67 | 44.01 | +8.2% |
| 16 MiB | 22.67 | 23.61 | +4.1% | 48.45 | 55.86 | +15.3% |
| 32 MiB | 32.55 | 37.46 | +15.1% | 60.64 | 76.89 | +26.8% |
| 64 MiB | 37.83 | 65.86 | +74.1% | 78.10 | 86.23 | +10.4% |
| 128 MiB | 54.43 | 82.14 | +50.9% | 87.95 | 92.46 | +5.1% |
| 256 MiB | 80.66 | 93.40 | +15.8% | 94.83 | 94.83 | +0.0% |
| 512 MiB | 96.13 | 96.08 | -0.1% | 95.90 | 95.84 | -0.1% |
| 1 GiB | 96.54 | 96.56 | +0.0% | 96.56 | 96.58 | +0.0% |

### AllReduce `ctring` (float32) — per-scale tables (out-of-place BusBW, GB/s; interleave gated)

#### n=2 (4 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring |
|---|---|---|---|
| 4 KiB | 0.05 | 0.05 | +0.0% |
| 8 KiB | 0.10 | 0.10 | +0.0% |
| 16 KiB | 0.19 | 0.19 | +0.0% |
| 32 KiB | 0.36 | 0.36 | +0.0% |
| 64 KiB | 0.63 | 0.61 | -3.2% |
| 128 KiB | 1.18 | 1.18 | +0.0% |
| 256 KiB | 2.42 | 2.40 | -0.8% |
| 512 KiB | 4.84 | 4.82 | -0.4% |
| 1 MiB | 9.19 | 9.06 | -1.4% |
| 2 MiB | 16.29 | 16.60 | +1.9% |
| 4 MiB | 26.32 | 30.28 | +15.0% |
| 8 MiB | 37.79 | 47.89 | +26.7% |
| 16 MiB | 56.82 | 67.57 | +18.9% |
| 32 MiB | 82.61 | 82.69 | +0.1% |
| 64 MiB | 87.75 | 87.85 | +0.1% |
| 128 MiB | 90.27 | 90.32 | +0.1% |
| 256 MiB | 90.71 | 90.77 | +0.1% |
| 512 MiB | 90.92 | 90.96 | +0.0% |
| 1 GiB | 91.07 | 91.13 | +0.1% |

#### n=4 (8 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring |
|---|---|---|---|
| 4 KiB | 0.03 | 0.03 | +0.0% |
| 8 KiB | 0.06 | 0.06 | +0.0% |
| 16 KiB | 0.12 | 0.12 | +0.0% |
| 32 KiB | 0.23 | 0.23 | +0.0% |
| 64 KiB | 0.42 | 0.42 | +0.0% |
| 128 KiB | 0.71 | 0.72 | +1.4% |
| 256 KiB | 1.38 | 1.38 | +0.0% |
| 512 KiB | 2.78 | 2.79 | +0.4% |
| 1 MiB | 5.57 | 5.57 | +0.0% |
| 2 MiB | 10.49 | 10.44 | -0.5% |
| 4 MiB | 18.40 | 18.77 | +2.0% |
| 8 MiB | 29.33 | 34.39 | +17.3% |
| 16 MiB | 41.28 | 53.68 | +30.0% |
| 32 MiB | 55.54 | 75.44 | +35.8% |
| 64 MiB | 89.46 | 89.79 | +0.4% |
| 128 MiB | 92.60 | 92.76 | +0.2% |
| 256 MiB | 94.05 | 94.12 | +0.1% |
| 512 MiB | 94.29 | 94.36 | +0.1% |
| 1 GiB | 94.44 | 94.50 | +0.1% |

#### n=8 (16 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring |
|---|---|---|---|
| 4 KiB | 0.02 | 0.02 | +0.0% |
| 8 KiB | 0.03 | 0.03 | +0.0% |
| 16 KiB | 0.07 | 0.07 | +0.0% |
| 32 KiB | 0.13 | 0.13 | +0.0% |
| 64 KiB | 0.25 | 0.25 | +0.0% |
| 128 KiB | 0.45 | 0.45 | +0.0% |
| 256 KiB | 0.75 | 0.75 | +0.0% |
| 512 KiB | 1.45 | 1.46 | +0.7% |
| 1 MiB | 2.94 | 2.94 | +0.0% |
| 2 MiB | 5.88 | 5.90 | +0.3% |
| 4 MiB | 11.00 | 11.01 | +0.1% |
| 8 MiB | 19.34 | 19.60 | +1.3% |
| 16 MiB | 30.55 | 36.15 | +18.3% |
| 32 MiB | 41.96 | 56.22 | +34.0% |
| 64 MiB | 60.88 | 78.35 | +28.7% |
| 128 MiB | 92.70 | 92.55 | -0.2% |
| 256 MiB | 95.25 | 95.26 | +0.0% |
| 512 MiB | 96.09 | 96.09 | +0.0% |
| 1 GiB | 96.25 | 96.25 | +0.0% |

#### n=16 (32 GPUs)

| Size | ctring OFF | ctring ON | Δ ctring |
|---|---|---|---|
| 4 KiB | 0.01 | 0.01 | +0.0% |
| 8 KiB | 0.02 | 0.02 | +0.0% |
| 16 KiB | 0.04 | 0.04 | +0.0% |
| 32 KiB | 0.07 | 0.07 | +0.0% |
| 64 KiB | 0.13 | 0.13 | +0.0% |
| 128 KiB | 0.25 | 0.25 | +0.0% |
| 256 KiB | 0.45 | 0.45 | +0.0% |
| 512 KiB | 0.76 | 0.76 | +0.0% |
| 1 MiB | 1.47 | 1.47 | +0.0% |
| 2 MiB | 2.95 | 2.96 | +0.3% |
| 4 MiB | 5.89 | 5.87 | -0.3% |
| 8 MiB | 10.98 | 11.06 | +0.7% |
| 16 MiB | 19.26 | 19.60 | +1.8% |
| 32 MiB | 30.80 | 36.32 | +17.9% |
| 64 MiB | 44.44 | 56.27 | +26.6% |
| 128 MiB | 77.74 | 79.13 | +1.8% |
| 256 MiB | 94.52 | 94.66 | +0.1% |
| 512 MiB | 95.38 | 95.44 | +0.1% |
| 1 GiB | 96.35 | 96.39 | +0.0% |
