# IBRC Benchmark Results

Run date: 2026-06-24

Hosts used: `rtptest2333.nha6.facebook.com`, `rtptest2349.nha6.facebook.com`

Launcher:

```bash
python3 comms/testinfra/ncclx_test_launcher.py \
  --launcher mpi \
  --nnode 2 \
  --ppn 1 \
  --hosts rtptest2333.nha6.facebook.com,rtptest2349.nha6.facebook.com \
  --ifname eth0 \
  --env 'NCCL_DEBUG=WARN;NCCL_IB_HCA=mlx5_0,mlx5_1' \
  --gtest_filter 'IbBackend/IbgdaBenchmarkFixture.PutSignalWaitCounter/*:IbBackend/IbgdaBenchmarkFixture.PutSignalComparison/*:IbBackend/IbgdaBenchmarkFixture.PutFlush/*:IbBackend/IbgdaBenchmarkFixture.SignalOnly/*' \
  --testname /data/users/zhiyongww/fbsource/buck-out/v2/art/fbcode/5b4cba3f80737e3b/comms/prims/benchmarks/__ibgda_benchmark_binary__/ibgda_benchmark_binary
```

The `rtptest2333`/`rtptest2349` pair was used because both hosts had CUDA ready (`cuInit=0`, two devices, no compute processes) and the MPI hostname smoke test passed. Before launching, the `2803:*` IPv6 address on `eth0` was temporarily removed on both hosts to avoid Open MPI advertising a peer address that did not match DNS. The `2803:*` addresses were restored after the run. The `rtptest2348`/`rtptest2350` pair was rejected because CUDA initialization returned `802`.

## Send/Recv Benchmark

Run date: 2026-07-01, with exact KB reruns on 2026-07-02

Hosts: `rtptest2347.nha6.facebook.com`, `rtptest2348.nha6.facebook.com`

Setup: one rank per host on GPU 0, using the rebased `aarch64` GB200 binary. Values are rank 0 P2P bandwidth in GB/s; bidirectional counts traffic in both directions. KB rows use exact 1000-iteration reruns.

Blocking API:

| Size | Bidir IBGDA | Bidir IBRC | Unidir IBGDA | Unidir IBRC |
| ---: | ---: | ---: | ---: | ---: |
| 1KB | 0.052 | 0.035 | 0.063 | 0.041 |
| 2KB | 0.105 | 0.063 | 0.119 | 0.082 |
| 4KB | 0.210 | 0.129 | 0.236 | 0.164 |
| 8KB | 0.417 | 0.254 | 0.485 | 0.320 |
| 16KB | 0.828 | 0.497 | 0.969 | 0.656 |
| 32KB | 1.57 | 0.944 | 1.94 | 1.12 |
| 64KB | 3.11 | 1.79 | 3.65 | 2.08 |
| 128KB | 5.83 | 3.44 | 6.40 | 3.87 |
| 256KB | 10.41 | 7.11 | 11.24 | 9.84 |
| 512KB | 5.57 | 3.65 | 7.00 | 4.54 |
| 1MB | 25.51 | 18.30 | 33.46 | 24.69 |
| 2MB | 33.07 | 26.71 | 51.28 | 39.54 |
| 4MB | 39.50 | 34.28 | 69.66 | 56.38 |
| 8MB | 43.67 | 40.59 | 59.97 | 55.63 |
| 16MB | 69.60 | 64.37 | 59.14 | 55.21 |
| 32MB | 85.60 | 79.21 | 58.78 | 57.40 |
| 64MB | 97.67 | 89.76 | 58.60 | 57.41 |
| 128MB | 105.40 | 97.46 | 58.50 | 55.36 |
| 256MB | 111.47 | 102.32 | 57.81 | 56.05 |
| 512MB | 114.46 | 107.16 | 57.71 | 57.48 |
| 1GB | 112.37 | 108.70 | 57.69 | 57.98 |
| 2GB | 115.88 | 111.14 | 58.22 | 60.24 |
| 4GB | 117.58 | 111.39 | 57.69 | 59.55 |

Progress API:

| Size | Bidir IBGDA | Bidir IBRC | Unidir IBGDA | Unidir IBRC |
| ---: | ---: | ---: | ---: | ---: |
| 1KB | 0.050 | 0.031 | 0.054 | 0.049 |
| 2KB | 0.101 | 0.063 | 0.111 | 0.074 |
| 4KB | 0.200 | 0.138 | 0.221 | 0.143 |
| 8KB | 0.401 | 0.247 | 0.444 | 0.290 |
| 16KB | 0.798 | 0.494 | 0.889 | 0.592 |
| 32KB | 1.52 | 0.946 | 1.65 | 1.06 |
| 64KB | 2.91 | 1.82 | 3.22 | 2.68 |
| 128KB | 5.22 | 3.50 | 5.58 | 3.83 |
| 256KB | 9.44 | 7.19 | 10.24 | 9.61 |
| 512KB | 5.06 | 3.80 | 6.04 | 4.35 |
| 1MB | 24.00 | 19.00 | 31.28 | 24.54 |
| 2MB | 32.21 | 27.35 | 48.70 | 39.26 |
| 4MB | 39.86 | 35.19 | 67.23 | 56.02 |
| 8MB | 44.24 | 41.25 | 59.73 | 55.44 |
| 16MB | 69.15 | 64.73 | 59.06 | 55.01 |
| 32MB | 84.96 | 79.86 | 58.66 | 54.58 |
| 64MB | 96.60 | 90.81 | 58.43 | 55.88 |
| 128MB | 104.10 | 97.88 | 57.62 | 56.83 |
| 256MB | 110.46 | 103.44 | 57.57 | 56.08 |
| 512MB | 112.53 | 108.19 | 57.52 | 57.51 |
| 1GB | 112.41 | 110.29 | 57.53 | 57.55 |
| 2GB | 115.33 | 113.14 | 58.08 | 58.86 |
| 4GB | 119.67 | 112.19 | 57.53 | 59.31 |

## Ring ReduceScatter Benchmark

Run date: 2026-06-26

Hosts used: `rtptest2329.nha6.facebook.com`, `rtptest2330.nha6.facebook.com`, `rtptest2344.nha6.facebook.com`, `rtptest2345.nha6.facebook.com`

Benchmark target:

```bash
fbcode//comms/prims/collectives/benchmarks:ring_reduce_scatter_benchmark_binary
```

Run command:

```bash
RUN_ID=codex_4host_ppn1_b_20260626_115913 \
  HOSTS=rtptest2329.nha6.facebook.com,rtptest2330.nha6.facebook.com,rtptest2344.nha6.facebook.com,rtptest2345.nha6.facebook.com \
  NNODES=4 \
  PPN=1 \
  GPU_ID=0 \
  /tmp/run_ibrc_reduce_scatter_benchmark.sh
```

Full log: `/tmp/ibrc_reduce_scatter_codex_4host_ppn1_b.log`

The benchmark was run with four ranks total, one rank per host on GPU 0. Values are bandwidth in GB/s and latency in microseconds. The `Size` column is total bytes across the four ranks; `chunk_elements` is per-rank output elements.

| Test | Size | Rings | IBGDA BW | IBRC BW | Speedup | IBGDA Latency (us) | IBRC Latency (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256K_8B | 1MB | 1 | 9.00 | 5.81 | 0.64x | 116.4 | 180.5 |
| 1M_16B | 4MB | 1 | 22.88 | 18.05 | 0.79x | 183.3 | 232.4 |
| 4M_16B | 16MB | 1 | 43.88 | 40.34 | 0.92x | 382.3 | 415.9 |
| 16M_16B | 64MB | 1 | 52.07 | 53.07 | 1.02x | 1288.9 | 1264.7 |
| 64M_16B | 256MB | 1 | 60.61 | 61.78 | 1.02x | 4428.7 | 4344.7 |
| 128M_16B | 512MB | 1 | 61.34 | 62.84 | 1.02x | 8752.1 | 8544.1 |
| 256M_32B | 1GB | 1 | 62.95 | 62.86 | 1.00x | 17055.8 | 17081.4 |
| 4M_16B_2R | 16MB | 2 | 40.42 | 37.01 | 0.92x | 415.0 | 453.3 |
| 16M_16B_2R | 64MB | 2 | 55.37 | 52.38 | 0.95x | 1212.0 | 1281.2 |
| 64M_16B_2R | 256MB | 2 | 59.11 | 58.34 | 0.99x | 4541.6 | 4601.6 |
| 256M_32B_2R | 1GB | 2 | 57.01 | 58.16 | 1.02x | 18832.7 | 18462.8 |

### 8 ranks, 4 nodes

Run command:

```bash
/home/zhiyongww/worktrees/IBRC/buck-out/v2/art/fbcode/70552be409c74e42/comms/testinfra/__ncclx_test_launcher__/ncclx_test_launcher.par \
  --launcher mpi \
  --nnode 4 \
  --ppn 2 \
  --interleave \
  --hosts rtptest2329.nha6.facebook.com,rtptest2330.nha6.facebook.com,rtptest2344.nha6.facebook.com,rtptest2345.nha6.facebook.com \
  --ifname eth0 \
  --env 'NCCL_DEBUG=WARN;NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_3,mlx5_4' \
  --gtest_filter RingReduceScatterBenchmarkFixture.IbrcVsIbgda \
  --testname /home/zhiyongww/worktrees/IBRC/buck-out/v2/art/fbcode/7923733aec21c17e/comms/prims/collectives/benchmarks/__ring_reduce_scatter_benchmark_binary__/ring_reduce_scatter_benchmark_binary
```

Full log: `/tmp/ibrc_reduce_scatter_codex_4node_8rank_interleave_retry.log`

The benchmark was run with eight ranks total, two ranks per host on GPUs 0 and 1. `--interleave` is required for this host layout; the default `ppr:2:node` rank ordering reached the benchmark header but did not produce a result row. Values are bandwidth in GB/s and latency in microseconds. The `Size` column is total bytes across the eight ranks.

| Test | Size | Rings | IBGDA BW | IBRC BW | Speedup | IBGDA Latency (us) | IBRC Latency (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256K_8B | 2MB | 1 | 8.26 | 4.66 | 0.56x | 254.0 | 450.0 |
| 1M_16B | 8MB | 1 | 23.48 | 14.04 | 0.60x | 357.3 | 597.6 |
| 4M_16B | 32MB | 1 | 43.09 | 34.34 | 0.80x | 778.7 | 977.1 |
| 16M_16B | 128MB | 1 | 48.85 | 47.73 | 0.98x | 2747.3 | 2811.9 |
| 64M_16B | 512MB | 1 | 52.79 | 54.41 | 1.03x | 10170.8 | 9867.6 |
| 128M_16B | 1GB | 1 | 53.04 | 54.85 | 1.03x | 20242.7 | 19575.8 |
| 256M_32B | 2GB | 1 | 53.37 | 54.72 | 1.03x | 40235.5 | 39246.8 |
| 4M_16B_2R | 32MB | 2 | 19.04 | 30.55 | 1.60x | 1762.0 | 1098.4 |
| 16M_16B_2R | 128MB | 2 | 47.98 | 46.08 | 0.96x | 2797.1 | 2912.5 |
| 64M_16B_2R | 512MB | 2 | 52.04 | 51.52 | 0.99x | 10315.7 | 10421.1 |
| 256M_32B_2R | 2GB | 2 | 48.87 | 49.52 | 1.01x | 43939.0 | 43366.6 |

## CTRAN AllGather Benchmark

Run date: 2026-07-01

Benchmark target:

```bash
fbcode//comms/ctran/benchmarks:allgather_bench_4x1_binary
```

The benchmark was run with `cthierarchical_ring`, `NCCL_CTRAN_ENABLE=1`, `NCCL_CTRAN_USE_PIPES=1`, `NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1`, and `NCCL_CTRAN_PIPES_IB_MODE={ibgda,ibrc}`. Values are rank 0 CTRAN bandwidth in GB/s from the `Finished <size>: NCCL=... ctran=...` lines. The 2-node run used `rtptest2329.nha6.facebook.com` and `rtptest2344.nha6.facebook.com` with 4 ranks total, 2 ranks per host, and `NCCL_CTRAN_IB_DEVICES_PER_RANK=1`. The 4-node run used `rtptest2356.nha6.facebook.com`, `rtptest2357.nha6.facebook.com`, `rtptest2359.nha6.facebook.com`, and `rtptest2360.nha6.facebook.com` with 8 ranks total, 2 ranks per host, `NCCL_CTRAN_IB_DEVICES_PER_RANK=2`, and `NCCL_CTRAN_IB_DEVICE_STRIDE=0`.

Full logs:

| Run | Log |
| --- | --- |
| 2-node IBGDA | `/tmp/rtptest_gb200_allgather_ibgda_sweep.txt` |
| 2-node IBRC | `/tmp/rtptest_gb200_allgather_ibrc_sweep.txt` |
| 4-node IBGDA, two-NIC config | `/tmp/rtptest_gb200_4node_dualnic_allgather_ibgda_sweep.txt` |
| 4-node IBRC, two-NIC config | `/tmp/rtptest_gb200_4node_dualnic_allgather_ibrc_sweep.txt` |

| Size | 2-node IBGDA | 2-node IBRC | 4-node IBGDA | 4-node IBRC |
| ---: | ---: | ---: | ---: | ---: |
| 8KB | 0.075 | 0.071 | 0.028 | 0.027 |
| 16KB | 0.151 | 0.149 | 0.052 | 0.053 |
| 32KB | 0.300 | 0.304 | 0.110 | 0.107 |
| 64KB | 0.620 | 0.586 | 0.223 | 0.211 |
| 128KB | 1.14 | 1.14 | 0.453 | 0.443 |
| 256KB | 2.37 | 2.39 | 0.881 | 0.844 |
| 512KB | 4.73 | 4.63 | 1.74 | 1.65 |
| 1MB | 8.72 | 8.65 | 3.57 | 3.45 |
| 2MB | 15.23 | 15.01 | 6.23 | 6.12 |
| 4MB | 25.98 | 25.66 | 11.34 | 10.94 |
| 8MB | 34.37 | 36.82 | 17.82 | 18.72 |
| 16MB | 46.12 | 47.39 | 25.80 | 26.84 |
| 32MB | 50.81 | 53.00 | 33.65 | 35.87 |
| 64MB | 58.60 | 58.33 | 40.84 | 41.62 |
| 128MB | 60.61 | 60.34 | 44.41 | 45.35 |
| 256MB | 62.87 | 62.80 | 45.69 | 47.02 |
| 512MB | 64.51 | 61.33 | 45.22 | 46.02 |
| 1GB | 64.07 | 63.77 | 46.96 | 48.40 |

The 4-node two-NIC setting did not double the CTRAN bandwidth for this path. A follow-up run with `NCCL_CTRAN_IB_QP_INTERLEAVE_DEVICES_ENABLE=1` produced similar top-end results (`47.02 GB/s` for IBGDA and `48.25 GB/s` for IBRC at 1GB), which is consistent with `cthierarchical_ring` using the prims hierarchical fused send/recv path instead of the CTRAN-IB QP interleave path.

## Current Counter-Slot Results

These results are from the 2026-06-24 rerun after pinning the IBRC progress thread. The initial filtered run covered `PutFlush`, `PutSignalWaitCounter`, `SignalOnly`, and `PutSignalComparison`. `PutWaitCounter` required a local benchmark test fix from `TEST_F` to `TEST_P` so the backend parameterization works; it was then rebuilt for `aarch64` and rerun separately on the same hosts.

Current logs:

| Run | Log |
| --- | --- |
| 2026-06-24 filtered rerun on `rtptest2333`/`rtptest2349` | `/tmp/ibrc-benchmark-direct-2333-2349-no2803-20260624-151520.log` |
| 2026-06-24 fixed `PutWaitCounter` rerun on `rtptest2333`/`rtptest2349` | `/tmp/ibrc-benchmark-putwait-fixed-2333-2349-no2803-20260624-160334.log` |

Summary:

| Benchmark | IBGDA | IBRC |
| --- | ---: | ---: |
| `PutFlush`, 1GB | 22197.61 us, 48.37 GB/s | 22203.10 us, 48.36 GB/s |
| `PutWaitCounter`, 1GB | 22204.56 us, 48.36 GB/s | 22205.06 us, 48.36 GB/s |
| `PutSignalWaitCounter`, 1GB | 22204.38 us, 48.36 GB/s | 22211.54 us, 48.34 GB/s |
| `PutSignalComparison`, 16MB | 365.32 us | 374.78 us |
| `SignalOnly`, average latency | 12.25 us | 30.33 us |
| `MultiPeerCounterFanOut`, 64KB per peer | Not rerun in 2026-06-24 filtered run | N/A: explicit shared-counter-buffer benchmark |

## PutWaitCounter

Rerun separately on 2026-06-24 using a locally rebuilt `aarch64` benchmark binary after fixing the test declaration to `TEST_P(IbgdaBenchmarkFixture, PutWaitCounter)`. The previous `TEST_F` declaration aborted before measuring because it called `GetParam()` outside a parameterized test.

| Size | IBGDA Latency (us) | IBGDA BW (GB/s) | IBRC Latency (us) | IBRC BW (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| 8B | 15.63 | 0.00 | 19.88 | 0.00 |
| 64B | 16.77 | 0.00 | 20.01 | 0.00 |
| 256B | 17.75 | 0.01 | 19.63 | 0.01 |
| 1KB | 17.94 | 0.06 | 20.10 | 0.05 |
| 4KB | 18.03 | 0.23 | 19.99 | 0.20 |
| 8KB | 18.30 | 0.45 | 20.14 | 0.41 |
| 16KB | 18.53 | 0.88 | 20.31 | 0.81 |
| 32KB | 19.06 | 1.72 | 22.65 | 1.45 |
| 64KB | 19.75 | 3.32 | 22.76 | 2.88 |
| 128KB | 21.97 | 5.97 | 22.99 | 5.70 |
| 256KB | 24.66 | 10.63 | 27.89 | 9.40 |
| 512KB | 30.06 | 17.44 | 33.58 | 15.61 |
| 1MB | 40.80 | 25.70 | 42.43 | 24.71 |
| 2MB | 62.35 | 33.64 | 66.19 | 31.68 |
| 4MB | 105.49 | 39.76 | 109.14 | 38.43 |
| 8MB | 191.77 | 43.74 | 195.05 | 43.01 |
| 16MB | 364.31 | 46.05 | 368.27 | 45.56 |
| 32MB | 709.46 | 47.30 | 713.38 | 47.04 |
| 64MB | 1400.05 | 47.93 | 1403.72 | 47.81 |
| 128MB | 2786.51 | 48.17 | 2785.43 | 48.19 |
| 256MB | 5567.58 | 48.21 | 5568.85 | 48.20 |
| 512MB | 11113.65 | 48.31 | 11114.50 | 48.30 |
| 1GB | 22204.56 | 48.36 | 22205.06 | 48.36 |

## PutSignalWaitCounter

| Size | IBGDA Latency (us) | IBGDA BW (GB/s) | IBRC Latency (us) | IBRC BW (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| 8B | 16.76 | 0.00 | 27.75 | 0.00 |
| 64B | 18.90 | 0.00 | 27.74 | 0.00 |
| 256B | 18.98 | 0.01 | 27.80 | 0.01 |
| 1KB | 19.08 | 0.05 | 27.80 | 0.04 |
| 4KB | 19.31 | 0.21 | 27.77 | 0.15 |
| 8KB | 19.50 | 0.42 | 27.73 | 0.30 |
| 16KB | 19.67 | 0.83 | 27.86 | 0.59 |
| 32KB | 20.11 | 1.63 | 27.81 | 1.18 |
| 64KB | 20.80 | 3.15 | 30.52 | 2.15 |
| 128KB | 23.05 | 5.69 | 31.22 | 4.20 |
| 256KB | 25.77 | 10.17 | 33.78 | 7.76 |
| 512KB | 31.19 | 16.81 | 41.46 | 12.65 |
| 1MB | 42.00 | 24.97 | 50.49 | 20.77 |
| 2MB | 63.53 | 33.01 | 72.62 | 28.88 |
| 4MB | 106.72 | 39.30 | 115.90 | 36.19 |
| 8MB | 192.90 | 43.49 | 201.96 | 41.54 |
| 16MB | 365.26 | 45.93 | 374.67 | 44.78 |
| 32MB | 710.34 | 47.24 | 719.73 | 46.62 |
| 64MB | 1400.89 | 47.90 | 1410.88 | 47.57 |
| 128MB | 2786.67 | 48.16 | 2793.80 | 48.04 |
| 256MB | 5567.98 | 48.21 | 5575.46 | 48.15 |
| 512MB | 11113.82 | 48.31 | 11121.18 | 48.27 |
| 1GB | 22204.38 | 48.36 | 22211.54 | 48.34 |

## PutSignalComparison

| Size | IBGDA Counter Latency (us) | IBRC Counter Latency (us) |
| --- | ---: | ---: |
| 8B | 16.78 | 27.42 |
| 64B | 18.80 | 27.46 |
| 1KB | 18.94 | 27.38 |
| 64KB | 20.83 | 31.11 |
| 1MB | 41.94 | 50.80 |
| 16MB | 365.32 | 374.78 |

## PutFlush Baseline

`PutFlush` completed for both backends. This path does not use the counter API.

| Size | IBGDA Latency (us) | IBGDA BW (GB/s) | IBRC Latency (us) | IBRC BW (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| 8B | 11.81 | 0.00 | 22.67 | 0.00 |
| 64B | 11.83 | 0.01 | 20.11 | 0.00 |
| 256B | 12.27 | 0.02 | 20.08 | 0.01 |
| 1KB | 12.37 | 0.08 | 20.04 | 0.05 |
| 4KB | 12.58 | 0.33 | 20.13 | 0.20 |
| 8KB | 12.80 | 0.64 | 19.98 | 0.41 |
| 16KB | 12.99 | 1.26 | 20.09 | 0.82 |
| 32KB | 13.48 | 2.43 | 21.50 | 1.52 |
| 64KB | 14.16 | 4.63 | 22.28 | 2.94 |
| 128KB | 16.37 | 8.01 | 22.62 | 5.79 |
| 256KB | 19.07 | 13.74 | 26.74 | 9.80 |
| 512KB | 24.48 | 21.42 | 33.40 | 15.70 |
| 1MB | 35.27 | 29.73 | 42.25 | 24.82 |
| 2MB | 56.86 | 36.89 | 65.69 | 31.92 |
| 4MB | 100.02 | 41.93 | 108.51 | 38.65 |
| 8MB | 186.34 | 45.02 | 194.58 | 43.11 |
| 16MB | 359.00 | 46.73 | 366.93 | 45.72 |
| 32MB | 704.31 | 47.64 | 712.63 | 47.09 |
| 64MB | 1394.93 | 48.11 | 1402.98 | 47.83 |
| 128MB | 2777.88 | 48.32 | 2784.92 | 48.19 |
| 256MB | 5561.42 | 48.27 | 5567.89 | 48.21 |
| 512MB | 11107.01 | 48.34 | 11113.06 | 48.31 |
| 1GB | 22197.61 | 48.37 | 22203.10 | 48.36 |

## SignalOnly Baseline

`SignalOnly` completed for both backends. This path does not use the counter API.

| Metric | IBGDA | IBRC |
| --- | ---: | ---: |
| Average latency (us) | 12.25 | 30.33 |
| Batch iterations | 1000 | 1000 |
