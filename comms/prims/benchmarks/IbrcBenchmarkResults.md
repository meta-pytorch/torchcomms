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
