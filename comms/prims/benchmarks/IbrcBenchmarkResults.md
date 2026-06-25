# IBRC Benchmark Results

Run date: 2026-06-16

Hosts used: `rtptest2330.nha6`, `rtptest2334.nha6`

Launcher:

```bash
python3 comms/testinfra/ncclx_test_launcher.py \
  --launcher mpi \
  --nnode 2 \
  --ppn 1 \
  --hosts rtptest2330.nha6,rtptest2334.nha6 \
  --ifname eth1 \
  --env 'NCCL_DEBUG=WARN' \
  --testname /home/zhiyongww/worktrees/IBRC/buck-out/v2/art/fbcode/065941cadf27ce30/comms/prims/benchmarks/__ibgda_benchmark_binary__/ibgda_benchmark_binary
```

The initially suggested hosts, `rtptest2347.nha6` and `rtptest2346.nha6`, were not usable for this run because CUDA context creation failed with `CUDA-capable device(s) is/are busy or unavailable`. The results below use the clean GB200 pair listed above.

## Current Counter-Slot Results

These results are after switching the counter benchmarks from explicit counter buffers to the transport-owned counter-slot API. IBRC `PutWaitCounter`, `PutSignalWaitCounter`, and `PutSignalComparison` pass with this path.

Current logs:

| Run | Log |
| --- | --- |
| Expanded sweep through 1GB | `/tmp/ibrc_1gb_sweep.log` |
| IBRC counter-slot retest | `/tmp/ibrc_counter_slot_tests.log` |
| IBGDA counter-slot retest | `/tmp/ibgda_counter_slot_tests.log` |

Summary:

| Benchmark | IBGDA | IBRC |
| --- | ---: | ---: |
| `PutFlush`, 1GB | 22202.67 us, 48.36 GB/s | 23039.32 us, 46.60 GB/s |
| `PutWaitCounter`, 1GB | 22209.76 us, 48.35 GB/s | 22987.54 us, 46.71 GB/s |
| `PutSignalWaitCounter`, 1GB | 22210.01 us, 48.34 GB/s | 22943.53 us, 46.80 GB/s |
| `PutSignalComparison`, 16MB | 368.99 us | 704.07 us |
| `MultiPeerCounterFanOut`, 64KB per peer | Not rerun in current retest | N/A: explicit shared-counter-buffer benchmark |

## PutWaitCounter

| Size | IBGDA Latency (us) | IBGDA BW (GB/s) | IBRC Latency (us) | IBRC BW (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| 8B | 21.27 | 0.00 | 38.28 | 0.00 |
| 64B | 23.02 | 0.00 | 38.27 | 0.00 |
| 256B | 23.07 | 0.01 | 38.26 | 0.01 |
| 1KB | 23.17 | 0.04 | 38.25 | 0.03 |
| 4KB | 23.48 | 0.17 | 38.31 | 0.11 |
| 8KB | 23.71 | 0.35 | 38.29 | 0.21 |
| 16KB | 23.95 | 0.68 | 42.00 | 0.39 |
| 32KB | 24.43 | 1.34 | 43.70 | 0.75 |
| 64KB | 25.10 | 2.61 | 44.16 | 1.48 |
| 128KB | 27.32 | 4.80 | 49.02 | 2.67 |
| 256KB | 30.02 | 8.73 | 52.46 | 5.00 |
| 512KB | 35.44 | 14.79 | 65.69 | 7.98 |
| 1MB | 46.23 | 22.68 | 86.75 | 12.09 |
| 2MB | 67.83 | 30.92 | 129.09 | 16.25 |
| 4MB | 110.97 | 37.80 | 216.09 | 19.41 |
| 8MB | 197.30 | 42.52 | 384.12 | 21.84 |
| 16MB | 369.71 | 45.38 | 695.64 | 24.12 |
| 32MB | 714.95 | 46.93 | 1271.31 | 26.39 |
| 64MB | 1405.48 | 47.75 | 2322.96 | 28.89 |
| 128MB | 2787.44 | 48.15 | 3521.63 | 38.11 |
| 256MB | 5572.73 | 48.17 | 6944.67 | 38.65 |
| 512MB | 11118.86 | 48.28 | 13613.28 | 39.44 |
| 1GB | 22209.76 | 48.35 | 22987.54 | 46.71 |

## PutSignalWaitCounter

| Size | IBGDA Latency (us) | IBGDA BW (GB/s) | IBRC Latency (us) | IBRC BW (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| 8B | 22.05 | 0.00 | 65.29 | 0.00 |
| 64B | 23.09 | 0.00 | 65.33 | 0.00 |
| 256B | 23.22 | 0.01 | 64.58 | 0.00 |
| 1KB | 23.35 | 0.04 | 65.23 | 0.02 |
| 4KB | 23.68 | 0.17 | 65.67 | 0.06 |
| 8KB | 23.94 | 0.34 | 64.94 | 0.13 |
| 16KB | 24.18 | 0.68 | 64.80 | 0.25 |
| 32KB | 24.55 | 1.33 | 64.63 | 0.51 |
| 64KB | 25.24 | 2.60 | 65.06 | 1.01 |
| 128KB | 27.41 | 4.78 | 72.07 | 1.82 |
| 256KB | 30.24 | 8.67 | 78.78 | 3.33 |
| 512KB | 35.43 | 14.80 | 87.25 | 6.01 |
| 1MB | 46.14 | 22.72 | 111.76 | 9.38 |
| 2MB | 67.67 | 30.99 | 149.81 | 14.00 |
| 4MB | 110.66 | 37.90 | 237.82 | 17.64 |
| 8MB | 196.82 | 42.62 | 406.69 | 20.63 |
| 16MB | 369.59 | 45.39 | 731.31 | 22.94 |
| 32MB | 714.92 | 46.93 | 1281.58 | 26.18 |
| 64MB | 1405.51 | 47.75 | 2324.56 | 28.87 |
| 128MB | 2787.41 | 48.15 | 3527.65 | 38.05 |
| 256MB | 5573.03 | 48.17 | 6940.11 | 38.68 |
| 512MB | 11119.03 | 48.28 | 13555.27 | 39.61 |
| 1GB | 22210.01 | 48.34 | 22943.53 | 46.80 |

## PutSignalComparison

| Size | IBGDA Counter Latency (us) | IBRC Counter Latency (us) |
| --- | ---: | ---: |
| 8B | 21.48 | 64.89 |
| 64B | 22.26 | 64.81 |
| 1KB | 22.54 | 64.01 |
| 64KB | 24.50 | 65.44 |
| 1MB | 45.78 | 111.09 |
| 16MB | 368.99 | 704.07 |

## PutFlush Baseline

`PutFlush` completed for both backends. This path does not use the counter API.

| Size | IBGDA Latency (us) | IBGDA BW (GB/s) | IBRC Latency (us) | IBRC BW (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| 8B | 16.89 | 0.00 | 38.64 | 0.00 |
| 64B | 17.17 | 0.00 | 39.58 | 0.00 |
| 256B | 17.25 | 0.01 | 38.55 | 0.01 |
| 1KB | 17.35 | 0.06 | 38.55 | 0.03 |
| 4KB | 17.67 | 0.23 | 38.54 | 0.11 |
| 8KB | 17.79 | 0.46 | 38.53 | 0.21 |
| 16KB | 18.10 | 0.90 | 39.22 | 0.42 |
| 32KB | 18.47 | 1.77 | 38.60 | 0.85 |
| 64KB | 19.15 | 3.42 | 43.39 | 1.51 |
| 128KB | 21.39 | 6.13 | 44.10 | 2.97 |
| 256KB | 24.09 | 10.88 | 50.02 | 5.24 |
| 512KB | 29.48 | 17.78 | 61.50 | 8.52 |
| 1MB | 40.28 | 26.03 | 83.85 | 12.51 |
| 2MB | 61.86 | 33.90 | 127.60 | 16.44 |
| 4MB | 105.03 | 39.94 | 207.77 | 20.19 |
| 8MB | 191.35 | 43.84 | 375.71 | 22.33 |
| 16MB | 364.00 | 46.09 | 695.80 | 24.11 |
| 32MB | 709.30 | 47.31 | 1280.02 | 26.21 |
| 64MB | 1399.92 | 47.94 | 2269.87 | 29.57 |
| 128MB | 2782.83 | 48.23 | 3535.06 | 37.97 |
| 256MB | 5566.01 | 48.23 | 6928.67 | 38.74 |
| 512MB | 11111.88 | 48.32 | 13582.13 | 39.53 |
| 1GB | 22202.67 | 48.36 | 23039.32 | 46.60 |

## SignalOnly Baseline

`SignalOnly` completed for both backends. This path does not use the counter API.

| Metric | IBGDA | IBRC |
| --- | ---: | ---: |
| Average latency (us) | 17.75 | 40.90 |
| Batch iterations | 1000 | 1000 |
