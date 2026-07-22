# Send/Recv Benchmark Results

These results compare NCCL and the MCCL IBGDA blocking send/recv path using the
same `nccl_sendrecv_perf` binary. All rows are out-of-place results.

## Configuration

| Field | Value |
| --- | --- |
| Run date | 2026-07-22 |
| Ranks | 2 local ranks |
| Topology | `IB_ONLY` |
| Data type | `uint8` |
| Message sizes | 1 B through 64 MiB, factor 2 |
| Warmup iterations | 5 |
| Timed iterations | 20 |
| Registered memory | Disabled |
| Validation | Enabled |

## H100

Hardware: 2 NVIDIA H100 80GB HBM3 GPUs on `devgpu005`.

### Command

```bash
fbcode/comms/mccl/benchmarks/benchmark_sendrecv_nccl_tests.sh \
  --variants nccl,mccl-ibgda \
  --topologies IB_ONLY \
  --arch h100 \
  --np 2 \
  --hosts localhost:2 \
  --min-bytes 1 \
  --max-bytes 64M \
  --factor 2 \
  --iters 20 \
  --warmup 5 \
  --register 0
```

### Results

| Size | NCCL latency (us) | NCCL BW (GB/s) | IBGDA latency (us) | IBGDA BW (GB/s) | Wrong |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 B | 31.70 | 0.00 | 52.96 | 0.00 | 0 |
| 2 B | 29.01 | 0.00 | 35.98 | 0.00 | 0 |
| 4 B | 29.23 | 0.00 | 48.73 | 0.00 | 0 |
| 8 B | 27.43 | 0.00 | 86.27 | 0.00 | 0 |
| 16 B | 26.79 | 0.00 | 61.90 | 0.00 | 0 |
| 32 B | 31.72 | 0.00 | 64.89 | 0.00 | 0 |
| 64 B | 25.50 | 0.00 | 35.96 | 0.00 | 0 |
| 128 B | 29.51 | 0.00 | 36.87 | 0.00 | 0 |
| 256 B | 48.06 | 0.01 | 39.17 | 0.01 | 0 |
| 512 B | 31.64 | 0.02 | 36.18 | 0.01 | 0 |
| 1 KiB | 31.29 | 0.03 | 36.34 | 0.03 | 0 |
| 2 KiB | 42.07 | 0.05 | 36.25 | 0.06 | 0 |
| 4 KiB | 26.80 | 0.15 | 36.36 | 0.11 | 0 |
| 8 KiB | 33.74 | 0.24 | 38.01 | 0.22 | 0 |
| 16 KiB | 41.81 | 0.39 | 48.96 | 0.33 | 0 |
| 32 KiB | 36.34 | 0.90 | 44.55 | 0.74 | 0 |
| 64 KiB | 41.85 | 1.57 | 43.81 | 1.50 | 0 |
| 128 KiB | 135.6 | 0.97 | 47.70 | 2.75 | 0 |
| 256 KiB | 57.35 | 4.57 | 49.58 | 5.29 | 0 |
| 512 KiB | 61.58 | 8.51 | 60.23 | 8.70 | 0 |
| 1 MiB | 620.5 | 1.69 | 80.31 | 13.06 | 0 |
| 2 MiB | 98.83 | 21.22 | 128.5 | 16.32 | 0 |
| 4 MiB | 199.7 | 21.01 | 152.6 | 27.49 | 0 |
| 8 MiB | 374.2 | 22.42 | 259.7 | 32.30 | 0 |
| 16 MiB | 1,477.9 | 11.35 | 472.2 | 35.53 | 0 |
| 32 MiB | 1,511.5 | 22.20 | 900.7 | 37.26 | 0 |
| 64 MiB | 2,181.1 | 30.77 | 1,756.4 | 38.21 | 0 |

The isolated H100 sweep contains several NCCL latency outliers. Use repeated
runs before treating these NCCL rows as a regression baseline.

## GB300

Hardware: 2 NVIDIA GB300 GPUs on `twshared0104.33.nlh2.facebook.com`.
The IBGDA path used the host's Data-Direct-capable NICs.

### Command

```bash
fbcode/comms/mccl/benchmarks/benchmark_sendrecv_nccl_tests.sh \
  --variants nccl,mccl-ibgda \
  --topologies IB_ONLY \
  --arch gb300 \
  --np 2 \
  --hosts localhost:2 \
  --remote-host twshared0104.33.nlh2.facebook.com \
  --min-bytes 1 \
  --max-bytes 64M \
  --factor 2 \
  --iters 20 \
  --warmup 5 \
  --register 0
```

### Results

| Size | NCCL latency (us) | NCCL BW (GB/s) | IBGDA latency (us) | IBGDA BW (GB/s) | Wrong |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 B | 25.78 | 0.00 | 34.03 | 0.00 | 0 |
| 2 B | 25.04 | 0.00 | 33.94 | 0.00 | 0 |
| 4 B | 25.13 | 0.00 | 33.79 | 0.00 | 0 |
| 8 B | 25.18 | 0.00 | 33.93 | 0.00 | 0 |
| 16 B | 24.97 | 0.00 | 33.64 | 0.00 | 0 |
| 32 B | 25.10 | 0.00 | 33.80 | 0.00 | 0 |
| 64 B | 25.33 | 0.00 | 34.45 | 0.00 | 0 |
| 128 B | 25.18 | 0.01 | 35.35 | 0.00 | 0 |
| 256 B | 25.47 | 0.01 | 35.48 | 0.01 | 0 |
| 512 B | 25.56 | 0.02 | 35.49 | 0.01 | 0 |
| 1 KiB | 25.74 | 0.04 | 35.54 | 0.03 | 0 |
| 2 KiB | 25.81 | 0.08 | 35.71 | 0.06 | 0 |
| 4 KiB | 25.47 | 0.16 | 35.94 | 0.11 | 0 |
| 8 KiB | 26.04 | 0.31 | 36.79 | 0.22 | 0 |
| 16 KiB | 31.59 | 0.52 | 37.97 | 0.43 | 0 |
| 32 KiB | 32.39 | 1.01 | 39.58 | 0.83 | 0 |
| 64 KiB | 35.19 | 1.86 | 42.37 | 1.55 | 0 |
| 128 KiB | 44.65 | 2.94 | 46.46 | 2.82 | 0 |
| 256 KiB | 40.68 | 6.44 | 47.78 | 5.49 | 0 |
| 512 KiB | 44.96 | 11.66 | 57.91 | 9.05 | 0 |
| 1 MiB | 56.34 | 18.61 | 74.61 | 14.05 | 0 |
| 2 MiB | 70.58 | 29.71 | 118.9 | 17.63 | 0 |
| 4 MiB | 99.77 | 42.04 | 146.3 | 28.67 | 0 |
| 8 MiB | 159.1 | 52.73 | 260.2 | 32.24 | 0 |
| 16 MiB | 276.9 | 60.58 | 469.0 | 35.77 | 0 |
| 32 MiB | 476.3 | 70.45 | 890.7 | 37.67 | 0 |
| 64 MiB | 899.2 | 74.63 | 1,746.9 | 38.42 | 0 |

Both variants completed every size with zero wrong counts.
