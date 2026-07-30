# IBGDA Benchmark Results

## Local-Completion Comparison

This benchmark compares serialized raw puts completed through the loopback
counter path with puts completed through the local-completion ticket returned
by `put()`. It isolates the completion primitive and does not represent the
overlapped pipeline behavior of blocking send/recv.

## H100

### Hardware and Configuration

| Field | Value |
| --- | --- |
| Run date | 2026-07-22 |
| Host | `devgpu005` |
| GPU | NVIDIA H100 80GB HBM3 |
| Ranks | 2 local ranks |
| Backend | IBGDA |
| Warmup iterations | 10 |
| Timed iterations | 1,000 |
| Timing | Batched GPU kernel; kernel-launch overhead excluded |

### Command

```bash
buck run @fbcode//mode/opt -c comms.hosts=localhost \
  fbcode//comms/prims/benchmarks:ibgda_benchmark -- \
  --gtest_filter='IbBackend/IbgdaBenchmarkFixture.PutCompletionComparison/IBGDA'
```

### Results

Latency reduction is `(wait_counter - wait_local) / wait_counter`.

| Message size | `put + wait_counter` (us) | `put + wait_local` (us) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 8 B | 13.13 | 9.67 | 26.4% |
| 1 KiB | 13.66 | 9.78 | 28.4% |
| 64 KiB | 15.33 | 11.46 | 25.2% |
| 1 MiB | 37.48 | 33.77 | 9.9% |
| 8 MiB | 188.53 | 184.99 | 1.9% |
| 64 MiB | 1,398.54 | 1,395.25 | 0.2% |
| 1 GiB | 22,186.48 | 22,176.71 | <0.1% |

The local-completion ticket removes approximately 3.5-3.9 us through 1 MiB.
The relative benefit becomes negligible once payload transfer time dominates.

## GB300

### Hardware and Configuration

| Field | Value |
| --- | --- |
| Run date | 2026-07-22 |
| Host | `twshared0104.33.nlh2.facebook.com` |
| GPU | NVIDIA GB300 |
| Ranks | 2 local ranks |
| Backend | IBGDA |
| NIC | `mlx5_4`, 400 Gb/s |
| Data Direct | Inactive on the selected NIC |
| Warmup iterations | 10 |
| Timed iterations | 1,000 |
| Timing | Batched GPU kernel; kernel-launch overhead excluded |

The benchmark uses the non-Data-Direct `mlx5_4` rail because the host rejects
Data-Direct DMA-BUF registration for the benchmark's 8-byte allocation. The
selected rail uses ordinary GPUDirect RDMA registration.

### Command

```bash
buck2 run @fbcode//mode/opt \
  -c hpc_comms.use_ncclx=2.29 \
  -c fbcode.arch=aarch64 \
  -c fbcode.enable_gpu_sections=true \
  -c fbcode.nvcc_arch=b300 \
  -c fbcode.platform010-aarch64_clang=17 \
  -c fbcode.platform010_cuda_version=13.0 \
  -m ovr_config//third-party/cuda/constraints:13.0 \
  -c comms.hosts=twshared0104.33.nlh2.facebook.com \
  -c comms.ifname=lo \
  -c 'comms.envs=NCCL_IB_HCA=mlx5_4' \
  fbcode//comms/prims/benchmarks:ibgda_benchmark -- \
  --gtest_filter='IbBackend/IbgdaBenchmarkFixture.PutCompletionComparison/IBGDA'
```

### Results

| Message size | `put + wait_counter` (us) | `put + wait_local` (us) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 8 B | 11.38 | 7.90 | 30.6% |
| 1 KiB | 12.41 | 8.01 | 35.5% |
| 64 KiB | 13.87 | 9.55 | 31.1% |
| 1 MiB | 35.25 | 31.06 | 11.9% |
| 8 MiB | 182.91 | 178.94 | 2.2% |
| 64 MiB | 1,371.40 | 1,362.10 | 0.7% |
| 1 GiB | 22,021.93 | 22,060.31 | -0.2% |

The fixed-cost reduction is approximately 3.5-4.4 us through 1 MiB. At large
sizes the difference is within run-to-run noise because transfer time
dominates.
