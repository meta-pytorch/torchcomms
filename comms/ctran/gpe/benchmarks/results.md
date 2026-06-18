# CUDA graph memcpy granularity

The benchmark uses fixed 2 GiB `cudaMalloc` allocations and copies the first `--bytes` bytes. By default, source and destination are on CUDA device 0. With `--p2p`, the source is on device 0 and the destination is on device 1.

Eager command template: `buck run @fbcode//mode/opt -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=gb200 fbcode//comms/ctran/gpe/benchmarks:copy_graph_memcpy_bench -- --bytes <BYTES>`

Graph command template: `buck run @fbcode//mode/opt -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=gb200 fbcode//comms/ctran/gpe/benchmarks:copy_graph_memcpy_bench -- --bytes <BYTES> --graph`

P2P command template: add `--p2p` to either command.

Each run uses `copies_per_graph=100`, `launches=100`, and `warmups=100`.

| Size | Bytes | Eager GB/s | Graph GB/s |
|---|---:|---:|---:|
| 1 GiB | 1073741824 | 3362.3 | 1563.7 |
| 1 GiB + 8 KiB | 1073750016 | 3362.3 | 1564.7 |
| 1 GiB + 64 KiB | 1073807360 | 3405.4 | 3433.9 |
| 1 GiB + 1 MiB | 1074790400 | 3403.6 | 3433.9 |
| 1 GiB + 2 MiB | 1075838976 | 3366.1 | 1564.6 |
| 1 GiB + 4 MiB | 1077936128 | 3367.0 | 1564.5 |
| 1 GiB + 8 MiB | 1082130432 | 3365.4 | 1564.5 |
| 1 GiB + 16 MiB | 1090519040 | 3411.4 | 3434.7 |
| 1 GiB + 32 MiB | 1107296256 | 3362.2 | 1564.5 |
| 1 GiB + 64 MiB | 1140850688 | 3374.2 | 1564.4 |
| 1 GiB + 128 MiB | 1207959552 | 3372.0 | 1564.5 |
| 1 GiB + 256 MiB | 1342177280 | 3377.4 | 1565.0 |
| 1 GiB + 512 MiB | 1610612736 | 3392.1 | 1565.2 |
| 1 GiB + 1 GiB | 2147483648 | 3407.3 | 1565.5 |

Eager memcpy stays near peak bandwidth. CUDA graph memcpy flips between a fast path and a ~1.56 TB/s path depending on copy length.

## 256 MiB Base

| Size | Bytes | Eager GB/s | Graph GB/s |
|---|---:|---:|---:|
| 256 MiB | 268435456 | 3277.6 | 3389.4 |
| 256 MiB + 8 KiB | 268443648 | 3277.6 | 3389.5 |
| 256 MiB + 64 KiB | 268500992 | 3278.3 | 3389.7 |
| 256 MiB + 1 MiB | 269484032 | 3289.2 | 3390.9 |
| 256 MiB + 2 MiB | 270532608 | 3300.9 | 3390.9 |
| 256 MiB + 4 MiB | 272629760 | 3279.2 | 3389.9 |
| 256 MiB + 8 MiB | 276824064 | 3295.8 | 3391.6 |
| 256 MiB + 16 MiB | 285212672 | 3312.5 | 3393.3 |
| 256 MiB + 32 MiB | 301989888 | 3285.2 | 3398.9 |
| 256 MiB + 64 MiB | 335544320 | 3329.4 | 3401.1 |
| 256 MiB + 128 MiB | 402653184 | 3331.7 | 3402.5 |
| 256 MiB + 256 MiB | 536870912 | 3277.7 | 1560.6 |
| 256 MiB + 512 MiB | 805306368 | 3390.3 | 3427.5 |
| 256 MiB + 1 GiB | 1342177280 | 3377.6 | 1565.0 |

## P2P Device 0 to Device 1

The devgpu used for these runs has two GB200 GPUs connected by `NV18`. P2P mode enables peer access in both directions, creates the timing stream on device 1, and submits the copy with `cudaMemcpyAsync(..., cudaMemcpyDefault, stream)`.

| Size | Bytes | Eager GB/s | Graph GB/s |
|---|---:|---:|---:|
| 1 GiB | 1073741824 | 776.9 | 779.8 |
| 1 GiB + 64 KiB | 1073807360 | 776.9 | 779.8 |
| 1 GiB + 1 MiB | 1074790400 | 776.9 | 779.8 |
| 1 GiB + 2 MiB | 1075838976 | 776.9 | 779.9 |
| 1 GiB + 16 MiB | 1090519040 | 777.7 | 779.8 |
| 1 GiB + 32 MiB | 1107296256 | 776.8 | 779.8 |
| 256 MiB + 256 MiB | 536870912 | 770.7 | 775.6 |
| 256 MiB + 512 MiB | 805306368 | 776.1 | 778.8 |

P2P copies are NVLink-limited and do not show the local-device graph fast/slow cliff.

NCU graph profiling command template:
`/usr/local/cuda-12.8/bin/ncu --target-processes all --graph-profiling graph --launch-count 1 --csv --page raw --metrics <METRICS> <BENCHMARK> --bytes <BYTES> --p2p --graph`

For one graph launch, `nvlrx__bytes_data_user.sum` matched `bytes * copies_per_graph`, `nvltx__bytes_data_user.sum` was zero on the destination-device profile, and PCIe traffic was negligible relative to the NVLink payload. The `syslts__*_srcunit_ce_aperture_peer*` read/write counters were zero in these runs, and the generic CE sector counters were small control-path-sized counts rather than payload-sized counts. Unlike local graph memcpy, P2P graph rows did not expose kernel-like block/grid geometry in NCU.
