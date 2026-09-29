# P2P NVLink Send/Recv Benchmark

D110094643 validated the matched active-channel bidirectional CTA benchmark on
H100 using NCCLX 2.29 over NVLink. The benchmark uses `partition_interleaved(2)`
with one 512-thread CTA per channel, two 256-thread groups per CTA, 512KB
per-channel staging, pipeline depth 4, and `maxNumChannels` equal to the
NCCL-like active channel count. Bandwidth is aggregate bidirectional bandwidth
(`2 * message_size / time`), matching the benchmark output.

Command:

```bash
GTEST_FILTER=P2pSendRecvBenchmarkFixture.MatchedBidirCtaBenchmark NCCL_PROTO=Simple NCCL_MIN_P2P_NCHANNELS=32 NCCL_MAX_P2P_NCHANNELS=32 buck2 run @fbcode//mode/opt -c hpc_comms.use_ncclx=stable fbcode//comms/prims/benchmarks:p2p_nvl_send_recv_benchmark
```

Result on `devgpu005` (CUDA 12.4, NCCLX 2.29): passed.

```text
Test Name           Msg Size     Staging   PD   Chunk Blocks Threads    NCCL BW     P2P BW  Speedup   NCCL Lat    P2P Lat Lat Reduc
MatchedCh1_1B             1B       512KB    4   512KB      1     512       0.00       0.00    0.71x        6.3        8.9       -2.6
MatchedCh1_2B             2B       512KB    4   512KB      1     512       0.00       0.00    0.53x        6.4       12.2       -5.8
MatchedCh1_4B             4B       512KB    4   512KB      1     512       0.00       0.00    0.74x        6.3        8.6       -2.3
MatchedCh1_8B             8B       512KB    4   512KB      1     512       0.00       0.00    0.74x        6.3        8.6       -2.3
MatchedCh1_16B           16B       512KB    4   512KB      1     512       0.01       0.00    0.76x        6.3        8.3       -2.0
MatchedCh1_32B           32B       512KB    4   512KB      1     512       0.01       0.01    0.55x        6.5       11.8       -5.3
MatchedCh1_64B           64B       512KB    4   512KB      1     512       0.02       0.01    0.33x        6.3       19.0      -12.7
MatchedCh1_128B         128B       512KB    4   512KB      1     512       0.04       0.03    0.76x        6.3        8.3       -2.0
MatchedCh1_256B         256B       512KB    4   512KB      1     512       0.08       0.06    0.76x        6.3        8.3       -2.0
MatchedCh1_512B         512B       512KB    4   512KB      1     512       0.16       0.12    0.76x        6.3        8.3       -2.0
MatchedCh1_1KB           1KB       512KB    4   512KB      1     512       0.30       0.25    0.81x        6.7        8.3       -1.6
MatchedCh1_2KB           2KB       512KB    4   512KB      1     512       0.64       0.49    0.76x        6.4        8.3       -2.0
MatchedCh1_4KB           4KB       512KB    4   512KB      1     512       1.18       0.98    0.83x        6.9        8.4       -1.4
MatchedCh1_8KB           8KB       512KB    4   512KB      1     512       2.13       1.84    0.86x        7.7        8.9       -1.2
MatchedCh1_16KB         16KB       512KB    4   512KB      1     512       3.62       3.35    0.92x        9.1        9.8       -0.7
MatchedCh1_32KB         32KB       512KB    4   512KB      1     512       5.35       7.00    1.31x       12.2        9.4        2.9
MatchedCh1_64KB         64KB       512KB    4   512KB      1     512       9.53      11.95    1.25x       13.8       11.0        2.8
MatchedCh2_128KB       128KB         1MB    4   512KB      2     512      18.15      22.30    1.23x       14.4       11.8        2.7
MatchedCh4_256KB       256KB         2MB    4   512KB      4     512      35.50      43.65    1.23x       14.8       12.0        2.8
MatchedCh8_512KB       512KB         4MB    4   512KB      8     512      69.93      77.44    1.11x       15.0       13.5        1.5
MatchedCh16_1MB          1MB         8MB    4   512KB     16     512     127.98     153.21    1.20x       16.4       13.7        2.7
MatchedCh16_2MB          2MB         8MB    4   512KB     16     512     194.18     222.90    1.15x       21.6       18.8        2.8
MatchedCh16_4MB          4MB         8MB    4   512KB     16     512     263.83     300.31    1.14x       31.8       27.9        3.9
MatchedCh16_8MB          8MB         8MB    4   512KB     16     512     318.19     368.27    1.16x       52.7       45.6        7.2
MatchedCh16_16MB        16MB         8MB    4   512KB     16     512     410.61     454.52    1.11x       81.7       73.8        7.9
MatchedCh16_32MB        32MB         8MB    4   512KB     16     512     456.95     531.13    1.16x      146.9      126.4       20.5
MatchedCh16_64MB        64MB         8MB    4   512KB     16     512     523.90     597.76    1.14x      256.2      224.5       31.7
MatchedCh16_128MB      128MB         8MB    4   512KB     16     512     572.39     632.87    1.11x      469.0      424.2       44.8
MatchedCh16_256MB      256MB         8MB    4   512KB     16     512     605.29     660.70    1.09x      887.0      812.6       74.4
MatchedCh32_512MB      512MB        16MB    4   512KB     32     512     712.35     722.24    1.01x     1507.3     1486.7       20.6
MatchedCh32_1GB          1GB        16MB    4   512KB     32     512     723.16     730.17    1.01x     2969.6     2941.1       28.5
```
