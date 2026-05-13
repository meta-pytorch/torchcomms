# Ctran Hierarchical AllGatherP Optimization Log

## Phase 0: 76.7 GB/s Ceiling Root Cause

### Context

Baseline diff: D104948958 (`cthierarchical_pipes`)

Baseline run:

```text
Hosts: rtptest2333, rtptest2335, rtptest2337, rtptest2339
PPN: 1
CUDA_VISIBLE_DEVICES: 1
Warmup/iters: 5/50
Hier env: NCCL_CTRAN_USE_PIPES=1;NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432;NCCL_MNNVL_ENABLE=0
NCCL env: NCCL_MNNVL_ENABLE=0;NCCL_NVLS_ENABLE=0;NCCL_P2P_DISABLE=1;NCCL_ALGO=Ring
```

### Findings

The 64MB+ plateau is not from a fully single-NIC path. The baseline log shows the Pipes NIC discovery seeing four filtered HCAs and ranking `mlx5_3` and `mlx5_4` as the equal best-affinity NODE/400Gbps NICs for the selected GPU. `MultipeerIbgdaTransport` uses all NICs returned by `getBestAffinityNics()`, so this run should instantiate two NIC rails.

The remaining ceiling is still a resource-parallelism limit:

```text
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS = 16
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH = 2
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC = 1
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES = 1048576
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE = 33554432
active NICs per peer = 2
active QP slots per peer = 2 NICs * 1 QP/NIC = 2
CUDA block groups = 16
groups per QP slot = 8
```

`P2pIbgdaTransportDevice::nic_qp_for_group()` maps `group_id % numNics` to NIC and `(group_id / numNics) % qps_per_nic` to QP. With one QP per NIC, every peer has only two QP slots, so the 16 block leaders serialize WQE posting over two QPs. This is below NCCL's effective channel parallelism and matches the observed plateau: bus bandwidth reaches only ~57.5 GB/s while NCCL Ring reaches ~90 GB/s bus bandwidth.

Pipeline/window math for the baseline:

```text
perBlockSlot = dataBufferSize / active_blocks = 32MB / 16 = 2MB
pipeline_window = perBlockSlot * pipelineDepth = 4MB
signal chunk = min(1MB, perBlockSlot) = 1MB
```

For 64MB+ per-rank sends, each block loops over multiple 4MB windows and can keep the pipeline full, producing the stable ~76.7 GB/s algorithmic plateau. The 32MB anomaly lands at exactly a 2MB tile per block: it creates two 1MB signaled chunks but does not use the second staging slot as a full pipeline window, so it pays extra signaling/step overhead without enough in-flight data. 64MB is the first size where each block reaches a full 4MB pipeline window.

Small-message latency is dominated by ctran/Pipes setup and GPE/device launch path, not network bandwidth. The `cthierarchical_pipes` path submits an empty GPE op group plus one CUDA kernel per operation. NCCL's small-message ring is far below that launch/control overhead.

### Next

Experiment 1 should increase QPs per peer per NIC from 1 to 4 without changing block count, pipeline depth, or buffer size. This creates 8 active NIC/QP slots per peer, matching NCCL's reported 8-channel parallelism while keeping the kernel and staging math unchanged.

## Experiment 1: Hierarchical AllGather QPs Per Peer Per NIC = 4

### Hypothesis

The large-message plateau is caused by too few IBGDA QP slots per peer. Raising `NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC` from 1 to 4 should increase active slots from 2 to 8 on the two best-affinity NICs and reduce WQE posting contention across the 16 block groups. Predicted gain: 64MB+ should move from ~76.7 GB/s toward the 110-120 GB/s NCCL range if QP serialization is the dominant limiter.

### Change

CVAR-only experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4

Held fixed:
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=16
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=2
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

```text
buck2 build @fbcode/mode/opt -m cuda128 -c fbcode.arch=aarch64 -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 -c fbcode.platform010-aarch64_clang=17 fbcode//comms/ctran/benchmarks:allgatherp_bench_4x1_binary
Result: BUILD SUCCEEDED
```

After promoting the default in `utils/cvars/nccl_cvars.yaml`, rebuilt the same target:

```text
Result: BUILD SUCCEEDED
```

### Run

Detached overnight process:

```text
pid=3051154
log=/tmp/hier_ag_opt/exp_1_qps4.log
command=/tmp/hier_ag_opt/exp_1_qps4.command.txt
status=/tmp/hier_ag_opt/exp_1_qps4.status
```

Recovery:

```bash
ps -p $(cat /tmp/hier_ag_opt/exp_1_qps4.pid) -o pid,ppid,sid,stat,etime,cmd
tail -f /tmp/hier_ag_opt/exp_1_qps4.log
```

The process was launched with `setsid`, stdin detached, and stdout/stderr redirected. It reparented to PID 1, so it should survive loss of SSH to the devgpu running Codex.

Default verification run after source change:

```text
pid=3750466
log=/tmp/hier_ag_opt/exp_1_verify_default_qps4.log
command=/tmp/hier_ag_opt/exp_1_verify_default_qps4.command.txt
```

### Results

```text
Size      Baseline GB/s  Exp1 GB/s  NCCL GB/s  Verdict
8KB       0.056          0.045      0.679      worse
16KB      0.113          0.111      1.131      worse
32KB      0.200          0.224      1.235      worse
64KB      0.435          0.447      2.222      worse
128KB     0.759          0.895      3.483      worse
256KB     1.478          1.485      10.669     worse
512KB     3.041          3.580      21.375     worse
1MB       3.752          6.907      41.482     worse
2MB       13.529         13.583     28.200     worse
4MB       25.466         27.566     53.342     worse
8MB       32.717         45.778     87.989     worse
16MB      33.236         89.835     103.343    worse
32MB      26.590         110.344    106.526    beats NCCL
64MB      76.268         116.417    108.487    beats NCCL
128MB     76.649         121.900    110.383    beats NCCL
256MB     76.748         124.318    114.268    beats NCCL
512MB     76.706         125.665    117.311    beats NCCL
1GB       76.665         126.326    119.969    beats NCCL
```

Raw log: `/tmp/hier_ag_opt/exp_1_qps4.log`

Default verification, no QP override:

```text
Size      Default GB/s  NCCL GB/s  Default/NCCL
8KB       0.055         0.679      0.08x
16KB      0.111         1.131      0.10x
32KB      0.210         1.235      0.17x
64KB      0.441         2.222      0.20x
128KB     0.897         3.483      0.26x
256KB     1.757         10.669     0.16x
512KB     3.511         21.375     0.16x
1MB       5.999         41.482     0.14x
2MB       14.119        28.200     0.50x
4MB       27.946        53.342     0.52x
8MB       44.732        87.989     0.51x
16MB      89.307        103.343    0.86x
32MB      107.320       106.526    1.01x
64MB      116.284       108.487    1.07x
128MB     121.666       110.383    1.10x
256MB     124.360       114.268    1.09x
512MB     125.699       117.311    1.07x
1GB       126.333       119.969    1.05x
```

Raw default-verification log: `/tmp/hier_ag_opt/exp_1_verify_default_qps4.log`

### Verdict

✅ Keep. This breaks the 76.7 GB/s ceiling and satisfies the termination targets at 64MB, 128MB, 256MB, 512MB, and 1GB. It also fixes the 32MB anomaly and beats the NCCL baseline from 32MB upward. The no-override verification confirms the source default change takes effect. It does not address small-message latency; 8KB-16MB remain below NCCL.

### Insight

The plateau was QP-slot contention, not missing NIC discovery. The baseline had two NICs but only one QP per NIC, so 16 block groups contended over two QP slots per peer. Four QPs per NIC create eight QP slots per peer and bring bus bandwidth from ~57.5 GB/s to ~94.7 GB/s, above the NCCL Ring IB-only baseline.

### Next

Run formatter/lint on `utils/cvars/nccl_cvars.yaml` and `ctran/OPTIMIZATION_LOG.md`, submit/update the draft diff, then notify the win. Continue with a separate small-message latency experiment only after the large-message winning default is captured.

### Diff

`utils/cvars/nccl_cvars.yaml`: default `NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC` changed from 1 to 4.

## Experiment 2: Hierarchial all gather 1 CUDA Block For Small Messages

### Hypothesis

For `<32MB`, the QPs=4 path is dominated by a ~0.6ms startup/launch/protocol floor rather than NIC bandwidth. The current default launches 16 CUDA block groups even when the message is only 8KB. Reducing `NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS` to 1 should reduce block scheduling, per-block transport state, and per-peer WQE/signaling fanout. Predicted gain: smaller messages should move below the ~0.6ms floor; large-message throughput is expected to regress and is not the goal of this experiment.

### Change

CVAR-only experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=1

Held fixed:
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=2
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=145146
log=/tmp/hier_ag_opt/exp_2_blocks1_small.log
command=/tmp/hier_ag_opt/exp_2_blocks1_small.command.txt
status=/tmp/hier_ag_opt/exp_2_blocks1_small.status
```

### Results

```text
Size      Exp1 default GB/s  Exp2 GB/s  NCCL GB/s  Verdict
8KB       0.055              0.057      0.679      flat
16KB      0.111              0.111      1.131      flat
32KB      0.210              0.224      1.235      slight win
64KB      0.441              0.454      2.222      slight win
128KB     0.897              0.886      3.483      flat
256KB     1.757              1.720      10.669     worse
512KB     3.511              3.443      21.375     worse
1MB       5.999              6.155      41.482     flat
2MB       14.119             12.137     28.200     worse
4MB       27.946             18.321     53.342     worse
8MB       44.732             18.699     87.989     worse
16MB      89.307             18.861     103.343    worse
32MB      107.320            18.992     106.526    worse
```

Raw log: `/tmp/hier_ag_opt/exp_2_blocks1_small.log`

### Verdict

❌ Reject. One block does not reduce the ~0.6ms startup floor and destroys mid-size throughput. Do not keep.

### Insight

The small-message floor is not caused primarily by the number of CUDA block groups. Even with one group, 8KB-1MB stays at ~0.58-0.68ms. The floor is more likely GPE/launcher setup plus the IBGDA send/forward/recv protocol sequence. Block count is still important for 2MB+ pipeline efficiency.

### Next

Benchmark existing ctran AllGatherP algorithms (`ctdirect`, `ctpipeline`, `ctrdpipeline`) over `<32MB` with the same 4x1 methodology. If any existing ctran path is faster for small sizes, implement a size-thresholded dispatch before attempting a new eager/persistent hierarchical kernel.

### Diff

None.

## Experiment 9: Direct QP Threshold = 0 For 8MB-16MB

### Hypothesis

The existing direct path is closest to NCCL at 8MB. Forcing equal splitting across direct IB QPs may improve link utilization versus threshold-based WQE slicing.

### Change

Benchmark-only direct-path QP config. No source code change.

```text
Algorithm: ctdirect
Added env:
NCCL_CTRAN_IB_QP_CONFIG_XRACK=0,16,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XZONE=0,16,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XDC=0,16,spray,128
```

### Files

None.

### Results

```text
Size      Direct default GB/s  Exp9 GB/s  NCCL GB/s  Verdict
8MB       82.616               80.156     87.428     worse
16MB      84.202               82.631     106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_9_direct_qp_threshold0_8mb_16mb.log`

### Verdict

❌ Reject. Equal QP splitting regresses direct.

### Insight

Direct's 8MB gap is not from insufficient QP striping under the default threshold config.

### Next

Try larger direct QP chunks to reduce WQE fragmentation.

### Diff

None.

## Experiment 10: Direct QP Threshold = 2MB For 8MB-16MB

### Hypothesis

Larger direct chunks may reduce WQE overhead enough to close the 8MB direct gap while retaining enough QP parallelism.

### Change

Benchmark-only direct-path QP config. No source code change.

```text
Algorithm: ctdirect
Added env:
NCCL_CTRAN_IB_QP_CONFIG_XRACK=2097152,16,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XZONE=2097152,16,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XDC=2097152,16,spray,128
```

### Files

None.

### Results

```text
Size      Direct default GB/s  Exp10 GB/s  NCCL GB/s  Verdict
8MB       82.616               83.161      87.428     flat
16MB      84.202               84.334      106.633    flat
```

Raw log: `/tmp/hier_ag_opt/exp_10_direct_qp_threshold2mb_8mb_16mb.log`

### Verdict

❌ Reject. The small 8MB movement is not enough and 16MB remains far below NCCL.

### Insight

Direct chunk size is not the dominant remaining limiter.

### Next

Try 4MB direct chunks to complete the chunk-size sweep.

### Diff

None.

## Experiment 11: Direct QP Threshold = 4MB For 8MB-16MB

### Hypothesis

Further reducing direct WQE count may improve 16MB and possibly 8MB.

### Change

Benchmark-only direct-path QP config. No source code change.

```text
Algorithm: ctdirect
Added env:
NCCL_CTRAN_IB_QP_CONFIG_XRACK=4194304,16,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XZONE=4194304,16,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XDC=4194304,16,spray,128
```

### Files

None.

### Results

```text
Size      Direct default GB/s  Exp11 GB/s  NCCL GB/s  Verdict
8MB       82.616               80.581      87.428     worse
16MB      84.202               85.378      106.633    slight win, still below
```

Raw log: `/tmp/hier_ag_opt/exp_11_direct_qp_threshold4mb_8mb_16mb.log`

### Verdict

❌ Reject. 4MB chunks hurt 8MB and do not make 16MB competitive.

### Insight

The direct path's 8MB/16MB gap is not solved by coarser QP slicing.

### Next

Try increasing direct max QPs while restoring 1MB chunks.

### Diff

None.

## Experiment 12: Direct Max QPs = 32 For 8MB-16MB

### Hypothesis

Direct may need more than 16 QPs to match NCCL's 8-channel ring at mid sizes.

### Change

Benchmark-only direct-path QP config. No source code change.

```text
Algorithm: ctdirect
Added env:
NCCL_CTRAN_IB_QP_CONFIG_XRACK=1048576,32,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XZONE=1048576,32,spray,128
NCCL_CTRAN_IB_QP_CONFIG_XDC=1048576,32,spray,128
```

### Files

None.

### Results

```text
Size      Direct default GB/s  Exp12 GB/s  NCCL GB/s  Verdict
8MB       82.616               81.769      87.428     worse
16MB      84.202               84.781      106.633    flat
```

Raw log: `/tmp/hier_ag_opt/exp_12_direct_maxqps32_8mb_16mb.log`

### Verdict

❌ Reject. More direct QPs do not close the 8MB/16MB gap.

### Insight

Direct is not QP-count limited at these sizes.

### Next

Temporarily remove direct's per-call IB ctrl sync to test whether the residual gap is startup/control latency.

### Diff

None.

## Experiment 13: Direct Skip IB Ctrl Sync For 8MB-16MB

### Hypothesis

Direct spends a host control-sync round before issuing IB puts. If IB notify accounting is already sufficient, skipping that sync may reduce direct latency enough to beat NCCL at 8MB.

### Change

Temporary source experiment in `ctran/algos/AllGatherP/DirectImpl.cc`: skipped the IB ctrl send/recv loop inside `gpnFn`.

### Files

```text
ctran/algos/AllGatherP/DirectImpl.cc
```

### Build

```text
buck2 build @fbcode/mode/opt -m cuda128 -c fbcode.arch=aarch64 -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 -c fbcode.platform010-aarch64_clang=17 fbcode//comms/ctran/benchmarks:allgatherp_bench_4x1_binary
Result: BUILD SUCCEEDED
```

### Results

```text
Size      Direct default GB/s  Exp13 GB/s  NCCL GB/s  Verdict
8MB       82.616               83.233      87.428     flat
16MB      84.202               85.430      106.633    flat
```

Raw log: `/tmp/hier_ag_opt/exp_13_direct_skip_ctrl_8mb_16mb.log`

### Verdict

❌ Reject and revert. The gain is too small for a correctness-sensitive change.

### Insight

Direct's remaining 8MB gap is mostly data path, not the explicit ctrl sync.

### Next

Keep direct only as a fallback where it is already a clear win; implement fallback inside `cthierarchical_pipes` for small messages.

### Diff

Reverted.

## Experiment 14: Hierarchial all gather Direct Fallback <= 4MB

### Hypothesis

Existing ctran direct AllGatherP beats hierarchical and generally matches/beats NCCL for small messages, while QPs=4 hierarchical beats NCCL from 32MB upward. Dispatching `cthierarchical_pipes` to direct for `sendBytes <= 4MB` should remove the small-message 0.58ms floor without affecting large-message throughput.

### Change

Source change in `ctran/algos/AllGatherP/HierarchicalPipesImpl.cc`: after `waitInit()`, route `sendBytes <= 4 * 1024 * 1024` to `execDirect(...)`; otherwise run the existing hierarchical Pipes kernel.

### Files

```text
ctran/algos/AllGatherP/HierarchicalPipesImpl.cc
```

### Build

```text
buck2 build @fbcode/mode/opt -m cuda128 -c fbcode.arch=aarch64 -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 -c fbcode.platform010-aarch64_clang=17 fbcode//comms/ctran/benchmarks:allgatherp_bench_4x1_binary
Result: BUILD SUCCEEDED
```

### Results

```text
Size      Exp14 GB/s  NCCL GB/s  Verdict
8KB       0.796       0.679      beats
16KB      1.613       1.131      beats
32KB      3.200       1.235      beats
64KB      6.101       2.222      beats
128KB     10.820      3.483      beats
256KB     18.870      10.669     beats
512KB     28.326      21.375     beats
1MB       39.379      41.482     slightly below baseline NCCL
2MB       46.830      28.200     beats
4MB       65.383      53.342     beats
8MB       44.798      87.989     below
16MB      91.164      103.343    below
32MB      109.655     106.526    beats
64MB      116.293     108.487    beats
128MB     119.848     110.383    beats
256MB     124.377     114.268    beats
512MB     125.693     117.311    beats
1GB       126.322     119.969    beats
```

Raw log: `/tmp/hier_ag_opt/exp_14_hier_direct_fallback_4mb_full.log`

### Verdict

✅ Keep as a partial win. This removes the small-message hierarchical latency floor and preserves the large-message QPs=4 win. It does not solve 1MB robustly in this run, and 8MB/16MB remain below NCCL.

### Insight

The best current composed path is direct for most small sizes and hierarchical ring for 32MB+. The unsolved band is now concentrated at 1MB, 8MB, and 16MB, with 8MB/16MB requiring a deeper data-path change.

### Next

Rerun direct 1MB-4MB to separate noise from fallback overhead, then continue 8MB/16MB experiments.

### Diff

Kept.

## Experiment 15: Direct 1MB-4MB Rerun

### Hypothesis

The 1MB miss in Experiment 14 may be run-to-run variance or fallback overhead. A focused direct rerun should establish the current direct path's small-band stability.

### Change

Benchmark-only direct rerun. No source code change.

### Files

None.

### Results

```text
Size      Exp15 Direct GB/s  NCCL GB/s  Verdict
1MB       36.296             41.482     below
2MB       53.112             28.200     beats
4MB       60.887             53.342     beats
```

Raw log: `/tmp/hier_ag_opt/exp_15_direct_1mb_4mb_rerun.log`

### Verdict

⚠️ Diagnostic only. Direct is noisy and not a robust 1MB solution in this run, but still clearly improves over hierarchical baseline.

### Insight

The small-message direct path needs its own latency optimization if strict 1MB parity is required.

### Next

Return to 8MB/16MB, where the gap is much larger.

### Diff

None.

## Experiment 16: Hierarchial all gather 32 Blocks + 8 QPs For 8MB-16MB

### Hypothesis

Experiment 4 showed 32 blocks helped 8MB but hurt 16MB. Pairing 32 blocks with 8 QPs may preserve the 8MB block parallelism while reducing QP-slot contention.

### Change

Benchmark-only interaction experiment.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=32
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=8
```

### Files

None.

### Results

```text
Size      Exp1 default GB/s  Exp16 GB/s  NCCL GB/s  Verdict
8MB       46.177             52.281      87.428     better but still below
16MB      90.340             81.996      106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_16_hier_blocks32_qps8_8mb_16mb.log`

### Verdict

❌ Reject. The interaction improves 8MB but remains far below NCCL and regresses 16MB.

### Insight

The 8MB path benefits modestly from more block-level parallelism, but this approach cannot close the gap.

### Next

Check direct `dqplb` routing because direct remains closer at 8MB than hierarchical.

### Diff

None.

## Experiment 17: Direct DQPLB For 8MB-16MB

### Hypothesis

Changing direct from spray to DQPLB routing may improve adaptive QP selection and close the direct 8MB gap.

### Change

Benchmark-only direct-path QP config.

```text
Algorithm: ctdirect
Added env:
NCCL_CTRAN_IB_QP_CONFIG_XRACK=1048576,16,dqplb,128
NCCL_CTRAN_IB_QP_CONFIG_XZONE=1048576,16,dqplb,128
NCCL_CTRAN_IB_QP_CONFIG_XDC=1048576,16,dqplb,128
```

### Files

None.

### Results

```text
Size      Direct default GB/s  Exp17 GB/s  NCCL GB/s  Verdict
8MB       82.616               80.867      87.428     worse
16MB      84.202               84.373      106.633    flat
```

Raw log: `/tmp/hier_ag_opt/exp_17_direct_dqplb_8mb_16mb.log`

### Verdict

❌ Reject. DQPLB does not improve direct for this benchmark.

### Insight

Direct's 8MB gap is not a simple VC routing-mode issue.

### Next

Try a device-side direct IB path inside the hierarchical kernel to avoid ring forwarding latency without falling back to host direct.

### Diff

None.

## Experiment 18: Hierarchial all gather Device-Direct IB For 8MB-16MB

### Hypothesis

For `ppn=1`, hierarchical degenerates to a 4-node IB ring. A device-side direct IB allgather using all peer IBGDA transports may reduce ring forwarding latency at 8MB/16MB while staying in the Pipes kernel path.

### Change

Temporary source experiment:

```text
comms/pipes/collectives/DirectNvlTypes.h: added direct IB peer handles to hierarchical args
ctran/algos/AllGatherP/HierarchicalPipesImpl.cc: filled peer IBGDA handles
ctran/algos/AllGatherP/HierarchicalPipes.cu: added guarded nvl_size==1 direct IB path for sendcount <=16MB
```

### Files

```text
pipes/collectives/DirectNvlTypes.h
ctran/algos/AllGatherP/HierarchicalPipesImpl.cc
ctran/algos/AllGatherP/HierarchicalPipes.cu
```

### Build

```text
buck2 build @fbcode/mode/opt -m cuda128 -c fbcode.arch=aarch64 -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 -c fbcode.platform010-aarch64_clang=17 fbcode//comms/ctran/benchmarks:allgatherp_bench_4x1_binary
Result: BUILD SUCCEEDED
```

### Results

```text
Size      Exp1 default GB/s  Exp18 GB/s  NCCL GB/s  Verdict
8MB       46.177             47.725      87.428     flat
16MB      90.340             82.783      106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_18_hier_device_direct_ib_8mb_16mb.log`

### Verdict

❌ Reject and revert. Device-side direct IB is not better than the ring for 16MB and does not help 8MB enough.

### Insight

Naive all-peer device sends increase pressure/ordering overhead enough to lose to the staged ring. The remaining 8MB/16MB gap likely needs a more NCCL-like multi-channel ring or a specialized medium-message protocol rather than a direct all-to-all substitution.

### Next

Rebuild after reverting Exp18 and continue only if a new medium-message strategy is identified. Failure count is approaching the protocol stop threshold.

### Diff

Reverted.

## Experiment 8: Hierarchial all gather 64 Blocks For 8MB-16MB

### Hypothesis

Experiment 4 showed that 32 blocks improved 8MB but hurt 16MB. Increasing to 64 blocks may continue improving 8MB by adding more block-level channel parallelism, while exposing whether the 16MB regression is from too many small tiles.

### Change

Benchmark-only CVAR experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=64

Held fixed:
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=2
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=2645870
log=/tmp/hier_ag_opt/exp_8_blocks64_8mb_16mb.log
command=/tmp/hier_ag_opt/exp_8_blocks64_8mb_16mb.command.txt
status=/tmp/hier_ag_opt/exp_8_blocks64_8mb_16mb.status
```

### Results

```text
Size      Exp1 default GB/s  Exp8 GB/s  NCCL GB/s  Verdict
8MB       46.177             47.655     87.428     slight win, still below 32-block result
16MB      90.340             80.527     106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_8_blocks64_8mb_16mb.log`

### Verdict

❌ Reject. 64 blocks does not continue the 32-block 8MB improvement and regresses 16MB.

### Insight

The 8MB/16MB gap is not solved by simply adding more device-side block channels. At high block counts, smaller per-block tiles and more WQEs outweigh extra parallelism.

### Next

Pivot to a dispatch strategy for small/mid sizes. Existing ctran `ctdirect` already beats NCCL through 4MB and is the closest path at 8MB, so test direct path IB QP slicing/VC settings for 8MB and 16MB.

### Diff

None.

## Experiment 7: Hierarchial all gather Pipeline Depth = 4 For 8MB-16MB

### Hypothesis

The 8MB/16MB band may be underfilled because each block has only two IB staging slots. Increasing pipeline depth from 2 to 4 may allow more in-flight device-side send/forward work without changing block count or QP count.

### Change

Benchmark-only CVAR experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=4

Held fixed:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=16
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=2222722
log=/tmp/hier_ag_opt/exp_7_pipeline_depth4_8mb_16mb.log
command=/tmp/hier_ag_opt/exp_7_pipeline_depth4_8mb_16mb.command.txt
status=/tmp/hier_ag_opt/exp_7_pipeline_depth4_8mb_16mb.status
```

### Results

```text
Size      Exp1 default GB/s  Exp7 GB/s  NCCL GB/s  Verdict
8MB       46.177             45.085     87.428     worse
16MB      90.340             91.495     106.633    slight win, still below NCCL
```

Raw log: `/tmp/hier_ag_opt/exp_7_pipeline_depth4_8mb_16mb.log`

### Verdict

❌ Reject as a default. It slightly improves 16MB but regresses 8MB and does not close the NCCL gap.

### Insight

For 8MB/16MB, each 16-block tile is already no larger than the 1MB signaling chunk and below the default pipeline window, so increasing pipeline depth mostly changes staging allocation rather than exposing much more overlap.

### Next

Try `NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=64` for 8MB/16MB to see whether the 32-block 8MB improvement continues with more block-level parallelism.

### Diff

None.

## Experiment 6: Hierarchial all gather 8 Blocks For 8MB-16MB

### Hypothesis

The 1-block experiment removed too much parallelism and 32 blocks shrank the per-block window too far. An 8-block middle point might preserve enough in-flight IB work while reducing block scheduling/window overhead for 8MB and 16MB.

### Change

Benchmark-only CVAR experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=8

Held fixed:
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=2
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=1710139
log=/tmp/hier_ag_opt/exp_6_blocks8_8mb_16mb.log
command=/tmp/hier_ag_opt/exp_6_blocks8_8mb_16mb.command.txt
status=/tmp/hier_ag_opt/exp_6_blocks8_8mb_16mb.status
```

### Results

```text
Size      Exp1 default GB/s  Exp6 GB/s  NCCL GB/s  Verdict
8MB       46.177             44.762     87.428     worse
16MB      90.340             54.920     106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_6_blocks8_8mb_16mb.log`

### Verdict

❌ Reject. 8 blocks is worse at both target sizes.

### Insight

The mid-size band is not improved by lowering block count. Together with Experiments 2 and 4, the 16-block default remains the best tested block geometry for 8MB/16MB.

### Next

Try `NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=4` at default 16 blocks and QPs=4. The remaining hypothesis is insufficient in-flight staging per block, not block count or QP slot count.

### Diff

None.

## Experiment 5: Hierarchial all gather QPs Per Peer Per NIC = 8 For 8MB-16MB

### Hypothesis

The 8MB/16MB gap may still include QP contention: QPs=4 gives 8 NIC/QP slots across two NICs, while the kernel has 16 block groups. Raising QPs per peer per NIC to 8 gives 16 slots, so each block can map to a distinct NIC/QP slot. Predicted gain: improve 8MB and 16MB without changing the block/window geometry.

### Change

CVAR-only experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=8

Held fixed:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=16
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=2
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=1214088
log=/tmp/hier_ag_opt/exp_5_qps8_8mb_16mb.log
command=/tmp/hier_ag_opt/exp_5_qps8_8mb_16mb.command.txt
status=/tmp/hier_ag_opt/exp_5_qps8_8mb_16mb.status
```

### Results

```text
Size      Exp1 default GB/s  Exp5 GB/s  NCCL GB/s  Verdict
8MB       46.177             47.160     87.428     flat
16MB      90.340             85.336     106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_5_qps8_8mb_16mb.log`

### Verdict

❌ Reject. QPs=8 does not close the 8MB/16MB gap and regresses 16MB.

### Insight

Residual 8MB/16MB performance is not from QP-slot contention. With 16 blocks and 2 NICs, QPs=4 already provides enough QP parallelism for the mid-size regime.

### Next

Inspect and implement a thresholded dispatch using existing lower-overhead ctran algorithms for 8KB-4MB, then continue targeted 8MB/16MB work. For 8MB, direct is close to NCCL; investigate whether direct can be tuned or combined. For 16MB, hierarchical QPs=4 remains closest.

### Diff

None.

## Experiment 4: Hierarchial all gather 32 Blocks For 8MB-16MB

### Hypothesis

The remaining gap at 8MB and 16MB may come from insufficient hierarchical Pipes parallelism in the mid-size regime. Raising `NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS` from 16 to 32 should increase GPU copy/posting parallelism and make each block's tile smaller. Predicted gain: improve 8MB and 16MB, with possible risk from smaller per-block staging windows.

### Change

CVAR-only experiment. No source code change.

```text
Added env:
NCCL_CTRAN_HIER_AG_IB_NUM_BLOCKS=32

Held fixed:
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_PIPELINE_DEPTH=2
NCCL_CTRAN_HIER_AG_IB_SIGNAL_BYTES=1048576
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=901085
log=/tmp/hier_ag_opt/exp_4_blocks32_8mb_16mb.log
command=/tmp/hier_ag_opt/exp_4_blocks32_8mb_16mb.command.txt
status=/tmp/hier_ag_opt/exp_4_blocks32_8mb_16mb.status
```

### Results

```text
Size      Exp1 default GB/s  Exp4 GB/s  NCCL GB/s  Verdict
8MB       46.177             51.216     87.428     better but still below NCCL
16MB      90.340             73.658     106.633    worse
```

Raw log: `/tmp/hier_ag_opt/exp_4_blocks32_8mb_16mb.log`

### Verdict

❌ Reject. 32 blocks gives a small 8MB improvement but makes 16MB much worse and does not approach NCCL.

### Insight

More blocks alone is not the mid-size solution. At 32 blocks, per-block staging windows shrink and the 16MB path loses the overlap that made the 16-block QPs=4 default competitive.

### Next

Try `NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=8` at the default 16 blocks for 8MB/16MB. This keeps the 16-block tile/window math but removes residual QP contention by giving each block its own NIC/QP slot across two NICs.

### Diff

None.

## Experiment 3: Existing ctran Small-Message Algorithm Sweep

### Hypothesis

The hierarchical Pipes path has a fixed ~0.6ms floor for `<32MB`. Existing ctran AllGatherP algorithms may have lower startup overhead for small sizes because they use different GPE/IB protocols and avoid the Pipes staged ring. If an existing path beats hierarchical for small messages, a size-thresholded dispatch can improve `<32MB` without changing the large-message QPs=4 win.

### Change

Benchmark-only diagnostic. No source code change.

```text
Benchmark arg:
--algo=all
--min_bytes=8192
--max_bytes=33554432

Held fixed:
NCCL_CTRAN_USE_PIPES=1
NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=33554432
NCCL_CTRAN_HIER_AG_IB_QPS_PER_PEER_PER_NIC=4 (source default)
NCCL_MNNVL_ENABLE=0
CUDA_VISIBLE_DEVICES=1
warmup_iters=5
bench_iters=50
hosts=rtptest2333,rtptest2335,rtptest2337,rtptest2339
```

### Files

None.

### Build

Reuses the successful QPs=4 default build from Experiment 1.

### Run

Detached process:

```text
pid=573089
log=/tmp/hier_ag_opt/exp_3_existing_algos_small.log
command=/tmp/hier_ag_opt/exp_3_existing_algos_small.command.txt
status=/tmp/hier_ag_opt/exp_3_existing_algos_small.status
```

### Results

```text
Size      Direct GB/s  Pipeline GB/s  RecDbl GB/s  Hier GB/s  NCCL GB/s  Best ctran
8KB       0.771        0.496          0.528        0.056      0.673      Direct beats NCCL
16KB      1.577        0.980          1.045        0.111      1.127      Direct beats NCCL
32KB      3.295        1.932          2.061        0.222      1.237      Direct beats NCCL
64KB      6.302        3.733          4.021        0.444      2.250      Direct beats NCCL
128KB     11.307       6.587          7.314        0.881      3.475      Direct beats NCCL
256KB     19.389       11.889         13.317       1.739      10.519     Direct beats NCCL
512KB     32.714       19.628         21.377       3.522      20.938     Direct beats NCCL
1MB       41.293       29.795         31.206       6.991      40.280     Direct beats NCCL
2MB       38.031       38.251         43.453       14.043     28.087     RecDbl beats NCCL
4MB       76.523       47.719         59.897       27.819     53.081     Direct beats NCCL
8MB       82.616       63.260         70.136       46.177     87.428     Direct closest, below NCCL
16MB      84.202       78.788         73.899       90.340     106.633    Hier closest, below NCCL
32MB      81.437       75.224         72.979       110.104    106.325    Hier beats NCCL
```

Raw log: `/tmp/hier_ag_opt/exp_3_existing_algos_small.log`

### Verdict

⚠️ Use as a dispatch insight, not a standalone change. `AllGatherP_Direct` is much better for 8KB-1MB and beats NCCL through 1MB. `RecDblPipeline` is best at 2MB. `Direct` beats NCCL again at 4MB. No existing ctran path beats NCCL at 8MB or 16MB.

### Insight

The small-message gap is specific to the hierarchical Pipes path, not an inherent ctran/NIC limitation. Existing direct/recursive-doubling implementations already have much lower launch/protocol overhead and can beat NCCL for 8KB-4MB. The remaining unsolved band is 8MB-16MB, where the direct path is close at 8MB and hierarchical is closest at 16MB.

### Next

Try a hierarchical tuning experiment focused only on 8MB and 16MB before implementing dispatch: increase hierarchical block count to 32 while keeping QPs=4. If that does not close the 8MB/16MB gap, implement size-thresholded dispatch to direct/rec-dbl for 8KB-4MB and continue a separate 8MB/16MB tuning pass.

### Diff

None.
