# SendRecvTileVariants benchmark

Point-to-point tile send/recv: plain and ANS-compressed, one-way and two-way,
against an NCCL `ncclSend`/`ncclRecv` baseline.

## How to run

Build and run the compressed sweeps (both one-way and two-way):

```
buck2 build @fbcode//mode/opt -c hpc_comms.use_ncclx=stable \
    -m ovr_config//third-party/cuda/constraints:13.0 -c fbcode.enable_gpu_sections=true \
    -c fbcode.nvcc_arch=h100a \
    fbcode//comms/prims/collectives/benchmarks:sendrecv_tile_benchmark_binary

mpirun -np 2 -host <host>:2 --allow-run-as-root --map-by ppr:2:node \
    --gmca oob_tcp_if_include eth1 --gmca btl self,vader \
    -x NCCL_SOCKET_IFNAME=eth1 --prefix /usr/local/fbcode \
    -x NCCL_P2P_DISABLE=1 \
    ./sendrecv_tile_benchmark_binary --gtest_filter='*IbSweepCompressed*'
```

Available filters: `*NvlSweep*`, `*IbSweep*`, `*IbSweepTwoWay*`,
`*IbSweepCompressed*`, `*IbSweepCompressedTwoWay*`.

**Run the NVL and IB sweeps in SEPARATE invocations.** The NCCL communicator is
built once from process-global settings, so a single `NCCL_P2P_DISABLE` value
cannot be right for both: unset, rows labelled `NCCL-IB` may still ride
NVLink; set to 1, `NvlSweep`'s NCCL baseline is forced onto the network. The
benchmark prints the effective NCCL configuration at startup so a result is
attributable after the fact.

### Knobs

| env | meaning |
|---|---|
| `PIPES_SENDRECV_BENCH_BLOCKS` | grid blocks (default 256) |
| `PIPES_SENDRECV_BENCH_MIN_BLOCKS_PER_SM` | `__launch_bounds__` m; 2 or 8 |
| `PIPES_SENDRECV_BENCH_NUM_SMS` | green-context SM partition; 0 = full GPU |
| `PIPES_SENDRECV_BENCH_PLAIN_PCT` | % of blocks forced to plain `Memcpy` |
| `PIPES_SENDRECV_BENCH_DATA_BUF_BYTES` | IB staging buffer |
| `PIPES_SENDRECV_BENCH_MAX_SIGNAL_BYTES` | signal granularity; 0 = auto-safe |

## Results

Recorded 2026-09-11, `rtptest001.dkl2`, 2x H100 80GB HBM3, mlx5 RoCE,
CUDA 13, `NCCL_P2P_DISABLE=1`, 2 ranks (paired), 100 iterations, 256 blocks x
256 threads, `min_blocks_per_sm=2`, full GPU, `plain_block_pct=0`.

Bandwidth is GB/s. `wireRatio` is the WHOLE-TRANSFER ratio
(`logical / (plain + compressed payload)`), not the ANS-payload-only ratio;
`n/a (all plain)` means nothing reached the compressor, which below 4 MiB is
expected because `CopyOp::kActivationThreshold` ships smaller peers raw.

### One-way compressed, 0% sparsity

```
Size           NCCL-IB            Tile  Best        vs NCCL     wireRatio
--------    ----------  --------------  ---------- --------  ------------
8KB               0.58       0.2 (256)  nccl          0.40x  n/a (all plain)
16KB              0.93       0.5 (256)  nccl          0.49x  n/a (all plain)
32KB              1.78       0.9 (256)  nccl          0.52x  n/a (all plain)
64KB              3.28       1.8 (256)  nccl          0.54x  n/a (all plain)
128KB             5.82       3.6 (256)  nccl          0.63x  n/a (all plain)
256KB             8.21       7.3 (256)  nccl          0.89x  n/a (all plain)
512KB            13.10      14.2 (256)  tile          1.08x  n/a (all plain)
1MB              17.39      26.5 (256)  tile          1.53x  n/a (all plain)
2MB              20.70      40.5 (256)  tile          1.96x  n/a (all plain)
4MB              28.40      39.7 (256)  tile          1.40x         0.95x
8MB              30.91      42.7 (256)  tile          1.38x         0.96x
16MB             32.19      47.2 (256)  tile          1.47x         0.97x
32MB             32.28      47.6 (256)  tile          1.47x         0.98x
64MB             32.15      47.8 (256)  tile          1.49x         0.98x
```

### One-way compressed, 90% sparsity

```
Size           NCCL-IB            Tile  Best        vs NCCL     wireRatio
--------    ----------  --------------  ---------- --------  ------------
8KB               0.58       0.2 (256)  nccl          0.39x  n/a (all plain)
16KB              0.93       0.5 (256)  nccl          0.50x  n/a (all plain)
32KB              1.78       0.9 (256)  nccl          0.50x  n/a (all plain)
64KB              3.28       1.8 (256)  nccl          0.54x  n/a (all plain)
128KB             5.82       3.6 (256)  nccl          0.62x  n/a (all plain)
256KB             8.21       7.2 (256)  nccl          0.88x  n/a (all plain)
512KB            13.10      14.2 (256)  tile          1.09x  n/a (all plain)
1MB              17.39      27.5 (256)  tile          1.58x  n/a (all plain)
2MB              20.70      41.3 (256)  tile          2.00x  n/a (all plain)
4MB              28.40      43.7 (256)  tile          1.54x         7.14x
8MB              30.91      77.4 (256)  tile          2.51x         7.30x
16MB             32.19     105.9 (256)  tile          3.29x         7.38x
32MB             32.28     113.2 (256)  tile          3.51x         7.42x
64MB             32.15     121.7 (256)  tile          3.78x         7.44x
```

### Two-way compressed, 0% sparsity (per-direction BW)

```
Size           NCCL-IB            Tile  Best        vs NCCL     wireRatio
--------    ----------  --------------  ---------- --------  ------------
8KB               0.34       0.2 (256)  nccl          0.46x  n/a (all plain)
16KB              0.58       0.3 (256)  nccl          0.54x  n/a (all plain)
32KB              1.10       0.6 (256)  nccl          0.57x  n/a (all plain)
64KB              2.04       1.2 (256)  nccl          0.60x  n/a (all plain)
128KB             3.49       2.5 (256)  nccl          0.70x  n/a (all plain)
256KB             5.70       4.8 (256)  nccl          0.84x  n/a (all plain)
512KB             9.03       9.7 (256)  tile          1.07x  n/a (all plain)
1MB              12.51      17.7 (256)  tile          1.41x  n/a (all plain)
2MB              14.96      27.1 (256)  tile          1.81x  n/a (all plain)
4MB              18.44      15.0 (256)  nccl          0.82x         0.96x
8MB              19.14      17.9 (256)  nccl          0.94x         0.97x
16MB             19.51      20.2 (256)  tile          1.04x         0.98x
32MB             19.57      20.2 (256)  tile          1.03x         0.98x
64MB             19.72      27.9 (256)  tile          1.42x         0.98x
```

## Reading these numbers

- The NCCL baseline is **deliberately constrained** -- one net channel per peer
  and two QPs, to sit near the Tile transport's shape. It is not a tuned NCCL
  result, and `vs NCCL` should be read as "against NCCL held at this shape".
- The payload is a hash, i.e. close to incompressible, so `wireRatio` below 1
  at 0% sparsity is expected: ANS adds size-header and alignment overhead it
  cannot win back. The ratio climbs with sparsity, which is what the sweep is
  for.
- Every configuration runs one **untimed correctness iteration** before the
  timed loop -- rank-distinct hashed payload, destination poisoned with 0xA5,
  received bytes compared. A wrong peer, wrong offset or truncated transfer
  fails the run instead of showing up as a speed-up.
