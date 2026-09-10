# Collective Performance Benchmarks

A unified benchmarking suite for PyTorch distributed collectives. It runs
the same collective across three comm adapters, producing CSV results
alongside side-by-side comparison reports and plots.

Comm adapters:

- `torchcomms`
- `c10d`
- `c10d_torchcomms` (c10d routed through torchcomms)

## Contents

- [Prerequisites](#prerequisites)
- [Supported Collectives](#supported-collectives)
- [Running a Benchmark](#running-a-benchmark)
- [Command-Line Options](#command-line-options)
- [CSV Output](#csv-output)
- [Comparing Comm Adapters](#comparing-comm-adapters)
- [Analyzing Results](#analyzing-results)
- [Adding a New Collective](#adding-a-new-collective)

## Prerequisites

- A working torchcomms install (see the repo root README / CLAUDE.md for build instructions).
- PyTorch build with the desired backend (NCCL, RCCL, XCCL, Gloo, …).
- For analysis: `pip install -r requirements.txt` (matplotlib, pandas).

## Supported Collectives

### Reduction collectives

Accept `--reduce-op`: `all_reduce`, `reduce`, `reduce_scatter`,
`reduce_scatter_single`.

### Data-movement collectives

`all_gather`, `all_gather_single`, `all_to_all`, `all_to_all_single`,
`broadcast`, `scatter`, `gather`.

### Point-to-point

`send_recv` (paired ping-pong).

### Synchronization

`barrier`.

To add a new collective see [Adding a New Collective](#adding-a-new-collective).

## Running a Benchmark

Set `TEST_BACKEND` to the backend name passed to `new_comm` /
`init_process_group` (e.g. `nccl`, `xccl`, `gloo`). Optionally set
`TEST_DEVICE` to a torch device string (`cuda`, `xpu`, `cpu`; defaults to
`cuda`).

Optional: `TEST_FAST_INIT_MODE` — forwarded to torchcomms as the `fastInitMode` hint.

Launch with `torchrun`:

```bash
TEST_BACKEND=nccl torchrun --nproc_per_node=8 \
collective_perf_test.py all_reduce --csv
```

## Command-Line Options

```bash
python collective_perf_test.py <collective> [options]
```

`<collective>` is the name of the collective to benchmark (e.g. `all_reduce`,
`all_to_all`, `barrier`). Pass `all` to sweep every supported collective; reduction collectives are
additionally swept across all supported reduction ops (see `--reduce-op`). See
[Supported Collectives](#supported-collectives) for the full list.

### Comm adapter selection

By default the `torchcomms` adapter is used. Pass one of the flags below to
switch adapters or run all three.

| Option | Default | Description |
|---|---|---|
| `--c10d` | off | Use `torch.distributed` instead of `torchcomms` |
| `--c10d-torchcomms` | off | Use `torch.distributed` with `use_torchcomms=True` |
| `--all-comm-adapters` | off | Run sequentially against all three adapters in a single invocation; overrides `--c10d` / `--c10d-torchcomms` |

### Sweep config

| Option | Default | Description |
|---|---|---|
| `--min-size <n>` | 4 | Minimum message size in bytes |
| `--max-size <n>` | 67108864 | Maximum message size in bytes (64 MB); must equal `min_size * size_scaling_factor^n` and divide evenly by the dtype element size |
| `--size-scaling-factor <n>` | 2 | Multiplier between sizes (must be ≥ 2) |
| `--dtype <type>` | float32 | `float32`, `float16`, `bfloat16`, `float64`, `int32`, `int64` |
| `--reduce-op <op>` | sum | Reduction collectives only: `sum`, `min`, `max`, `avg`, `product`, or `all` to sweep every supported op |

### Timing

| Option | Default | Description |
|---|---|---|
| `--warmup <n>` | 5 | Warmup iterations (not timed) |
| `--iters <n>` | 1000 | Measurement iterations |
| `--sync-interval <n>` | 0 | `0` = bulk timing, `1` = per-iteration (min/max), `N>1` = sync every N iters |
| `--async` | off | Run collectives async (`async_op=True` + `wait()`); default is sync |

`--sync-interval` modes:

- `0` — single start/stop around the whole loop. Lowest overhead; min/max latency equal the average.
- `1` — per-iteration sync. Accurate min/max at the cost of per-call overhead.
- `N > 1` — windowed sync every N iterations. Trade-off between the two.

### Output

| Option | Default | Description |
|---|---|---|
| `--csv` | off | Write per-collective CSVs (one file per collective) |
| `--output-dir <path>` | `./<adapter>/` | Directory for CSV output. With `--all-comm-adapters`, a per-adapter subdir is created underneath the given path |
| `--quiet`, `-q` | off | Suppress terminal output |
| `--help`, `-h` | — | Print usage |

## CSV Output

With `--csv`, each collective writes to `<adapter>/<collective>[_<op>].csv`
by default (e.g. `torchcomms/all_reduce_sum.csv`), so separate runs of
different adapters don't collide. Pass `--output-dir <path>` to redirect
the output (the per-adapter subdir is then only used under
`--all-comm-adapters`, where collision avoidance is needed).

The schema covers test identity, benchmark config, results, and environment
metadata. See `_CSV_COLUMNS` in [perf_test_helpers.py](perf_test_helpers.py)
for the full list.

## Comparing Comm Adapters

The simplest way is `--all-comm-adapters`, which runs the full benchmark against
`torchcomms`, `c10d`, and `c10d_torchcomms` in a single invocation and writes
one CSV directory per adapter:

```bash
# 1. Run against all three comm adapters in one go
TEST_BACKEND=nccl torchrun --nproc_per_node=8 \
    collective_perf_test.py all --all-comm-adapters --csv

# 2. Analyze — baseline is c10d, others compared against it
python analyze_perf.py \
    --dir c10d:./c10d \
    --dir c10d_torchcomms:./c10d_torchcomms \
    --dir torchcomms:./torchcomms \
    --baseline c10d
```

The same workflow can also be split into three separate runs. Each run
writes to its own `./<adapter>/` directory by default, so they don't
collide:

```bash
# torchcomms (default)
TEST_BACKEND=nccl torchrun --nproc_per_node=8 \
    collective_perf_test.py all --csv

# c10d
TEST_BACKEND=nccl torchrun --nproc_per_node=8 \
    collective_perf_test.py all --c10d --csv

# c10d_torchcomms
TEST_BACKEND=nccl torchrun --nproc_per_node=8 \
    collective_perf_test.py all --c10d-torchcomms --csv
```

Then analyze the same way as above.

## Analyzing Results

`analyze_perf.py` consumes the CSVs produced above and writes reports under
`./perf_results/` (or `--outdir`):

```
perf_results/
├── perf_summary_<adapter>.{txt,csv}     # per-adapter; latency + bus BW per (collective, message size)
├── perf_comparison.{txt,csv}            # absolute values + LatChange(%) / BwChange(%) vs baseline
├── plots/<collective>/                  # one PNG per collective per metric (latency, bus BW);
│   └── ...                              #   relative (% change) plots also here when --baseline is set
└── <adapter>_vs_<baseline>/             # one subdir per pairwise comparison
    ├── perf_regress.{txt,csv}           # rows whose latency regressed beyond --regression-threshold
    ├── perf_improve.{txt,csv}           # rows whose latency improved beyond --improvement-threshold
    └── perf_highlights.{txt,csv}        # counts and averages of improved / regressed / neutral points
```

### Key options

| Option | Default | Description |
|---|---|---|
| `--dir LABEL:PATH` | `torchcomms:./torchcomms` | Result directory; repeat for multiple adapters |
| `--baseline LABEL` | none | Enables relative plots and comparison reports |
| `--outdir PATH` | `./perf_results/` | Where to write figures and reports |
| `--regression-threshold PCT` | 5.0 | Latency increase to count as a regression |
| `--improvement-threshold PCT` | 5.0 | Latency decrease to count as an improvement |
| `--summary-only` | off | Skip figures, write only summary reports |

## Adding a New Collective

1. Create `<collective>_perf.py` with a `run_<collective>_perf(comm, record, device)` function.
2. Inside, define two callbacks and delegate to `run_bench_sweep`:
   - `setup_fn(num_elements, rank, num_ranks, device, dtype) -> (args, kwargs)`
   - `bus_bw_fn(num_elements, element_size, num_ranks, avg_time_us) -> gbps`
3. Register it in `COLLECTIVE_RUNNERS` in [collective_perf_test.py](collective_perf_test.py).
4. If it accepts a reduction operator, add it to `REDUCE_COLLECTIVES` in the
   same file and to `_REDUCE_COLLECTIVES` in [perf_test_helpers.py](perf_test_helpers.py).

See [all_reduce_perf.py](all_reduce_perf.py) for a minimal example.
