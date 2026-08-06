# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import collections.abc
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.distributed.config as dist_config
import torchcomms
from torch import distributed as dist


class CommAdapter:
    def __init__(
        self,
        name: Literal["torchcomms", "c10d", "c10d_torchcomms"],
        comm_name: Optional[str] = "collective_perf_test",
    ):
        self.name = name
        self.comm_name = comm_name
        self.initialized = False
        self.dist_comm = None

    def init(
        self,
        backend: str,
        device: torch.device,
        hints: Optional[collections.abc.Mapping] = None,
    ) -> None:
        if self.name == "torchcomms":
            self.dist_comm = torchcomms.new_comm(
                backend,
                device,
                name=self.comm_name,
                hints=hints,
            )
        elif self.name.startswith("c10d"):
            if self.name == "c10d_torchcomms":
                dist_config.use_torchcomms = True
            dist.init_process_group(backend=backend, init_method="env://")
            self.dist_comm = dist
        self.initialized = True

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("Comm adapter not initialized. Call init() first.")

    # c10d uses different names for some collectives
    _C10D_NAME_MAP = {
        "all_gather_single": "all_gather_into_tensor",
        "reduce_scatter_single": "reduce_scatter_tensor",
    }

    def run_collective(self, collective_fn, *args, **kwargs) -> None:
        """Dispatch a collective on the underlying adapter.

        ``async_op`` is consumed here (not forwarded to torchcomms / c10d
        verbatim): when True, the adapter awaits the returned handle before
        returning. For c10d send/recv this also rewrites the call to
        ``isend`` / ``irecv``. Callers must pass ``async_op`` as a keyword.

        For hot loops, prefer :meth:`resolve_collective` to lift the dispatch
        and lookup overhead out of the timing loop.
        """
        async_op = kwargs.pop("async_op", False)
        bound, args, kwargs = self._resolve(collective_fn, args, kwargs, async_op)
        handle = bound(*args, **kwargs)
        if async_op and handle is not None:
            handle.wait()

    def resolve_collective(self, collective_fn, *args, **kwargs):
        """Pre-resolve a collective into a zero-overhead callable for hot loops.

        Returns ``(call, args, kwargs)`` where ``call(*args, **kwargs)`` runs
        the collective and waits on the handle (matching :meth:`run_collective`
        semantics). All adapter-specific name mapping, arg reshaping, and
        ``async_op`` handling is performed once here, so the returned callable
        can be invoked tightly in a measurement loop without per-iteration
        Python overhead.

        ``async_op`` must be passed as a keyword.
        """
        self._require_initialized()
        async_op = kwargs.pop("async_op", False)
        bound, args, kwargs = self._resolve(collective_fn, args, kwargs, async_op)
        if async_op:

            def call(*a, **kw):
                handle = bound(*a, **kw)
                if handle is not None:
                    handle.wait()
        else:
            call = bound
        return call, args, kwargs

    def _resolve(self, collective_fn, args, kwargs, async_op):
        """Compute the bound callable plus adjusted args/kwargs for one call."""
        self._require_initialized()
        fn_name = collective_fn
        if self.name.startswith("c10d"):
            fn_name = self._C10D_NAME_MAP.get(collective_fn, collective_fn)
            # gather: torchcomms takes (output_list, input_tensor, root, ...)
            #         c10d takes (tensor, gather_list, dst, ...)
            if collective_fn == "gather" and len(args) >= 2:
                args = (args[1], args[0]) + args[2:]
            # c10d send/recv have no async_op; use isend/irecv for async
            if collective_fn in ("send", "recv"):
                if async_op:
                    fn_name = "i" + collective_fn
            else:
                kwargs["async_op"] = async_op
        else:
            # torchcomms: pass async_op positionally to match the pre-refactor
            # call shape exactly. The pybind11 binding has fewer overheads on
            # the positional path than on the kwargs path.
            args = args + (async_op,)

        try:
            bound = getattr(self.dist_comm, fn_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Collective operation '{fn_name}' not found on "
                f"comm adapter '{self.name}'."
            ) from exc
        return bound, args, kwargs

    def get_rank(self) -> int:
        self._require_initialized()
        return self.dist_comm.get_rank()

    def get_size(self) -> int:
        self._require_initialized()
        if self.name == "torchcomms":
            return self.dist_comm.get_size()
        elif self.name.startswith("c10d"):
            return self.dist_comm.get_world_size()
        raise NotImplementedError(f"get_size not implemented for adapter '{self.name}'")

    def get_backend(self) -> str:
        self._require_initialized()
        return self.dist_comm.get_backend()

    def get_backend_version(self) -> str:
        self._require_initialized()
        if self.name in ["torchcomms", "c10d_torchcomms"]:
            try:
                return self.dist_comm.get_backend_version()
            except (AttributeError, NotImplementedError, RuntimeError):
                return "n/a"
        elif self.name.startswith("c10d"):
            backend = self.get_backend()
            if backend == "nccl" and torch.cuda.is_available():
                return ".".join(str(v) for v in torch.cuda.nccl.version())
            return "n/a"
        return "n/a"

    _REDUCE_OP_MAP_TORCHCOMMS = {
        "sum": torchcomms.ReduceOp.SUM,
        "min": torchcomms.ReduceOp.MIN,
        "max": torchcomms.ReduceOp.MAX,
        "avg": torchcomms.ReduceOp.AVG,
        "product": torchcomms.ReduceOp.PRODUCT,
    }
    _REDUCE_OP_MAP_C10D = {
        "sum": dist.ReduceOp.SUM,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "avg": dist.ReduceOp.AVG,
        "product": dist.ReduceOp.PRODUCT,
    }

    def get_reduce_op(self, op_name: str):
        """Map a reduce op name to the appropriate ReduceOp type."""
        self._require_initialized()
        op_map = (
            self._REDUCE_OP_MAP_TORCHCOMMS
            if self.name == "torchcomms"
            else self._REDUCE_OP_MAP_C10D
        )
        if op_name not in op_map:
            raise ValueError(
                f"Unknown reduce op '{op_name}'. Supported: {', '.join(op_map.keys())}"
            )
        return op_map[op_name]

    def finalize(self) -> None:
        if self.initialized:
            if self.name == "torchcomms":
                self.dist_comm.finalize()
            elif self.name.startswith("c10d"):
                self.dist_comm.destroy_process_group()
            self.initialized = False


@dataclass
class EnvInfo:
    device_type: str = None
    device_name: str = None
    torch_version: str = torch.__version__
    torchcomms_version: str = "n/a"
    comm_adapter_name: Literal["torchcomms", "c10d", "c10d_torchcomms"] = None
    comm_backend_name: str = None
    comm_backend_version: str = None


@dataclass
class BenchConfig:
    run_all_collectives: bool = False
    run_all_comm_adapters: bool = False
    run_all_reduce_ops: bool = False
    reduce_op_explicit: bool = False
    # Message size range in bytes (powers of 2)
    min_size: int = 4  # 4 bytes
    max_size: int = 67108864 if os.environ.get("TEST_FULL_SWEEP") != "0" else 1024  # 64 MiB or 1 KiB
    # Scaling factor for message sizes (default 2 = powers of 2)
    size_scaling_factor: int = 2
    # Whether to save results to CSV files
    csv_enabled: bool = False
    # Whether to suppress terminal output
    quiet: bool = False
    # CSV output file path (set dynamically per collective)
    csv_file: Optional[str] = None
    # Base directory for CSV output (per-adapter subdir created underneath)
    output_dir: Optional[str] = None

@dataclass
class BenchParams:
    comm_adapter: Literal["torchcomms", "c10d", "c10d_torchcomms"] = "torchcomms"
    collective_name: str = None
    num_ranks: int = 0
    async_op: bool = False
    dispatch_mode: str = "sync"
    reduce_op: str = "sum"    
    warmup_iterations: int = 5 if os.environ.get("TEST_FULL_SWEEP") != "0" else 1
    measure_iterations: int = 1000 if os.environ.get("TEST_FULL_SWEEP") != "0" else 3
    # Number of iterations between stream synchronizations during measurement.
    # If 0, only synchronize after all iterations complete.
    sync_interval: int = 0
    # Data type
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)


@dataclass
class BenchMetrics:
    message_size_bytes: int = 0
    total_time_us: float = 0.0
    avg_time_us: float = 0.0
    min_time_us: float = 0.0
    max_time_us: float = 0.0
    bus_bw_gbps: float = 0.0


@dataclass
class BenchRecord:
    info: EnvInfo = field(default_factory=EnvInfo)
    config: BenchConfig = field(default_factory=BenchConfig)
    params: BenchParams = field(default_factory=BenchParams)
    metrics: BenchMetrics = field(default_factory=BenchMetrics)


class PerfTimer:
    def __init__(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._running = False

    def start(self):
        self._start_time = time.perf_counter()
        self._running = True

    def stop(self):
        self._end_time = time.perf_counter()
        self._running = False

    def reset(self):
        self._running = False

    def elapsed_us(self) -> float:
        return (self._end_time - self._start_time) * 1e6

    def elapsed_ms(self) -> float:
        return self.elapsed_us() / 1000.0


# Column definitions for terminal output and CSV files.
# Each entry is (header_name, column_width). The width is used only for
# terminal alignment; CSV output ignores it.
# Ordered as: test identity → benchmark config → results → environment,
# so the most frequently compared fields appear first when scrolling right.
_CSV_COLUMNS = [
    # Test identity
    ("Collective", 25),
    ("DType", 10),
    ("ReduceOp", 12),
    ("SendMsgSize(B)", 15),
    ("Ranks", 10),
    # Benchmark config
    ("DispatchMode", 14),
    ("WarmupIters", 12),
    ("MeasureIters", 14),
    ("SyncInterval", 14),
    # Results
    ("LatAvg(us)", 15),
    ("LatMin(us)", 15),
    ("LatMax(us)", 15),
    ("BusBw(GB/s)", 15),
    # Environment
    ("DeviceType", 12),
    ("DeviceName", 40),
    ("CommAdapter", 20),
    ("CommBackend", 13),
    ("CommBackendVersion", 20),
    ("TorchVersion", 30),
    ("TorchCommsVersion", 20),
]


# Collectives that accept a reduction operator (e.g. sum, min, max).
# Used in log_perf_result to decide whether to display the ReduceOp column
# value or substitute "n/a" for collectives that do not perform a reduction.
_REDUCE_COLLECTIVES = {
    "all_reduce",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_single",
}


def collective_to_camel_case(s: str) -> str:
    return "".join(word.capitalize() for word in s.split("_"))


def init_csv_file(csv_file: str) -> None:
    """Create the CSV file and write the header row."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _ in _CSV_COLUMNS])


def log_perf_header(record: BenchRecord, rank: int) -> None:
    if rank != 0 or record.config.quiet:
        return
    print(
        f"\n=== {collective_to_camel_case(record.params.collective_name)} Performance ===\n"
    )
    header = "".join(f"{name:<{width}}" for name, width in _CSV_COLUMNS)
    print(header)
    print("-" * len(header))


def log_perf_result(record: BenchRecord, rank: int) -> None:
    if rank != 0:
        return
    reduce_op_display = (
        record.params.reduce_op
        if record.params.collective_name in _REDUCE_COLLECTIVES
        else "n/a"
    )
    values = [
        # Test identity
        record.params.collective_name,
        dtype_to_string(record.params.dtype),
        reduce_op_display,
        record.metrics.message_size_bytes,
        record.params.num_ranks,
        # Benchmark config
        record.params.dispatch_mode,
        record.params.warmup_iterations,
        record.params.measure_iterations,
        record.params.sync_interval,
        # Results
        f"{record.metrics.avg_time_us:.2f}",
        f"{record.metrics.min_time_us:.2f}",
        f"{record.metrics.max_time_us:.2f}",
        f"{record.metrics.bus_bw_gbps:.2f}",
        # Environment
        record.info.device_type,
        record.info.device_name,
        record.info.comm_adapter_name,
        record.info.comm_backend_name,
        record.info.comm_backend_version,
        record.info.torch_version,
        record.info.torchcomms_version,
    ]
    if not record.config.quiet:
        print("".join(f"{v:<{w}}" for v, (_, w) in zip(values, _CSV_COLUMNS)))

    csv_file = record.config.csv_file
    if csv_file:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(values)


def create_tensor(
    num_elements: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.ones(num_elements, dtype=dtype, device=device) * float(rank + 1)


def sync_device() -> None:
    """Synchronize the device stream if it's an accelerator."""
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def run_bench_sweep(
    comm: CommAdapter,
    record: BenchRecord,
    device: torch.device,
    collective_name: str,
    setup_fn: Callable[
        [int, int, int, torch.device, torch.dtype],
        Tuple[tuple, dict],
    ],
    bus_bw_fn: Callable[[int, int, int, float], float],
) -> None:
    """Generic message-size sweep with warmup, measurement, and logging.

    Args:
        comm: The communication adapter.
        record: Benchmark record (config + params + metrics).
        device: Torch device.
        collective_name: Name passed to ``comm.run_collective``.
        setup_fn: ``(num_elements, rank, num_ranks, device, dtype) -> (args, kwargs)``
            returning positional args and keyword args for the collective call.
            ``async_op`` is appended automatically and must not be included.
        bus_bw_fn: ``(num_elements, element_size, num_ranks, avg_time_us) -> bus_bw_gbps``
            computing the bus bandwidth in GB/s.
    """
    rank = comm.get_rank()
    num_ranks = comm.get_size()
    params = record.params
    config = record.config

    element_size = torch.tensor([], dtype=params.dtype).element_size()

    msg_size = config.min_size
    while msg_size <= config.max_size:
        num_elements = msg_size // element_size
        if num_elements == 0:
            num_elements = 1

        raw_args, raw_kwargs = setup_fn(
            num_elements, rank, num_ranks, device, params.dtype
        )

        # Pre-resolve once so the timing loop calls a bound method directly.
        call, call_args, call_kwargs = comm.resolve_collective(
            collective_name, *raw_args, **raw_kwargs, async_op=params.async_op
        )

        # Warmup
        for _ in range(params.warmup_iterations):
            call(*call_args, **call_kwargs)

        comm.run_collective("barrier", async_op=False)

        # Measure
        timer = PerfTimer()

        if params.sync_interval == 1:
            # Per-iteration timing: sync after every iteration to get min/max.
            iter_times: list[float] = []
            sync_device()
            for _ in range(params.measure_iterations):
                timer.start()
                call(*call_args, **call_kwargs)
                sync_device()
                timer.stop()
                iter_times.append(timer.elapsed_us())

            total = sum(iter_times)
            avg_time = total / params.measure_iterations
            min_time = min(iter_times)
            max_time = max(iter_times)
        else:
            # Bulk timing: single start/stop around all iterations.
            sync_device()
            timer.start()

            for i in range(params.measure_iterations):
                call(*call_args, **call_kwargs)
                if params.sync_interval > 0 and (i + 1) % params.sync_interval == 0:
                    sync_device()

            sync_device()
            timer.stop()

            total = timer.elapsed_us()
            avg_time = total / params.measure_iterations
            min_time = avg_time
            max_time = avg_time

        bus_bw_gbps = bus_bw_fn(num_elements, element_size, num_ranks, avg_time)

        record.metrics.message_size_bytes = num_elements * element_size
        record.metrics.total_time_us = total
        record.metrics.avg_time_us = avg_time
        record.metrics.min_time_us = min_time
        record.metrics.max_time_us = max_time
        record.metrics.bus_bw_gbps = bus_bw_gbps

        log_perf_result(record, rank)

        msg_size *= config.size_scaling_factor


def print_usage(program_name: str) -> None:
    print(f"""Usage: {program_name} <collective> [options]

Collectives:
  all_reduce             - AllReduce collective
  all_gather             - AllGather collective (tensor list output)
  all_gather_single      - AllGather collective (single tensor output)
  reduce_scatter         - ReduceScatter collective (tensor list input)
  reduce_scatter_single  - ReduceScatter (single tensor input)
  all_to_all             - AllToAll collective (tensor list)
  all_to_all_single      - AllToAll collective (single tensor)
  broadcast              - Broadcast collective
  reduce                 - Reduce collective
  scatter                - Scatter collective
  gather                 - Gather collective
  send_recv              - Send/Recv point-to-point (ping-pong)
  barrier                - Barrier collective
  all                    - Run all collectives

Options:
  --async                - Run async mode (default: sync)
  --warmup <n>           - Number of warmup iterations (default: 5)
  --iters <n>            - Number of measurement iterations (default: 1000)
  --sync-interval <n>    - Iterations between stream syncs (default: 0)
  --min-size <n>         - Min message size in bytes (default: 4)
  --max-size <n>         - Max message size in bytes (default: 67108864)
  --size-scaling-factor <n> - Size multiplier between tests (default: 2)
  --dtype <type>         - Data type: float32, float16, bfloat16, float64,
                           int32, int64 (default: float32)
  --reduce-op <op>       - Reduction operation: sum, min, max, avg, product,
                           or 'all' to sweep every supported op (reduction
                           collectives only) (default: sum)
  --c10d                 - Use c10d instead of torchcomms
  --c10d-torchcomms      - Use c10d with torchcomms backend
  --all-comm-adapters    - Run against all comm adapters (torchcomms, c10d,
                           c10d_torchcomms)
  --csv                  - Save results to CSV files (one per collective)
  --output-dir <path>    - Base dir for CSV output (default: cwd); a
                           per-adapter subdirectory is created underneath
  --quiet, -q            - Suppress terminal output
  --help, -h             - Show this help message
""")


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "half": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "double": torch.float64,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int": torch.int32,
        "int32": torch.int32,
        "long": torch.int64,
        "int64": torch.int64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return dtype_map[dtype_str]


def dtype_to_string(dtype: torch.dtype) -> str:
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype}")
    return dtype_map[dtype]


def validate_bench_params(record: BenchRecord) -> Optional[str]:
    config = record.config
    params = record.params

    if not config.run_all_collectives and params.collective_name is None:
        return "No collective specified"

    if config.quiet and not config.csv_enabled:
        return "--quiet/-q requires --csv; otherwise results would be discarded"

    if config.reduce_op_explicit and config.run_all_collectives:
        return (
            "--reduce-op is implicit when running 'all'; reduction collectives "
            "are always swept across every supported op"
        )

    if config.size_scaling_factor < 2:
        return "size_scaling_factor must be at least 2"

    if config.min_size <= 0:
        return "min_size must be positive"

    if config.max_size < config.min_size:
        return "max_size must be >= min_size"

    # Validate that max_size is reachable from min_size via scaling factor
    size = config.min_size
    while size < config.max_size:
        size *= config.size_scaling_factor
    if size != config.max_size and config.min_size != config.max_size:
        return "max_size must be min_size * size_scaling_factor^n for some integer n"

    # Validate dtype divides sizes evenly
    element_size = torch.tensor([], dtype=params.dtype).element_size()
    if config.min_size % element_size != 0:
        return (
            f"min_size must be divisible by dtype element size ({element_size} bytes)"
        )
    if config.max_size % element_size != 0:
        return (
            f"max_size must be divisible by dtype element size ({element_size} bytes)"
        )

    if (
        config.reduce_op_explicit
        and not config.run_all_collectives
        and params.collective_name not in _REDUCE_COLLECTIVES
    ):
        return (
            f"--reduce-op is only valid for reduction collectives "
            f"({', '.join(sorted(_REDUCE_COLLECTIVES))}); "
            f"got '{params.collective_name}'"
        )

    valid_reduce_ops = {"sum", "min", "max", "avg", "product"}
    if (
        not config.run_all_reduce_ops
        and params.reduce_op not in valid_reduce_ops
        and (
            config.run_all_collectives or params.collective_name in _REDUCE_COLLECTIVES
        )
    ):
        return (
            f"Unknown reduce op '{params.reduce_op}' for collective "
            f"'{params.collective_name}'. Supported: "
            f"{', '.join(sorted(valid_reduce_ops))}"
        )

    if params.warmup_iterations < 0:
        return "warmup_iterations must be non-negative"
    if params.measure_iterations <= 0:
        return "measure_iterations must be positive"
    if params.sync_interval < 0:
        return "sync_interval must be non-negative"

    return None  # Valid
