# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import os
import sys
from typing import Optional, Tuple

import torch
import torchcomms
from torchcomms.tests.perf.py.all_gather_perf import run_all_gather_perf
from torchcomms.tests.perf.py.all_gather_single_perf import run_all_gather_single_perf
from torchcomms.tests.perf.py.all_reduce_perf import run_all_reduce_perf
from torchcomms.tests.perf.py.all_to_all_perf import run_all_to_all_perf
from torchcomms.tests.perf.py.all_to_all_single_perf import run_all_to_all_single_perf
from torchcomms.tests.perf.py.barrier_perf import run_barrier_perf
from torchcomms.tests.perf.py.broadcast_perf import run_broadcast_perf
from torchcomms.tests.perf.py.gather_perf import run_gather_perf
from torchcomms.tests.perf.py.perf_test_helpers import (
    BenchRecord,
    CommAdapter,
    dtype_to_string,
    init_csv_file,
    log_perf_header,
    parse_dtype,
    print_usage,
    validate_bench_params,
)
from torchcomms.tests.perf.py.reduce_perf import run_reduce_perf
from torchcomms.tests.perf.py.reduce_scatter_perf import run_reduce_scatter_perf
from torchcomms.tests.perf.py.reduce_scatter_single_perf import (
    run_reduce_scatter_single_perf,
)
from torchcomms.tests.perf.py.scatter_perf import run_scatter_perf
from torchcomms.tests.perf.py.send_recv_perf import run_send_recv_perf


# Map collective names to their perf functions
COLLECTIVE_RUNNERS = {
    "all_reduce": run_all_reduce_perf,
    "all_gather": run_all_gather_perf,
    "all_gather_single": run_all_gather_single_perf,
    "reduce_scatter": run_reduce_scatter_perf,
    "reduce_scatter_single": run_reduce_scatter_single_perf,
    "all_to_all": run_all_to_all_perf,
    "all_to_all_single": run_all_to_all_single_perf,
    "broadcast": run_broadcast_perf,
    "reduce": run_reduce_perf,
    "scatter": run_scatter_perf,
    "gather": run_gather_perf,
    "send_recv": run_send_recv_perf,
    "barrier": run_barrier_perf,
}

# Collectives that accept a reduce op
REDUCE_COLLECTIVES = {"all_reduce", "reduce_scatter", "reduce_scatter_single", "reduce"}

# All supported reduce ops
REDUCE_OPS = ["sum", "min", "max", "avg", "product"]


VALUE_FLAGS = {
    "--warmup",
    "--iters",
    "--sync-interval",
    "--min-size",
    "--max-size",
    "--size-scaling-factor",
    "--dtype",
    "--reduce-op",
    "--output-dir",
}


def parse_args(args: list) -> Tuple[BenchRecord, Optional[str]]:
    """Parse command-line arguments and return (record, error)."""
    record = BenchRecord()

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in VALUE_FLAGS:
            if i + 1 >= len(args):
                return record, f"Missing value for {arg}"
            value = args[i + 1]
            i += 1
            if arg == "--warmup":
                record.params.warmup_iterations = int(value)
            elif arg == "--iters":
                record.params.measure_iterations = int(value)
            elif arg == "--sync-interval":
                record.params.sync_interval = int(value)
            elif arg == "--min-size":
                record.config.min_size = int(value)
            elif arg == "--max-size":
                record.config.max_size = int(value)
            elif arg == "--size-scaling-factor":
                record.config.size_scaling_factor = int(value)
            elif arg == "--dtype":
                record.params.dtype = parse_dtype(value)
            elif arg == "--reduce-op":
                value = value.lower()
                record.config.reduce_op_explicit = True
                if value == "all":
                    record.config.run_all_reduce_ops = True
                else:
                    record.params.reduce_op = value
            elif arg == "--output-dir":
                record.config.output_dir = value
        elif arg in ("--help", "-h"):
            return record, "help"
        elif arg == "--async":
            record.params.async_op = True
        elif arg == "--csv":
            record.config.csv_enabled = True
        elif arg in ("--quiet", "-q"):
            record.config.quiet = True
        elif arg == "--c10d":
            record.params.comm_adapter = "c10d"
        elif arg == "--c10d-torchcomms":
            record.params.comm_adapter = "c10d_torchcomms"
        elif arg == "--all-comm-adapters":
            record.config.run_all_comm_adapters = True
        elif not arg.startswith("-"):
            if arg == "all":
                record.config.run_all_collectives = True
            elif arg in COLLECTIVE_RUNNERS:
                record.params.collective_name = arg
            else:
                return record, f"Unknown collective '{arg}'"
        else:
            return record, f"Unknown argument '{arg}'"

        i += 1

    return record, None


def run_collectives(
    comm: CommAdapter,
    record: BenchRecord,
    device: torch.device,
) -> None:
    """Run the specified collective performance test(s)."""
    config = record.config
    rank = comm.get_rank()
    csv_dir = None

    if config.csv_enabled and rank == 0:
        if config.output_dir:
            csv_dir = (
                os.path.join(config.output_dir, record.info.comm_adapter_name)
                if config.run_all_comm_adapters
                else config.output_dir
            )
        else:
            csv_dir = record.info.comm_adapter_name
        os.makedirs(csv_dir, exist_ok=True)

    if config.run_all_collectives:
        for collective_name, collective_runner in COLLECTIVE_RUNNERS.items():
            record.params.collective_name = collective_name
            if collective_name in REDUCE_COLLECTIVES:
                ops = REDUCE_OPS
            else:
                ops = [None]
            for op in ops:
                if op is not None:
                    record.params.reduce_op = op
                if csv_dir:
                    filename = collective_name
                    if op is not None:
                        filename = f"{collective_name}_{op}"
                    config.csv_file = os.path.join(csv_dir, f"{filename}.csv")
                    init_csv_file(config.csv_file)
                log_perf_header(record, rank)
                try:
                    collective_runner(comm, record, device)
                except Exception as e:
                    print(
                        f"Error running {collective_name}"
                        f"{f' ({op})' if op else ''}: {e}",
                        file=sys.stderr,
                    )
    elif record.params.collective_name in COLLECTIVE_RUNNERS:
        collective_name = record.params.collective_name
        if config.run_all_reduce_ops and collective_name in REDUCE_COLLECTIVES:
            ops = REDUCE_OPS
        else:
            ops = [None]
        for op in ops:
            if op is not None:
                record.params.reduce_op = op
            if csv_dir:
                filename = collective_name
                if op is not None:
                    filename = f"{collective_name}_{op}"
                config.csv_file = os.path.join(csv_dir, f"{filename}.csv")
                init_csv_file(config.csv_file)
            log_perf_header(record, rank)
            if len(ops) > 1:
                try:
                    COLLECTIVE_RUNNERS[collective_name](comm, record, device)
                except Exception as e:
                    print(
                        f"Error running {collective_name} ({op}): {e}",
                        file=sys.stderr,
                    )
            else:
                COLLECTIVE_RUNNERS[collective_name](comm, record, device)


def main() -> int:
    record, parse_output = parse_args(sys.argv[1:])

    if parse_output == "help":
        print_usage(sys.argv[0])
        return 0

    if parse_output is not None:
        print(f"Error: {parse_output}", file=sys.stderr)
        print(f"Run '{sys.argv[0]} --help' for usage.", file=sys.stderr)
        return 1

    error = validate_bench_params(record)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        print(f"Run '{sys.argv[0]} --help' for usage.", file=sys.stderr)
        return 1

    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    test_backend = os.environ.get("TEST_BACKEND")
    if not test_backend:
        print("Error: TEST_BACKEND environment variable is not set", file=sys.stderr)
        return 1

    hints = {}
    fast_init_mode = os.environ.get("TEST_FAST_INIT_MODE")
    if fast_init_mode:
        hints["fastInitMode"] = fast_init_mode

    info = record.info
    params = record.params
    config = record.config

    info.device_type = device.type
    if info.device_type == "xpu":
        info.device_name = torch.xpu.get_device_name()
    elif info.device_type == "cuda":
        info.device_name = torch.cuda.get_device_name()
    else:
        info.device_name = "n/a"

    try:
        torchcomms_version = torchcomms.__version__
    except AttributeError:
        torchcomms_version = "unknown"

    if config.run_all_comm_adapters:
        adapters = ["torchcomms", "c10d", "c10d_torchcomms"]
    else:
        adapters = [params.comm_adapter]

    rank = 0
    for adapter_name in adapters:
        params.comm_adapter = adapter_name
        info.comm_adapter_name = adapter_name

        comm = CommAdapter(info.comm_adapter_name)
        comm.init(test_backend, device, hints)

        rank = comm.get_rank()
        num_ranks = comm.get_size()

        params.num_ranks = num_ranks
        params.dispatch_mode = "async" if params.async_op else "sync"

        info.comm_backend_name = comm.get_backend()
        info.comm_backend_version = comm.get_backend_version()

        if torch.accelerator.is_available():
            device_count = torch.accelerator.device_count()
            if device_count > 0:
                torch.accelerator.set_device_index(rank % device_count)

        if rank == 0 and not record.config.quiet:
            print("Collective Performance Test")
            print("======================================")
            print(f"Comm adapter: {adapter_name}")
            if record.config.csv_enabled:
                if record.config.output_dir:
                    csv_path = (
                        os.path.join(
                            record.config.output_dir, record.info.comm_adapter_name
                        )
                        if record.config.run_all_comm_adapters
                        else record.config.output_dir
                    )
                else:
                    csv_path = record.info.comm_adapter_name
                print(f"CSV output: {csv_path}/")

        run_collectives(comm, record, device)

        comm.finalize()

    if rank == 0:
        print("\nPerformance test completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
