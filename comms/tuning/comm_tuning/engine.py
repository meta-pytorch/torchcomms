# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Generic parent/child runner for Triton comm tuning."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from comm_tuning.adapter import CommKernelTuningAdapter
from comm_tuning.schemas import TuningWorkItem


def benchmark_callable(
    fn,
    *,
    num_warmup: int,
    num_iters: int,
    use_cuda_graph: bool,
) -> float:
    """Return average latency in microseconds."""
    if use_cuda_graph:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(num_warmup):
                fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(num_iters):
                    fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / num_iters

    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / num_iters


def build_worklist(
    adapter: CommKernelTuningAdapter,
    *,
    world_size: int,
) -> list[TuningWorkItem]:
    worklist: list[TuningWorkItem] = []
    for spec in adapter.enumerate_input_specs(world_size):
        key = adapter.make_key(spec, world_size)
        spec_type, spec_json = adapter.spec_to_json(spec)
        key_type, key_json = adapter.key_to_json(key)

        for config in adapter.enumerate_candidate_configs(spec, key):
            config_type, config_json = adapter.config_to_json(config)
            worklist.append(
                TuningWorkItem(
                    work_item_id=len(worklist),
                    kind="candidate",
                    spec_type=spec_type,
                    spec=spec_json,
                    key_type=key_type,
                    key=key_json,
                    config_type=config_type,
                    config=config_json,
                )
            )

        for baseline in adapter.enumerate_baselines(spec, key):
            worklist.append(
                TuningWorkItem(
                    work_item_id=len(worklist),
                    kind="baseline",
                    spec_type=spec_type,
                    spec=spec_json,
                    key_type=key_type,
                    key=key_json,
                    baseline=baseline,
                )
            )
    return worklist


def default_run_name(args: argparse.Namespace, adapter_name: str) -> str:
    if args.run_name != "local":
        return args.run_name
    env_run_name = os.environ.get("COMM_TUNER_RUN_NAME")
    if env_run_name:
        return env_run_name
    mast_job = os.environ.get("MAST_JOB_NAME") or os.environ.get("MAST_HPC_JOB_NAME")
    if mast_job:
        return f"{mast_job}_{os.environ.get('MASTER_PORT', 'unknown')}"
    return f"{adapter_name}_local_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir(args: argparse.Namespace, adapter_name: str, run_name: str) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    dump_dir = os.environ.get("DUMP_DIR")
    if dump_dir:
        return Path(dump_dir) / "comm_tuning" / adapter_name / run_name
    return Path("/tmp") / "comm_tuning" / adapter_name / run_name


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _validate_work_item_port_range(
    args: argparse.Namespace,
    worklist: list[TuningWorkItem],
) -> None:
    if args.candidate_port_mode != "deterministic" or not worklist:
        return
    max_port = args.base_master_port + (len(worklist) - 1) * args.master_port_stride
    if max_port > 65535:
        raise ValueError(f"work item port range exceeds 65535: {max_port=}")


def _work_item_port(
    *,
    args: argparse.Namespace,
    rank: int,
    work_item: TuningWorkItem,
) -> int:
    if args.candidate_port_mode == "dynamic":
        port_obj: list[int | None] = [_get_free_tcp_port() if rank == 0 else None]
        dist.broadcast_object_list(port_obj, src=0)
        # pyrefly: ignore [bad-argument-type]
        return int(port_obj[0])
    return args.base_master_port + work_item.work_item_id * args.master_port_stride


def _has_work_item_result(path: Path, work_item_id: int) -> bool:
    if not path.exists():
        return False
    with path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("record_type") == "work_item_result"
                and record.get("work_item_id") == work_item_id
            ):
                return True
    return False


def _child_args(
    *,
    args: argparse.Namespace,
    entrypoint_path: str,
    work_item: TuningWorkItem,
    rank_output_path: Path,
    run_name: str,
) -> list[str]:
    child_args = [
        sys.executable,
        entrypoint_path,
        "--mode",
        "child",
        "--work-item-json",
        json.dumps(work_item.to_json_dict(), separators=(",", ":")),
        "--rank-output-path",
        str(rank_output_path),
        "--run-name",
        run_name,
        "--num-warmup",
        str(args.num_warmup),
        "--num-iters",
        str(args.num_iters),
        "--pg-timeout-sec",
        str(args.pg_timeout_sec),
    ]
    child_args.append("--cuda-graph" if args.cuda_graph else "--no-cuda-graph")
    return child_args


def _run_parent_work_item(
    *,
    args: argparse.Namespace,
    entrypoint_path: str,
    work_item: TuningWorkItem,
    rank_output_path: Path,
    run_name: str,
    child_log_path: Path,
    work_item_port: int,
) -> tuple[int, bool, float]:
    child_env = os.environ.copy()
    child_env["MASTER_PORT"] = str(work_item_port)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.pop("TORCHELASTIC_USE_AGENT_STORE", None)

    timeout_sec = args.work_item_timeout_sec
    if timeout_sec is None:
        timeout_sec = args.candidate_timeout_sec

    started = time.time()
    timed_out = False
    returncode = 0
    with child_log_path.open("w") as child_log:
        try:
            completed = subprocess.run(
                _child_args(
                    args=args,
                    entrypoint_path=entrypoint_path,
                    work_item=work_item,
                    rank_output_path=rank_output_path,
                    run_name=run_name,
                ),
                env=child_env,
                stdout=child_log,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
    return returncode, timed_out, time.time() - started


def _record_parent_child_failure(
    *,
    rank_output_path: Path,
    adapter_name: str,
    run_name: str,
    rank: int,
    world_size: int,
    work_item: TuningWorkItem,
    returncode: int,
    timed_out: bool,
    child_log_path: Path,
    elapsed_sec: float,
) -> None:
    if _has_work_item_result(rank_output_path, work_item.work_item_id):
        return
    ended = time.time()
    append_jsonl(
        rank_output_path,
        {
            "schema_version": 1,
            "record_type": "parent_child_failure",
            "adapter": adapter_name,
            "run_name": run_name,
            "rank": rank,
            "world_size": world_size,
            "work_item_id": work_item.work_item_id,
            "kind": work_item.kind,
            "spec_type": work_item.spec_type,
            "spec": work_item.spec,
            "key_type": work_item.key_type,
            "key": work_item.key,
            "config_type": work_item.config_type,
            "config": work_item.config,
            "baseline": work_item.baseline,
            "status": "error",
            "returncode": returncode,
            "timeout": timed_out,
            "child_log_path": str(child_log_path),
            "timing": {
                "start_time_unix": ended - elapsed_sec,
                "end_time_unix": ended,
                "elapsed_sec": elapsed_sec,
            },
        },
    )


def run_parent(
    args: argparse.Namespace,
    adapter: CommKernelTuningAdapter,
    *,
    entrypoint_path: str,
) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_name = default_run_name(args, adapter.name)
    output_dir = run_dir(args, adapter.name, run_name)
    results_dir = output_dir / "results"
    child_logs_dir = output_dir / "child_logs"
    worklist = build_worklist(adapter, world_size=world_size)
    _validate_work_item_port_range(args, worklist)

    parent_pg_initialized = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(seconds=args.pg_timeout_sec),
        )
        parent_pg_initialized = True

    if rank == 0:
        write_json(
            output_dir / "metadata.json",
            {
                "schema_version": 1,
                "adapter": adapter.name,
                "run_name": run_name,
                "world_size": world_size,
                "num_warmup": args.num_warmup,
                "num_iters": args.num_iters,
                "cuda_graph": args.cuda_graph,
                "work_item_count": len(worklist),
                "base_master_port": args.base_master_port,
                "master_port_stride": args.master_port_stride,
                "candidate_port_mode": args.candidate_port_mode,
            },
        )
        write_json(
            output_dir / "worklist.json",
            [work_item.to_json_dict() for work_item in worklist],
        )
        print(
            "[comm-tuner] parent_start "
            f"adapter={adapter.name} run_name={run_name} world_size={world_size} "
            f"work_item_count={len(worklist)} output_dir={output_dir} "
            f"cuda_graph={args.cuda_graph} num_warmup={args.num_warmup} "
            f"num_iters={args.num_iters}",
            flush=True,
        )

    rank_output_path = results_dir / f"rank_{rank:03d}.jsonl"
    local_ok_count = 0
    local_error_count = 0
    try:
        for work_item in worklist:
            port = _work_item_port(args=args, rank=rank, work_item=work_item)
            child_log_path = (
                child_logs_dir
                / f"rank_{rank:03d}_work_item_{work_item.work_item_id:06d}.log"
            )
            child_log_path.parent.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                print(
                    "[comm-tuner] work_item_start "
                    f"index={work_item.work_item_id + 1}/{len(worklist)} "
                    f"kind={work_item.kind} port={port} child_log={child_log_path} "
                    f"payload={json.dumps(work_item.to_json_dict(), sort_keys=True)}",
                    flush=True,
                )

            returncode, timed_out, elapsed_sec = _run_parent_work_item(
                args=args,
                entrypoint_path=entrypoint_path,
                work_item=work_item,
                rank_output_path=rank_output_path,
                run_name=run_name,
                child_log_path=child_log_path,
                work_item_port=port,
            )
            status = "timeout" if timed_out else ("ok" if returncode == 0 else "error")
            if returncode == 0:
                local_ok_count += 1
            else:
                local_error_count += 1
            if rank == 0:
                print(
                    "[comm-tuner] work_item_done "
                    f"index={work_item.work_item_id + 1}/{len(worklist)} "
                    f"work_item_id={work_item.work_item_id} kind={work_item.kind} "
                    f"status={status} returncode={returncode} "
                    f"elapsed_sec={elapsed_sec:.3f} child_log={child_log_path}",
                    flush=True,
                )
            if returncode != 0:
                _record_parent_child_failure(
                    rank_output_path=rank_output_path,
                    adapter_name=adapter.name,
                    run_name=run_name,
                    rank=rank,
                    world_size=world_size,
                    work_item=work_item,
                    returncode=returncode,
                    timed_out=timed_out,
                    child_log_path=child_log_path,
                    elapsed_sec=elapsed_sec,
                )
    finally:
        if rank == 0:
            print(
                "[comm-tuner] parent_done "
                f"adapter={adapter.name} run_name={run_name} "
                f"local_ok_count={local_ok_count} "
                f"local_error_count={local_error_count} results={results_dir}",
                flush=True,
            )
        if parent_pg_initialized:
            dist.destroy_process_group()


def run_child(args: argparse.Namespace, adapter: CommKernelTuningAdapter) -> None:
    work_item = TuningWorkItem.from_json_dict(json.loads(args.work_item_json))
    spec = adapter.spec_from_json(work_item.spec_type, work_item.spec)
    config = (
        None
        if work_item.config_type is None or work_item.config is None
        else adapter.config_from_json(work_item.config_type, work_item.config)
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    started = time.time()
    status = "ok"
    error = None
    correctness: dict[str, Any] = {"status": "not_applicable"}
    perf = None

    try:
        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=world_size,
            is_master=rank == 0,
            timeout=timedelta(seconds=args.pg_timeout_sec),
        )
        dist.init_process_group(
            backend="nccl",
            store=store,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=args.pg_timeout_sec),
        )

        inputs = adapter.make_inputs(
            spec,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if work_item.kind == "candidate":
            assert config is not None
            reference_output = adapter.run_reference(inputs, dist.group.WORLD)
            candidate_output = adapter.run_candidate(inputs, config, dist.group.WORLD)
            torch.cuda.synchronize()
            dist.barrier()
            correctness = adapter.check_correctness(candidate_output, reference_output)

            dist.barrier()
            torch.cuda.synchronize()
            latency_us = benchmark_callable(
                lambda: adapter.run_candidate(inputs, config, dist.group.WORLD),
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                use_cuda_graph=args.cuda_graph,
            )
        elif work_item.kind == "baseline":
            assert work_item.baseline is not None
            dist.barrier()
            torch.cuda.synchronize()
            latency_us = benchmark_callable(
                lambda: adapter.run_baseline(
                    inputs,
                    work_item.baseline,
                    dist.group.WORLD,
                ),
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                use_cuda_graph=args.cuda_graph,
            )
        else:
            raise ValueError(f"unsupported work item kind: {work_item.kind}")

        dist.barrier()
        torch.cuda.synchronize()
        if rank == 0:
            perf = {
                "latency_us": latency_us,
                "num_warmup": args.num_warmup,
                "num_iters": args.num_iters,
                "cuda_graph": args.cuda_graph,
            }
    except Exception as exc:
        status = "error"
        error = f"{exc!r}\n{traceback.format_exc()}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    ended = time.time()
    record = {
        "schema_version": 1,
        "record_type": "work_item_result",
        "adapter": adapter.name,
        "run_name": args.run_name,
        "rank": rank,
        "world_size": world_size,
        "work_item_id": work_item.work_item_id,
        "kind": work_item.kind,
        "spec_type": work_item.spec_type,
        "spec": work_item.spec,
        "key_type": work_item.key_type,
        "key": work_item.key,
        "config_type": work_item.config_type,
        "config": work_item.config,
        "baseline": work_item.baseline,
        "status": status,
        "correctness": correctness,
        "perf": perf,
        "timing": {
            "start_time_unix": started,
            "end_time_unix": ended,
            "elapsed_sec": ended - started,
        },
        "error": error,
    }
    append_jsonl(Path(args.rank_output_path), record)
    if status != "ok":
        raise RuntimeError(error)
