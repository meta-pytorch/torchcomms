# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Dev tool: measure NCCL all_to_all_single's real active-SM grid (kernel gridDim) per
per-rank message size (single node). Provenance for the values in
``_bench_common._NCCL_A2A_GRID_TABLE`` (the actual launched grid per (size, cap); run once per
cap with NCCL_MAX_NCHANNELS=NCCL_MIN_NCHANNELS=cap to fill the matrix); re-run when
NCCL / hardware changes:

    buck2 run @fbcode//mode/opt //comms/dsl/tests:_probe_nccl_grid

Method: run a2a under torch.profiler and read the max gridDim.x over the NCCL kernels from the
chrome trace -- the actual launched grid (ncclDevKernel_SendRecv, block=640, send+recv fused in
one CTA), not the allocated channel count. Cross-checked against nsys cuda_gpu_trace (64KB ->
grid 8, block 640; both agree). Do NOT run this under nsys -- nsys and torch.profiler conflict
(the chrome trace comes back without grid); use one or the other.
"""

import json
import os
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_DTYPE = torch.bfloat16
# per-rank message sizes to probe (the estimated rows + a few known ones to validate method).
_SIZES = [
    ("64KB", 65536),  # known natural=8
    ("1MB", 1048576),  # known natural=16
    ("8MB", 8388608),  # known natural=32
    ("48MB", 50331648),  # was ESTIMATED
    ("64MB", 67108864),  # known natural=32
    ("96MB", 100663296),  # was ESTIMATED
    ("256MB", 268435456),  # known natural=32
    ("512MB", 536870912),  # was ESTIMATED
    ("1GB", 1073741824),  # was ESTIMATED
    ("2GB", 2147483648),  # was fallback (not in table)
]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _grid_from_trace(path: str) -> int:
    """Max gridDim.x over NCCL a2a/sendrecv kernels in a chrome trace."""
    with open(path) as f:
        tr = json.load(f)
    best = 0
    for e in tr.get("traceEvents", []):
        name = str(e.get("name", "")).lower()
        if "nccl" not in name and "sendrecv" not in name and "alltoall" not in name:
            continue
        g = e.get("args", {}).get("grid")
        if isinstance(g, list) and g:
            best = max(best, int(g[0]))
    return best


def _worker(rank: int, ws: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    device = torch.device(f"cuda:{rank}")
    try:
        for label, msg in _SIZES:
            numel = max(msg // 2, ws)
            numel -= numel % ws
            inp = torch.randn(numel, dtype=_DTYPE, device=device)
            out = torch.empty_like(inp)
            for _ in range(5):  # warm up / let NCCL settle channels
                dist.all_to_all_single(out, inp)
            torch.cuda.synchronize()
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tpath = tf.name
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(3):
                        dist.all_to_all_single(out, inp)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(tpath)
                grid = _grid_from_trace(tpath)
            finally:
                if os.path.exists(tpath):
                    os.remove(tpath)
            # A 0 means no NCCL kernel grid was found (renamed kernel / changed trace schema),
            # NOT a real measurement -- fail loud so a bad probe is never copied into the table.
            if grid == 0:
                raise RuntimeError(
                    f"probe failed for {label} (msg={msg}): no NCCL kernel gridDim found in "
                    "the profiler trace -- kernel name filter or trace schema may have changed."
                )
            if rank == 0:
                print(f"PROBE {label:>6} msg={msg} nccl_grid={grid}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    ws = min(torch.cuda.device_count(), 8)
    print(
        f"probing NCCL a2a grid on {ws} GPUs ({torch.cuda.get_device_name(0)})",
        flush=True,
    )
    mp.spawn(_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)


if __name__ == "__main__":
    main()
