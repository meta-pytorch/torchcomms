# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Shared Nsight Compute (NCU) profiling boilerplate for the a2a benchmark.

One place that builds the ``ncu`` wrapper argv, so the local path
(``benchmark_a2a_cute --ncu`` re-execs itself under ncu) and the MAST path
(``mast_launch --ncu`` wraps the torchrun program) emit an identical invocation.
Mirrors the validated genai ``ops/mast.py --ncu`` mechanism (D104906945).

NCU version caveat: the MAST base image ships Nsight Compute 2025.3.1, which does NOT
support the ``--communicator`` / ``--lockstep-kernel-launch`` distributed replay flags
(need 2026.1.1+). So multi-pass metrics that require replaying a kernel will DEADLOCK on
the a2a comm kernel (a replayed rank spins on signals from peers that are not replaying in
lockstep). Default to single-pass **launch** metrics (``launch__registers_per_thread``,
grid/occupancy), which ncu collects without replay -- safe for the comm kernel.
"""

from __future__ import annotations

import os
import shutil

# Single-pass launch metrics: collected WITHOUT replaying the kernel body, so they are safe
# on the signal-spinning a2a comm kernel under the current (2025.3.1) MAST NCU.
NCU_DEFAULT_METRICS: str = (
    "launch__registers_per_thread,launch__grid_size,launch__block_size,"
    "launch__waves_per_multiprocessor"
)

# NCU binary candidates, newest CUDA / NCU first (mirrors genai's discovery order).
_NCU_CANDIDATES: tuple[str, ...] = (
    "/usr/local/cuda-13.0/bin/ncu",
    "/usr/local/cuda-12.8/bin/ncu",
    "/usr/local/cuda/bin/ncu",
    "/opt/nvidia/nsight-compute/2025.3.1/ncu",
)


def resolve_ncu_bin(explicit: str = "") -> str:
    """Absolute path to an ``ncu`` binary: explicit > $PATH > known install dirs.

    Raises ``FileNotFoundError`` if none is found so the caller fails loud instead of
    launching an un-profiled run that silently produces no report.
    """
    if explicit:
        return explicit
    found = shutil.which("ncu")
    if found:
        return found
    for cand in _NCU_CANDIDATES:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "no ncu binary found (looked on $PATH and "
        f"{', '.join(_NCU_CANDIDATES)}); pass --ncu-path or use a MAST image with "
        "Nsight Compute (mast_launch --ncu swaps to one)."
    )


# The driver libs NCU needs to connect to the CUDA driver inside a MAST container.
_NCU_DRIVER_LIBS: tuple[str, ...] = (
    "libcuda.so",
    "libcuda.so.1",
    "libnvidia-ml.so",
    "libnvidia-ml.so.1",
    "libnvidia-ptxjitcompiler.so",
    "libnvidia-ptxjitcompiler.so.1",
)


def _materialize_driver_links(platform_driver_lib_dir: str, shim_dir: str) -> None:
    os.makedirs(shim_dir, exist_ok=True)
    for lib in _NCU_DRIVER_LIBS:
        src = os.path.join(platform_driver_lib_dir, lib)
        dst = os.path.join(shim_dir, lib)
        if not os.path.exists(src):
            continue
        if os.path.lexists(dst) and not os.path.exists(dst):
            try:
                os.unlink(dst)
            except FileNotFoundError:
                pass
        if not os.path.lexists(dst):
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass


def _find_driver_lib(shim_dir: str, base: str) -> str | None:
    for candidate in (base, base + ".1"):
        path = os.path.join(shim_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def build_driver_shim(
    platform_driver_lib_dir: str, shim_dir: str = "/tmp/cuda_driver_ncu"
) -> dict[str, str]:
    """Create a narrow CUDA-driver shim dir and return the env NCU needs (mirrors genai).

    Symlinks ONLY the driver / NVML / ptxjit libs from the platform driver dir into
    ``shim_dir``, and returns ``LD_LIBRARY_PATH`` / ``LD_PRELOAD`` / ``TRITON_LIBCUDA_PATH``
    pointing at it. Putting the FULL platform driver dir on ``LD_LIBRARY_PATH`` crashes
    Python (D104906945); the narrow shim lets NCU connect without that. Returns ``{}`` if
    the platform driver dir is absent (e.g. local dev where the driver is already resolvable).
    """
    if not os.path.isdir(platform_driver_lib_dir):
        return {}
    _materialize_driver_links(platform_driver_lib_dir, shim_dir)
    libcuda = _find_driver_lib(shim_dir, "libcuda.so")
    if libcuda is None:
        raise FileNotFoundError(
            f"{platform_driver_lib_dir} contains neither libcuda.so nor libcuda.so.1"
        )
    preload = [libcuda]
    nvml = _find_driver_lib(shim_dir, "libnvidia-ml.so")
    if nvml is not None:
        preload.append(nvml)
    existing_path = os.environ.get("LD_LIBRARY_PATH")
    library_path = f"{shim_dir}:{existing_path}" if existing_path else shim_dir
    existing_preload = os.environ.get("LD_PRELOAD")
    if existing_preload:
        preload.append(existing_preload)
    return {
        "LD_LIBRARY_PATH": library_path,
        "LD_PRELOAD": ":".join(preload),
        "TRITON_LIBCUDA_PATH": libcuda,
    }


def ncu_wrap_argv(
    program_argv: list[str],
    *,
    ncu_bin: str,
    out_prefix: str,
    metrics: str = NCU_DEFAULT_METRICS,
    launch_count: int = 1,
    kernel_regex: str = "",
) -> list[str]:
    """Wrap ``program_argv`` with an ``ncu`` invocation.

    ``out_prefix`` gets ``_%p`` appended (one report per profiled process). Uses
    ``--target-processes all`` so ncu profiles torchrun / mp.spawn child ranks, and
    ``--launch-count`` to bound the number of profiled launches. Returns the full argv
    (``[ncu, ...ncu-flags..., *program_argv]``).
    """
    argv = [
        ncu_bin,
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--launch-count",
        str(launch_count),
        "--metrics",
        metrics,
        "-f",
        "-o",
        f"{out_prefix}_%p",
    ]
    if kernel_regex:
        argv += ["-k", kernel_regex, "--kernel-name-base", "function"]
    return argv + list(program_argv)
