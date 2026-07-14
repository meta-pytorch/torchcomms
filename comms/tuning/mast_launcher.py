# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""MAST launcher for source-based Triton comm tuning jobs."""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torchx.specs import AppDef

_BLACKWELL_AARCH64_INSTANCE_TYPES = {
    "gb200",
    "gb300",
    "gcp_gb300",
}

_H100_INSTANCE_ALIASES = {
    "h100": "grandteton_80g_roce",
}
_H100_INSTANCE_TYPES = {
    "grandteton_80g_roce",
    "grandteton_80g_ib",
    "oci_h100",
    "oci_h100_8",
}

_NCU_BASE_IMAGE_FBPKG = "tupperware.image.sendstream.mast_environments:stable"
_MOUNT_SCRIPT = "/packages/conda_mast_core/mount/mount.sh"


def _normalize_instance_type(instance_type: str) -> str:
    return _H100_INSTANCE_ALIASES.get(instance_type, instance_type)


def _default_region(instance_type: str) -> str:
    return {
        "grandteton_80g_roce": "eag",
        "grandteton_80g_ib": "pci",
        "t1": "eag",
        "gb200": "nao",
        "gb300": "lco",
        "gcp_gb300": "oce",
        "oci_h100": "oas",
        "oci_h100_8": "oas",
    }.get(instance_type, "pci")


def _driver_paths(
    instance_type: str,
    conda_package: str,
    *,
    preload_gb200_glog: bool,
) -> tuple[str, str, str, str | None, str, str]:
    if instance_type in _BLACKWELL_AARCH64_INSTANCE_TYPES:
        platform_driver_lib_dir = "/usr/local/fbcode/platform010-aarch64/lib"
        compiler_triple = "aarch64-conda-linux-gnu"
        triton_ptxas_path = f"/packages/{conda_package}/conda/bin/ptxas"
    else:
        platform_driver_lib_dir = "/usr/local/fbcode/platform010/lib"
        compiler_triple = "x86_64-conda-linux-gnu"
        triton_ptxas_path = None

    ld_preload = (
        f"{platform_driver_lib_dir}/libcuda.so:"
        f"{platform_driver_lib_dir}/libnvidia-ml.so"
    )
    if instance_type == "gb200" and preload_gb200_glog:
        ld_preload += f":/packages/{conda_package}/conda/lib/libglog.so"

    triton_libcuda_path = f"{platform_driver_lib_dir}/libcuda.so"
    cc_path = f"/packages/{conda_package}/conda/bin/{compiler_triple}-gcc"
    cxx_path = f"/packages/{conda_package}/conda/bin/{compiler_triple}-g++"
    return (
        platform_driver_lib_dir,
        ld_preload,
        triton_libcuda_path,
        triton_ptxas_path,
        cc_path,
        cxx_path,
    )


def _parse_env_vars(env: str) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not env:
        return env_vars
    for entry in env.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Expected KEY=VALUE in --env entry, got {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"Expected non-empty env var name in --env entry {entry!r}"
            )
        env_vars[key] = value
    return env_vars


def _cuda_env_setup(
    platform_driver_lib_dir: str,
    cuda_lib_path: str,
    ld_preload: str,
    triton_libcuda_path: str,
) -> str:
    return (
        f"export LD_LIBRARY_PATH={shlex.quote(cuda_lib_path)}:${{LD_LIBRARY_PATH:-}} && "
        f"export LD_PRELOAD={shlex.quote(ld_preload)} && "
        f"export TRITON_LIBCUDA_PATH={shlex.quote(triton_libcuda_path)}"
    )


def _build_ncu_workload_cmd(
    ncu_path: str,
    ncu_metrics: str,
    ncu_kernel_regex: str,
    ncu_launch_count: int,
    python_workload_cmd: str,
    workspace_package_name: str,
) -> str:
    if ncu_path:
        ncu_bin_setup = [f"NCU_BIN={shlex.quote(ncu_path)}"]
    else:
        ncu_bin_setup = [
            'NCU_BIN="$(command -v ncu || true)"',
            (
                'if [ -z "$NCU_BIN" ]; then for candidate in '
                "/usr/local/cuda-13.0/bin/ncu "
                "/usr/local/cuda-12.8/bin/ncu "
                "/opt/nvidia/nsight-compute/2025.3.1/ncu "
                "/opt/nvidia/nsight-compute/2025.1.1/ncu "
                "/usr/local/cuda*/bin/ncu "
                "/opt/nvidia/nsight-compute/*/ncu "
                "/usr/local/NVIDIA-Nsight-Compute*/ncu; do "
                'if [ -x "$candidate" ]; then NCU_BIN="$candidate"; break; fi; '
                "done; fi"
            ),
            'if [ -z "$NCU_BIN" ]; then echo "ncu not found" >&2; exit 1; fi',
        ]

    ncu_setup_parts = [
        "mkdir -p $DUMP_DIR/ncu",
        *ncu_bin_setup,
        'echo "NCU_BIN=$NCU_BIN"',
        '"$NCU_BIN" --version',
    ]
    ncu_cmd_parts = [
        "env TORCH_SYMM_MEM_DISABLE_MULTICAST=1",
        '"$NCU_BIN"',
        "--replay-mode kernel",
        "--target-processes all",
        "-f",
        "-o",
        "$DUMP_DIR/ncu/ncu_%p",
        "--metrics",
        shlex.quote(ncu_metrics),
        "--launch-count",
        str(ncu_launch_count),
    ]
    if ncu_kernel_regex:
        ncu_cmd_parts.extend(
            ["-k", shlex.quote(ncu_kernel_regex), "--kernel-name-base", "function"]
        )
    ncu_cmd_parts.append(
        f"env PYTHONPATH=/packages/{shlex.quote(workspace_package_name)} "
        f"{python_workload_cmd}"
    )
    return " && ".join(ncu_setup_parts + [" ".join(ncu_cmd_parts)])


def _copytree(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {
            name for name in names if name == "__pycache__" or name.endswith(".pyc")
        }

    shutil.copytree(src, dst, ignore=ignore)


@contextmanager
def _packaging_workspace(
    *,
    workspace_root: Path,
    workspace_toplevels: tuple[str, ...],
) -> Iterator[Path]:
    """Build a minimal source workspace containing caller code and comm_tuning."""
    with tempfile.TemporaryDirectory(prefix="comm_tuning_workspace_") as tmp:
        workspace = Path(tmp)
        for name in workspace_toplevels:
            src = workspace_root / name
            if not src.exists():
                raise FileNotFoundError(
                    f"workspace top-level path does not exist: {src}"
                )
            dst = workspace / name
            if src.is_dir():
                _copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        comm_tuning_dst = workspace / "comm_tuning"
        _copytree(Path(__file__).resolve().parent / "comm_tuning", comm_tuning_dst)
        yield workspace


def _scheduler_args(
    *,
    conda_fbpkg_id: str,
    hpc_cluster_uuid: str,
    hpc_identity: str,
    region: str,
    tenant: str,
    workspace_package_name: str,
    workspace_fbpkg_id: Optional[str],
    priority: Optional[str],
) -> dict[str, bool | str]:
    scheduler_args: dict[str, bool | str] = {
        "activate_conda": False,
        "conda_fbpkg_id": conda_fbpkg_id,
        "conda_pack_ignore_missing_files": True,
        "conda_path_in_fbpkg": "conda",
        "git": False,
        "hpcClusterUuid": hpc_cluster_uuid,
        "hpcIdentity": hpc_identity,
        "hpcJobOncall": "ncclx",
        "localityConstraints": f"region;{region};rmAttribution={tenant}",
        "use_caf": False,
        "workspace_fbpkg_name": workspace_package_name,
        "fbpkg_ids": "cif:stable,clicat:stable,torchx_cli:stable,dsi.logger.cat:prod,oil.oilfs:prod,conda_mast_core:stable",
    }
    if workspace_fbpkg_id is not None:
        scheduler_args["workspace_fbpkg_id"] = workspace_fbpkg_id
    if priority is not None:
        scheduler_args["jobPriority"] = priority
    return scheduler_args


def _target_kwargs(args: argparse.Namespace, conda_package_name: str) -> dict[str, str]:
    target_kwargs = {
        "--script": args.script,
        "--instance_type": _normalize_instance_type(args.instance_type),
        "--region": args.region,
        "--nnodes": str(args.nnodes),
        "--nproc_per_node": str(args.nproc_per_node),
        "--max_domain_multiple": str(args.max_domain_multiple),
        "--conda_package": conda_package_name,
        "--workspace_package_name": args.workspace_package_name,
    }
    if args.oilfs_profile:
        target_kwargs["--oilfs_profile"] = args.oilfs_profile
    if args.preload_gb200_glog:
        target_kwargs["--preload_gb200_glog"] = "true"
    if args.script_args:
        target_kwargs["--script_args"] = args.script_args
    if args.env:
        target_kwargs["--env"] = args.env
    if args.ncu:
        target_kwargs.update(
            {
                "--ncu": "true",
                "--ncu_path": args.ncu_path,
                "--ncu_metrics": args.ncu_metrics,
                "--ncu_kernel_regex": args.ncu_kernel_regex,
                "--ncu_launch_count": str(args.ncu_launch_count),
            }
        )
    if args.hpc_base_image_fbpkg_id:
        target_kwargs["--hpc_base_image_fbpkg_id"] = args.hpc_base_image_fbpkg_id
    return target_kwargs


def _torchx_cmd(
    *,
    scheduler: str,
    scheduler_args: dict[str, bool | str],
    target_kwargs: dict[str, str],
    workspace: Path,
    dry_run: bool,
) -> list[str]:
    scheduler_args_str = ",".join(f"{k}={v}" for k, v in scheduler_args.items())
    target = f"{Path(__file__).resolve()}:submit_job"
    cmd = [
        "torchx",
        "run",
        f"--scheduler={scheduler}",
        f"--scheduler_args={scheduler_args_str}",
        "--workspace",
        str(workspace),
    ]
    if dry_run:
        cmd.append("--dryrun")
    cmd.append(target)
    for k, v in target_kwargs.items():
        cmd.append(f"{k}={v}")
    return cmd


def _run_torchx_cmd(cmd: list[str], interactive: bool) -> tuple[str, str]:
    env = os.environ.copy()
    with tempfile.NamedTemporaryFile() as torchx_config:
        env["TORCHXCONFIG"] = torchx_config.name
        env.pop("LD_PRELOAD", None)
        env.pop("LD_LIBRARY_PATH", None)
        if interactive:
            env["TORCHX_INTERACTIVE"] = "1"

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    return result.stdout, result.stderr


def submit_job(
    script: str,
    instance_type: str = "gb200",
    region: str = "nao",
    nnodes: int = 1,
    nproc_per_node: int = 2,
    max_domain_multiple: int = 16,
    script_args: str = "",
    conda_package: str = "",
    workspace_package_name: str = "",
    oilfs_profile: str = "",
    preload_gb200_glog: bool = False,
    ncu: bool = False,
    ncu_path: str = "",
    ncu_metrics: str = "sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread",
    ncu_kernel_regex: str = "",
    ncu_launch_count: int = 1,
    hpc_base_image_fbpkg_id: str = "",
    env: str = "",
) -> "AppDef":
    import torchx.components.fb.conda as conda
    from torchx.specs.fb.named_resources import MAST_WHOLE_HOST_FEATURE

    instance_type = _normalize_instance_type(instance_type)
    if not conda_package:
        raise ValueError("conda_package is required")
    if not workspace_package_name:
        raise ValueError("workspace_package_name is required")

    script_name = os.path.splitext(os.path.basename(script))[0]
    script_args_list = script_args.split() if script_args else []
    script_args_list.append("--torchx")
    script_args_str = " ".join(script_args_list)

    python_path = f"/packages/{conda_package}/conda/bin/python"
    cuda_lib_path = f"/packages/{conda_package}/conda/lib"

    (
        platform_driver_lib_dir,
        ld_preload,
        triton_libcuda_path,
        triton_ptxas_path,
        cc_path,
        cxx_path,
    ) = _driver_paths(
        instance_type,
        conda_package,
        preload_gb200_glog=preload_gb200_glog,
    )

    python_workload_cmd = (
        f"{shlex.quote(python_path)} -u ops/{shlex.quote(script)} {script_args_str}"
    )
    if ncu:
        workload_cmd = _build_ncu_workload_cmd(
            ncu_path,
            ncu_metrics,
            ncu_kernel_regex,
            ncu_launch_count,
            python_workload_cmd,
            workspace_package_name,
        )
    else:
        workload_cmd = (
            f"PYTHONPATH=/packages/{shlex.quote(workspace_package_name)} "
            f"{python_workload_cmd}"
        )

    cuda_env_setup = _cuda_env_setup(
        platform_driver_lib_dir,
        cuda_lib_path,
        ld_preload,
        triton_libcuda_path,
    )
    bash_cmd = (
        f"{cuda_env_setup} && "
        "mkdir -p $DUMP_DIR && "
        f"cd /packages/{shlex.quote(workspace_package_name)} && "
        f"{workload_cmd}"
    )

    job_env = {
        "DUMP_DIR": "/mnt/wsfuse/outputs/${app_id}",
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "CC": cc_path,
        "CXX": cxx_path,
        "NCCL_DEBUG": "WARN",
        "TORCH_CPP_LOG_LEVEL": "WARNING",
    }
    if oilfs_profile:
        job_env["OILFS_PROFILE"] = oilfs_profile
    job_env.update(_parse_env_vars(env))

    job_spec = conda.torchrun(
        "--tee",
        "3",
        "--role",
        "trainer",
        "--nnodes",
        f"{nnodes}",
        "--nproc-per-node",
        f"{nproc_per_node}",
        "--no-python",
        "bash",
        "-c",
        bash_cmd,
        h=instance_type,
        run_as_root=True,
        env=job_env,
    )
    if triton_ptxas_path is not None:
        job_spec.roles[0].env["TRITON_PTXAS_PATH"] = triton_ptxas_path

    job_spec.name = script_name
    if not hpc_base_image_fbpkg_id and ncu:
        hpc_base_image_fbpkg_id = _NCU_BASE_IMAGE_FBPKG

    if instance_type in _BLACKWELL_AARCH64_INSTANCE_TYPES and (
        nnodes * nproc_per_node >= 2
    ):
        job_spec.roles[0].resource.capabilities[MAST_WHOLE_HOST_FEATURE] = True

    hpc_task_group_spec = (
        job_spec.roles[0]
        .metadata.setdefault("mast", {})
        .setdefault("HpcTaskGroupSpec", {})
    )
    if hpc_base_image_fbpkg_id:
        hpc_task_group_spec["baseImage"] = {
            "baseImagePackage": {"fbpkgIdentifier": hpc_base_image_fbpkg_id}
        }
    if instance_type in _BLACKWELL_AARCH64_INSTANCE_TYPES:
        hpc_task_group_spec["topologyConstraints"] = [
            {
                "domain": {
                    "multiple": min(nnodes, max_domain_multiple),
                    "quorumBuffer": 0,
                }
            }
        ]

    job_spec.roles[0].env["TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"] = "[${rank}]:"
    mount_wsfuse = (
        f"({_MOUNT_SCRIPT} || "
        'echo "WARNING: failed to run /mnt/wsfuse mount helper; outputs may not persist." >&2) && '
        "if ! mountpoint -q /mnt/wsfuse 2>/dev/null && ! grep -q ' /mnt/wsfuse ' /proc/mounts; then "
        'echo "WARNING: /mnt/wsfuse is not mounted. Outputs under $DUMP_DIR may not persist." >&2; '
        'echo "WARNING: check warm-storage ACLs for FUSE_SRC=${FUSE_SRC:-unset} and hpcIdentity if outputs are missing." >&2; '
        "fi"
    )
    job_spec.roles[0].entrypoint = f"{mount_wsfuse} && {job_spec.roles[0].entrypoint}"
    job_spec.metadata["tags"] = ",".join(["comm_tuning", script])
    return job_spec


def launch_mast(args: argparse.Namespace) -> tuple[str, str]:
    instance_type = _normalize_instance_type(args.instance_type)
    if args.region is None:
        args.region = _default_region(instance_type)

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    workspace_toplevels = tuple(
        part.strip() for part in args.workspace_toplevels.split(",") if part.strip()
    )
    conda_package_name = args.conda_fbpkg_id.split(":")[0]
    scheduler_args = _scheduler_args(
        conda_fbpkg_id=args.conda_fbpkg_id,
        hpc_cluster_uuid=args.hpc_cluster_uuid,
        hpc_identity=args.hpcIdentity,
        region=args.region,
        tenant=args.tenant,
        workspace_package_name=args.workspace_package_name,
        workspace_fbpkg_id=args.workspace_fbpkg_id,
        priority=args.priority,
    )
    target_kwargs = _target_kwargs(args, conda_package_name)

    with _packaging_workspace(
        workspace_root=workspace_root,
        workspace_toplevels=workspace_toplevels,
    ) as workspace:
        cmd = _torchx_cmd(
            scheduler=args.scheduler,
            scheduler_args=scheduler_args,
            target_kwargs=target_kwargs,
            workspace=workspace,
            dry_run=args.dry_run,
        )
        print(f"\nFull torchx run command:\n{' '.join(cmd)}\n")
        stdout_content, stderr_content = _run_torchx_cmd(cmd, args.interactive)

    if stdout_content:
        print(stdout_content, end="")
    if stderr_content:
        print(stderr_content, end="", file=sys.stderr)

    if args.dry_run:
        return "<dry_run>", "<dry_run>"

    pat = rf"{re.escape(args.scheduler)}://torchx/([^\s]+)"
    for line in stdout_content.splitlines():
        if match := re.search(pat, line):
            job_id = match.group(1)
            return job_id, f"/mnt/wsfuse/outputs/{job_id}"
    raise ValueError("Unable to extract job name from torchx run output.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Triton comm tuning jobs on MAST.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instance_type", type=str, required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--script_args", type=str, default="")
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--tenant", type=str, required=True)
    parser.add_argument("--hpcIdentity", type=str, required=True)
    parser.add_argument("--hpc_cluster_uuid", type=str, required=True)
    parser.add_argument("--conda_fbpkg_id", type=str, required=True)
    parser.add_argument("--workspace_package_name", type=str, required=True)
    parser.add_argument("--workspace_fbpkg_id", type=str, default=None)
    parser.add_argument(
        "--workspace_root",
        type=str,
        required=True,
        help="Caller source workspace to package.",
    )
    parser.add_argument(
        "--workspace_toplevels",
        type=str,
        required=True,
        help="Comma-separated top-level workspace paths copied into the MAST package.",
    )
    parser.add_argument("--oilfs_profile", type=str, default="")
    parser.add_argument("--max_domain_multiple", type=int, default=32)
    parser.add_argument("--priority", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--preload_gb200_glog",
        action="store_true",
        help="Preload conda libglog on GB200. Disabled by default due duplicate flag registration failures.",
    )
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--ncu_path", type=str, default="")
    parser.add_argument(
        "--ncu_metrics",
        type=str,
        default="sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread",
    )
    parser.add_argument("--ncu_kernel_regex", type=str, default="")
    parser.add_argument("--ncu_launch_count", type=int, default=1)
    parser.add_argument("--hpc_base_image_fbpkg_id", type=str, default=None)
    parser.add_argument("--env", type=str, default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    job_id, dump_dir = launch_mast(args)
    print("\n" + "=" * 80)
    print("MAST job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"\nOutputs written to: {dump_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
