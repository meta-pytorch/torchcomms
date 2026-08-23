# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""MAST launcher for the comms/dsl CuTe a2a benchmark (GB300).

One launcher, two image-delivery paths selected by ``--delivery``:

- ``--delivery fbpkg`` (canonical): submit a pre-built aarch64 GB300 ``fbpkg`` via
  torchx ``fb.dist.hpc`` one-process-per-rank (scheduler ``mast``). Needs a full
  ``buck2`` cross-build of the package (``--img`` required). Best for a stable,
  reproducible perf run.
- ``--delivery conda`` (fast iteration, default): the torchx ``mast_conda`` scheduler with a
  :class:`~torchx.specs.Workspace` source overlay -- the remote conda env comes from a
  pre-built conda ``fbpkg`` and the local ``fbcode/comms`` source is shipped at
  submit-time as an ephemeral workspace fbpkg (CAF upload, no buck compile). Iterate on
  the kernel and re-launch in seconds-to-a-minute.

Capacity / access: ``--entitlement`` (rmAttribution, a tenant leaf that holds GPU quota) and
``--identity`` / ``--oncall`` are caller/team-specific and have **no baked-in default** -- pass
them as flags or via the ``$MAST_RM_ATTRIBUTION`` / ``$MAST_HPC_IDENTITY`` / ``$MAST_HPC_ONCALL``
env vars (required for ``--submit``). ``--hw gb300_dsf`` is the sub-type MastProdCluster registers
(the quota path). ``--flex-pool-id`` defaults to '' (the quota path); pass a pool id only if you
have one -- flexpools are typically temporary, so none is hardcoded here.

NVL co-placement: MAST has no NVLink-domain locality flag, so the trainer role overlays
``HpcTaskGroupSpec.topologyConstraints = [{"domain": {"multiple": <nvl_hosts>}}]``.
``multiple`` is a HOST count, so 2 GB300 hosts (8 GPUs) in one NVL72 domain is
``--nvl-hosts 2``.

Dry-run by default (assembles + prints the MAST job spec -- and for conda, builds the
workspace fbpkg -- but consumes NO capacity). Pass ``--submit`` to schedule.

Examples::

    # conda fast-iteration dry-run (validate spec + workspace build; consumes no capacity)
    buck2 run @fbcode//mode/opt --prefer-local //comms/dsl/tests:mast_launch -- \\
        --delivery conda --env "A2A_CAPS=32"

    # fbpkg submit (needs a published --img and your entitlement/identity/oncall)
    buck2 run @fbcode//mode/opt --prefer-local //comms/dsl/tests:mast_launch -- \\
        --delivery fbpkg --img comms.dsl.benchmark_a2a.aarch64.gb300:<ver> \\
        --entitlement <tenant> --identity <data-project> --oncall <oncall> --submit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from typing import Any

import torchx.components.fb.conda as conda
from torchx.components.fb.dist import hpc as fb_dist_hpc
from torchx.runner import get_runner
from torchx.specs import AppDef, CfgVal, named_resources, Workspace
from torchx.specs.fb.named_resources import MAST_WHOLE_HOST_FEATURE

logger: logging.Logger = logging.getLogger(__name__)

_MAST_TIER_MAPPING: dict[str, str] = {
    "PROD": "mast.api.write",
    "STAGING": "mast.api.write.staging",
    "RC": "mast.api.write.rc",
}

# The conda workspace overlay names the ephemeral *workspace* fbpkg after role.image[0];
# reuse the already-ACL'd benchmark package so no separate `fbpkg create` is needed.
_DEFAULT_WS_PKG = "comms.dsl.benchmark_a2a.aarch64.gb300"
# Pre-built GB300 (aarch64, CUDA 13.0) conda env.
# @oss-disable[end= ]: _DEFAULT_CONDA = "xlformers_msl_rl_gb300_conda:20"
_DEFAULT_MODULE = "comms.dsl.tests.benchmark_a2a_cute"
# Source dirs overlaid into the ephemeral workspace, as ``relpath:remote`` (relpath is under
# fbcode). The benchmark only needs comms/dsl. `comms` is a PEP-420 namespace package, so
# mapping to "comms/dsl" keeps ``import comms.dsl.*`` working.
_DEFAULT_OVERLAYS = ["comms/dsl:comms/dsl"]


def _parse_envs(env_str: str | None) -> dict[str, str]:
    """Parse a ``K=V;K2=V2`` string into a dict (forwarded to every rank)."""
    envs: dict[str, str] = {}
    if not env_str:
        return envs
    for raw in env_str.split(";"):
        entry = raw.strip()
        if not entry:
            continue
        if "=" not in entry:
            logger.warning("skipping malformed --env entry (no '='): %r", entry)
            continue
        key, _, val = entry.partition("=")
        envs[key] = val
    return envs


def _job_name(args: argparse.Namespace) -> str:
    if args.job_name:
        return args.job_name
    tag = "conda-gb300" if args.delivery == "conda" else "gb300"
    return f"a2a-{tag}-n{args.nnode}x{args.ppn}-{uuid.uuid4().hex[0:6]}"


def _locality(args: argparse.Namespace) -> list[str] | None:
    """Region localityConstraints, or None. A flexpool already pins placement to its hosts,
    so a region constraint is dropped when a flexpool is set (avoids a region/pool clash)."""
    if args.flex_pool_id:
        return None
    return ["region", args.region] if args.region else None


def _fbcode_root() -> str:
    """Absolute path to this checkout's SOURCE ``fbcode`` dir (works in any worktree).

    The workspace overlay does a literal ``shutil.copytree`` of the project dir, so it
    MUST point at the source tree -- NOT the ``buck-out`` link-tree where ``__file__``
    lives when run via ``buck2 run``. ``buck2 run`` preserves the caller's cwd, so derive
    the root from cwd (or the buck-provided project root). Override with ``--src-fbcode``.
    """
    candidates = [os.environ.get("BUCK_PROJECT_ROOT"), os.getcwd()]
    for base in candidates:
        if not base:
            continue
        p = os.path.abspath(base)
        # cwd may be fbcode itself, a subdir of it, or the repo root (has fbcode/).
        while True:
            if os.path.basename(p) == "fbcode" and os.path.isdir(
                os.path.join(p, "comms")
            ):
                return p
            if os.path.isdir(os.path.join(p, "fbcode", "comms")):
                return os.path.join(p, "fbcode")
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
    raise RuntimeError(
        "could not locate source fbcode root from cwd; pass --src-fbcode <path>"
    )


def _build_fbpkg(args: argparse.Namespace) -> tuple[AppDef, dict[str, CfgVal]]:
    """Assemble the torchx AppDef + MAST cfg for the pre-built fbpkg delivery path."""
    if not args.img:
        raise ValueError("--delivery fbpkg requires --img (the benchmark fbpkg)")
    tier = _MAST_TIER_MAPPING.get(args.tier, args.tier)
    if not tier.startswith("mast."):
        raise ValueError(f"unrecognized MAST tier: {args.tier}")

    rsrc = named_resources[args.hw]
    env = _parse_envs(args.env)
    env.setdefault("LOGLEVEL", "INFO")
    # The fbpkg mount is read-only on the host; the CuTe JIT must write its compiled
    # artifacts to a writable cache dir.
    env.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    env["PPN"] = str(args.ppn)
    env["LOCAL_SIZE"] = str(rsrc.gpu)

    bench_args = args.bench_args.split() if args.bench_args else []
    job_name = _job_name(args)
    module = args.module

    app = fb_dist_hpc(
        *bench_args,
        m=module,
        img=args.img,
        name=job_name,
        h=args.hw,
        j=f"{args.nnode}x{args.ppn}",
        env=env,
        max_retries=args.max_retries,
        metadata={
            "rmAttribution": args.entitlement,
            "hpcIdentity": args.dp,
            "hpcClusterUuid": args.cluster,
            "hpcJobOncall": args.oncall,
            "smcTier": tier,
        },
    )

    # NVL co-placement: pin all hosts into one NVLink (NVL72) domain. multiple is a HOST
    # count (GB300 = 4 GPU/host), so 2 hosts = 8 GPUs in one domain.
    trainer = app.roles[0]  # pyre-ignore[16]
    tg_spec: dict[str, Any] = {
        "topologyConstraints": [{"domain": {"multiple": args.nvl_hosts}}],
    }
    if args.flex_pool_id:
        tg_spec["flexPoolId"] = args.flex_pool_id
    trainer.metadata["mast"] = {"HpcTaskGroupSpec": tg_spec}
    # Domain/topology constraints require whole-host allocation; ppn == GPUs/host already
    # selects the full resource, but set it explicitly for clarity.
    trainer.resource.capabilities[MAST_WHOLE_HOST_FEATURE] = True

    cfg: dict[str, CfgVal] = {
        "localityConstraints": _locality(args),
        "rmAttribution": args.entitlement,
        "hpcIdentity": args.dp,
        "hpcClusterUuid": args.cluster,
        "hpcJobOncall": args.oncall,
        "smcTier": tier,
        "runningTimeoutSec": args.timeout,
        "useStrictName": True,
        "enableGracefulPreemption": True,
        "runtimeAppMetadataJson": json.dumps(app.metadata),  # pyre-ignore[16]
    }
    if args.flex_pool_id:
        cfg["forceSingleRegion"] = False
    return app, cfg


def _build_conda(
    args: argparse.Namespace,
) -> tuple[AppDef, dict[str, CfgVal]]:
    """Assemble the torchx AppDef (conda.torchrun + workspace overlay) + mast_conda cfg."""
    tier = _MAST_TIER_MAPPING.get(args.tier, args.tier)
    if not tier.startswith("mast."):
        raise ValueError(f"unrecognized MAST tier: {args.tier}")

    env = _parse_envs(args.env)
    env.setdefault("LOGLEVEL", "INFO")
    # JIT must write its compiled artifacts to a writable dir (conda fbpkg is read-only).
    env.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    # The conda_on_mast default mounts an OILFS warm-storage dir; the a2a benchmark needs no
    # warm storage, and our region has no WS cluster mapping, so the scheduler rejects the
    # spec ("WarmStorageSpec ... missing clusters field"). Disable it.
    env.setdefault("DISABLE_OILFS", "1")
    env["PPN"] = str(args.ppn)

    bench_args = args.bench_args.split() if args.bench_args else []
    job_name = _job_name(args)
    module = args.module

    # torchrun runs `python -m <module> <prog_args>` one proc/rank.
    torchrun_args = [
        "--tee",
        "3",
        "--nnodes",
        str(args.nnode),
        "--nproc-per-node",
        str(args.ppn),
        "-m",
        module,
        *bench_args,
    ]

    job_spec = conda.torchrun(
        *torchrun_args,
        name=job_name,
        h=args.hw,
        env=env,
        run_as_root=True,
        torchrun_fbpkg_id="torchx_torchrun:stable",
        conda_mast_core_fbpkg_id="conda_mast_core:stable",
    )

    role = job_spec.roles[0]  # pyre-ignore[16]

    # Leave role.image as conda.torchrun produced it (torchx_torchrun + conda_mast_core).
    # The conda workspace scheduler ADDS the conda env (conda_fbpkg_id) and the freshly
    # built source-overlay (workspace_fbpkg_name) to the image itself via _get_image. Do
    # NOT also put them in role.image ("Can't have 2 versions of the same package!").

    # Source overlay: ship only the needed source dirs (fast CAF upload). Each --overlay is
    # `relpath:remote` (relpath under fbcode).
    fbcode_root = args.src_fbcode or _fbcode_root()
    projects: dict[str, str] = {}
    for entry in args.overlay:
        relpath, _, remote = entry.partition(":")
        projects[os.path.join(fbcode_root, relpath)] = remote or relpath
    role.workspace = Workspace(projects=projects)

    # NVL co-placement: pin all hosts into one NVLink (NVL72) domain (whole-host a2a).
    tg_spec: dict[str, Any] = {
        "topologyConstraints": [{"domain": {"multiple": args.nvl_hosts}}],
    }
    if args.flex_pool_id:
        tg_spec["flexPoolId"] = args.flex_pool_id
    role.metadata["mast"] = {"HpcTaskGroupSpec": tg_spec}
    role.resource.capabilities[MAST_WHOLE_HOST_FEATURE] = True

    cfg: dict[str, CfgVal] = {
        "localityConstraints": _locality(args),
        "rmAttribution": args.entitlement,
        "hpcIdentity": args.dp,
        "hpcClusterUuid": args.cluster,
        "hpcJobOncall": args.oncall,
        "smcTier": tier,
        "runningTimeoutSec": args.timeout,
        "useStrictName": True,
        "enableGracefulPreemption": True,
        # --- conda workspace scheduler knobs ---
        "conda_fbpkg_id": args.conda,
        "conda_path_in_fbpkg": "conda",
        "workspace_fbpkg_name": args.ws_pkg,
        "activate_conda": True,
        "git": False,
        "fbpkg_ids": [],
    }
    if args.flex_pool_id:
        cfg["forceSingleRegion"] = False
    return job_spec, cfg


def build_app_and_cfg(
    args: argparse.Namespace,
) -> tuple[AppDef, dict[str, CfgVal], str, str]:
    """Dispatch on ``--delivery``; return (app, cfg, scheduler, resolved_module)."""
    if args.delivery == "fbpkg":
        app, cfg = _build_fbpkg(args)
        return app, cfg, "mast", args.module
    app, cfg = _build_conda(args)
    return app, cfg, "mast_conda", args.module


def _print_spec_summary(args: argparse.Namespace, app: object, module: str) -> None:
    """Human-readable summary of the assembled job (shown even in dry-run)."""
    role = app.roles[0]  # pyre-ignore[16]
    tg = role.metadata["mast"]["HpcTaskGroupSpec"]
    print("=" * 80)
    print(f"MAST job spec (assembled, delivery={args.delivery}):")
    print(f"  name             = {app.name}")  # pyre-ignore[16]
    print(f"  module           = {module}")
    if args.delivery == "fbpkg":
        print(f"  img              = {args.img}")
    else:
        print(f"  overlay          = {args.overlay}")
        print(f"  role image       = {role.image}")
        print(f"  conda fbpkg      = {args.conda}")
        print(f"  workspace fbpkg  = {args.ws_pkg} (overlay: {role.workspace})")
    print(
        f"  hw / j           = {args.hw} / {args.nnode}x{args.ppn} "
        f"(world_size={args.nnode * args.ppn})"
    )
    print(f"  cluster          = {args.cluster}")
    print(f"  region           = {args.region}")
    print(f"  entitlement      = {args.entitlement}")
    print(f"  hpcIdentity      = {args.dp}")
    print(f"  oncall           = {args.oncall}")
    print(f"  topologyConstraints = {tg.get('topologyConstraints')}")
    print(f"  flexPoolId       = {tg.get('flexPoolId')}")
    print(f"  env              = {args.env}")
    print(f"  bench_args       = {args.bench_args}")
    print("=" * 80)


def launch(args: argparse.Namespace) -> None:
    # The access knobs are caller-specific (no baked default), so a real submit needs them; a
    # dry-run does not (it only assembles + prints the spec). Fail fast with a clear message.
    if args.submit:
        missing = [
            flag
            for flag, val in (
                ("--entitlement ($MAST_RM_ATTRIBUTION)", args.entitlement),
                ("--identity ($MAST_HPC_IDENTITY)", args.dp),
                ("--oncall ($MAST_HPC_ONCALL)", args.oncall),
            )
            if not val
        ]
        if missing:
            raise SystemExit(
                "--submit requires caller-specific access knobs with no default: "
                + ", ".join(missing)
                + ". Pass the flag(s) or export the env var(s)."
            )
    app, cfg, scheduler, module = build_app_and_cfg(args)
    _print_spec_summary(args, app, module)
    runner = get_runner()
    try:
        dryrun_info = runner.dryrun(app=app, scheduler=scheduler, cfg=cfg)
    except ValueError as e:
        # fbpkg dry-run: an unpublished fbpkg is expected (the spec itself is valid); surface
        # it instead of crashing so the summary above is still useful.
        if args.delivery == "fbpkg" and not args.submit and "do not exist" in str(e):
            print(f"DRY-RUN: spec validated up to packaging; fbpkg not published: {e}")
            return
        raise
    print(f"MAST dry-run (validated job request, scheduler={scheduler}):")
    print(dryrun_info)
    print("=" * 80)
    if not args.submit:
        print("DRY-RUN ONLY (pass --submit to schedule). No capacity consumed.")
        return
    app_handle = runner.schedule(dryrun_info)
    status = runner.status(app_handle)
    assert status, f"failed to get job status for {app_handle}"
    print(f"SUBMITTED: {app_handle}")
    print(f"UI: {status.ui_url}")


def _init_argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MAST launcher for the comms/dsl CuTe a2a benchmark (GB300)"
    )
    p.add_argument(
        "--delivery",
        default="conda",
        choices=["fbpkg", "conda"],
        help="image delivery: 'fbpkg' (pre-built package, needs --img) or 'conda' "
        "(fast-iteration conda env + workspace source overlay). Default: conda.",
    )
    p.add_argument("--nnode", type=int, default=2, help="number of hosts")
    p.add_argument(
        "--ppn",
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help="processes/GPUs per host (GB300 = 4)",
    )
    p.add_argument(
        "--nvl-hosts",
        type=int,
        default=2,
        help="hosts that must share one NVLink (NVL72) domain "
        "(topologyConstraints domain.multiple; HOST count, not GPU)",
    )
    p.add_argument(
        "--hw",
        default="gb300_dsf",
        help="torchx named resource. `gb300_dsf` is the sub-type MastProdCluster registers "
        "(the quota path).",
    )
    p.add_argument("--cluster", default="MastProdCluster", help="hpcClusterUuid")
    p.add_argument(
        "--region",
        default="uco",
        help="region for localityConstraints (ignored when --flex-pool-id is set)",
    )
    # Caller/team-specific access knobs: NO hardcoded default (this launcher lands for everyone).
    # Supply via flag or the $MAST_* env var; required for --submit (validated in launch()).
    p.add_argument(
        "--dp",
        "--identity",
        dest="dp",
        default=os.environ.get("MAST_HPC_IDENTITY"),
        help="hpcIdentity (data project / ACL). No default; flag or $MAST_HPC_IDENTITY.",
    )
    p.add_argument(
        "--entitlement",
        "--rm-attribution",
        dest="entitlement",
        default=os.environ.get("MAST_RM_ATTRIBUTION"),
        help="rmAttribution (entitlement = tenant leaf holding GPU quota). No default; flag "
        "or $MAST_RM_ATTRIBUTION.",
    )
    p.add_argument(
        "--oncall",
        default=os.environ.get("MAST_HPC_ONCALL"),
        help="hpcJobOncall. No default; flag or $MAST_HPC_ONCALL.",
    )
    p.add_argument(
        "--tier",
        default="PROD",
        choices=["PROD", "STAGING", "RC"],
        help="MAST environment",
    )
    p.add_argument("--timeout", type=int, default=3600, help="runningTimeoutSec")
    p.add_argument("--max-retries", type=int, default=1, help="MAST retries (fbpkg)")
    p.add_argument(
        "--flex-pool-id",
        default="",
        help="optional flexpool id for placement; default '' uses the quota path "
        "(--entitlement). Pass a pool id only if you have one.",
    )
    p.add_argument(
        "--module",
        default=_DEFAULT_MODULE,
        help=f"python -m module to run under torchrun (default: {_DEFAULT_MODULE})",
    )
    # --- fbpkg delivery ---
    p.add_argument(
        "--img",
        default=None,
        help="benchmark fbpkg (REQUIRED for --delivery fbpkg), e.g. "
        "comms.dsl.benchmark_a2a.aarch64.gb300:<ver>",
    )
    # --- conda delivery ---
    p.add_argument(
        "--conda",
        default=_DEFAULT_CONDA,
        help=f"conda env fbpkg (conda delivery; default: {_DEFAULT_CONDA})",
    )
    p.add_argument(
        "--ws-pkg",
        default=_DEFAULT_WS_PKG,
        help="fbpkg NAME used for the ephemeral source-overlay workspace build "
        f"(must be ACL-writable; default: {_DEFAULT_WS_PKG})",
    )
    p.add_argument(
        "--overlay",
        action="append",
        default=None,
        help="source dir to overlay as 'relpath:remote' (relpath under fbcode); "
        f"repeatable (conda delivery). Default: {_DEFAULT_OVERLAYS}.",
    )
    p.add_argument(
        "--src-fbcode",
        default=None,
        help="source fbcode dir to overlay comms/ from (default: auto-detect from cwd)",
    )
    p.add_argument("--job-name", default=None, help="override the generated job name")
    p.add_argument("--env", default=None, help="extra env as 'K=V;K2=V2' (per rank)")
    p.add_argument(
        "--bench-args",
        default=None,
        help="args forwarded to the module (space-separated)",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        default=False,
        help="actually schedule the job (default: dry-run only, no capacity used)",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _init_argparse().parse_args()
    if args.overlay is None:
        args.overlay = _DEFAULT_OVERLAYS
    launch(args)


if __name__ == "__main__":
    main()
