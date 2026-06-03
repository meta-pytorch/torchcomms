#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


logger = logging.getLogger(__name__)

DEFAULT_TARGET = "fbcode//comms/uniflow/benchmarks:uniflow_disagg_bench_mast"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_WORKSPACE = Path("/tmp/mast_empty_workspace")
DEFAULT_BHA_CONFIG = Path("/tmp/bha_uniflow_split_2x8.json")
RESULT_LOG_REGEX = (
    "CORRECTNESS|PERFORMANCE|ALL ACCURACY|FAILED|Traceback|RuntimeError|"
    "Exception|ValueError"
)
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
FAILURE_STATES = {"FAILED", "CANCELLED"}


CONNECTORS = {
    "uniflow": "UniflowConnector",
    "nixl": "NixlConnector",
}


@dataclass(frozen=True)
class Workload:
    model: str
    tp: int
    warmup_iters: int
    measure_iters: int
    prompt_len: int
    num_prompts: int
    output_len: int
    gpu_memory_utilization: float
    mode: str
    timeout: int


@dataclass(frozen=True)
class MastConfig:
    workspace: Path
    bha_config_path: Path
    hpc_identity: str
    hpc_cluster_uuid: str
    rm_attribution: str
    hpc_job_oncall: str
    locality_constraints: str
    model_type_name: str
    opec_tag: str
    host_type: str
    presto_identity: str
    poll_interval_sec: int
    job_timeout_sec: int

    @property
    def scheduler_args(self) -> str:
        values = {
            "hpcIdentity": self.hpc_identity,
            "hpcClusterUuid": self.hpc_cluster_uuid,
            "rmAttribution": self.rm_attribution,
            "hpcJobOncall": self.hpc_job_oncall,
            "localityConstraints": self.locality_constraints,
            "modelTypeName": self.model_type_name,
            "enableGracefulPreemption": "false",
            "maxJobFailures": "0",
            "opecTag": self.opec_tag,
        }
        return ",".join(f"{key}={value}" for key, value in values.items())

    @property
    def common_env(self) -> str:
        values = {
            "HYDRA_MAIN_MODULE": "vllm.fb.standalone_benchmark.disagg_entry",
            "LOGLEVEL": "WARNING",
            "PRESTO_CLIENT_IDENTITY": self.presto_identity,
            "PROCESS_MEMORY_AUTO_MEASUREMENT": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
        return ",".join(f"{key}={value}" for key, value in values.items())


@dataclass(frozen=True)
class JobSpec:
    connector_key: str
    connector_class: str
    layout: str
    name: str
    session_id: str

    @property
    def is_cross_node(self) -> bool:
        return self.layout == "cn"


@dataclass
class JobResult:
    spec: JobSpec
    app_uri: str
    state: str
    log_lines: list[str]

    @property
    def passed(self) -> bool:
        output = "\n".join(self.log_lines)
        return (
            self.state == "SUCCEEDED"
            and "CORRECTNESS PASSED" in output
            and "ALL ACCURACY TESTS PASSED" in output
            and "PERFORMANCE:" in output
            and "Traceback" not in output
            and "RuntimeError" not in output
            and "ValueError" not in output
        )


class CommandRunner:
    def __init__(self, *, quiet_warnings: bool) -> None:
        self.quiet_warnings = quiet_warnings

    def run_capture(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        suppress_stderr: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        stderr = subprocess.DEVNULL if suppress_stderr else subprocess.PIPE
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr,
            check=False,
        )

    def run_stream(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> tuple[int, list[str]]:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
        )
        assert process.stdout is not None
        lines: list[str] = []
        for line in process.stdout:
            line = line.rstrip()
            lines.append(line)
            if self._should_print(line):
                logger.info("%s", line)
        return process.wait(), lines

    def _should_print(self, line: str) -> bool:
        if not self.quiet_warnings:
            return True
        noisy_tokens = (
            "ThriftPyDeprecatedWarning",
            "Future automatic thrift-py-deprecated",
            "NumExpr",
        )
        return not any(token in line for token in noisy_tokens)


class FbpkgBuilder:
    def __init__(self, runner: CommandRunner, fbcode_dir: Path, target: str) -> None:
        self.runner = runner
        self.fbcode_dir = fbcode_dir
        self.target = target

    def build(self) -> str:
        logger.info("Building fbpkg target: %s", self.target)
        code, lines = self.runner.run_stream(
            ["fbpkg", "build", self.target], cwd=self.fbcode_dir
        )
        if code != 0:
            raise RuntimeError(
                f"`fbpkg build {self.target}` failed with exit code {code}"
            )
        package = self._extract_package(lines)
        logger.info("Built package: %s", package)
        return package

    @staticmethod
    def _extract_package(lines: Sequence[str]) -> str:
        pattern = re.compile(r"\b(uniflow_disagg_bench_mast:[0-9a-f]{32})\b")
        for line in reversed(lines):
            match = pattern.search(line)
            if match:
                return match.group(1)
        raise RuntimeError(
            "Unable to find `uniflow_disagg_bench_mast:<hash>` in fbpkg output"
        )


class MastBenchmarkRunner:
    def __init__(
        self,
        runner: CommandRunner,
        config: MastConfig,
        workload: Workload,
        package: str,
    ) -> None:
        self.runner = runner
        self.config = config
        self.workload = workload
        self.package = package

    def launch(self, spec: JobSpec) -> str:
        self._prepare_workspace()
        cmd = self._launch_command(spec)
        logger.info(
            "Launching %s %s: %s",
            spec.layout.upper(),
            spec.connector_class,
            spec.name,
        )
        code, lines = self.runner.run_stream(
            cmd, cwd=self.config.workspace, env=self._torchx_env()
        )
        if code != 0:
            raise RuntimeError(
                f"`torchx run` failed for {spec.name} with exit code {code}"
            )
        app_uri = self._extract_app_uri(lines)
        logger.info("Launched %s: %s", spec.name, app_uri)
        return app_uri

    def wait_for_result(self, spec: JobSpec, app_uri: str) -> JobResult:
        deadline = time.monotonic() + self.config.job_timeout_sec
        last_state = "UNKNOWN"
        while time.monotonic() < deadline:
            state = self._status(app_uri)
            if state != last_state:
                logger.info("%s: %s", spec.name, state)
                last_state = state
            if state in TERMINAL_STATES:
                return JobResult(
                    spec=spec,
                    app_uri=app_uri,
                    state=state,
                    log_lines=self._collect_logs(spec, app_uri),
                )
            time.sleep(self.config.poll_interval_sec)
        return JobResult(
            spec=spec,
            app_uri=app_uri,
            state=f"TIMEOUT_AFTER_{self.config.job_timeout_sec}s",
            log_lines=self._collect_logs(spec, app_uri),
        )

    def _launch_command(self, spec: JobSpec) -> list[str]:
        layout = "2x8" if spec.is_cross_node else "1x2"
        rdzv_backend = "zeus" if spec.is_cross_node else "c10d"
        env = f"{self.config.common_env},TORCHX_INTERNAL_SESSION_ID={spec.session_id}"
        cmd = [
            "torchx",
            "run",
            "--no-repro-info",
            "-s",
            "mast",
            "--scheduler_args",
            self.config.scheduler_args,
            "fb.dist.ddp",
            "--name",
            spec.name,
            "--img",
            self.package,
            "-h",
            self.config.host_type,
            "-j",
            layout,
            "--env",
            env,
            "--rdzv_backend",
            rdzv_backend,
            "-m",
            "vllm.fb.standalone_benchmark.disagg_entry",
            "--",
            "--connector",
            spec.connector_class,
            "--tp",
            str(self.workload.tp),
            "--model",
            self.workload.model,
            "--gpu-memory-utilization",
            str(self.workload.gpu_memory_utilization),
            "--warmup-iters",
            str(self.workload.warmup_iters),
            "--measure-iters",
            str(self.workload.measure_iters),
            "--prompt-len",
            str(self.workload.prompt_len),
            "--num-prompts",
            str(self.workload.num_prompts),
            "--output-len",
            str(self.workload.output_len),
            "--mode",
            self.workload.mode,
            "--timeout",
            str(self.workload.timeout),
        ]
        if spec.is_cross_node:
            cmd.insert(cmd.index("--env"), "--bha_config_path")
            cmd.insert(cmd.index("--env"), str(self.config.bha_config_path))
        return cmd

    def _prepare_workspace(self) -> None:
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        if not self.config.bha_config_path.exists():
            self.config.bha_config_path.write_text(
                '{\n  "trainer": [\n    {\n      "domain": {\n        "multiple": 2,\n'
                '        "maxGroupCount": 1\n      }\n    }\n  ]\n}\n'
            )

    def _status(self, app_uri: str) -> str:
        result = self.runner.run_capture(
            ["torchx", "status", app_uri],
            cwd=self.config.workspace,
            env=self._torchx_env(),
            suppress_stderr=True,
        )
        if result.returncode != 0:
            return "STATUS_ERROR"
        match = re.search(r"State:\s+([A-Z_]+)", result.stdout)
        return match.group(1) if match else "UNKNOWN"

    def _collect_logs(self, spec: JobSpec, app_uri: str) -> list[str]:
        logs: list[str] = []
        for role in self._trainer_roles(spec):
            result = self.runner.run_capture(
                ["torchx", "log", "--regex", RESULT_LOG_REGEX, f"{app_uri}/{role}"],
                cwd=self.config.workspace,
                env=self._torchx_env(),
                suppress_stderr=True,
            )
            if result.stdout:
                logs.extend(line for line in result.stdout.splitlines() if line)
        return logs

    @staticmethod
    def _trainer_roles(spec: JobSpec) -> list[str]:
        return ["trainer/0", "trainer/1"] if spec.is_cross_node else ["trainer/0"]

    @staticmethod
    def _extract_app_uri(lines: Sequence[str]) -> str:
        for line in lines:
            match = re.search(r"(mast://torchx/\S+)", line)
            if match:
                return match.group(1)
        raise RuntimeError("Unable to find launched `mast://torchx/...` app URI")

    @staticmethod
    def _torchx_env() -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["PYTHON_EXEC"] = "./mast_wrapper.sh"
        return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the uniflow benchmark fbpkg, launch MAST jobs, poll status, and summarize results."
    )
    parser.add_argument(
        "--connector", choices=["uniflow", "nixl", "both"], default="both"
    )
    parser.add_argument("--layout", choices=["sn", "cn", "both"], default="both")
    parser.add_argument(
        "--package",
        help="Existing `uniflow_disagg_bench_mast:<hash>` package. Skips build when set.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Require --package and skip fbpkg build.",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--fbcode-dir", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--bha-config-path", type=Path, default=DEFAULT_BHA_CONFIG)
    parser.add_argument("--job-prefix", default="acc-h100")
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--job-timeout-sec", type=int, default=3600)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=250)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--mode", default="accuracy")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--host-type", default="grandteton_80g_roce")
    parser.add_argument("--hpc-identity", default="networkai_mast_job_identity")
    parser.add_argument("--hpc-cluster-uuid", default="MastGenAICluster")
    parser.add_argument("--rm-attribution", default="msl_infra_optimization")
    parser.add_argument("--hpc-job-oncall", default="ncclx")
    parser.add_argument("--locality-constraints", default="region;eag")
    parser.add_argument("--model-type-name", default="gen_ai_default")
    parser.add_argument("--opec-tag", default="DEDICATED_ONLY")
    parser.add_argument(
        "--presto-identity", default="svc:chronos_secgrp_networkai_mast_job_identity"
    )
    parser.add_argument(
        "--quiet-warnings", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def make_job_specs(args: argparse.Namespace, package: str) -> list[JobSpec]:
    connectors = list(CONNECTORS) if args.connector == "both" else [args.connector]
    layouts = ["sn", "cn"] if args.layout == "both" else [args.layout]
    package_suffix = package_hash_suffix(package)
    run_suffix = time.strftime("%m%d%H%M%S")
    specs: list[JobSpec] = []
    for connector in connectors:
        for layout in layouts:
            connector_short = "uni" if connector == "uniflow" else "nixl"
            name = f"{args.job_prefix}-{layout}-{connector_short}-{package_suffix}-{run_suffix}"
            session_id = f"codex-{package_suffix}-{layout}-{connector}"
            specs.append(
                JobSpec(
                    connector_key=connector,
                    connector_class=CONNECTORS[connector],
                    layout=layout,
                    name=name,
                    session_id=session_id,
                )
            )
    return specs


def package_hash_suffix(package: str) -> str:
    parts = package.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"--package must use the fbpkg `name:hash` format, got {package!r}"
        )
    return parts[1][:7]


def print_summary(package: str, results: Sequence[JobResult]) -> None:
    logger.info("\nSummary")
    logger.info("Package: %s", package)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "%s: %s %s %s %s",
            status,
            result.spec.layout.upper(),
            result.spec.connector_class,
            result.state,
            result.app_uri,
        )
        for line in result.log_lines:
            logger.info("  %s", line)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.skip_build and not args.package:
        raise SystemExit("--skip-build requires --package")

    runner = CommandRunner(quiet_warnings=args.quiet_warnings)
    package = args.package or FbpkgBuilder(runner, args.fbcode_dir, args.target).build()
    workload = Workload(
        model=args.model,
        tp=args.tp,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        prompt_len=args.prompt_len,
        num_prompts=args.num_prompts,
        output_len=args.output_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mode=args.mode,
        timeout=args.timeout,
    )
    mast_config = MastConfig(
        workspace=args.workspace,
        bha_config_path=args.bha_config_path,
        hpc_identity=args.hpc_identity,
        hpc_cluster_uuid=args.hpc_cluster_uuid,
        rm_attribution=args.rm_attribution,
        hpc_job_oncall=args.hpc_job_oncall,
        locality_constraints=args.locality_constraints,
        model_type_name=args.model_type_name,
        opec_tag=args.opec_tag,
        host_type=args.host_type,
        presto_identity=args.presto_identity,
        poll_interval_sec=args.poll_interval_sec,
        job_timeout_sec=args.job_timeout_sec,
    )
    mast = MastBenchmarkRunner(runner, mast_config, workload, package)

    launched: list[tuple[JobSpec, str]] = []
    for spec in make_job_specs(args, package):
        launched.append((spec, mast.launch(spec)))

    results = [mast.wait_for_result(spec, app_uri) for spec, app_uri in launched]
    print_summary(package, results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
