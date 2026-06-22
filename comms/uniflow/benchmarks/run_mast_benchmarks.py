#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TypedDict


logger = logging.getLogger(__name__)

DEFAULT_TARGET = "fbcode//comms/uniflow/benchmarks:uniflow_disagg_bench_mast"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_WORKSPACE = Path("/tmp/mast_empty_workspace")
DEFAULT_BHA_CONFIG = Path("/tmp/bha_uniflow_split_2x8.json")
RESULT_LOG_REGEX = (
    "BENCHMARK_RESULT_JSON|BENCHMARK_PROGRESS_JSON|UNIFLOW_TRANSFER_STATS_JSON|"
    "CORRECTNESS|PERFORMANCE|ALL ACCURACY|DECODE"
)
FAILURE_LOG_REGEX = (
    "FAILED|Traceback|RuntimeError|Exception|ValueError|MISMATCH|"
    "Retrying UniFlow|Recovered UniFlow|Uniflow transfer failure|"
    "UniFlow transfer telemetry export failed|expired|Xid|CUDA error|"
    "CUDA-capable|NCCL.*(error|failed|failure|timeout|abort)|OOM|"
    "OutOfMemory|TimeoutError|Assertion|ChildFailedError|died unexpectedly|"
    "cuda_ipc_md|ucs_fatal"
)
TRACEBACK_LOG_REGEX = (
    'Traceback|File ".*", line|\\]:    |RuntimeError|Exception|ValueError|'
    "TimeoutError|AssertionError|ChildFailedError|Root Cause|Failures:|"
    "exitcode|exit code|Signal|SIG[A-Z]+|rank *:|local_rank|pid *:|"
    "error_file|ProcessGroupGloo|DistNetworkError|Gloo|connectFullMesh|"
    "Address already in use|Connection refused|Connection reset|Broken pipe|"
    "No route to host|Timed out|timed out"
)
PROGRESS_LOG_REGEX = "BENCHMARK_PROGRESS_JSON|Processed prompts|read_blocks"
WATCHDOG_PROGRESS_LOG_REGEX = "BENCHMARK_PROGRESS_JSON"
WATCHDOG_FALLBACK_PROGRESS_LOG_REGEX = "read_blocks"
BENCHMARK_RESULT_PREFIX = "BENCHMARK_RESULT_JSON: "
UNIFLOW_TRANSFER_STATS_PREFIX = "UNIFLOW_TRANSFER_STATS_JSON: "
ALLOC_CONF_TOKEN_RE = re.compile(r"-(?P<token>[A-Za-z0-9_]+)alloc-")
ALLOC_CONF_TOKEN_PREFIX = "hex_"
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
FAILURE_STATES = {"FAILED", "CANCELLED"}
MAX_CONSECUTIVE_STATUS_ERRORS = 10
MAX_PROGRESS_LOG_LINES = 80
TORCHX_CLI_RETRIES = 3
TORCHX_CLI_RETRY_BACKOFF_SEC = 5
TORCHX_CANCEL_TIMEOUT_SEC = 120
TORCHX_LOG_TIMEOUT_SEC = 300
TORCHX_STATUS_TIMEOUT_SEC = 120


CONNECTORS = {
    "uniflow": "UniflowConnector",
    "nixl": "NixlConnector",
}
NIXL_KV_BUFFER_DEVICES = ("auto", "cuda", "cpu")


class AllocConfToken:
    @staticmethod
    def encode(pytorch_cuda_alloc_conf: str) -> str:
        if pytorch_cuda_alloc_conf == "expandable_segments:True":
            return "expandable"
        if pytorch_cuda_alloc_conf == "default":
            return "default"
        return f"{ALLOC_CONF_TOKEN_PREFIX}{pytorch_cuda_alloc_conf.encode().hex()}"

    @staticmethod
    def decode(token: str) -> str:
        if token == "default":
            return "default"
        if not token.startswith(ALLOC_CONF_TOKEN_PREFIX):
            raise ValueError(f"Unrecognized alloc-conf token `{token}`")
        encoded = token[len(ALLOC_CONF_TOKEN_PREFIX) :]
        try:
            return bytes.fromhex(encoded).decode()
        except ValueError as exc:
            raise ValueError(f"Invalid alloc-conf token `{token}`") from exc


class LogSummaryRecord(TypedDict):
    structured_result: dict[str, object] | None
    result_usable_for_comparison: bool
    correctness_passed: bool
    all_accuracy_passed: bool
    performance_seen: bool
    median_s: float | None
    req_per_s: float | None
    completed_requests: int | None
    completed_output_tokens: int | None
    mismatch_count: int
    transfer_failures: int
    retry_count: int
    recovery_count: int
    telemetry_export_failures: int
    expired_kv_requests: int
    transfer_stats: dict[str, int | float]
    fatal_runtime_errors: list[str]
    post_result_runtime_errors: list[str]


class GateResultRecord(TypedDict):
    passed: bool
    failures: list[str]
    warnings: list[str]


class JobRecord(TypedDict):
    key: str
    connector: str
    connector_class: str
    layout: str
    tp: int
    nixl_kv_buffer_device: str
    pytorch_cuda_alloc_conf: str
    name: str
    app_uri: str
    state: str
    legacy_passed: bool
    summary: LogSummaryRecord
    gate: GateResultRecord
    log_lines: list[str]


@dataclass(frozen=True)
class BenchmarkProfile:
    """Named workload shape for repeatable MAST validation."""

    name: str
    warmup_iters: int
    measure_iters: int
    prompt_len: int
    num_prompts: int
    output_len: int
    concurrency: int
    mode: str
    timeout: int
    job_timeout_sec: int
    progress_idle_timeout_sec: int
    progress_check_interval_sec: int
    require_structured_result: bool = False
    min_measured_requests: int = 0
    min_measured_output_tokens: int = 0


def get_benchmark_profiles() -> dict[str, BenchmarkProfile]:
    return {
        "default": BenchmarkProfile(
            name="default",
            warmup_iters=3,
            measure_iters=3,
            prompt_len=128,
            num_prompts=250,
            output_len=32,
            concurrency=1,
            mode="accuracy",
            timeout=0,
            job_timeout_sec=3600,
            progress_idle_timeout_sec=1800,
            progress_check_interval_sec=300,
        ),
        "soak-short": BenchmarkProfile(
            name="soak-short",
            warmup_iters=3,
            measure_iters=10,
            prompt_len=128,
            num_prompts=500,
            output_len=64,
            concurrency=1,
            mode="accuracy",
            timeout=0,
            job_timeout_sec=7200,
            progress_idle_timeout_sec=3600,
            progress_check_interval_sec=300,
            require_structured_result=True,
            min_measured_requests=5000,
            min_measured_output_tokens=320000,
        ),
        "soak-long": BenchmarkProfile(
            name="soak-long",
            warmup_iters=5,
            measure_iters=30,
            prompt_len=128,
            num_prompts=1000,
            output_len=64,
            concurrency=1,
            mode="accuracy",
            timeout=0,
            job_timeout_sec=25200,
            progress_idle_timeout_sec=7200,
            progress_check_interval_sec=600,
            require_structured_result=True,
            min_measured_requests=30000,
            min_measured_output_tokens=1920000,
        ),
        "load-soak-short": BenchmarkProfile(
            name="load-soak-short",
            warmup_iters=3,
            measure_iters=12,
            prompt_len=256,
            num_prompts=500,
            output_len=64,
            concurrency=4,
            mode="accuracy",
            timeout=0,
            job_timeout_sec=10800,
            progress_idle_timeout_sec=3600,
            progress_check_interval_sec=300,
            require_structured_result=True,
            min_measured_requests=6000,
            min_measured_output_tokens=384000,
        ),
        "load-soak-long": BenchmarkProfile(
            name="load-soak-long",
            warmup_iters=5,
            measure_iters=24,
            prompt_len=256,
            num_prompts=750,
            output_len=96,
            concurrency=8,
            mode="accuracy",
            timeout=0,
            job_timeout_sec=21600,
            progress_idle_timeout_sec=7200,
            progress_check_interval_sec=600,
            require_structured_result=True,
            min_measured_requests=18000,
            min_measured_output_tokens=1728000,
        ),
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
    concurrency: int
    gpu_memory_utilization: float
    mode: str
    timeout: int
    validate_performance_outputs: bool

    @property
    def measured_requests(self) -> int:
        return self.num_prompts * self.measure_iters

    @property
    def measured_output_tokens(self) -> int:
        return self.measured_requests * self.output_len


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
    progress_idle_timeout_sec: int
    progress_check_interval_sec: int

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
    tp: int
    nixl_kv_buffer_device: str
    pytorch_cuda_alloc_conf: str
    name: str
    session_id: str

    @property
    def is_cross_node(self) -> bool:
        return self.layout == "cn"


@dataclass(frozen=True)
class ProgressSnapshot:
    """Monotonic benchmark progress observed from sparse MAST logs."""

    key: str
    description: str


@dataclass(frozen=True)
class StructuredProgressRecord:
    """One structured benchmark progress event parsed from MAST logs."""

    completed_requests: int
    description: str


@dataclass
class JobResult:
    spec: JobSpec
    app_uri: str
    state: str
    log_lines: list[str]

    @property
    def passed(self) -> bool:
        summary = BenchmarkLogParser().parse(self.log_lines)
        if summary.structured_result is not None:
            return self.state == "SUCCEEDED" and summary.result_usable_for_comparison
        return (
            self.state == "SUCCEEDED"
            and summary.correctness_passed
            and summary.all_accuracy_passed
            and summary.performance_seen
            and summary.mismatch_count == 0
            and not summary.fatal_runtime_errors
        )


@dataclass(frozen=True)
class MastApp:
    spec: JobSpec
    app_uri: str
    owned: bool


@dataclass(frozen=True)
class LogSummary:
    """Stable signals extracted from MAST logs for soak gating.

    `result_usable_for_comparison` means the benchmark result has enough valid
    correctness and performance data for `--compare-json`, even if the runtime
    gate later fails on a post-result shutdown traceback.
    """

    structured_result: dict[str, object] | None
    result_usable_for_comparison: bool
    correctness_passed: bool
    all_accuracy_passed: bool
    performance_seen: bool
    median_s: float | None
    req_per_s: float | None
    completed_requests: int | None
    completed_output_tokens: int | None
    mismatch_count: int
    transfer_failures: int
    retry_count: int
    recovery_count: int
    telemetry_export_failures: int
    expired_kv_requests: int
    transfer_stats: dict[str, int | float]
    fatal_runtime_errors: list[str]
    post_result_runtime_errors: list[str]

    def to_dict(self) -> LogSummaryRecord:
        return {
            "structured_result": self.structured_result,
            "result_usable_for_comparison": self.result_usable_for_comparison,
            "correctness_passed": self.correctness_passed,
            "all_accuracy_passed": self.all_accuracy_passed,
            "performance_seen": self.performance_seen,
            "median_s": self.median_s,
            "req_per_s": self.req_per_s,
            "completed_requests": self.completed_requests,
            "completed_output_tokens": self.completed_output_tokens,
            "mismatch_count": self.mismatch_count,
            "transfer_failures": self.transfer_failures,
            "retry_count": self.retry_count,
            "recovery_count": self.recovery_count,
            "telemetry_export_failures": self.telemetry_export_failures,
            "expired_kv_requests": self.expired_kv_requests,
            "transfer_stats": self.transfer_stats,
            "fatal_runtime_errors": self.fatal_runtime_errors,
            "post_result_runtime_errors": self.post_result_runtime_errors,
        }


@dataclass
class LogParseState:
    """Mutable state accumulated while parsing MAST benchmark logs."""

    median_s: float | None = None
    req_per_s: float | None = None
    mismatch_count: int = 0
    fatal_runtime_errors: list[str] = field(default_factory=list)
    post_result_runtime_errors: list[str] = field(default_factory=list)
    structured_result: dict[str, object] | None = None
    transfer_stats_records: list[dict[str, object]] = field(default_factory=list)
    saw_structured_result: bool = False


@dataclass(frozen=True)
class GateResult:
    """Pass/fail decision for one MAST job."""

    passed: bool
    failures: list[str]
    warnings: list[str]

    def to_dict(self) -> GateResultRecord:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
        }


class BenchmarkLogParser:
    """Parses stable benchmark and connector signals from collected MAST logs."""

    PERFORMANCE_RE = re.compile(
        r"PERFORMANCE:\s+median=(?P<median>[0-9.]+)s.*\((?P<rps>[0-9.]+)\s+req/s\)"
    )
    DECODE_RE = re.compile(r"DECODE:\s+median=(?P<median>[0-9.]+)s")
    CORRECTNESS_FAILED_RE = re.compile(
        r"CORRECTNESS FAILED:\s+(?P<mismatches>\d+)/\d+\s+mismatches"
    )
    MISMATCH_RE = re.compile(
        r"\b(?P<mismatches>[1-9]\d*)\s+mismatches?\b", re.IGNORECASE
    )
    FATAL_RE = re.compile(
        r"Traceback|RuntimeError|ValueError|CUDA error|CUDA-capable|"
        r"\bNCCL\b.*\b(?:error|failed|failure|timeout|abort)\b|"
        r"Xid|OutOfMemory|out of memory|OOM|TimeoutError|"
        r"Assertion.*failed|cuda_ipc_md|ucs_fatal|ChildFailedError|"
        r"died unexpectedly|LOG_COLLECTION_FAILED|LOG_COLLECTION_STATUS_ERROR",
        re.IGNORECASE,
    )
    EXPIRED_KV_REQUEST_RE = re.compile(
        r"\b(?:expired[^\n]*(?:kv|key-value)|(?:kv|key-value)[^\n]*expired)\b",
        re.IGNORECASE,
    )

    def parse(self, log_lines: Sequence[str]) -> LogSummary:
        state = LogParseState()
        for line in log_lines:
            self.update_parse_state(state, line)
        return self.build_summary(state, "\n".join(log_lines))

    def update_parse_state(self, state: LogParseState, line: str) -> None:
        self.record_structured_result(state, line)
        self.record_transfer_stats(state, line)
        self.record_performance(state, line)
        self.record_mismatches(state, line)
        self.record_runtime_error(state, line)

    def record_structured_result(self, state: LogParseState, line: str) -> None:
        parsed_result = self.parse_structured_result(line)
        if parsed_result is None:
            return
        state.structured_result = parsed_result
        state.saw_structured_result = True

    def record_transfer_stats(self, state: LogParseState, line: str) -> None:
        parsed_transfer_stats = self.parse_transfer_stats(line)
        if parsed_transfer_stats is not None:
            state.transfer_stats_records.append(parsed_transfer_stats)

    def record_performance(self, state: LogParseState, line: str) -> None:
        performance_match = self.PERFORMANCE_RE.search(line)
        if performance_match is not None:
            state.median_s = float(performance_match.group("median"))
            state.req_per_s = float(performance_match.group("rps"))
            return

        decode_match = self.DECODE_RE.search(line)
        if decode_match is not None and state.median_s is None:
            state.median_s = float(decode_match.group("median"))

    def record_mismatches(self, state: LogParseState, line: str) -> None:
        # Text logs can contain one correctness summary per measured iter;
        # aggregate them until structured result JSON provides the final count.
        failed_match = self.CORRECTNESS_FAILED_RE.search(line)
        if failed_match is not None:
            state.mismatch_count += int(failed_match.group("mismatches"))
            return
        mismatch_match = self.MISMATCH_RE.search(line)
        if mismatch_match is not None:
            state.mismatch_count += int(mismatch_match.group("mismatches"))

    def record_runtime_error(self, state: LogParseState, line: str) -> None:
        if not self.FATAL_RE.search(line) or self.should_ignore_runtime_error(line):
            return
        state.fatal_runtime_errors.append(line)
        if state.saw_structured_result:
            state.post_result_runtime_errors.append(line)

    def build_summary(self, state: LogParseState, output: str) -> LogSummary:
        structured_signals = self.structured_signals(state.structured_result)
        mismatch_count = self.mismatch_count(state, structured_signals)
        median_s = self.median_s(state, structured_signals)
        req_per_s = self.req_per_s(state, structured_signals)
        completed_requests = structured_signals.get("completed_requests")
        completed_output_tokens = structured_signals.get("completed_output_tokens")
        correctness_passed = bool(structured_signals.get("correctness_passed")) or (
            "CORRECTNESS PASSED" in output
        )
        all_accuracy_passed = bool(structured_signals.get("all_accuracy_passed")) or (
            "ALL ACCURACY TESTS PASSED" in output
        )
        performance_seen = median_s is not None or "PERFORMANCE:" in output

        return LogSummary(
            structured_result=state.structured_result,
            result_usable_for_comparison=self.result_usable_for_comparison(
                correctness_passed=correctness_passed,
                all_accuracy_passed=all_accuracy_passed,
                performance_seen=performance_seen,
                median_s=median_s,
                req_per_s=req_per_s,
                mismatch_count=mismatch_count,
            ),
            correctness_passed=correctness_passed,
            all_accuracy_passed=all_accuracy_passed,
            performance_seen=performance_seen,
            median_s=median_s,
            req_per_s=req_per_s,
            completed_requests=completed_requests
            if isinstance(completed_requests, int)
            else None,
            completed_output_tokens=completed_output_tokens
            if isinstance(completed_output_tokens, int)
            else None,
            mismatch_count=mismatch_count,
            transfer_failures=output.count("Uniflow transfer failure"),
            retry_count=output.count("Retrying UniFlow"),
            recovery_count=output.count("Recovered UniFlow"),
            telemetry_export_failures=output.count(
                "UniFlow transfer telemetry export failed"
            ),
            expired_kv_requests=len(self.EXPIRED_KV_REQUEST_RE.findall(output)),
            transfer_stats=self.aggregate_transfer_stats(state.transfer_stats_records),
            fatal_runtime_errors=state.fatal_runtime_errors,
            post_result_runtime_errors=state.post_result_runtime_errors,
        )

    @staticmethod
    def mismatch_count(
        state: LogParseState,
        structured_signals: dict[str, object],
    ) -> int:
        structured_mismatches = structured_signals.get("mismatch_count")
        if isinstance(structured_mismatches, int):
            return structured_mismatches
        return state.mismatch_count

    @staticmethod
    def median_s(
        state: LogParseState,
        structured_signals: dict[str, object],
    ) -> float | None:
        structured_median = structured_signals.get("median_s")
        if isinstance(structured_median, int | float):
            return float(structured_median)
        return state.median_s

    @staticmethod
    def req_per_s(
        state: LogParseState,
        structured_signals: dict[str, object],
    ) -> float | None:
        structured_rps = structured_signals.get("req_per_s")
        if isinstance(structured_rps, int | float):
            return float(structured_rps)
        return state.req_per_s

    @staticmethod
    def result_usable_for_comparison(
        *,
        correctness_passed: bool,
        all_accuracy_passed: bool,
        performance_seen: bool,
        median_s: float | None,
        req_per_s: float | None,
        mismatch_count: int,
    ) -> bool:
        return (
            correctness_passed
            and all_accuracy_passed
            and performance_seen
            and median_s is not None
            and req_per_s is not None
            and mismatch_count == 0
        )

    @staticmethod
    def should_ignore_runtime_error(line: str) -> bool:
        return "fastcheck.rs" in line and "gpuxidcheck" in line

    @staticmethod
    def parse_structured_result(line: str) -> dict[str, object] | None:
        prefix_index = line.find(BENCHMARK_RESULT_PREFIX)
        if prefix_index < 0:
            return None
        payload = line[prefix_index + len(BENCHMARK_RESULT_PREFIX) :].strip()
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return result if isinstance(result, dict) else None

    @staticmethod
    def parse_transfer_stats(line: str) -> dict[str, object] | None:
        prefix_index = line.find(UNIFLOW_TRANSFER_STATS_PREFIX)
        if prefix_index < 0:
            return None
        payload = line[prefix_index + len(UNIFLOW_TRANSFER_STATS_PREFIX) :].strip()
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return result if isinstance(result, dict) else None

    @staticmethod
    def aggregate_transfer_stats(
        records: Sequence[dict[str, object]],
    ) -> dict[str, int | float]:
        transfers_initiated = 0
        transfers_completed = 0
        transfers_failed = 0
        bytes_transferred = 0
        blocks_transferred = 0
        total_duration_s = 0.0

        for record in records:
            transfers_initiated += BenchmarkLogParser.int_value(
                record.get("transfers_initiated")
            )
            transfers_completed += BenchmarkLogParser.int_value(
                record.get("transfers_completed")
            )
            transfers_failed += BenchmarkLogParser.int_value(
                record.get("transfers_failed")
            )
            bytes_transferred += BenchmarkLogParser.int_value(
                record.get("bytes_transferred")
            )
            blocks_transferred += BenchmarkLogParser.int_value(
                record.get("blocks_transferred")
            )
            total_duration_s += BenchmarkLogParser.float_value(
                record.get("total_duration_s")
            )

        average_duration_ms = (
            (total_duration_s / transfers_completed) * 1000
            if transfers_completed > 0
            else 0.0
        )
        bandwidth_gbps = (
            bytes_transferred / total_duration_s / 1e9 if total_duration_s > 0 else 0.0
        )
        return {
            "records": len(records),
            "transfers_initiated": transfers_initiated,
            "transfers_completed": transfers_completed,
            "transfers_failed": transfers_failed,
            "bytes_transferred": bytes_transferred,
            "blocks_transferred": blocks_transferred,
            "total_duration_s": total_duration_s,
            "average_duration_ms": average_duration_ms,
            "bandwidth_gbps": bandwidth_gbps,
        }

    @staticmethod
    def int_value(value: object) -> int:
        return value if isinstance(value, int) else 0

    @staticmethod
    def int_or_none(value: object) -> int | None:
        return value if isinstance(value, int) else None

    @staticmethod
    def float_value(value: object) -> float:
        return float(value) if isinstance(value, int | float) else 0.0

    @staticmethod
    def structured_signals(
        structured_result: dict[str, object] | None,
    ) -> dict[str, object]:
        if structured_result is None:
            return {}

        if structured_result.get("mode") == "offline":
            result = structured_result.get("result")
            if not isinstance(result, dict):
                return {}
            validation = result.get("validation")
            decode = result.get("decode")
            return BenchmarkLogParser.signals_from_result(
                status=str(structured_result.get("status", "")),
                correctness=validation if isinstance(validation, dict) else {},
                performance=decode if isinstance(decode, dict) else {},
                config=result.get("config"),
            )

        correctness = structured_result.get("correctness")
        performance = structured_result.get("performance")
        return BenchmarkLogParser.signals_from_result(
            status=str(structured_result.get("status", "")),
            correctness=correctness if isinstance(correctness, dict) else {},
            performance=performance if isinstance(performance, dict) else {},
            config=structured_result.get("config"),
        )

    @staticmethod
    def signals_from_result(
        *,
        status: str,
        correctness: dict[str, object],
        performance: dict[str, object],
        config: object,
    ) -> dict[str, object]:
        signals: dict[str, object] = {}
        passed = correctness.get("passed")
        if isinstance(passed, bool):
            signals["correctness_passed"] = passed
        signals["all_accuracy_passed"] = status == "passed"
        mismatches = correctness.get("mismatches")
        if isinstance(mismatches, int):
            signals["mismatch_count"] = mismatches
        median_s = performance.get("median_s")
        if isinstance(median_s, int | float):
            signals["median_s"] = float(median_s)
        req_per_s = performance.get("req_per_s")
        if isinstance(req_per_s, int | float):
            signals["req_per_s"] = float(req_per_s)
        completed_requests, completed_output_tokens = (
            BenchmarkLogParser.completed_workload_counts(config, performance)
        )
        if completed_requests is not None:
            signals["completed_requests"] = completed_requests
        if completed_output_tokens is not None:
            signals["completed_output_tokens"] = completed_output_tokens
        return signals

    @staticmethod
    def completed_workload_counts(
        config: object,
        performance: dict[str, object],
    ) -> tuple[int | None, int | None]:
        config_dict = config if isinstance(config, dict) else {}
        num_prompts = BenchmarkLogParser.int_or_none(
            performance.get("num_prompts")
        ) or BenchmarkLogParser.int_or_none(config_dict.get("num_prompts"))
        output_len = BenchmarkLogParser.int_or_none(config_dict.get("output_len"))
        completed_iters = BenchmarkLogParser.int_or_none(performance.get("count"))
        if completed_iters is None:
            times = performance.get("times_s")
            if isinstance(times, list):
                completed_iters = len(times)
        if num_prompts is None or completed_iters is None:
            return None, None

        completed_requests = num_prompts * completed_iters
        completed_output_tokens = (
            completed_requests * output_len if output_len is not None else None
        )
        return completed_requests, completed_output_tokens


class BenchmarkBaseline:
    """Optional baseline used for performance regression gates."""

    def __init__(self, median_by_key: dict[str, float]) -> None:
        self.median_by_key = median_by_key

    @classmethod
    def load(cls, path: Path | None) -> "BenchmarkBaseline":
        if path is None:
            return cls({})
        try:
            data = json.loads(path.read_text())
        except OSError as exc:
            raise SystemExit(f"Failed to read baseline JSON `{path}`: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse baseline JSON `{path}`: {exc}") from exc
        if not isinstance(data, dict):
            raise SystemExit(f"Baseline JSON `{path}` must contain a JSON object")
        medians: dict[str, float] = {}
        for job in data.get("jobs", []):
            if not isinstance(job, dict):
                continue
            key = job.get("key")
            summary = job.get("summary", {})
            if not isinstance(summary, dict):
                continue
            median_s = summary.get("median_s")
            if isinstance(key, str) and isinstance(median_s, int | float):
                medians[key] = float(median_s)
        return cls(medians)

    def median_for(self, spec: JobSpec) -> float | None:
        return self.median_by_key.get(job_key(spec))


class BenchmarkGateEvaluator:
    """Evaluates explicit soak pass/fail criteria from parsed log signals."""

    def __init__(
        self,
        *,
        enforce_gates: bool,
        allow_recovered_failures: bool,
        require_structured_result: bool,
        workload: Workload,
        min_measured_requests: int,
        min_measured_output_tokens: int,
        baseline: BenchmarkBaseline,
        max_median_regression_pct: float | None,
    ) -> None:
        self.enforce_gates = enforce_gates
        self.allow_recovered_failures = allow_recovered_failures
        self.require_structured_result = require_structured_result
        self.workload = workload
        self.min_measured_requests = min_measured_requests
        self.min_measured_output_tokens = min_measured_output_tokens
        self.baseline = baseline
        self.max_median_regression_pct = max_median_regression_pct

    def evaluate(self, result: JobResult, summary: LogSummary) -> GateResult:
        if not self.enforce_gates:
            return GateResult(passed=result.passed, failures=[], warnings=[])

        failures: list[str] = []
        warnings: list[str] = []

        self.add_state_failures(result, failures)
        self.add_workload_failures(summary, failures)
        self.add_correctness_failures(summary, failures)
        self.add_runtime_failures(summary, failures, warnings)
        self.add_transfer_failures(result, summary, failures)
        self.add_baseline_failures(result.spec, summary, failures, warnings)
        return GateResult(passed=not failures, failures=failures, warnings=warnings)

    def add_state_failures(
        self,
        result: JobResult,
        failures: list[str],
    ) -> None:
        if result.state != "SUCCEEDED":
            failures.append(f"job ended in state {result.state}")

    def add_workload_failures(
        self,
        summary: LogSummary,
        failures: list[str],
    ) -> None:
        if self.require_structured_result and summary.structured_result is None:
            failures.append("missing structured benchmark result")
        if self.min_measured_requests > 0 and summary.completed_requests is None:
            failures.append("missing measured request volume in structured result")
        elif (
            summary.completed_requests is not None
            and summary.completed_requests < self.min_measured_requests
        ):
            failures.append(
                f"measured request volume {summary.completed_requests} below "
                f"required {self.min_measured_requests}"
            )
        if (
            self.min_measured_output_tokens > 0
            and summary.completed_output_tokens is None
        ):
            failures.append("missing measured output token volume in structured result")
        elif (
            summary.completed_output_tokens is not None
            and summary.completed_output_tokens < self.min_measured_output_tokens
        ):
            failures.append(
                f"measured output token volume "
                f"{summary.completed_output_tokens} below required "
                f"{self.min_measured_output_tokens}"
            )

    @staticmethod
    def add_correctness_failures(
        summary: LogSummary,
        failures: list[str],
    ) -> None:
        if not summary.correctness_passed:
            failures.append("missing `CORRECTNESS PASSED`")
        if not summary.all_accuracy_passed:
            failures.append("missing `ALL ACCURACY TESTS PASSED`")
        if not summary.performance_seen:
            failures.append("missing performance summary")
        if summary.mismatch_count:
            failures.append(f"{summary.mismatch_count} correctness mismatches")

    @staticmethod
    def add_runtime_failures(
        summary: LogSummary,
        failures: list[str],
        warnings: list[str],
    ) -> None:
        if summary.fatal_runtime_errors:
            failures.append(
                f"{len(summary.fatal_runtime_errors)} fatal runtime log lines"
            )
        if summary.result_usable_for_comparison and summary.post_result_runtime_errors:
            warnings.append(
                f"{len(summary.post_result_runtime_errors)} post-result "
                "runtime log lines"
            )
        if summary.telemetry_export_failures:
            failures.append(
                f"{summary.telemetry_export_failures} telemetry export failures"
            )
        if summary.expired_kv_requests:
            failures.append(f"{summary.expired_kv_requests} expired KV request lines")

    def add_transfer_failures(
        self,
        result: JobResult,
        summary: LogSummary,
        failures: list[str],
    ) -> None:
        if summary.transfer_failures and not self.allow_recovered_failures:
            failures.append(f"{summary.transfer_failures} transfer failure lines")
        if result.state == "SUCCEEDED" and summary.retry_count > summary.recovery_count:
            failures.append(
                f"{summary.retry_count} retries but only "
                f"{summary.recovery_count} recoveries"
            )

    def add_baseline_failures(
        self,
        spec: JobSpec,
        summary: LogSummary,
        failures: list[str],
        warnings: list[str],
    ) -> None:
        if self.max_median_regression_pct is None:
            return
        baseline_median = self.baseline.median_for(spec)
        if baseline_median is None:
            warnings.append(f"no baseline median for {job_key(spec)}")
            return
        if summary.median_s is None:
            failures.append("cannot evaluate median regression without median")
            return

        allowed = baseline_median * (1 + self.max_median_regression_pct / 100)
        if summary.median_s > allowed:
            failures.append(
                f"median {summary.median_s:.3f}s regressed beyond "
                f"{self.max_median_regression_pct:.1f}% baseline "
                f"{baseline_median:.3f}s"
            )


class BenchmarkReport:
    """Machine-readable run report for review evidence and automation."""

    def __init__(
        self,
        *,
        package: str,
        profile: BenchmarkProfile,
        workload: Workload,
        results: Sequence[JobResult],
        parser: BenchmarkLogParser,
        evaluator: BenchmarkGateEvaluator,
    ) -> None:
        self.package = package
        self.profile = profile
        self.workload = workload
        self.results = results
        self.parser = parser
        self.evaluator = evaluator
        self._job_records: list[JobRecord] | None = None

    def job_records(self) -> list[JobRecord]:
        if self._job_records is not None:
            return self._job_records

        records: list[JobRecord] = []
        for result in self.results:
            summary = self.parser.parse(result.log_lines)
            gate = self.evaluator.evaluate(result, summary)
            records.append(
                {
                    "key": job_key(result.spec),
                    "connector": result.spec.connector_key,
                    "connector_class": result.spec.connector_class,
                    "layout": result.spec.layout,
                    "tp": result.spec.tp,
                    "nixl_kv_buffer_device": result.spec.nixl_kv_buffer_device,
                    "pytorch_cuda_alloc_conf": result.spec.pytorch_cuda_alloc_conf,
                    "name": result.spec.name,
                    "app_uri": result.app_uri,
                    "state": result.state,
                    "legacy_passed": result.passed,
                    "summary": summary.to_dict(),
                    "gate": gate.to_dict(),
                    "log_lines": result.log_lines,
                }
            )
        self._job_records = records
        return records

    def passed(self) -> bool:
        return all(job["gate"]["passed"] for job in self.job_records())

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_unix_s": int(time.time()),
            "package": self.package,
            "profile": self.profile.name,
            "workload": {
                "model": self.workload.model,
                "tp": self.workload.tp,
                "warmup_iters": self.workload.warmup_iters,
                "measure_iters": self.workload.measure_iters,
                "prompt_len": self.workload.prompt_len,
                "num_prompts": self.workload.num_prompts,
                "output_len": self.workload.output_len,
                "concurrency": self.workload.concurrency,
                "measured_requests": self.workload.measured_requests,
                "measured_output_tokens": self.workload.measured_output_tokens,
                "gpu_memory_utilization": self.workload.gpu_memory_utilization,
                "mode": self.workload.mode,
                "timeout": self.workload.timeout,
            },
            "jobs": self.job_records(),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class BenchmarkMarkdownWriter:
    """Writes a compact reviewer-facing MAST benchmark report."""

    def __init__(self, report: BenchmarkReport) -> None:
        self.report = report

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())

    def render(self) -> str:
        workload = self.report.workload
        lines = [
            "# MAST Benchmark Report",
            "",
            f"- Package: `{self.report.package}`",
            f"- Profile: `{self.report.profile.name}`",
            f"- Model: `{workload.model}`",
            f"- Workload: `{workload.measured_requests}` measured requests, "
            f"`{workload.measured_output_tokens}` measured output tokens "
            f"({workload.measure_iters} iters x {workload.num_prompts} prompts x "
            f"{workload.output_len} output tokens, concurrency={workload.concurrency})",
            "",
            "| Job | State | Gate | Result | Median | Req/s | Xfer GB | Blocks | Xfer GB/s | Mismatches | Issue |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for job in self.report.job_records():
            summary = job["summary"]
            gate = job["gate"]
            transfer_stats = summary["transfer_stats"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{job['key']}`",
                        f"`{job['state']}`",
                        "`PASS`" if gate["passed"] else "`FAIL`",
                        "`USABLE`"
                        if summary["result_usable_for_comparison"]
                        else "`UNUSABLE`",
                        self.format_value(summary["median_s"], "s"),
                        self.format_value(summary["req_per_s"], ""),
                        self.format_gb(transfer_stats["bytes_transferred"]),
                        str(transfer_stats["blocks_transferred"]),
                        self.format_value(transfer_stats["bandwidth_gbps"], ""),
                        str(summary["mismatch_count"]),
                        self.first_issue(gate),
                    ]
                )
                + " |"
            )

        lines.extend(["", "## MAST Apps", ""])
        for job in self.report.job_records():
            lines.append(f"- `{job['key']}`: `{job['app_uri']}`")
        return "\n".join(lines) + "\n"

    @staticmethod
    def format_value(value: object, suffix: str) -> str:
        if not isinstance(value, int | float):
            return ""
        return f"{value:.3f}{suffix}"

    @staticmethod
    def format_gb(value: object) -> str:
        if not isinstance(value, int | float):
            return ""
        return f"{value / 1e9:.3f}"

    @staticmethod
    def first_issue(gate: GateResultRecord) -> str:
        failures = gate["failures"]
        if failures:
            return str(failures[0])
        warnings = gate["warnings"]
        if warnings:
            return str(warnings[0])
        return ""


class BenchmarkComparisonReport:
    """Compares completed benchmark JSON reports by job key and profile."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)
        self.reports = [self.load_report(path) for path in self.paths]

    @staticmethod
    def load_report(path: Path) -> dict[str, object]:
        try:
            data = json.loads(path.read_text())
        except OSError as exc:
            raise SystemExit(f"Failed to read comparison JSON `{path}`: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Failed to parse comparison JSON `{path}`: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise SystemExit(f"Comparison JSON `{path}` must contain a JSON object")
        return data

    def passed(self) -> bool:
        rows = self.rows()
        if not rows:
            return True
        for row in rows:
            gate = row.get("gate", {})
            if isinstance(gate, dict) and not gate.get("passed", False):
                return False
        return True

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for path, report in zip(self.paths, self.reports):
            profile = report.get("profile", "")
            workload = report.get("workload", {})
            if not isinstance(workload, dict):
                workload = {}
            jobs = report.get("jobs", [])
            if not isinstance(jobs, list):
                continue
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                summary = job.get("summary", {})
                gate = job.get("gate", {})
                transfer_stats = (
                    summary.get("transfer_stats", {})
                    if isinstance(summary, dict)
                    else {}
                )
                if not isinstance(transfer_stats, dict):
                    transfer_stats = {}
                result_usable = (
                    summary.get("result_usable_for_comparison")
                    if isinstance(summary, dict)
                    else False
                )
                if result_usable is None:
                    result_usable = (
                        bool(summary.get("correctness_passed"))
                        and bool(summary.get("all_accuracy_passed"))
                        and bool(summary.get("performance_seen"))
                        and isinstance(summary.get("median_s"), int | float)
                        and isinstance(summary.get("req_per_s"), int | float)
                        and summary.get("mismatch_count") == 0
                    )
                rows.append(
                    {
                        "source": str(path),
                        "profile": profile,
                        "layout": job.get("layout", ""),
                        "connector": job.get("connector", ""),
                        "tp": job.get("tp", ""),
                        "concurrency": workload.get("concurrency", ""),
                        "measured_requests": workload.get("measured_requests", ""),
                        "state": job.get("state", ""),
                        "gate": gate,
                        "result_usable_for_comparison": result_usable,
                        "median_s": summary.get("median_s")
                        if isinstance(summary, dict)
                        else None,
                        "req_per_s": summary.get("req_per_s")
                        if isinstance(summary, dict)
                        else None,
                        "mismatches": summary.get("mismatch_count")
                        if isinstance(summary, dict)
                        else None,
                        "transfer_failures": summary.get("transfer_failures")
                        if isinstance(summary, dict)
                        else None,
                        "xfer_gb": transfer_stats.get("bytes_transferred", 0) / 1e9
                        if isinstance(
                            transfer_stats.get("bytes_transferred"), int | float
                        )
                        else None,
                        "xfer_blocks": transfer_stats.get("blocks_transferred"),
                        "xfer_gbps": transfer_stats.get("bandwidth_gbps"),
                    }
                )
        return rows

    def render(self) -> str:
        lines = [
            "# MAST Benchmark Comparison",
            "",
            "| Profile | Layout | Connector | TP | Concurrency | Requests | State | Gate | Result | Median | Req/s | Xfer GB | Blocks | Xfer GB/s | Mismatches | Transfer Failures |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in self.rows():
            gate = row.get("gate", {})
            gate_passed = isinstance(gate, dict) and gate.get("passed", False)
            usable = bool(row["result_usable_for_comparison"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['profile']}`",
                        f"`{row['layout']}`",
                        f"`{row['connector']}`",
                        str(row["tp"]),
                        str(row["concurrency"]),
                        str(row["measured_requests"]),
                        f"`{row['state']}`",
                        "`PASS`" if gate_passed else "`FAIL`",
                        "`USABLE`" if usable else "`UNUSABLE`",
                        BenchmarkMarkdownWriter.format_value(row["median_s"], "s"),
                        BenchmarkMarkdownWriter.format_value(row["req_per_s"], ""),
                        BenchmarkMarkdownWriter.format_value(row["xfer_gb"], ""),
                        str(row["xfer_blocks"]),
                        BenchmarkMarkdownWriter.format_value(row["xfer_gbps"], ""),
                        str(row["mismatches"]),
                        str(row["transfer_failures"]),
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Sources", ""])
        for path in self.paths:
            lines.append(f"- `{path}`")
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())


def job_key(spec: JobSpec) -> str:
    key = f"{spec.layout}:{spec.connector_key}:tp{spec.tp}"
    if spec.connector_key == "nixl" and spec.nixl_kv_buffer_device != "cuda":
        key += f":kv-{spec.nixl_kv_buffer_device}"
    if spec.pytorch_cuda_alloc_conf != "expandable_segments:True":
        key += f":alloc-{alloc_conf_key(spec.pytorch_cuda_alloc_conf)}"
    return key


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
        timeout_sec: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        stderr = subprocess.DEVNULL if suppress_stderr else subprocess.PIPE
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=stderr,
                check=False,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout or ""
            )
            timeout_stderr = "" if suppress_stderr else str(exc)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=124,
                stdout=stdout,
                stderr=timeout_stderr,
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


class TorchXClient:
    """Small retrying adapter around the TorchX CLI."""

    def __init__(
        self,
        runner: CommandRunner,
        *,
        workspace: Path,
        env: dict[str, str],
        retries: int = TORCHX_CLI_RETRIES,
        retry_backoff_sec: int = TORCHX_CLI_RETRY_BACKOFF_SEC,
    ) -> None:
        self.runner = runner
        self.workspace = workspace
        self.env = env
        self.retries = retries
        self.retry_backoff_sec = retry_backoff_sec

    def status(self, app_uri: str) -> subprocess.CompletedProcess[str]:
        return self.run_with_retries(
            ["torchx", "status", app_uri],
            timeout_sec=TORCHX_STATUS_TIMEOUT_SEC,
            suppress_stderr=True,
        )

    def log(
        self, app_uri: str, role: str, regex: str
    ) -> subprocess.CompletedProcess[str]:
        return self.run_with_retries(
            ["torchx", "log", "--regex", regex, f"{app_uri}/{role}"],
            timeout_sec=TORCHX_LOG_TIMEOUT_SEC,
        )

    def cancel(self, app_uri: str) -> subprocess.CompletedProcess[str]:
        return self.run_with_retries(
            ["torchx", "cancel", app_uri],
            timeout_sec=TORCHX_CANCEL_TIMEOUT_SEC,
        )

    def run_with_retries(
        self,
        cmd: Sequence[str],
        *,
        timeout_sec: int,
        suppress_stderr: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        last_result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, self.retries + 1):
            result = self.runner.run_capture(
                cmd,
                cwd=self.workspace,
                env=self.env,
                suppress_stderr=suppress_stderr,
                timeout_sec=timeout_sec,
            )
            if result.returncode == 0:
                return result
            last_result = result
            if attempt < self.retries:
                time.sleep(self.retry_backoff_sec * attempt)
        assert last_result is not None
        return last_result


@dataclass(frozen=True)
class LogCollectionPolicy:
    """Controls how much log data is retained in reports."""

    progress_tail_lines: int = MAX_PROGRESS_LOG_LINES


class MastLogCollector:
    """Collects staged MAST log signals without dumping full progress logs."""

    def __init__(self, torchx: TorchXClient, policy: LogCollectionPolicy) -> None:
        self.torchx = torchx
        self.policy = policy

    def collect(self, spec: JobSpec, app_uri: str, *, state: str) -> list[str]:
        logs: list[str] = []
        for role in self.trainer_roles(spec):
            logs.extend(self.collect_role(app_uri, role, state=state))
        return dedupe_preserve_order(logs)

    def collect_role(self, app_uri: str, role: str, *, state: str) -> list[str]:
        lines: list[str] = []
        result_lines = self.collect_category(app_uri, role, RESULT_LOG_REGEX, "result")
        failure_lines = self.collect_category(
            app_uri,
            role,
            FAILURE_LOG_REGEX,
            "failure",
        )
        lines.extend(result_lines)
        lines.extend(failure_lines)
        if state != "SUCCEEDED" or not self.has_structured_result(result_lines):
            traceback_lines = self.collect_category(
                app_uri,
                role,
                TRACEBACK_LOG_REGEX,
                "traceback",
            )
            progress_lines = self.collect_category(
                app_uri,
                role,
                PROGRESS_LOG_REGEX,
                "progress",
            )
            lines.extend(traceback_lines)
            lines.extend(self.tail(progress_lines, self.policy.progress_tail_lines))
        return dedupe_preserve_order(lines)

    def collect_category(
        self,
        app_uri: str,
        role: str,
        regex: str,
        category: str,
    ) -> list[str]:
        result = self.torchx.log(app_uri, role, regex)
        lines: list[str] = []
        if result.stdout:
            lines.extend(line for line in result.stdout.splitlines() if line)
        if result.returncode != 0:
            lines.append(
                "LOG_COLLECTION_FAILED: "
                f"`torchx log` exited {result.returncode} for "
                f"{app_uri}/{role} category={category}"
            )
            if result.stderr:
                lines.extend(
                    f"LOG_COLLECTION_STDERR: {line}"
                    for line in result.stderr.splitlines()
                    if line
                )
        return lines

    @staticmethod
    def has_structured_result(lines: Sequence[str]) -> bool:
        return any(BENCHMARK_RESULT_PREFIX in line for line in lines)

    @staticmethod
    def tail(lines: Sequence[str], max_lines: int) -> list[str]:
        if len(lines) <= max_lines:
            return list(lines)
        omitted = len(lines) - max_lines
        return [f"LOG_COLLECTION_TRUNCATED: omitted {omitted} progress lines"] + list(
            lines[-max_lines:]
        )

    @staticmethod
    def trainer_roles(spec: JobSpec) -> list[str]:
        return ["trainer/0", "trainer/1"] if spec.is_cross_node else ["trainer/0"]


def dedupe_preserve_order(lines: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return deduped


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
        self.torchx = TorchXClient(
            runner,
            workspace=config.workspace,
            env=self._torchx_env(),
        )
        self.log_collector = MastLogCollector(
            self.torchx,
            LogCollectionPolicy(),
        )

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

    def cancel(self, app: MastApp, reason: str) -> None:
        if not app.owned:
            logger.info(
                "Skipping cancel for externally provided app %s: %s",
                app.app_uri,
                reason,
            )
            return

        state = self._status(app.app_uri)
        if state in TERMINAL_STATES:
            logger.info("Skipping cancel for %s: already %s", app.spec.name, state)
            return

        logger.info("Cancelling %s: %s", app.spec.name, reason)
        result = self.torchx.cancel(app.app_uri)
        if result.returncode != 0:
            logger.warning(
                "`torchx cancel` failed for %s with exit code %d",
                app.app_uri,
                result.returncode,
            )
            if result.stderr:
                for line in result.stderr.splitlines():
                    if line:
                        logger.warning("  %s", line)

    def wait_for_result(self, spec: JobSpec, app_uri: str) -> JobResult:
        nominal_deadline = time.monotonic() + self.config.job_timeout_sec
        last_state = "UNKNOWN"
        last_progress_key = ""
        last_progress_time = time.monotonic()
        next_progress_check = last_progress_time
        nominal_timeout_logged = False
        consecutive_status_errors = 0
        while True:
            now = time.monotonic()
            state = self._status(app_uri)
            if state != last_state:
                logger.info("%s: %s", spec.name, state)
                last_state = state
                if state == "RUNNING":
                    last_progress_time = now
                    next_progress_check = now

            if state == "STATUS_ERROR":
                consecutive_status_errors += 1
                status_error_result = self._status_error_result(
                    spec, app_uri, consecutive_status_errors
                )
                if status_error_result is not None:
                    return status_error_result
            else:
                consecutive_status_errors = 0

            if state in TERMINAL_STATES:
                return self.snapshot_result(spec, app_uri, state=state)

            if state == "RUNNING" and now >= next_progress_check:
                progress = self._progress_snapshot(spec, app_uri)
                if progress is not None and progress.key != last_progress_key:
                    last_progress_key = progress.key
                    last_progress_time = now
                    logger.info("%s progress: %s", spec.name, progress.description)
                next_progress_check = now + self.config.progress_check_interval_sec
            idle_s = now - last_progress_time
            if state == "RUNNING" and idle_s > self.config.progress_idle_timeout_sec:
                return self._no_progress_result(spec, app_uri, idle_s)

            if now >= nominal_deadline and not last_progress_key:
                return self._timeout_result(spec, app_uri)

            if (
                now >= nominal_deadline
                and last_progress_key
                and not nominal_timeout_logged
            ):
                logger.info(
                    "%s exceeded nominal timeout, continuing because benchmark "
                    "progress is advancing",
                    spec.name,
                )
                nominal_timeout_logged = True
            time.sleep(self.config.poll_interval_sec)

    def _status_error_result(
        self,
        spec: JobSpec,
        app_uri: str,
        consecutive_status_errors: int,
    ) -> JobResult | None:
        if consecutive_status_errors < MAX_CONSECUTIVE_STATUS_ERRORS:
            return None
        return JobResult(
            spec=spec,
            app_uri=app_uri,
            state="STATUS_ERROR",
            log_lines=self._collect_logs(spec, app_uri, state="STATUS_ERROR")
            + [
                "LOG_COLLECTION_STATUS_ERROR: "
                f"`torchx status` failed {consecutive_status_errors} consecutive times"
            ],
        )

    def _no_progress_result(
        self,
        spec: JobSpec,
        app_uri: str,
        idle_s: float,
    ) -> JobResult:
        reason = f"NO_PROGRESS_AFTER_{self.config.progress_idle_timeout_sec}s"
        return JobResult(
            spec=spec,
            app_uri=app_uri,
            state=reason,
            log_lines=self._collect_logs(spec, app_uri, state=reason)
            + [
                "LOG_COLLECTION_NO_PROGRESS: "
                f"benchmark progress did not advance for {int(idle_s)}s"
            ],
        )

    def _timeout_result(self, spec: JobSpec, app_uri: str) -> JobResult:
        state = f"TIMEOUT_AFTER_{self.config.job_timeout_sec}s"
        return JobResult(
            spec=spec,
            app_uri=app_uri,
            state=state,
            log_lines=self._collect_logs(spec, app_uri, state=state),
        )

    def snapshot_result(
        self,
        spec: JobSpec,
        app_uri: str,
        *,
        state: str | None = None,
    ) -> JobResult:
        if state is None:
            state = self._status(app_uri)
        return JobResult(
            spec=spec,
            app_uri=app_uri,
            state=state,
            log_lines=self._collect_logs(spec, app_uri, state=state),
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
            str(spec.tp),
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
            "--concurrency",
            str(self.workload.concurrency),
            "--mode",
            self.workload.mode,
            "--timeout",
            str(self.workload.timeout),
            "--barrier-timeout",
            str(self.config.job_timeout_sec),
            "--pytorch-cuda-alloc-conf",
            spec.pytorch_cuda_alloc_conf,
        ]
        if self.workload.validate_performance_outputs:
            cmd.append("--validate-performance-outputs")
        if spec.connector_key == "nixl":
            cmd.extend(
                [
                    "--nixl-kv-buffer-device",
                    spec.nixl_kv_buffer_device,
                ]
            )
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
        result = self.torchx.status(app_uri)
        if result.returncode != 0:
            return "STATUS_ERROR"
        match = re.search(r"State:\s+([A-Z_]+)", result.stdout)
        return match.group(1) if match else "UNKNOWN"

    def _collect_logs(self, spec: JobSpec, app_uri: str, *, state: str) -> list[str]:
        return self.log_collector.collect(spec, app_uri, state=state)

    def _progress_snapshot(
        self, spec: JobSpec, app_uri: str
    ) -> ProgressSnapshot | None:
        structured_snapshots: list[ProgressSnapshot] = []
        fallback_snapshots: list[ProgressSnapshot] = []
        for role in MastLogCollector.trainer_roles(spec):
            result = self.torchx.log(app_uri, role, WATCHDOG_PROGRESS_LOG_REGEX)
            if result.returncode == 0 and result.stdout:
                snapshot = self._structured_progress_snapshot(
                    result.stdout.splitlines()
                )
                if snapshot is not None:
                    structured_snapshots.append(snapshot)
                    continue

            result = self.torchx.log(
                app_uri, role, WATCHDOG_FALLBACK_PROGRESS_LOG_REGEX
            )
            if result.returncode != 0 or not result.stdout:
                continue
            snapshot = self._read_blocks_progress_snapshot(result.stdout.splitlines())
            if snapshot is not None:
                fallback_snapshots.append(snapshot)
        if structured_snapshots:
            return max(structured_snapshots, key=lambda snapshot: snapshot.key)
        if fallback_snapshots:
            return max(fallback_snapshots, key=lambda snapshot: snapshot.key)
        return None

    @staticmethod
    def _structured_progress_snapshot(
        lines: Sequence[str],
    ) -> ProgressSnapshot | None:
        attempt_index = 0
        previous_completed_requests: int | None = None
        latest_completed_requests = -1
        latest_description = ""
        for line in lines:
            progress = MastBenchmarkRunner.parse_structured_progress_line(line)
            if progress is None:
                continue
            if (
                previous_completed_requests is not None
                and progress.completed_requests < previous_completed_requests
            ):
                attempt_index += 1
            previous_completed_requests = progress.completed_requests
            latest_completed_requests = progress.completed_requests
            latest_description = progress.description

        if latest_completed_requests < 0:
            return None
        if attempt_index:
            latest_description = f"attempt {attempt_index + 1}, {latest_description}"
        return ProgressSnapshot(
            key=f"structured:{attempt_index:04d}:{latest_completed_requests:012d}",
            description=latest_description,
        )

    @staticmethod
    def parse_structured_progress_line(
        line: str,
    ) -> StructuredProgressRecord | None:
        prefix_index = line.find("BENCHMARK_PROGRESS_JSON: ")
        if prefix_index < 0:
            return None
        payload = line[prefix_index + len("BENCHMARK_PROGRESS_JSON: ") :].strip()
        try:
            progress_record = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(progress_record, dict):
            return None
        progress = progress_record.get("progress")
        if not isinstance(progress, dict):
            return None
        completed_requests = progress.get("completed_requests")
        if not isinstance(completed_requests, int):
            return None
        return StructuredProgressRecord(
            completed_requests=completed_requests,
            description=MastBenchmarkRunner.progress_description(progress),
        )

    @staticmethod
    def progress_description(progress: dict[str, object]) -> str:
        completed_requests = progress["completed_requests"]
        total_requests = progress.get("total_requests")
        completed_iters = progress.get("completed_iters")
        total_iters = progress.get("total_iters")
        request_text = (
            f"{completed_requests}/{total_requests} requests"
            if isinstance(total_requests, int)
            else f"{completed_requests} requests"
        )
        if isinstance(completed_iters, int) and isinstance(total_iters, int):
            return f"{completed_iters}/{total_iters} iters, {request_text}"
        return request_text

    @staticmethod
    def _read_blocks_progress_snapshot(lines: Sequence[str]) -> ProgressSnapshot | None:
        max_block_index = -1
        for line in lines:
            match = re.search(r"\bread_blocks\s+(?P<index>\d+)-", line)
            if match is not None:
                max_block_index = max(max_block_index, int(match.group("index")))
        if max_block_index < 0:
            return None
        return ProgressSnapshot(
            key=f"read_blocks:{max_block_index:012d}",
            description=f"read_blocks index {max_block_index}",
        )

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


class MastBenchmarkBatchRunner:
    """Owns MAST app lifecycle for one benchmark script invocation."""

    def __init__(
        self,
        mast: MastBenchmarkRunner,
        specs: Sequence[JobSpec],
        existing_apps: Sequence[tuple[JobSpec, str]],
    ) -> None:
        self.mast = mast
        self.specs = specs
        self.existing_apps = existing_apps
        self.apps: list[MastApp] = []

    def run(self) -> list[JobResult]:
        results: list[JobResult] = []
        completed_app_uris: set[str] = set()
        try:
            self.attach_existing_apps()
            self.launch_owned_apps()
            for app in self.apps:
                result = self.mast.wait_for_result(app.spec, app.app_uri)
                results.append(result)
                completed_app_uris.add(app.app_uri)
                if (
                    result.state in FAILURE_STATES
                    or result.state not in TERMINAL_STATES
                ):
                    self.cancel_unfinished_apps(
                        f"{app.spec.name} ended in state {result.state}"
                    )
                    results.extend(self.snapshot_missing_apps(completed_app_uris))
                    break
        except KeyboardInterrupt:
            self.cancel_unfinished_apps("local benchmark runner interrupted")
            raise
        except Exception:
            self.cancel_unfinished_apps("local benchmark runner failed")
            raise
        return results

    def attach_existing_apps(self) -> None:
        for spec, app_uri in self.existing_apps:
            self.apps.append(MastApp(spec=spec, app_uri=app_uri, owned=False))

    def launch_owned_apps(self) -> None:
        for spec in self.specs:
            self.apps.append(
                MastApp(spec=spec, app_uri=self.mast.launch(spec), owned=True)
            )

    def cancel_unfinished_apps(self, reason: str) -> None:
        for app in self.apps:
            self.mast.cancel(app, reason)

    def snapshot_missing_apps(self, completed_app_uris: set[str]) -> list[JobResult]:
        snapshots: list[JobResult] = []
        for app in self.apps:
            if app.app_uri not in completed_app_uris:
                snapshots.append(self.mast.snapshot_result(app.spec, app.app_uri))
        return snapshots


def parse_args() -> argparse.Namespace:
    benchmark_profiles = get_benchmark_profiles()
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
    parser.add_argument(
        "--profile",
        choices=sorted(benchmark_profiles),
        default="default",
        help="Named workload profile. Explicit workload flags override profile values.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write structured job summaries and gate decisions to this path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Write a compact reviewer-facing benchmark report to this path.",
    )
    parser.add_argument(
        "--compare-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Render a comparison table from existing benchmark JSON reports. "
            "May be specified multiple times. Skips build and launch."
        ),
    )
    parser.add_argument(
        "--output-comparison-markdown",
        type=Path,
        help="Write the --compare-json Markdown comparison table to this path.",
    )
    parser.add_argument(
        "--app-uri",
        action="append",
        default=[],
        help=(
            "Collect status/logs for an existing MAST app instead of launching "
            "new jobs. May be specified multiple times. The script infers "
            "layout, connector, and TP from the app name."
        ),
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Fail if correctness, runtime, transfer, telemetry, or perf gates fail.",
    )
    parser.add_argument(
        "--require-structured-result",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require `BENCHMARK_RESULT_JSON` in logs. Defaults to the selected "
            "profile's requirement."
        ),
    )
    parser.add_argument(
        "--min-measured-requests",
        type=int,
        help="Fail gates if the workload has fewer measured requests.",
    )
    parser.add_argument(
        "--min-measured-output-tokens",
        type=int,
        help="Fail gates if the workload has fewer measured output tokens.",
    )
    parser.add_argument(
        "--allow-recovered-failures",
        action="store_true",
        help="Do not fail solely on transfer failure lines when recovery is allowed.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Previous --output-json report used for median regression gates.",
    )
    parser.add_argument(
        "--max-median-regression-pct",
        type=float,
        help="Fail gates when median latency exceeds matching baseline by this percent.",
    )
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--job-timeout-sec", type=int)
    parser.add_argument(
        "--progress-idle-timeout-sec",
        type=int,
        help=(
            "Fail a running app when benchmark progress has not advanced for this "
            "many seconds. Defaults to the selected profile value."
        ),
    )
    parser.add_argument(
        "--progress-check-interval-sec",
        type=int,
        help=(
            "How often to sample sparse benchmark progress logs while an app is "
            "running. Defaults to the selected profile value."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--tp-values",
        type=int,
        nargs="+",
        help=(
            "Tensor-parallel values to run. Defaults to the single --tp value. "
            "Same-node MAST validation supports TP=1; TP>1 runs are cross-node only."
        ),
    )
    parser.add_argument(
        "--nixl-kv-buffer-device",
        choices=NIXL_KV_BUFFER_DEVICES,
        default="cuda",
        help=(
            "NIXL KV transfer buffer device passed to the benchmark entry point. "
            "The default `cuda` keeps CUDA-direct registration. Pass `cpu` to "
            "stage through host memory for diagnostics, or `auto` to stage TP>1 "
            "through host memory."
        ),
    )
    parser.add_argument(
        "--pytorch-cuda-alloc-conf",
        default="expandable_segments:True",
        help=(
            "PYTORCH_CUDA_ALLOC_CONF passed to the benchmark entry point. "
            "Pass `default` to leave the variable unset inside the benchmark."
        ),
    )
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--measure-iters", type=int)
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        help=(
            "Maximum in-flight disaggregated HTTP requests for accuracy "
            "performance mode. Defaults to the selected profile value."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--mode")
    parser.add_argument("--timeout", type=int)
    parser.add_argument(
        "--validate-performance-outputs",
        action="store_true",
        help=(
            "Validate every measured accuracy-mode performance output against "
            "standalone baseline outputs. This adds baseline work and is intended "
            "for publishable correctness runs."
        ),
    )
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
    if args.tp_values is not None and args.tp != 1:
        logger.warning(
            "--tp=%d is ignored because --tp-values=%s was supplied",
            args.tp,
            args.tp_values,
        )
    tp_values = args.tp_values or [args.tp]
    package_suffix = package_hash_suffix(package)
    run_suffix = time.strftime("%m%d%H%M%S")
    specs: list[JobSpec] = []
    for tp, connector, layout in itertools.product(tp_values, connectors, layouts):
        if layout == "sn" and tp > 1:
            logger.info(
                "Skipping SN %s TP=%d: same-node multi-TP benchmark orchestration "
                "is not supported",
                CONNECTORS[connector],
                tp,
            )
            continue
        specs.append(
            build_job_spec(
                args=args,
                package_suffix=package_suffix,
                run_suffix=run_suffix,
                tp=tp,
                connector=connector,
                layout=layout,
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


def build_job_spec(
    *,
    args: argparse.Namespace,
    package_suffix: str,
    run_suffix: str,
    tp: int,
    connector: str,
    layout: str,
) -> JobSpec:
    connector_short = "uni" if connector == "uniflow" else "nixl"
    nixl_kv_buffer_device = nixl_kv_buffer_device_for_job(
        connector=connector,
        tp=tp,
        requested_device=args.nixl_kv_buffer_device,
    )
    pytorch_cuda_alloc_conf = args.pytorch_cuda_alloc_conf
    connector_part = connector_short
    if connector == "nixl" and nixl_kv_buffer_device != "cuda":
        connector_part = f"{connector_short}-{nixl_kv_buffer_device}kv"
    alloc_key = alloc_conf_key(pytorch_cuda_alloc_conf)
    alloc_part = "" if alloc_key == "expandable" else f"-{alloc_key}alloc"
    name = (
        f"{args.job_prefix}-{layout}-{connector_part}-tp{tp}{alloc_part}-"
        f"{package_suffix}-{run_suffix}"
    )
    session_id_parts = ["codex", package_suffix, layout, connector]
    if connector == "nixl" and nixl_kv_buffer_device != "cuda":
        session_id_parts.append(f"{nixl_kv_buffer_device}kv")
    if alloc_key != "expandable":
        session_id_parts.append(f"{alloc_key}alloc")
    session_id_parts.append(f"tp{tp}")
    session_id = "-".join(session_id_parts)
    return JobSpec(
        connector_key=connector,
        connector_class=CONNECTORS[connector],
        layout=layout,
        tp=tp,
        nixl_kv_buffer_device=nixl_kv_buffer_device,
        pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
        name=name,
        session_id=session_id,
    )


def nixl_kv_buffer_device_for_job(
    *,
    connector: str,
    tp: int,
    requested_device: str,
) -> str:
    if connector != "nixl":
        return "cuda"
    if requested_device != "auto":
        return requested_device
    return "cpu" if tp > 1 else "cuda"


def make_existing_app_specs(app_uris: Sequence[str]) -> list[tuple[JobSpec, str]]:
    specs: list[tuple[JobSpec, str]] = []
    for app_uri in app_uris:
        spec = infer_job_spec_from_app_uri(app_uri)
        specs.append((spec, app_uri))
    return specs


def infer_job_spec_from_app_uri(app_uri: str) -> JobSpec:
    name = app_uri.rstrip("/").rsplit("/", 1)[-1]
    if name.startswith("torchx-"):
        name = name.removeprefix("torchx-")

    layout = infer_layout_from_name(name)
    connector_key = infer_connector_from_name(name)
    tp = infer_tp_from_name(name)
    nixl_kv_buffer_device = infer_nixl_kv_buffer_device_from_name(name)
    pytorch_cuda_alloc_conf = infer_pytorch_cuda_alloc_conf_from_name(name)
    return JobSpec(
        connector_key=connector_key,
        connector_class=CONNECTORS[connector_key],
        layout=layout,
        tp=tp,
        nixl_kv_buffer_device=nixl_kv_buffer_device,
        pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
        name=name,
        session_id="existing-mast-app",
    )


def infer_layout_from_name(name: str) -> str:
    if "-cn-" in name:
        return "cn"
    if "-sn-" in name:
        return "sn"
    raise ValueError(f"Could not infer layout from MAST app name `{name}`")


def infer_connector_from_name(name: str) -> str:
    if "-uni-" in name or "-uniflow-" in name:
        return "uniflow"
    if "-nixl-" in name:
        return "nixl"
    raise ValueError(f"Could not infer connector from MAST app name `{name}`")


def infer_tp_from_name(name: str) -> int:
    match = re.search(r"-tp(?P<tp>\d+)(?:-|$)", name)
    if match is None:
        raise ValueError(f"Could not infer TP from MAST app name `{name}`")
    return int(match.group("tp"))


def infer_nixl_kv_buffer_device_from_name(name: str) -> str:
    match = re.search(r"-nixl-(?P<device>cpu|cuda)kv-", name)
    if match is None:
        return "cuda"
    return match.group("device")


def infer_pytorch_cuda_alloc_conf_from_name(name: str) -> str:
    match = ALLOC_CONF_TOKEN_RE.search(name)
    if match is None:
        return "expandable_segments:True"
    return AllocConfToken.decode(match.group("token"))


def alloc_conf_key(pytorch_cuda_alloc_conf: str) -> str:
    return AllocConfToken.encode(pytorch_cuda_alloc_conf)


def print_summary(package: str, results: Sequence[JobResult]) -> None:
    logger.info("\nSummary")
    logger.info("Package: %s", package)
    parser = BenchmarkLogParser()
    for result in results:
        summary = parser.parse(result.log_lines)
        comparison_status = "USABLE" if result.passed else "UNUSABLE"
        runtime_status = "CLEAN" if not summary.fatal_runtime_errors else "ISSUES"
        logger.info(
            "comparison=%s runtime=%s layout=%s connector=%s tp=%d state=%s app_uri=%s",
            comparison_status,
            runtime_status,
            result.spec.layout.upper(),
            result.spec.connector_class,
            result.spec.tp,
            result.state,
            result.app_uri,
        )
        if summary.post_result_runtime_errors:
            logger.info(
                "  post_result_runtime_errors=%d",
                len(summary.post_result_runtime_errors),
            )
        for line in result.log_lines:
            logger.info("  %s", line)


def print_gate_summary(report: BenchmarkReport) -> None:
    logger.info("\nGate Summary")
    for job in report.job_records():
        gate = job["gate"]
        status = "PASS" if gate["passed"] else "FAIL"
        logger.info("%s: %s %s %s", status, job["key"], job["state"], job["app_uri"])
        for failure in gate["failures"]:
            logger.info("  failure: %s", failure)
        for warning in gate["warnings"]:
            logger.info("  warning: %s", warning)


def profile_value(args: argparse.Namespace, profile: BenchmarkProfile, name: str):
    value = getattr(args, name)
    return value if value is not None else getattr(profile, name)


def make_workload(args: argparse.Namespace, profile: BenchmarkProfile) -> Workload:
    return Workload(
        model=args.model,
        tp=args.tp,
        warmup_iters=profile_value(args, profile, "warmup_iters"),
        measure_iters=profile_value(args, profile, "measure_iters"),
        prompt_len=profile_value(args, profile, "prompt_len"),
        num_prompts=profile_value(args, profile, "num_prompts"),
        output_len=profile_value(args, profile, "output_len"),
        concurrency=profile_value(args, profile, "concurrency"),
        gpu_memory_utilization=args.gpu_memory_utilization,
        mode=profile_value(args, profile, "mode"),
        timeout=profile_value(args, profile, "timeout"),
        validate_performance_outputs=args.validate_performance_outputs,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.compare_json:
        comparison = BenchmarkComparisonReport(args.compare_json)
        rendered = comparison.render()
        logger.info("%s", rendered.rstrip())
        if args.output_comparison_markdown is not None:
            comparison.write_markdown(args.output_comparison_markdown)
            logger.info(
                "Wrote comparison Markdown report: %s",
                args.output_comparison_markdown,
            )
        return 0 if comparison.passed() else 1

    if args.skip_build and not args.package:
        raise SystemExit("--skip-build requires --package")
    if args.max_median_regression_pct is not None and args.baseline_json is None:
        raise SystemExit("--max-median-regression-pct requires --baseline-json")

    runner = CommandRunner(quiet_warnings=args.quiet_warnings)
    package = args.package or (
        "existing-mast-app"
        if args.app_uri
        else FbpkgBuilder(runner, args.fbcode_dir, args.target).build()
    )
    profile = get_benchmark_profiles()[args.profile]
    workload = make_workload(args, profile)
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
        job_timeout_sec=profile_value(args, profile, "job_timeout_sec"),
        progress_idle_timeout_sec=profile_value(
            args, profile, "progress_idle_timeout_sec"
        ),
        progress_check_interval_sec=profile_value(
            args, profile, "progress_check_interval_sec"
        ),
    )
    mast = MastBenchmarkRunner(runner, mast_config, workload, package)

    specs: list[JobSpec]
    existing_apps: list[tuple[JobSpec, str]]
    if args.app_uri:
        specs = []
        existing_apps = make_existing_app_specs(args.app_uri)
    else:
        specs = make_job_specs(args, package)
        existing_apps = []

    results = MastBenchmarkBatchRunner(mast, specs, existing_apps).run()
    report = BenchmarkReport(
        package=package,
        profile=profile,
        workload=workload,
        results=results,
        parser=BenchmarkLogParser(),
        evaluator=BenchmarkGateEvaluator(
            enforce_gates=args.enforce_gates,
            allow_recovered_failures=args.allow_recovered_failures,
            require_structured_result=profile_value(
                args, profile, "require_structured_result"
            ),
            workload=workload,
            min_measured_requests=profile_value(args, profile, "min_measured_requests"),
            min_measured_output_tokens=profile_value(
                args, profile, "min_measured_output_tokens"
            ),
            baseline=BenchmarkBaseline.load(args.baseline_json),
            max_median_regression_pct=args.max_median_regression_pct,
        ),
    )
    print_summary(package, results)
    print_gate_summary(report)
    if args.output_json is not None:
        report.write_json(args.output_json)
        logger.info("Wrote JSON report: %s", args.output_json)
    if args.output_markdown is not None:
        BenchmarkMarkdownWriter(report).write(args.output_markdown)
        logger.info("Wrote Markdown report: %s", args.output_markdown)
    return 0 if report.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
