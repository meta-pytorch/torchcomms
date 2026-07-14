# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from comms.uniflow.benchmarks import run_mast_benchmarks


class RunMastBenchmarksTest(unittest.TestCase):
    def test_default_nixl_kv_buffer_device_preserves_cuda_direct(self) -> None:
        with patch(
            "sys.argv",
            ["run_mast_benchmarks.py", "--package", "pkg:abcdef0"],
        ):
            args = run_mast_benchmarks.parse_args()

        self.assertEqual(args.nixl_kv_buffer_device, "cuda")

    def test_nixl_cpu_kv_buffer_device_is_encoded_and_passed(self) -> None:
        args = argparse.Namespace(
            job_prefix="bench",
            nixl_kv_buffer_device="cpu",
            pytorch_cuda_alloc_conf="expandable_segments:True",
        )
        spec = run_mast_benchmarks.build_job_spec(
            args=args,
            package_suffix="abcdef0",
            run_suffix="010203",
            tp=2,
            connector="nixl",
            layout="cn",
        )

        self.assertEqual(spec.name, "bench-cn-nixl-cpukv-tp2-abcdef0-010203")
        self.assertEqual(spec.session_id, "codex-abcdef0-cn-nixl-cpukv-tp2")
        self.assertEqual(spec.nixl_kv_buffer_device, "cpu")
        self.assertEqual(spec.pytorch_cuda_alloc_conf, "expandable_segments:True")
        self.assertEqual(run_mast_benchmarks.job_key(spec), "cn:nixl:tp2:kv-cpu")
        inferred = run_mast_benchmarks.infer_job_spec_from_app_uri(
            "mast://torchx/torchx-" + spec.name
        )
        self.assertEqual(inferred.nixl_kv_buffer_device, "cpu")
        self.assertEqual(run_mast_benchmarks.job_key(inferred), "cn:nixl:tp2:kv-cpu")

        command = self.launch_command(spec)

        self.assertIn("--nixl-kv-buffer-device", command)
        self.assertEqual(command[command.index("--nixl-kv-buffer-device") + 1], "cpu")
        self.assertIn("--pytorch-cuda-alloc-conf", command)
        self.assertEqual(
            command[command.index("--pytorch-cuda-alloc-conf") + 1],
            "expandable_segments:True",
        )

    def test_uniflow_job_does_not_get_nixl_kv_buffer_argument(self) -> None:
        args = argparse.Namespace(
            job_prefix="bench",
            nixl_kv_buffer_device="cpu",
            pytorch_cuda_alloc_conf="expandable_segments:True",
        )
        spec = run_mast_benchmarks.build_job_spec(
            args=args,
            package_suffix="abcdef0",
            run_suffix="010203",
            tp=2,
            connector="uniflow",
            layout="cn",
        )

        self.assertEqual(spec.name, "bench-cn-uni-tp2-abcdef0-010203")
        self.assertEqual(spec.nixl_kv_buffer_device, "cuda")
        self.assertEqual(run_mast_benchmarks.job_key(spec), "cn:uniflow:tp2")
        self.assertNotIn("--nixl-kv-buffer-device", self.launch_command(spec))

    def test_auto_nixl_kv_buffer_device_for_job(self) -> None:
        cases = [
            ("nixl", 1, "cuda"),
            ("nixl", 2, "cpu"),
            ("nixl", 4, "cpu"),
            ("uniflow", 2, "cuda"),
        ]
        for connector, tp, expected_device in cases:
            with self.subTest(connector=connector, tp=tp):
                self.assertEqual(
                    run_mast_benchmarks.nixl_kv_buffer_device_for_job(
                        connector=connector,
                        tp=tp,
                        requested_device="auto",
                    ),
                    expected_device,
                )

    def test_validate_performance_outputs_is_passed_to_benchmark(self) -> None:
        args = argparse.Namespace(
            job_prefix="bench",
            nixl_kv_buffer_device="cuda",
            pytorch_cuda_alloc_conf="expandable_segments:True",
        )
        spec = run_mast_benchmarks.build_job_spec(
            args=args,
            package_suffix="abcdef0",
            run_suffix="010203",
            tp=1,
            connector="uniflow",
            layout="cn",
        )

        command = self.launch_command(spec, validate_performance_outputs=True)

        self.assertIn("--validate-performance-outputs", command)

    def test_non_default_cuda_alloc_conf_is_encoded_and_passed(self) -> None:
        args = argparse.Namespace(
            job_prefix="bench",
            nixl_kv_buffer_device="cuda",
            pytorch_cuda_alloc_conf="default",
        )
        spec = run_mast_benchmarks.build_job_spec(
            args=args,
            package_suffix="abcdef0",
            run_suffix="010203",
            tp=2,
            connector="nixl",
            layout="cn",
        )

        self.assertEqual(spec.name, "bench-cn-nixl-tp2-defaultalloc-abcdef0-010203")
        self.assertEqual(spec.session_id, "codex-abcdef0-cn-nixl-defaultalloc-tp2")
        self.assertEqual(spec.pytorch_cuda_alloc_conf, "default")
        self.assertEqual(
            run_mast_benchmarks.job_key(spec),
            "cn:nixl:tp2:alloc-default",
        )
        inferred = run_mast_benchmarks.infer_job_spec_from_app_uri(
            "mast://torchx/torchx-" + spec.name
        )
        self.assertEqual(inferred.pytorch_cuda_alloc_conf, "default")
        self.assertEqual(
            run_mast_benchmarks.job_key(inferred),
            "cn:nixl:tp2:alloc-default",
        )

        command = self.launch_command(spec)

        self.assertIn("--pytorch-cuda-alloc-conf", command)
        self.assertEqual(
            command[command.index("--pytorch-cuda-alloc-conf") + 1],
            "default",
        )

    def test_custom_cuda_alloc_conf_round_trips_from_job_name(self) -> None:
        cuda_alloc_conf = "max_split_size_mb:256"
        args = argparse.Namespace(
            job_prefix="bench",
            nixl_kv_buffer_device="cuda",
            pytorch_cuda_alloc_conf=cuda_alloc_conf,
        )
        spec = run_mast_benchmarks.build_job_spec(
            args=args,
            package_suffix="abcdef0",
            run_suffix="010203",
            tp=2,
            connector="nixl",
            layout="cn",
        )
        alloc_key = run_mast_benchmarks.alloc_conf_key(cuda_alloc_conf)

        self.assertEqual(
            alloc_key,
            "hex_6d61785f73706c69745f73697a655f6d623a323536",
        )
        self.assertIn(f"-{alloc_key}alloc-", spec.name)
        self.assertEqual(
            run_mast_benchmarks.job_key(spec),
            f"cn:nixl:tp2:alloc-{alloc_key}",
        )
        inferred = run_mast_benchmarks.infer_job_spec_from_app_uri(
            "mast://torchx/torchx-" + spec.name
        )
        self.assertEqual(inferred.pytorch_cuda_alloc_conf, cuda_alloc_conf)
        self.assertEqual(
            run_mast_benchmarks.job_key(inferred),
            f"cn:nixl:tp2:alloc-{alloc_key}",
        )

        command = self.launch_command(spec)

        self.assertIn("--pytorch-cuda-alloc-conf", command)
        self.assertEqual(
            command[command.index("--pytorch-cuda-alloc-conf") + 1],
            cuda_alloc_conf,
        )

    def test_invalid_alloc_conf_token_fails_at_app_uri_parse(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid alloc-conf token"):
            run_mast_benchmarks.infer_job_spec_from_app_uri(
                "mast://torchx/torchx-bench-cn-nixl-tp2-hex_badalloc-abcdef0-010203"
            )

        with self.assertRaisesRegex(ValueError, "Unrecognized alloc-conf token"):
            run_mast_benchmarks.infer_job_spec_from_app_uri(
                "mast://torchx/torchx-bench-cn-nixl-tp2-unknownalloc-abcdef0-010203"
            )

    def test_expired_kv_requests_ignore_unrelated_expired_lines(self) -> None:
        summary = run_mast_benchmarks.BenchmarkLogParser().parse(
            [
                "trainer/0 [0]:certificate expired while fetching logs",
                "trainer/0 [0]:expired KV request for block table",
                "trainer/0 [0]:kv cache request expired before decode",
            ]
        )

        self.assertEqual(summary.expired_kv_requests, 2)

    def test_progress_snapshot_advances_after_restart(self) -> None:
        lines = [
            self.progress_line(completed_iters=9, completed_requests=4500),
            self.progress_line(completed_iters=10, completed_requests=5000),
            self.progress_line(completed_iters=1, completed_requests=500),
            self.progress_line(completed_iters=2, completed_requests=1000),
        ]

        snapshot = (
            run_mast_benchmarks.MastBenchmarkRunner._structured_progress_snapshot(lines)
        )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.key, "structured:0001:000000001000")
        self.assertEqual(
            snapshot.description, "attempt 2, 2/12 iters, 1000/6000 requests"
        )

    def test_post_result_traceback_keeps_comparison_result_usable(self) -> None:
        result = {
            "benchmark": "disagg_prefill_decode",
            "connector": "NixlConnector",
            "correctness": {"mismatches": 0, "passed": True, "total_checked": 2},
            "mode": "accuracy",
            "performance": {
                "concurrency": 4,
                "median_s": 312.6276,
                "num_prompts": 500,
                "req_per_s": 1.5993,
            },
            "schema_version": 1,
            "status": "passed",
        }
        lines = [
            "trainer/0 [0]:CORRECTNESS PASSED: 2/2 exact match",
            "trainer/0 [0]:PERFORMANCE: median=312.63s (1.6 req/s)",
            "trainer/0 [0]:BENCHMARK_RESULT_JSON: " + json.dumps(result),
            "trainer/0 [0]:ALL ACCURACY TESTS PASSED",
            "trainer/0 [0]:Exception in thread nixl_handshake_listener:",
            "trainer/0 [0]:Traceback (most recent call last):",
            "trainer/0 [0]:msgspec.DecodeError: MessagePack data is malformed",
        ]

        summary = run_mast_benchmarks.BenchmarkLogParser().parse(lines)

        self.assertTrue(summary.result_usable_for_comparison)
        self.assertEqual(summary.mismatch_count, 0)
        self.assertEqual(summary.median_s, 312.6276)
        self.assertEqual(summary.req_per_s, 1.5993)
        self.assertGreaterEqual(len(summary.fatal_runtime_errors), 1)
        self.assertEqual(len(summary.post_result_runtime_errors), 1)

    def test_post_result_traceback_still_fails_runtime_gate(self) -> None:
        result = {
            "benchmark": "disagg_prefill_decode",
            "connector": "NixlConnector",
            "correctness": {"mismatches": 0, "passed": True, "total_checked": 2},
            "mode": "accuracy",
            "performance": {
                "median_s": 87.1988,
                "req_per_s": 2.867,
            },
            "schema_version": 1,
            "status": "passed",
        }
        lines = [
            "trainer/0 [0]:CORRECTNESS PASSED: 2/2 exact match",
            "trainer/0 [0]:PERFORMANCE: median=87.20s (2.9 req/s)",
            "trainer/0 [0]:BENCHMARK_RESULT_JSON: " + json.dumps(result),
            "trainer/0 [0]:ALL ACCURACY TESTS PASSED",
            "trainer/1 [0]:Exception in thread nixl_handshake_listener:",
            "trainer/1 [0]:Traceback (most recent call last):",
        ]
        summary = run_mast_benchmarks.BenchmarkLogParser().parse(lines)
        failures: list[str] = []
        warnings: list[str] = []

        run_mast_benchmarks.BenchmarkGateEvaluator.add_runtime_failures(
            summary, failures, warnings
        )

        self.assertEqual(failures, ["1 fatal runtime log lines"])
        self.assertEqual(warnings, ["1 post-result runtime log lines"])

    def test_job_result_passed_uses_structured_result(self) -> None:
        result = {
            "benchmark": "disagg_prefill_decode",
            "connector": "UniflowConnector",
            "correctness": {"mismatches": 0, "passed": True, "total_checked": 2},
            "mode": "accuracy",
            "performance": {
                "median_s": 10.0,
                "req_per_s": 4.0,
            },
            "schema_version": 1,
            "status": "passed",
        }
        job = run_mast_benchmarks.JobResult(
            spec=self.job_spec(),
            app_uri="mast://torchx/torchx-bench-cn-uni-tp1",
            state="SUCCEEDED",
            log_lines=[
                "trainer/0 [0]:BENCHMARK_RESULT_JSON: " + json.dumps(result),
                "trainer/0 [0]:previous run had RuntimeError but recovered",
            ],
        )

        self.assertTrue(job.passed)

    def test_print_summary_distinguishes_usable_result_from_runtime_clean(
        self,
    ) -> None:
        result = {
            "benchmark": "disagg_prefill_decode",
            "connector": "UniflowConnector",
            "correctness": {"mismatches": 0, "passed": True, "total_checked": 2},
            "mode": "accuracy",
            "performance": {
                "median_s": 10.0,
                "req_per_s": 4.0,
            },
            "schema_version": 1,
            "status": "passed",
        }
        job = run_mast_benchmarks.JobResult(
            spec=self.job_spec(),
            app_uri="mast://torchx/torchx-bench-cn-uni-tp1",
            state="SUCCEEDED",
            log_lines=[
                "trainer/0 [0]:BENCHMARK_RESULT_JSON: " + json.dumps(result),
                "trainer/0 [0]:Traceback (most recent call last):",
            ],
        )

        with self.assertLogs(run_mast_benchmarks.logger, level="INFO") as logs:
            run_mast_benchmarks.print_summary("pkg:abcdef0", [job])

        rendered = "\n".join(logs.output)
        self.assertIn("comparison=USABLE runtime=ISSUES", rendered)
        self.assertIn("post_result_runtime_errors=1", rendered)

    def test_make_job_specs_logs_when_tp_values_shadow_tp(self) -> None:
        args = argparse.Namespace(
            connector="uniflow",
            layout="cn",
            tp=4,
            tp_values=[1, 2],
            job_prefix="bench",
            nixl_kv_buffer_device="cuda",
            pytorch_cuda_alloc_conf="expandable_segments:True",
        )

        with self.assertLogs(run_mast_benchmarks.logger, level="WARNING") as logs:
            specs = run_mast_benchmarks.make_job_specs(args, "pkg:abcdef0")

        self.assertEqual([spec.tp for spec in specs], [1, 2])
        self.assertIn("--tp=4 is ignored", "\n".join(logs.output))

    @staticmethod
    def job_spec() -> run_mast_benchmarks.JobSpec:
        return run_mast_benchmarks.JobSpec(
            connector_key="uniflow",
            connector_class="UniflowConnector",
            layout="cn",
            tp=1,
            nixl_kv_buffer_device="cuda",
            pytorch_cuda_alloc_conf="expandable_segments:True",
            name="bench-cn-uni-tp1",
            session_id="codex-abcdef0-cn-uniflow-tp1",
        )

    @staticmethod
    def progress_line(*, completed_iters: int, completed_requests: int) -> str:
        payload = {
            "benchmark": "disagg_prefill_decode",
            "connector": "NixlConnector",
            "mode": "accuracy",
            "phase": "performance",
            "progress": {
                "completed_iters": completed_iters,
                "completed_requests": completed_requests,
                "last_iter_s": 300.0,
                "total_iters": 12,
                "total_requests": 6000,
            },
            "schema_version": 1,
        }
        return "trainer/0 [0]:BENCHMARK_PROGRESS_JSON: " + json.dumps(payload)

    @staticmethod
    def launch_command(
        spec: run_mast_benchmarks.JobSpec,
        *,
        validate_performance_outputs: bool = False,
    ) -> list[str]:
        runner = run_mast_benchmarks.MastBenchmarkRunner.__new__(
            run_mast_benchmarks.MastBenchmarkRunner
        )
        runner.config = run_mast_benchmarks.MastConfig(
            workspace=Path("/tmp"),
            bha_config_path=Path("/tmp/bha_uniflow_split_2x8.json"),
            hpc_identity="identity",
            hpc_cluster_uuid="cluster",
            rm_attribution="attribution",
            hpc_job_oncall="ncclx",
            locality_constraints="constraints",
            model_type_name="model_type",
            opec_tag="opec",
            host_type="host",
            presto_identity="presto",
            poll_interval_sec=30,
            job_timeout_sec=300,
            progress_idle_timeout_sec=300,
            progress_check_interval_sec=300,
        )
        runner.workload = run_mast_benchmarks.Workload(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tp=spec.tp,
            warmup_iters=1,
            measure_iters=1,
            prompt_len=16,
            num_prompts=1,
            output_len=4,
            concurrency=1,
            gpu_memory_utilization=0.9,
            mode="accuracy",
            timeout=60,
            validate_performance_outputs=validate_performance_outputs,
        )
        runner.package = "pkg:abcdef0"
        return runner._launch_command(spec)
