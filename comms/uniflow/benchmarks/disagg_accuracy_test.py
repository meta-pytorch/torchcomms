# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Disaggregated serving accuracy and performance test.

Replicates vLLM's production test methodology:
  1. Standalone baseline server (no disagg) → reference outputs
  2. Disagg pipeline (prefill + decode + proxy) → exact match
  3. Performance benchmark through disagg proxy

Usage:
  torchx run fb.dist.ddp -j 1x2 -m comms.uniflow.benchmarks.disagg_accuracy_test \
    -- --connector UniflowConnector --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    logger.info(msg)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class ServerConfig:
    """Configuration for a single vLLM server instance."""

    model_path: str
    port: int
    gpu_id: str
    block_size: int = 128
    gpu_memory_utilization: float = 0.45
    max_model_len: int = 2048
    kv_transfer_config: str | None = None
    side_channel_port: int | None = None
    host_ip: str = "127.0.0.1"

    def to_cmd(self) -> list[str]:
        # In MAST PAR environment, use the PAR's Python to invoke
        # vllm.entrypoints.openai.api_server. On devservers, use vllm CLI.
        par_path = os.environ.get("FBPKG_ROOT", "")
        if par_path and (Path(par_path) / "penv.par").exists():
            python = str(Path(par_path) / "penv.par")
            cmd = [python, "-m", "vllm.entrypoints.openai.api_server"]
        else:
            cmd = ["vllm", "serve"]
        cmd += [
            self.model_path,
            "--port",
            str(self.port),
            "--block-size",
            str(self.block_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--enforce-eager",
            "--seed",
            "42",
        ]
        if self.kv_transfer_config:
            cmd += ["--kv-transfer-config", self.kv_transfer_config]
        return cmd

    def to_env(self) -> dict[str, str]:
        env = {
            "CUDA_VISIBLE_DEVICES": self.gpu_id,
            "VLLM_HOST_IP": self.host_ip,
        }
        if self.side_channel_port:
            env["VLLM_UNIFLOW_SIDE_CHANNEL_PORT"] = str(self.side_channel_port)
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(self.side_channel_port)
        return env


class ProcessStderrLog:
    """Owns a child process stderr file without risking pipe backpressure."""

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._file = None
        self.path: Path | None = None

    def open(self):
        self._file = tempfile.NamedTemporaryFile(
            prefix=self._prefix,
            suffix=".stderr.log",
            delete=False,
        )
        self.path = Path(self._file.name)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def tail(self, max_bytes: int = 4096) -> str:
        if self._file is not None:
            self._file.flush()
        if self.path is None or not self.path.exists():
            return ""
        with self.path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            return log_file.read().decode(errors="replace")


@dataclass
class TestConfig:
    """Top-level test configuration."""

    connector: str = "UniflowConnector"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    block_size: int = 128
    gpu_memory_utilization: float = 0.45
    max_model_len: int = 2048
    num_perf_prompts: int = 250
    output_len: int = 32
    perf_warmup: int = 3
    perf_iterations: int = 5
    skip_perf: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TestConfig:
        return cls(
            connector=args.connector,
            model=args.model,
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            num_perf_prompts=args.num_prompts,
            output_len=args.output_len,
            perf_warmup=args.perf_warmup,
            perf_iterations=args.perf_iterations,
            skip_perf=args.skip_perf,
        )

    def kv_transfer_config(self) -> str:
        if self.connector == "UniflowConnector":
            return json.dumps(
                {
                    "kv_connector": "UniflowConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "vllm.fb.distributed.vllm_kv_connector.uniflow_connector",
                }
            )
        return json.dumps(
            {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            }
        )


# ── Server Lifecycle ───────────────────────────────────────────────────


class VLLMServer:
    """Manages the lifecycle of a single vLLM server process."""

    def __init__(self, config: ServerConfig, label: str):
        self._config = config
        self._label = label
        self._process: subprocess.Popen | None = None
        self._stderr_log = ProcessStderrLog(f"disagg_{label}_")

    def start(self) -> None:
        env = {**os.environ, **self._config.to_env()}
        cmd = self._config.to_cmd()
        stderr_file = self._stderr_log.open()
        _log(
            f"[{self._label}] Starting: CUDA_VISIBLE_DEVICES={self._config.gpu_id} "
            f"port={self._config.port} stderr={self._stderr_log.path}"
        )
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )

    def wait_until_ready(self, timeout: int = 300) -> None:
        url = f"http://localhost:{self._config.port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=5)
                _log(f"[{self._label}] Ready on port {self._config.port}")
                return
            except (OSError, urllib.error.URLError):
                if self._process and self._process.poll() is not None:
                    raise RuntimeError(
                        f"[{self._label}] Server exited with code "
                        f"{self._process.returncode}:\n{self._stderr_log.tail()}"
                    )
                time.sleep(1)
        raise TimeoutError(f"[{self._label}] Not ready after {timeout}s")

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            _log(f"[{self._label}] Stopped")
        self._process = None
        self._stderr_log.close()


class ProxyServer:
    """Routes requests: client → prefill → decode."""

    def __init__(self, prefill_port: int, decode_port: int, proxy_port: int = 8192):
        self._prefill_port = prefill_port
        self._decode_port = decode_port
        self._proxy_port = proxy_port
        self._process: subprocess.Popen | None = None
        self._stderr_log = ProcessStderrLog("disagg_proxy_")

    @property
    def url(self) -> str:
        return f"http://localhost:{self._proxy_port}"

    def start(self) -> None:
        # Find the proxy script in the fbpkg
        proxy_script = self._find_proxy_script()
        par_path = os.environ.get("FBPKG_ROOT", "")
        if par_path and (Path(par_path) / "penv.par").exists():
            python = str(Path(par_path) / "penv.par")
        else:
            python = sys.executable
        cmd = [
            python,
            proxy_script,
            "--prefiller-host",
            "localhost",
            "--prefiller-port",
            str(self._prefill_port),
            "--decoder-host",
            "localhost",
            "--decoder-port",
            str(self._decode_port),
            "--host",
            "0.0.0.0",
            "--port",
            str(self._proxy_port),
        ]
        stderr_file = self._stderr_log.open()
        _log(
            f"[proxy] Starting on port {self._proxy_port} "
            f"stderr={self._stderr_log.path}"
        )
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
        self.wait_until_ready()

    def wait_until_ready(self, timeout: int = 60) -> None:
        url = f"http://localhost:{self._proxy_port}/healthcheck"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=5)
                _log(f"[proxy] Ready on port {self._proxy_port}")
                return
            except (OSError, urllib.error.URLError):
                if self._process and self._process.poll() is not None:
                    raise RuntimeError(
                        "[proxy] Server exited with code "
                        f"{self._process.returncode}:\n{self._stderr_log.tail()}"
                    )
                time.sleep(1)
        raise TimeoutError(f"[proxy] Not ready after {timeout}s")

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            _log("[proxy] Stopped")
        self._process = None
        self._stderr_log.close()

    def _find_proxy_script(self) -> str:
        candidates = [
            Path(__file__).parent / "toy_proxy_server.py",
        ]
        par_path = os.environ.get("FBPKG_ROOT", "")
        if par_path:
            candidates.append(Path(par_path) / "toy_proxy_server.py")
        for p in Path("/packages").glob("*/toy_proxy_server.py"):
            candidates.append(p)
        for p in candidates:
            if p.exists():
                return str(p)
        raise FileNotFoundError("toy_proxy_server.py not found")


# ── Test Client ────────────────────────────────────────────────────────


class TestClient:
    """Sends requests to a vLLM-compatible endpoint and collects outputs."""

    # Same prompts as vLLM's test_disagg_accuracy.py
    CORRECTNESS_PROMPTS = [
        "Red Hat is the best company in the world to work for because it works "
        "on open source software, which means that all the contributions are "
        "delivered to the community. As a result, when working on projects like "
        "vLLM we are able to meet many amazing people from various organizations "
        "like AMD, Google, NVIDIA, ",
        "We hold these truths to be self-evident, that all men are created "
        "equal, that they are endowed by their Creator with certain unalienable "
        "Rights, that among these are Life, Liberty and the pursuit of "
        "Happiness.--That to secure these rights, Governments are instituted "
        "among Men, deriving their just powers from the consent of the "
        "governed, ",
    ]

    def __init__(self, base_url: str, model: str):
        self._url = f"{base_url}/v1/completions"
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        body = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 42,
            }
        ).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["text"]

    def generate_batch(self, prompts: list[str], max_tokens: int = 30) -> list[str]:
        return [self.generate(p, max_tokens) for p in prompts]


# ── Test Phases ────────────────────────────────────────────────────────


class CorrectnessTest:
    """Exact-match check between captured baseline and disagg outputs."""

    def __init__(self, baseline_outputs: dict[str, str], client_disagg: TestClient):
        self._baseline_outputs = baseline_outputs
        self._disagg = client_disagg

    @staticmethod
    def collect_baseline(client_baseline: TestClient) -> dict[str, str]:
        prompts = TestClient.CORRECTNESS_PROMPTS
        _log("── Correctness: generating baseline outputs ──")
        baseline_outputs = {}
        for i, prompt in enumerate(prompts):
            text = client_baseline.generate(prompt)
            baseline_outputs[prompt] = text
            _log(f"  [baseline] prompt {i}: {text[:60]!r}")
        return baseline_outputs

    def run(self) -> bool:
        prompts = TestClient.CORRECTNESS_PROMPTS
        _log("── Correctness: generating disagg outputs ──")
        mismatches = 0
        for i, p in enumerate(prompts):
            text = self._disagg.generate(p)
            expected = self._baseline_outputs[p]
            match = text == expected
            _log(f"  [disagg]   prompt {i}: match={match} {text[:60]!r}")
            if not match:
                _log(f"    EXPECTED: {expected[:60]!r}")
                mismatches += 1

        if mismatches == 0:
            _log(f"CORRECTNESS PASSED: {len(prompts)}/{len(prompts)} exact match")
            return True
        _log(f"CORRECTNESS FAILED: {mismatches}/{len(prompts)} mismatches")
        return False


@dataclass
class PerfResult:
    """Performance measurement result."""

    num_prompts: int
    output_len: int
    times_s: list[float]
    median_s: float = 0.0
    mean_s: float = 0.0
    req_per_s: float = 0.0

    def __post_init__(self):
        if self.times_s:
            self.median_s = statistics.median(self.times_s)
            self.mean_s = statistics.mean(self.times_s)
            self.req_per_s = self.num_prompts / self.median_s if self.median_s else 0


class PerformanceBenchmark:
    """Phase 2: Throughput measurement through disagg proxy."""

    def __init__(self, client: TestClient, config: TestConfig):
        self._client = client
        self._config = config

    def run(self) -> PerfResult:
        n = self._config.num_perf_prompts
        out_len = self._config.output_len
        base = "The quick brown fox jumps over the lazy dog. "
        text = (base * 17)[:512]
        prompts = [f"{text} Question {i}: What comes next?" for i in range(n)]

        _log(f"── Performance: {n} prompts, {out_len} output tokens ──")

        _log("  Warmup...")
        for p in prompts[: min(self._config.perf_warmup, n)]:
            self._client.generate(p, out_len)

        times: list[float] = []
        for iteration in range(self._config.perf_iterations):
            t0 = time.perf_counter()
            for p in prompts:
                self._client.generate(p, out_len)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            _log(f"  Iter {iteration}: {elapsed:.2f}s ({n / elapsed:.1f} req/s)")

        result = PerfResult(num_prompts=n, output_len=out_len, times_s=times)
        _log(
            f"PERFORMANCE: median={result.median_s:.2f}s "
            f"mean={result.mean_s:.2f}s "
            f"({result.req_per_s:.1f} req/s)"
        )
        return result


# ── Orchestrator ───────────────────────────────────────────────────────


class Topology:
    """Detects the MAST rank topology for GPU assignment."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 2))
        self.local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size))
        )
        self.num_nodes = self.world_size // self.local_world_size
        self.gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", str(self.local_rank))

    @property
    def is_same_node(self) -> bool:
        return self.num_nodes == 1

    @property
    def is_driver(self) -> bool:
        return self.rank == 0

    def detect_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
                sock.connect(("2001:4860:4860::8888", 80))
                return sock.getsockname()[0]
        except OSError:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]


class DisaggAccuracyTest:
    """Main orchestrator — runs correctness + performance tests.

    Flow:
      1. Download model
      2. Start standalone baseline server → generate reference outputs → stop
      3. Start disagg pipeline (prefill + decode + proxy)
      4. Correctness: exact match against baseline
      5. Performance: throughput measurement
      6. Cleanup
    """

    BASELINE_PORT = 8000
    PREFILL_PORT = 8100
    DECODE_PORT = 8200
    PROXY_PORT = 8192

    def __init__(self, config: TestConfig, topo: Topology):
        self._config = config
        self._topo = topo
        self._servers: list[VLLMServer] = []
        self._proxy: ProxyServer | None = None
        self._model_path: str = ""

    def run(self) -> bool:
        if not self._topo.is_driver:
            return self._idle_non_driver_rank()
        if not self._topo.is_same_node:
            _log(
                "Cross-node online accuracy orchestration is not supported by this "
                "single-process test harness; use `run_mast_benchmarks.py` for "
                "cross-node MAST validation."
            )
            return False

        try:
            self._model_path = self._download_model()
            gpu_baseline, gpu_prefill, gpu_decode = self._assign_gpus()
            host_ip = "127.0.0.1"
            self._log_header()

            baseline_outputs = self._run_baseline_phase(gpu_baseline)
            disagg_client = self._start_disagg_phase(
                gpu_prefill,
                gpu_decode,
                host_ip,
            )

            _log("\n── Phase 2: Correctness Test ──")
            correctness = CorrectnessTest(baseline_outputs, disagg_client)
            passed = correctness.run()
            if not passed:
                _log("ABORTING: correctness test failed")
                return False

            if not self._config.skip_perf:
                self._run_performance_phase(disagg_client)

            _log("\n" + "=" * 60)
            _log("ALL TESTS PASSED")
            _log("=" * 60)
            return True

        finally:
            self._stop_all()

    def _idle_non_driver_rank(self) -> bool:
        _log(f"Rank {self._topo.rank}: idle (driver is rank 0)")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        signal.pause()
        return True

    def _log_header(self) -> None:
        _log("=" * 60)
        _log(f"DISAGG ACCURACY TEST — {self._config.connector}")
        _log(f"  Model: {self._config.model}")
        _log(f"  Topology: {'same-node' if self._topo.is_same_node else 'cross-node'}")
        _log("=" * 60)

    def _run_baseline_phase(self, gpu_baseline: str) -> dict[str, str]:
        _log("\n── Phase 1a: Standalone Baseline ──")
        self._start_server(
            "baseline",
            ServerConfig(
                model_path=self._model_path,
                port=self.BASELINE_PORT,
                gpu_id=gpu_baseline,
                block_size=self._config.block_size,
                gpu_memory_utilization=self._config.gpu_memory_utilization,
                max_model_len=self._config.max_model_len,
            ),
        )
        baseline_client = TestClient(
            f"http://localhost:{self.BASELINE_PORT}", self._model_path
        )
        baseline_outputs = CorrectnessTest.collect_baseline(baseline_client)
        self._stop_all()
        return baseline_outputs

    def _start_disagg_phase(
        self,
        gpu_prefill: str,
        gpu_decode: str,
        host_ip: str,
    ) -> TestClient:
        _log("\n── Phase 1b: Disagg Pipeline ──")
        kv_config = self._config.kv_transfer_config()
        self._start_server(
            "prefill",
            ServerConfig(
                model_path=self._model_path,
                port=self.PREFILL_PORT,
                gpu_id=gpu_prefill,
                block_size=self._config.block_size,
                gpu_memory_utilization=self._config.gpu_memory_utilization,
                max_model_len=self._config.max_model_len,
                kv_transfer_config=kv_config,
                side_channel_port=15580,
                host_ip=host_ip,
            ),
        )
        self._start_server(
            "decode",
            ServerConfig(
                model_path=self._model_path,
                port=self.DECODE_PORT,
                gpu_id=gpu_decode,
                block_size=self._config.block_size,
                gpu_memory_utilization=self._config.gpu_memory_utilization,
                max_model_len=self._config.max_model_len,
                kv_transfer_config=kv_config,
                side_channel_port=15680,
                host_ip=host_ip,
            ),
        )

        self._proxy = ProxyServer(self.PREFILL_PORT, self.DECODE_PORT, self.PROXY_PORT)
        self._proxy.start()
        return TestClient(f"http://localhost:{self.PROXY_PORT}", self._model_path)

    def _run_performance_phase(self, disagg_client: TestClient) -> None:
        _log("\n── Phase 3: Performance Benchmark ──")
        perf = PerformanceBenchmark(disagg_client, self._config)
        result = perf.run()
        _log(f"\nFinal: {result.req_per_s:.1f} req/s (median={result.median_s:.2f}s)")

    def _download_model(self) -> str:
        model_dir = Path("/tmp/disagg_models") / Path(self._config.model).name
        _log(f"Downloading {self._config.model}...")
        # Use the existing download infrastructure from disagg_entry
        try:
            from vllm.fb.standalone_benchmark.disagg_entry import download_model

            return download_model(self._config.model, 0)
        except ImportError:
            from mast_utils import download_model_from_manifold

            if model_dir.exists() and any(model_dir.iterdir()):
                _log(f"Model cached: {model_dir}")
                return str(model_dir)
            download_model_from_manifold(self._config.model, str(model_dir))
            return str(model_dir)

    def _assign_gpus(self) -> tuple[str, str, str]:
        if self._topo.is_same_node:
            return "0", "0", "1"
        raise RuntimeError("cross-node online accuracy orchestration is unsupported")

    def _start_server(self, label: str, config: ServerConfig) -> VLLMServer:
        server = VLLMServer(config, label)
        server.start()
        server.wait_until_ready()
        self._servers.append(server)
        return server

    def _stop_all(self) -> None:
        if self._proxy:
            self._proxy.stop()
            self._proxy = None
        for s in self._servers:
            s.stop()
        self._servers.clear()
        time.sleep(2)


# ── Entry Point ────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Disaggregated serving accuracy + performance test",
    )
    p.add_argument(
        "--connector",
        type=str,
        default=os.environ.get("CONNECTOR", "UniflowConnector"),
        choices=["UniflowConnector", "NixlConnector"],
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    p.add_argument(
        "--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", "128"))
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("GPU_MEM_UTIL", "0.45")),
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("MAX_MODEL_LEN", "2048")),
    )
    p.add_argument(
        "--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "250"))
    )
    p.add_argument(
        "--output-len", type=int, default=int(os.environ.get("OUTPUT_LEN", "32"))
    )
    p.add_argument(
        "--perf-warmup", type=int, default=int(os.environ.get("PERF_WARMUP", "3"))
    )
    p.add_argument(
        "--perf-iterations",
        type=int,
        default=int(os.environ.get("PERF_ITERATIONS", "5")),
    )
    p.add_argument("--skip-perf", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    config = TestConfig.from_args(args)
    topo = Topology()

    test = DisaggAccuracyTest(config, topo)
    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
