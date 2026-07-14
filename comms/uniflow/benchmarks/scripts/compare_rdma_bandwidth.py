#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# @noautodeps — standalone script, runs with system python3 (no buck target)
#
# Compare uniflow RDMA bandwidth against ib_write_bw (perftest).
# stdlib-only — runs with system python3, no buck build for the script itself.
# Binaries are cached in ~/.cache/uniflow_compare/<gpu>/ after first build.
#
# Usage: python3 compare_rdma_bandwidth.py [OPTIONS]

import argparse
import atexit
import base64
import csv
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

_verbose = False

REMOTE_DIR = "/tmp/uniflow_compare"

# SSH login user for sush2/suscp. Default "root" needs the root_<host> machine
# ACL; set to your unixname via --ssh-user to authenticate with your own login.
SSH_USER = "root"

IB_SERVER_INIT_DELAY_LOCAL = 3  # seconds for ib_write_bw server startup (local)
IB_SERVER_INIT_DELAY_REMOTE = 10  # seconds for ib_write_bw server startup (remote)
IB_PER_SIZE_DELAY = 3  # seconds between sizes in ib_write_bw loop

# Hard minimum wall-clock spacing between ANY two sush2/suscp invocations to a
# remote host. Bare-metal SSH gateways permanently blacklist identities that
# open connections too rapidly, so this is a safety floor, not a perf knob.
# _ssh_gate() sleeps as needed before every remote SSH call; --ssh-interval can
# raise it but is clamped to never go below this value.
MIN_SSH_INTERVAL = 30
_last_ssh_ts = 0.0


def _ssh_gate():
    """Block until >= MIN_SSH_INTERVAL seconds have elapsed since the last
    remote SSH call, then stamp the current time. Call immediately before every
    sush2/suscp subprocess launch to a remote host."""
    global _last_ssh_ts
    wait = MIN_SSH_INTERVAL - (time.monotonic() - _last_ssh_ts)
    if wait > 0:
        _log(
            f"SSH gate: sleeping {wait:.0f}s to keep >= {MIN_SSH_INTERVAL}s between SSH calls"
        )
        time.sleep(wait)
    _last_ssh_ts = time.monotonic()


GPU_PATTERNS = [
    ("H100", "h100"),
    ("H200", "h200"),
    ("GB300", "b300"),
    ("GB200", "b200"),
    ("B200", "b200"),
    ("A100", "a100"),
]

# Per-GPU extra buck flags. GB300 (b300) is Grace/aarch64 with the b200a nvcc
# arch and CUDA 13 (matches the uniflow_disagg/comms_rdma_bench MAST fbpkg).
_B200_FLAGS = (
    " -c fbcode.arch=aarch64"
    " -c fbcode.enable_gpu_sections=true"
    " -c fbcode.nvcc_arch=b200"
    " -c fbcode.platform010_cuda_version=12.8"
)
_B300_FLAGS = (
    " -c fbcode.arch=aarch64"
    " -c fbcode.enable_gpu_sections=true"
    " -c fbcode.nvcc_arch=b200a"
    " -c fbcode.platform010_cuda_version=13.0"
)

BUILD_SPECS = {
    "uniflow_bench": {
        "target": "fbcode//comms/uniflow/benchmarks:uniflow_bench",
        "arch_flags": {"b200": _B200_FLAGS, "b300": _B300_FLAGS},
        "supported_gpus": ("h100", "b200", "b300"),
    },
    "ib_write_bw": {
        "target": "fbsource//third-party/perftest/25.07.0-0.104:ib_write_bw",
        "arch_flags": {
            "b200": " -c fbcode.arch=aarch64",
            "b300": " -c fbcode.arch=aarch64",
        },
        "supported_gpus": None,
    },
}

# Setup scripts: plain bash with __PLACEHOLDER__ substitution, base64-encoded
# before execution to avoid sush2 quoting issues entirely.

SETUP_SCRIPT_HOST0 = """\
set +e
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU:$GPU"
NIC=$(ibstat -l 2>/dev/null | head -1 || ls /sys/class/infiniband/ 2>/dev/null | head -1)
echo "NIC:$NIC"
IP=$(ip -4 addr show eth1 2>/dev/null | awk '/inet /{split($2,a,"/"); print a[1]; exit}')
[ -z "$IP" ] && IP=$(ip -4 addr show eth0 2>/dev/null | awk '/inet /{split($2,a,"/"); print a[1]; exit}')
[ -z "$IP" ] && IP=$(hostname -i 2>/dev/null | awk '{print $1}')
echo "IP:$IP"
GID_NIC=__NIC_OVERRIDE__
[ -z "$GID_NIC" ] && GID_NIC=$NIC
GID=$(show_gids 2>/dev/null | awk "/$GID_NIC/ && /v2/ && !/fe80/ {print \\$3; exit}")
echo "GID:${GID:-3}"
test -x __REMOTE_DIR__/uniflow_bench && echo "HAS:uniflow_bench"
test -x __REMOTE_DIR__/ib_write_bw && echo "HAS:ib_write_bw"
echo "LINKAGE:$(cat __REMOTE_DIR__/uniflow_bench.linkage 2>/dev/null)"
pkill -9 -f uniflow_bench 2>/dev/null
pkill -9 -f ib_write_bw 2>/dev/null
mkdir -p __REMOTE_DIR__
echo "SETUP_OK"
"""

SETUP_SCRIPT_EXTRA_HOST = """\
set +e
GID=$(show_gids 2>/dev/null | awk "/__NIC__/ && /v2/ && !/fe80/ {print \\$3; exit}")
echo "GID:${GID:-3}"
test -x __REMOTE_DIR__/uniflow_bench && echo "HAS:uniflow_bench"
test -x __REMOTE_DIR__/ib_write_bw && echo "HAS:ib_write_bw"
echo "LINKAGE:$(cat __REMOTE_DIR__/uniflow_bench.linkage 2>/dev/null)"
pkill -9 -f uniflow_bench 2>/dev/null
pkill -9 -f ib_write_bw 2>/dev/null
mkdir -p __REMOTE_DIR__
echo "SETUP_OK"
"""


def _log(msg):
    if _verbose:
        print(f"  [debug] {msg}", file=sys.stderr)


def format_size(nbytes):
    if nbytes >= 1 << 30:
        return f"{nbytes >> 30} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes >> 20} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes >> 10} KB"
    return f"{nbytes} B"


def generate_sizes(min_size, max_size):
    for val, label in [(min_size, "--min-size"), (max_size, "--max-size")]:
        if val <= 0 or (val & (val - 1)) != 0:
            sys.exit(f"ERROR: {label} must be a power of 2, got {val}")
    if min_size > max_size:
        sys.exit(f"ERROR: --min-size ({min_size}) must be <= --max-size ({max_size})")
    sizes = []
    s = min_size
    while s <= max_size:
        sizes.append(s)
        s *= 2
    return sizes


def match_gpu(raw):
    """Match nvidia-smi output string to a canonical GPU type."""
    for pattern, name in GPU_PATTERNS:
        if pattern in raw:
            return name
    return ""


def _run_script_on_host(host, script):
    """Base64-encode a bash script and run it via sush2, avoiding quoting issues."""
    encoded = base64.b64encode(script.encode()).decode()
    return host.run(f"echo {encoded} | base64 -d | bash", timeout=30)


_SUSH2_BANNER = (
    "Meta authorized users only. Usage is subject to monitoring and recording."
)


class RemoteHost:
    """Manages all sush2 interactions with a single remote host."""

    def __init__(self, hostname):
        self.hostname = hostname
        self._bg_procs = []

    @property
    def is_local(self):
        return self.hostname == "localhost"

    def _wrap(self, cmd):
        if self.is_local:
            return cmd
        remote = f"{{ {cmd} ; }} 2>&1"
        return (
            f"sush2 --reason 'RDMA bandwidth comparison'"
            f" {SSH_USER}@{self.hostname} {shlex.quote(remote)}"
        )

    @staticmethod
    def _strip_banner(output):
        lines = output.splitlines()
        return "\n".join(ln for ln in lines if _SUSH2_BANNER not in ln).strip()

    def run(self, cmd, capture=True, timeout=None):
        _log(f"[{self.hostname}] {cmd}")
        full = self._wrap(cmd)
        if not capture:
            if not self.is_local:
                _ssh_gate()
            subprocess.run(full, shell=True, timeout=timeout)
            return ""
        # Retry on SSH throttling (empty stdout + non-zero rc).
        # Commands that legitimately return empty output have rc=0.
        out = ""
        for attempt in range(3):
            if attempt > 0:
                _log(f"sush2 retry {attempt + 1}/3 after SSH throttle delay")
            if not self.is_local:
                _ssh_gate()
            r = subprocess.run(
                full, shell=True, capture_output=True, text=True, timeout=timeout
            )
            if _verbose and r.stderr and r.stderr.strip():
                _log(f"stderr: {r.stderr.strip()[:500]}")
            out = r.stdout.strip()
            if not self.is_local and out:
                out = self._strip_banner(out)
            if out or r.returncode == 0:
                return out
        return out

    def run_bg(self, cmd, quiet=False):
        _log(f"[{self.hostname} bg] {cmd}")
        full = self._wrap(cmd)
        if not self.is_local:
            _ssh_gate()
        fd, log_path = tempfile.mkstemp(suffix=".log", prefix="uniflow_bg_", dir="/tmp")
        log_file = os.fdopen(fd, "w")
        p = subprocess.Popen(
            full,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        p.log_file = log_file
        p.log_path = log_path
        self._bg_procs.append(p)
        if not quiet:
            print(f"  bg log: {log_path}")
        return p

    def collect_bg(self, proc):
        proc.log_file.close()
        output = ""
        if os.path.exists(proc.log_path):
            with open(proc.log_path) as f:
                output = f.read()
            os.unlink(proc.log_path)
        return self._strip_banner(output).strip() if output else ""

    def _suscp(self, src, dst, reason, timeout=120):
        # Retry on transient SSH kex errors (MaxStartups throttling).
        cmd = f"suscp --reason {shlex.quote(reason)} {src} {dst}"
        for attempt in range(3):
            if attempt > 0:
                _log(f"suscp retry {attempt + 1}/3 after SSH throttle delay")
            _ssh_gate()
            rc = subprocess.run(cmd, shell=True, timeout=timeout).returncode
            if rc == 0:
                return
        # suscp may also return non-zero despite successful transfer when SSH
        # hits transient kex errors. Callers verify the file arrived.
        _log(f"suscp exited {rc}, caller will verify file")

    def copy_to(self, local_path, remote_name):
        self._suscp(
            local_path,
            f"{SSH_USER}@{self.hostname}:{REMOTE_DIR}/{remote_name}",
            "copy benchmark binary",
        )

    def copy_from(self, remote_name, local_path):
        self._suscp(
            f"{SSH_USER}@{self.hostname}:{REMOTE_DIR}/{remote_name}",
            local_path,
            "retrieve results",
            timeout=60,
        )

    def verify_and_chmod(self, names, uniflow_linkage=None):
        checks = " && ".join(
            f"test -f {REMOTE_DIR}/{n} && chmod +x {REMOTE_DIR}/{n}" for n in names
        )
        cmd = checks
        if uniflow_linkage and "uniflow_bench" in names:
            # Stamp the deployed linkage so a later run can detect a switch and
            # re-copy instead of silently running a stale build.
            cmd += (
                f" && echo {shlex.quote(uniflow_linkage)} >"
                f" {REMOTE_DIR}/uniflow_bench.linkage"
            )
        out = self.run(f"{cmd} && echo OK", timeout=15)
        if "OK" not in (out or ""):
            sys.exit(
                f"ERROR: Binaries not found on {self.hostname} after copy."
                f" Expected: {', '.join(names)}"
            )

    def setup_primary(self, nic_override=""):
        script = SETUP_SCRIPT_HOST0.replace("__REMOTE_DIR__", REMOTE_DIR).replace(
            "__NIC_OVERRIDE__", shlex.quote(nic_override or "")
        )
        return self._parse_setup(_run_script_on_host(self, script))

    def setup_secondary(self, nic):
        script = SETUP_SCRIPT_EXTRA_HOST.replace("__REMOTE_DIR__", REMOTE_DIR).replace(
            "__NIC__", nic
        )
        return self._parse_setup(_run_script_on_host(self, script))

    @staticmethod
    def _parse_setup(output):
        if "SETUP_OK" not in (output or ""):
            sys.exit(f"ERROR: Remote setup failed.\n  Output: {output!r}")

        info = {}
        has_bins = set()
        for line in output.splitlines():
            if line.startswith("HAS:"):
                has_bins.add(line[4:].strip())
            elif ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()

        info["has_bins"] = has_bins
        return info

    def cleanup(self):
        """Gracefully terminate bg sush2 processes.

        SIGTERM lets sush2 close the SSH channel, which sends SIGHUP to the
        remote process group. SIGKILL is a fallback if it doesn't exit in time.
        """
        for proc in self._bg_procs:
            try:
                proc.terminate()  # SIGTERM → sush2 propagates to remote
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()  # fallback
            except OSError:
                pass
        self._bg_procs.clear()

    def format_cmd(self, cmd):
        return cmd if self.is_local else self._wrap(cmd)


CACHE_DIR = os.path.expanduser("~/.cache/uniflow_compare")


def _cache_subdir(gpu, linkage):
    # Keep dynamic-linkage binaries in a separate dir so they never get confused
    # with the default static build for the same GPU.
    return gpu if linkage == "static" else f"{gpu}-{linkage}"


def get_cached_binary(name, gpu, linkage="static"):
    path = os.path.join(CACHE_DIR, _cache_subdir(gpu, linkage), name)
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


def build_binary(name, gpu, rebuild=False, linkage="static"):
    cached = get_cached_binary(name, gpu, linkage)
    if cached and not rebuild:
        print(f"  {name}: {cached} (cached)")
        return cached

    spec = BUILD_SPECS[name]
    if spec["supported_gpus"] and gpu not in spec["supported_gpus"]:
        sys.exit(f"ERROR: Unsupported GPU for {name}: {gpu}")

    print(f"  Building {name}...")
    extra = spec.get("arch_flags", {}).get(gpu, "")
    # Only uniflow_bench honors the rdma_linkage toggle; ib_write_bw ignores it.
    if name == "uniflow_bench" and linkage == "dynamic":
        extra += " -c uniflow.rdma_linkage=dynamic"
    cmd = f"buck2 build @fbcode//mode/opt{extra} {spec['target']} --show-full-output"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        sys.exit(f"ERROR: Build of {name} timed out after 10 minutes")
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-20:]) if r.stderr else "(no stderr)"
        sys.exit(f"ERROR: Build of {name} failed (exit {r.returncode}):\n{tail}")

    parts = r.stdout.strip().split()
    built = parts[1] if len(parts) >= 2 else ""
    if not built or not os.path.isfile(built):
        sys.exit(f"ERROR: Build of {name} produced no output binary")

    dest_dir = os.path.join(CACHE_DIR, _cache_subdir(gpu, linkage))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    shutil.copy2(built, dest)
    os.chmod(dest, 0o755)
    print(f"  {name}: {dest}")
    return dest


class UniflowBenchmark:
    name = "uniflow"
    binary_name = "uniflow_bench"

    def __init__(self, args, sizes, host0, rank1_host):
        self.args = args
        self.sizes = sizes
        self.host0 = host0
        self.rank1_host = rank1_host
        self.binary_path = f"{REMOTE_DIR}/{self.binary_name}"

    def _build_bench_flags(self):
        """Assemble the uniflow_bench CLI flags from args (shared by both ranks)."""
        a = self.args
        flags = (
            f"--benchmark rdma_bandwidth --transport rdma"
            f" --iterations {a.iterations} --warmup {a.warmup}"
            f" --min-size {a.min_size} --max-size {a.max_size}"
            f" --direction {a.direction} --batch-size {a.batch_size}"
            f" --tx-depth {a.tx_depth}"
            f" --chunk-size {a.chunk_size}"
            f" --format csv --output {REMOTE_DIR}/results.csv"
        )
        if a.num_nics > 0:
            flags += f" --num-nics {a.num_nics}"
        if a.use_cuda:
            if a.cuda_devices:
                flags += f" --cuda-devices {shlex.quote(a.cuda_devices)}"
            else:
                flags += " --cuda-device 0"
        if a.gpu_nics:
            # shlex.quote the value for the REMOTE shell (sush2 runs the command
            # through a shell on the far side): --gpu-nics contains ';' group
            # separators that would otherwise be treated as command terminators.
            flags += f" --gpu-nics {shlex.quote(a.gpu_nics)}"
        if a.nics:
            flags += f" --rdma-devices {shlex.quote(a.nics)}"
        if a.data_direct:
            flags += " --data-direct"
        return flags

    def run(self, dry_run=False):
        print("--- Running uniflow ---")
        a = self.args

        bench_flags = self._build_bench_flags()

        env = f"MASTER_ADDR={a.master_ip} MASTER_PORT={a.master_port} WORLD_SIZE=2"
        if _verbose:
            env = f'SPDLOG_LEVEL="uniflow=info" {env}'
        command_prefix = ""
        if a.numa_node >= 0:
            command_prefix = (
                f"numactl --cpunodebind={a.numa_node} --membind={a.numa_node} "
            )
        r0_cmd = (
            f"{env} RANK=0 LOCAL_RANK=0 "
            f"{command_prefix}{self.binary_path} {bench_flags}"
        )
        r1_cmd = (
            f"{env} RANK=1 LOCAL_RANK={a.rank1_local_rank}"
            f" {command_prefix}{self.binary_path} {bench_flags}"
        )

        if dry_run:
            print(f"  rank 0: {self.host0.format_cmd(r0_cmd)}")
            print(f"  rank 1: {self.rank1_host.format_cmd(r1_cmd)}")
            print()
            return {}

        print(f"  rank 0: {r0_cmd}")
        print(f"  rank 1: {r1_cmd}")
        print()

        # Prepend rm to the rank0 command to clear stale results within
        # the same sush2 session (zero extra SSH calls).
        r0_cmd_full = f"rm -f {REMOTE_DIR}/results.csv; {r0_cmd}"
        # rank 1 (RANK=1, the connecting side) launches first in the background;
        # rank 0 (RANK=0, the rendezvous server) follows. The 30s SSH gate makes
        # host0.run() below wait ~30s before starting rank 0, so rank 1 must
        # tolerate a ~30s rendezvous wait -- uniflow's TcpController connect-retry
        # window covers this today. Revisit if MIN_SSH_INTERVAL grows.
        p1 = self.rank1_host.run_bg(r1_cmd)
        time.sleep(1)
        self.host0.run(r0_cmd_full, capture=False)
        p1.wait()

        rank1_log = self.rank1_host.collect_bg(p1)
        if rank1_log:
            print("  [rank 1 log]")
            for line in rank1_log.splitlines():
                print(f"    {line}")
            print()

        results = self._collect_results()
        if results:
            print(f"  Parsed {len(results)} sizes")
        else:
            print("  WARNING: No results parsed from CSV")
        print()
        return results

    def _collect_results(self):
        a = self.args
        fd, csv_path = tempfile.mkstemp(suffix=".csv", prefix="uniflow_")
        os.close(fd)
        try:
            if self.host0.is_local:
                shutil.copy2(f"{REMOTE_DIR}/results.csv", csv_path)
            else:
                self.host0.copy_from("results.csv", csv_path)
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                remote_check = ""
                if not self.host0.is_local:
                    remote_check = self.host0.run(
                        f"ls -la {REMOTE_DIR}/results.csv 2>&1", timeout=10
                    )
                sys.exit(
                    f"ERROR: uniflow CSV results not retrieved."
                    f"\n  Remote: {remote_check or '(local mode)'}"
                )
            # Validate the CSV is from this run, not a stale file from a
            # previous chunk-size sweep.
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                first_row = next(reader, None)
                if first_row and "chunk_size" in first_row:
                    csv_chunk = int(first_row["chunk_size"])
                    if csv_chunk != a.chunk_size:
                        sys.exit(
                            f"ERROR: Stale results.csv detected"
                            f" (chunk_size={csv_chunk}, expected {a.chunk_size})."
                            f" The uniflow benchmark may have failed silently."
                        )
            if getattr(a, "save_csv", ""):
                os.makedirs(os.path.dirname(a.save_csv) or ".", exist_ok=True)
                shutil.copy2(csv_path, a.save_csv)
                print(f"  Raw CSV saved to {a.save_csv}")
            return self._parse_csv(csv_path)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    @staticmethod
    def _parse_csv(csv_path):
        results = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    size = int(row["size_bytes"])
                    entry = {"bw": float(row["bw_gbps"])}
                    for csv_key, dict_key in [
                        ("lat_avg_us", "avg_us"),
                        ("lat_p50_us", "p50_us"),
                        ("lat_p99_us", "p99_us"),
                        ("lat_min_us", "min_us"),
                        ("lat_max_us", "max_us"),
                        ("msg_rate_mops", "msg_rate_mops"),
                        ("batch_size", "batch_size"),
                        ("chunk_size", "chunk_size"),
                    ]:
                        if csv_key in row:
                            entry[dict_key] = float(row[csv_key])
                    results[size] = entry
                except (ValueError, KeyError):
                    continue
        return results


class IbWriteBwBenchmark:
    name = "ib_write_bw"
    binary_name = "ib_write_bw"

    def __init__(self, args, sizes, host0, rank1_host):
        self.args = args
        self.sizes = sizes
        self.host0 = host0
        self.rank1_host = rank1_host
        self.binary_path = f"{REMOTE_DIR}/{self.binary_name}"

    def run(self, dry_run=False):
        print("--- Running ib_write_bw ---")
        a = self.args
        base_port = random.randint(18000, 29999)
        cuda_flag = "--use_cuda=0 --use_cuda_dmabuf" if a.use_cuda else ""

        server_loop, client_loop = self._build_loop(base_port, cuda_flag)

        if dry_run:
            print(f"  server: {self.host0.format_cmd(server_loop)}")
            print(f"  client: {self.rank1_host.format_cmd(client_loop)}")
            print()
            return {}

        srv_proc = self.host0.run_bg(server_loop)
        delay = (
            IB_SERVER_INIT_DELAY_LOCAL
            if self.host0.is_local
            else IB_SERVER_INIT_DELAY_REMOTE
        )
        time.sleep(delay)
        self.rank1_host.run(client_loop, capture=False, timeout=1800)
        srv_proc.wait()
        srv_log = self.host0.collect_bg(srv_proc)

        ib_out = self.rank1_host.run(f"cat {REMOTE_DIR}/ib_results.txt 2>/dev/null")

        results = {}
        for size in self.sizes:
            bw = self._parse_bw(ib_out, size)
            if bw is not None:
                results[size] = bw

        missing = [s for s in self.sizes if s not in results]
        if missing:
            labels = ", ".join(format_size(s) for s in missing)
            print(f"  Missing results for: {labels}")
            if srv_log:
                print(f"  Server log (last 500 chars):\n    {srv_log[-500:]}")

        print(f"  Completed {len(results)}/{len(self.sizes)} sizes")
        print()
        return results

    def _build_loop(self, base_port, cuda_flag):
        a = self.args
        sizes_str = " ".join(str(s) for s in self.sizes)
        target = "127.0.0.1" if self.host0.is_local else a.master_ip
        total = len(self.sizes)

        common = f"--report_gbits --CPU-freq -n {a.iterations}"
        srv_flags = f"-x {a.gid0} {common} -d {shlex.quote(a.nic0)}"
        cli_flags = f"-x {a.gid1} {common} -d {shlex.quote(a.nic1)}"
        for flag in [a.ipv6_flag, cuda_flag]:
            if flag:
                srv_flags += f" {flag}"
                cli_flags += f" {flag}"

        results_file = f"{REMOTE_DIR}/ib_results.txt"

        # Server: suppress all output (it's only useful for debugging via bg log).
        server_loop = (
            f"P={base_port}; for S in {sizes_str}; do"
            f" {self.binary_path} {srv_flags} -p $P -s $S 2>&1;"
            f" P=$((P+1)); done"
        )
        # Client: tee raw output to results file for parsing, but only display
        # the data line (starts with whitespace+digits) to keep console clean.
        # Retries each size up to 5 times to handle server-not-ready.
        ib_cmd = f"{self.binary_path} {cli_flags}"
        # rm at the start clears stale results within the same sush2 session.
        client_loop = (
            f"rm -f {results_file};"
            f" set -o pipefail; P={base_port}; I=0; for S in {sizes_str}; do"
            f" I=$((I+1));"
            f' echo "  [$I/{total}] size=$S ...";'
            f" for RETRY in 1 2 3 4 5; do"
            f" sleep {IB_PER_SIZE_DELAY};"
            f" {ib_cmd} -p $P -s $S {target} 2>&1"
            f" | tee -a {results_file}"
            f" | grep -E '^\\s+[0-9]';"
            f" RC=${{PIPESTATUS[0]}}; [ $RC -eq 0 ] && break;"
            f' echo "    retry $RETRY/5 ...";'
            f" done;"
            f" [ $RC -eq 0 ] && echo '    ok' || echo \"    FAILED (rc=$RC)\";"
            f" P=$((P+1)); done"
        )
        return server_loop, client_loop

    @staticmethod
    def _parse_bw(output, size):
        if not output:
            return None
        unit_is_mb = "BW average[MB/sec]" in output
        for line in output.splitlines():
            m = re.match(r"\s*(\d+)\s+", line)
            if m and int(m.group(1)) == size:
                cols = line.split()
                if len(cols) >= 4:
                    try:
                        raw = float(cols[3])
                        return raw / 1024.0 if unit_is_mb else raw / 8.0
                    except ValueError:
                        pass
        return None


def _fmt(val, decimals=2):
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def print_table(tool, sizes, uniflow_results, ib_results):
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)

    if tool == "uniflow":
        print(
            f"| {'Size':<11} | {'BW (GB/s)':>9}"
            f" | {'Avg (us)':>10} | {'P50 (us)':>10} | {'P99 (us)':>10} |"
        )
        print(f"|{'-' * 13}|{'-' * 11}|{'-' * 12}|{'-' * 12}|{'-' * 12}|")
        for size in sizes:
            r = uniflow_results.get(size, {})
            print(
                f"| {format_size(size):<11} | {_fmt(r.get('bw')):>9}"
                f" | {_fmt(r.get('avg_us'), 1):>10}"
                f" | {_fmt(r.get('p50_us'), 1):>10}"
                f" | {_fmt(r.get('p99_us'), 1):>10} |"
            )

    elif tool == "ib":
        print(f"| {'Size':<11} | {'BW (GB/s)':>9} |")
        print(f"|{'-' * 13}|{'-' * 11}|")
        for size in sizes:
            print(f"| {format_size(size):<11} | {_fmt(ib_results.get(size)):>9} |")

    elif tool == "both":
        print(
            f"| {'Size':<11} | {'uniflow':>9}"
            f" | {'ib_write_bw':>11} | {'Gap':>6}"
            f" | {'Avg (us)':>10} | {'P99 (us)':>10} |"
        )
        print(f"|{'-' * 13}|{'-' * 11}|{'-' * 13}|{'-' * 8}|{'-' * 12}|{'-' * 12}|")
        for size in sizes:
            ur = uniflow_results.get(size, {})
            uf = ur.get("bw")
            ib = ib_results.get(size)

            if uf is not None and ib is not None and ib > 0:
                gap_pct = (ib - uf) / ib * 100
                gap_s = "*" if gap_pct > 50 else f"{gap_pct:.1f}%"
            else:
                gap_s = "N/A"

            print(
                f"| {format_size(size):<11} | {_fmt(uf):>9}"
                f" | {_fmt(ib):>11} | {gap_s:>6}"
                f" | {_fmt(ur.get('avg_us'), 1):>10}"
                f" | {_fmt(ur.get('p99_us'), 1):>10} |"
            )

        print()
        print("  Gap = (ib - uniflow) / ib. (*) = gap > 50% (small-message overhead).")
        print("  BW in GB/s. Latency from uniflow.")

    print()


def _parse_nics(nics_str):
    parts = [n.strip() for n in nics_str.split(",") if n.strip()]
    if not parts:
        return "", ""
    nic0 = parts[0]
    nic1 = parts[1] if len(parts) > 1 else nic0
    for n in (nic0, nic1):
        if not re.match(r"^[a-zA-Z0-9_-]+$", n):
            sys.exit(f"ERROR: Invalid NIC name: {n!r}")
    return nic0, nic1


def _setup_hosts(args):
    """Batched remote setup: 1 SSH call per host to avoid MaxStartups throttling."""
    hosts = {}

    h0 = RemoteHost(args.host0)
    print(f"Setting up {h0.hostname}...")
    nic_override = args.nic0 if args.nics else ""
    info0 = h0.setup_primary(nic_override=nic_override)

    if not args.gpu:
        args.gpu = match_gpu(info0.get("GPU", ""))
        if not args.gpu:
            sys.exit(
                f"ERROR: Could not auto-detect GPU on {h0.hostname}"
                f" (nvidia-smi returned: {info0.get('GPU', '')!r}). Specify --gpu"
            )

    if not args.nics:
        nic_raw = info0.get("NIC", "")
        args.nic0 = nic_raw.split()[0] if nic_raw.split() else ""
        if not args.nic0:
            sys.exit(f"ERROR: No RDMA NIC found on {h0.hostname}. Specify --nics")
        args.nic1 = args.nic0

    args.master_ip = info0.get("IP", "")
    if not args.master_ip:
        sys.exit(f"ERROR: Could not resolve IP for {h0.hostname}")
    args.gid0 = info0.get("GID", "3")
    args.ipv6_flag = "--ipv6-addr" if ":" in args.master_ip else ""

    print(f"  GPU: {args.gpu}, NIC: {args.nic0}, IP: {args.master_ip}")

    hosts[args.host0] = h0
    has_bins_all = dict.fromkeys(info0["has_bins"], True)
    # rdma-core linkage stamped on the remote uniflow_bench, so a linkage switch
    # forces a re-copy (the HAS check can't tell a static build from a dynamic
    # one). Empty when no binary/stamp is present or the hosts disagree.
    remote_linkage = info0.get("LINKAGE", "")

    if args.host1 != args.host0:
        h1 = RemoteHost(args.host1)
        print(f"Setting up {h1.hostname}...")
        info1 = h1.setup_secondary(nic=args.nic1)
        args.gid1 = info1.get("GID", "3")
        hosts[args.host1] = h1
        for name in ("uniflow_bench", "ib_write_bw"):
            if name not in info1["has_bins"]:
                has_bins_all.pop(name, None)
        if info1.get("LINKAGE", "") != remote_linkage:
            remote_linkage = ""  # hosts disagree → re-copy to both
    else:
        args.gid1 = args.gid0

    print(
        f"  GID: {args.host0}/{args.nic0}={args.gid0},"
        f" {args.host1}/{args.nic1}={args.gid1}"
    )

    args._hosts = hosts
    args._has_remote_bins = has_bins_all
    args._remote_uniflow_linkage = remote_linkage


def _copy_binaries(args, benchmarks):
    force = args.force_copy or ""
    has = args._has_remote_bins
    hosts = list(args._hosts.values())

    to_copy = []
    for b in benchmarks:
        need = b.binary_name not in has
        if force in ("both", b.name):
            need = True
        # uniflow_bench's build depends on --rdma-linkage, but the remote HAS
        # check only tests executability and can't tell a static build from a
        # dynamic one. Re-copy when the linkage stamped on the remote differs
        # from the requested one, so we never silently run a stale build.
        if b.name == "uniflow" and (
            getattr(args, "_remote_uniflow_linkage", "") != args.rdma_linkage
        ):
            need = True
        if need:
            to_copy.append(b)
        else:
            print(f"  {b.binary_name}: already on remote (--force-copy to re-copy)")

    if not to_copy:
        return

    print("  Copying binaries to remote hosts...")
    # copy_to (via _suscp) and verify_and_chmod each call _ssh_gate(), which
    # already enforces the MIN_SSH_INTERVAL floor between SSH calls, so no extra
    # inter-copy sleep is needed here.
    for host in hosts:
        for b in to_copy:
            host.copy_to(b.local_binary_path, b.binary_name)
        host.verify_and_chmod(
            [b.binary_name for b in to_copy], uniflow_linkage=args.rdma_linkage
        )


def _parse_args():
    global _verbose, SSH_USER, MIN_SSH_INTERVAL
    parser = argparse.ArgumentParser(
        description="Compare uniflow RDMA bandwidth against ib_write_bw (perftest)",
    )
    parser.add_argument(
        "--hosts",
        default="",
        help="Comma-separated hostnames (1 = intra-host remote, 2 = inter-host)",
    )
    parser.add_argument(
        "--gpu",
        default="",
        help="GPU type: h100 | b200 | b300 (default: auto-detect)",
    )
    parser.add_argument(
        "--nics",
        default="",
        help="RDMA device(s), comma-separated if different per rank"
        " (default: auto-detect)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Iterations per size (default: 10000)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Min message size, must be power of 2 (default: 1)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1073741824,
        help="Max message size, must be power of 2 (default: 1 GiB)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of requests per transport call (default: 1)",
    )
    parser.add_argument(
        "--tx-depth",
        type=int,
        default=128,
        help="Outstanding transport calls before waiting (default: 128)",
    )
    parser.add_argument(
        "--num-nics",
        type=int,
        default=0,
        help="Cap number of NICs to use, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=524288,
        help="RDMA transfer chunk size in bytes (default: 524288)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Control-plane port for uniflow benchmark rendezvous (default: 29500)",
    )
    parser.add_argument(
        "--numa-node",
        type=int,
        default=-1,
        help="Run uniflow_bench under numactl for the given NUMA node (default: disabled)",
    )
    parser.add_argument(
        "--direction",
        default="put",
        choices=["put", "get", "both"],
        help="RDMA transfer direction for uniflow: put | get | both (default: put)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU memory instead of GPU Direct RDMA (default: GPU Direct)",
    )
    parser.add_argument(
        "--tool",
        default="uniflow",
        choices=["uniflow", "ib", "both"],
        help="Which tool(s) to run (default: uniflow)",
    )
    parser.add_argument(
        "--cuda-devices",
        default="",
        help=(
            "Comma-separated GPU indices for single-process multi-GPU aggregate"
            " (e.g. 0,1,2,3), overriding the default single --cuda-device 0."
            " uniflow only; each rank drives all listed GPUs and BW is summed."
        ),
    )
    parser.add_argument(
        "--gpu-nics",
        default="",
        help=(
            "Per-GPU NIC map for --cuda-devices: ';'-separated groups of"
            " comma-separated NICs, one group per GPU (e.g."
            " 'mlx5_0,mlx5_1;mlx5_2,mlx5_3'). Omit to let the binary auto-select"
            " NICs per GPU."
        ),
    )
    parser.add_argument(
        "--ssh-interval",
        type=int,
        default=MIN_SSH_INTERVAL,
        help=(
            "Minimum seconds between remote SSH calls (default and hard floor:"
            f" {MIN_SSH_INTERVAL}). Values below the floor are clamped up to it to"
            " avoid a permanent SSH blacklist."
        ),
    )
    parser.add_argument(
        "--ssh-user",
        default="root",
        help=(
            "SSH login user for sush2/suscp (default: root, needs the"
            " root_<host> machine ACL). Pass your unixname to authenticate with"
            " your own login instead."
        ),
    )
    parser.add_argument(
        "--rdma-linkage",
        default="static",
        choices=["static", "dynamic"],
        help=(
            "uniflow rdma-core linkage. 'static' (default) links the pinned"
            " fbsource third-party rdma-core. 'dynamic' builds with"
            " -c uniflow.rdma_linkage=dynamic so the binary dlopens the HOST's"
            " /usr/lib64 libibverbs/libmlx5 at runtime — use when the host ships"
            " a newer mlx5 (e.g. GB300 Data Direct needing rdma-core > the pin)."
        ),
    )
    parser.add_argument(
        "--data-direct",
        action="store_true",
        help=(
            "Enable mlx5 Data Direct (NIC<->GPU HBM over PCIe, bypassing Grace"
            " C2C). Requires a DD-provisioned node and VRAM buffers; uniflow"
            " only."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of binaries even if cached",
    )
    parser.add_argument(
        "--force-copy",
        default="",
        nargs="?",
        const="both",
        choices=["uniflow", "ib", "both"],
        help="Force re-copy binaries: uniflow | ib | both (default: both)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every command to stderr for debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print benchmark commands without executing them",
    )
    parser.add_argument(
        "--save-csv",
        default="",
        help="Save raw uniflow CSV results to this path",
    )
    args = parser.parse_args()

    _verbose = args.verbose
    SSH_USER = args.ssh_user
    if args.ssh_interval < MIN_SSH_INTERVAL:
        print(
            f"WARNING: --ssh-interval {args.ssh_interval}s is below the {MIN_SSH_INTERVAL}s"
            f" safety floor; clamping to {MIN_SSH_INTERVAL}s to avoid an SSH blacklist."
        )
    else:
        MIN_SSH_INTERVAL = args.ssh_interval
    args.use_cuda = not args.no_cuda

    if args.data_direct and not args.use_cuda:
        parser.error(
            "--data-direct requires GPU buffers; do not combine with --no-cuda"
        )

    if args.hosts:
        parts = [h.strip() for h in args.hosts.split(",") if h.strip()]
        if not parts:
            sys.exit("ERROR: --hosts value is empty")
        for h in parts:
            if not re.match(r"^[a-zA-Z0-9._-]+$", h):
                sys.exit(f"ERROR: Invalid hostname: {h!r}")
        args.host0 = parts[0]
        args.host1 = parts[1] if len(parts) > 1 else parts[0]
        args.mode = "inter-host" if len(parts) > 1 else "intra-host-remote"
    else:
        args.host0 = args.host1 = "localhost"
        args.mode = "intra-host-local"

    if args.nics:
        args.nic0, args.nic1 = _parse_nics(args.nics)

    return args


def _setup_local(args):
    local = RemoteHost("localhost")
    if not args.gpu:
        raw = local.run(
            "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1"
        )
        args.gpu = match_gpu(raw)
        if not args.gpu:
            sys.exit("ERROR: Could not auto-detect GPU. Specify --gpu")
    if not args.nics:
        raw = local.run(
            "ibstat -l 2>/dev/null | head -1"
            " || ls /sys/class/infiniband/ 2>/dev/null | head -1"
        )
        args.nic0 = raw.split()[0] if raw.split() else ""
        if not args.nic0:
            sys.exit("ERROR: No RDMA NIC found. Specify --nics")
        args.nic1 = args.nic0
    args.master_ip = "127.0.0.1"
    args.gid0 = args.gid1 = "0"
    args.ipv6_flag = ""
    args._hosts = {"localhost": local}
    args._has_remote_bins = {}


def _create_benchmarks(args, sizes, host0, rank1_host):
    binary_names = []
    if args.tool in ("uniflow", "both"):
        binary_names.append("uniflow_bench")
    if args.tool in ("ib", "both"):
        binary_names.append("ib_write_bw")

    print()
    local_bins = {}
    for name in binary_names:
        linkage = args.rdma_linkage if name == "uniflow_bench" else "static"
        local_bins[name] = build_binary(
            name, args.gpu, rebuild=args.rebuild, linkage=linkage
        )

    benchmarks = []
    if args.tool in ("uniflow", "both"):
        benchmarks.append(UniflowBenchmark(args, sizes, host0, rank1_host))
    if args.tool in ("ib", "both"):
        benchmarks.append(IbWriteBwBenchmark(args, sizes, host0, rank1_host))

    for b in benchmarks:
        b.local_binary_path = local_bins[b.binary_name]

    if args.mode == "intra-host-local":
        os.makedirs(REMOTE_DIR, exist_ok=True)
        for b in benchmarks:
            b.binary_path = b.local_binary_path
    else:
        _copy_binaries(args, benchmarks)

    return benchmarks


def _print_config(args, sizes):
    print()
    print("=" * 60)
    print(f"  RDMA Bandwidth Benchmark (tool: {args.tool})")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    if args.mode == "intra-host-local":
        print("  Host:       localhost")
    else:
        print(f"  Rank 0:     {args.host0} ({args.master_ip})")
        print(f"  Rank 1:     {args.rank1_host}")
    print(f"  GPU:        {args.gpu}")
    nic_str = args.nic0
    if args.nic1 != args.nic0:
        nic_str += f", {args.nic1}"
    print(f"  NICs:       {nic_str}")
    print(
        f"  Sizes:      {format_size(sizes[0])} - {format_size(sizes[-1])}"
        f" ({len(sizes)} steps)"
    )
    print(f"  Iterations: {args.iterations}")
    print(f"  Direction:  {args.direction}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  TX depth:   {args.tx_depth}")
    print(f"  Port:       {args.master_port}")
    if args.numa_node >= 0:
        print(f"  NUMA node:  {args.numa_node}")
    if args.num_nics > 0:
        print(f"  Num NICs:   {args.num_nics}")
    if args.use_cuda:
        print("  Memory:     GPU 0 (GPU Direct)")
    else:
        print("  Memory:     CPU (DRAM)")
    print("=" * 60)
    print()


def _run_benchmarks(args, benchmarks):
    all_results = {}
    for b in benchmarks:
        all_results[b.name] = b.run(dry_run=args.dry_run)
    return all_results


def main():
    args = _parse_args()

    if args.mode == "intra-host-local":
        _setup_local(args)
    else:
        _setup_hosts(args)

    if args.mode == "inter-host":
        args.rank1_host = args.host1
        args.rank1_local_rank = 0
    else:
        args.rank1_host = args.host0
        args.rank1_local_rank = 1

    host0 = args._hosts[args.host0]
    rank1_host = args._hosts[args.rank1_host]

    def _cleanup():
        for h in args._hosts.values():
            h.cleanup()

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(143)))

    sizes = generate_sizes(args.min_size, args.max_size)
    benchmarks = _create_benchmarks(args, sizes, host0, rank1_host)
    _print_config(args, sizes)
    all_results = _run_benchmarks(args, benchmarks)

    if args.dry_run:
        print("(dry run — no commands were executed)")
        return

    print_table(
        args.tool,
        sizes,
        all_results.get("uniflow", {}),
        all_results.get("ib_write_bw", {}),
    )

    _cleanup()
    print("Done.")


if __name__ == "__main__":
    main()
