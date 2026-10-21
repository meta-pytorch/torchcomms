#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Build pinned ROCm 7.14.1/gfx950 resources; never execute native code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from comms.rcclx.develop.meta.relay.hipipc.packaging import (
        build_hipipc_relay_wheel as packager,
    )
else:
    spec = importlib.util.spec_from_file_location(
        "build_hipipc_relay_wheel",
        Path(__file__).with_name("build_hipipc_relay_wheel.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the sibling wheel packager")
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)

RELAY = "comms/rcclx/develop/meta/relay/"
HIPIPC = RELAY + "hipipc/"
NATIVE_INPUTS = (
    *packager.PUBLIC_HEADERS,
    RELAY + "registered_allreduce_relay_kernels.cu",
    HIPIPC + "hipipc_relay.cc",
    HIPIPC + "hipipc_probe.h",
    HIPIPC + "hipipc_probe.cu",
)
COMMON_FLAGS = (
    "-std=c++17",
    "-O3",
    "--offload-arch=gfx950",
    "-fPIC",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
)
HOST_FLAGS = ("-D__HIP_PLATFORM_AMD__=1", "-DHIP_UNCACHED_MEMORY")
SDK_NAMES = {
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-device-gfx950",
    "rocm-sdk-libraries",
}


class BuildError(RuntimeError):
    """A build or precondition failure."""


@dataclass(frozen=True)
class Artifact:
    name: str
    version: str
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class SDK:
    root: Path
    venv: Path
    hipcc: Path
    ar: Path
    hip_version: str


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_lock() -> tuple[list[Artifact], str]:
    if __package__:
        data = (
            importlib.resources.files(__package__)
            .joinpath("rocm714_sdk_lock.json")
            .read_bytes()
        )
    else:
        data = Path(__file__).with_name("rocm714_sdk_lock.json").read_bytes()
    lock = json.loads(data)
    if (lock["schema_version"], lock["version"], lock["architecture"]) != (
        1,
        "7.14.1",
        "gfx950",
    ):
        raise BuildError("Unsupported SDK lock identity")
    artifacts = [
        Artifact(**item) for item in lock["artifacts"] + lock["bootstrap_artifacts"]
    ]
    versions = dict.fromkeys(SDK_NAMES | {"rocm"}, "7.14.1")
    versions.update(setuptools="81.0.0", packaging="26.2")
    if len(artifacts) != 7 or {a.name for a in artifacts} != set(versions):
        raise BuildError(
            "SDK lock requires four payloads and three bootstrap artifacts"
        )
    for item in artifacts:
        url = urllib.parse.urlsplit(item.url)
        amd = item.name in SDK_NAMES | {"rocm"}
        origin = "repo.amd.com" if amd else "files.pythonhosted.org"
        if (
            item.version != versions[item.name]
            or Path(item.filename).name != item.filename
            or url.scheme != "https"
            or url.netloc != origin
            or not url.path.endswith("/" + item.filename)
            or url.query
            or url.fragment
            or not re.fullmatch(r"[0-9a-f]{64}", item.sha256)
        ):
            raise BuildError(f"Invalid pinned SDK artifact: {item.name}")
    return artifacts, hashlib.sha256(data).hexdigest()


def verify_artifact(path: Path, artifact: Artifact) -> None:
    # The checked-in lock pins complete original bytes, including package metadata.
    if path.is_symlink() or not path.is_file() or digest(path) != artifact.sha256:
        raise BuildError(f"SDK SHA256 mismatch or missing artifact: {path}")


def cached_artifact(directory: Path, artifact: Artifact, offline: bool) -> Path:
    path = directory / artifact.filename
    if path.exists() or path.is_symlink():
        verify_artifact(path, artifact)
        return path
    if offline:
        raise BuildError(f"Offline cache is missing verified artifact: {path}")
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=".partial", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as target:
            with urllib.request.urlopen(artifact.url, timeout=60) as response:
                url = urllib.parse.urlsplit(response.geturl())
                if url.scheme != "https" or url.netloc not in (
                    "repo.amd.com",
                    "files.pythonhosted.org",
                ):
                    raise BuildError(
                        "SDK download redirected outside pinned HTTPS origins"
                    )
                shutil.copyfileobj(response, target, length=1024 * 1024)
        verify_artifact(temporary, artifact)
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify_artifact(path, artifact)
        return path
    finally:
        temporary.unlink()


def managed_directory(root: Path, directory: Path) -> Path:
    if not directory.is_relative_to(root):
        raise BuildError(f"Managed directory escapes {root}: {directory}")
    for path in (directory, *directory.parents):
        if path.is_relative_to(root) and (
            path.is_symlink() or (path.exists() and not path.is_dir())
        ):
            raise BuildError(f"Managed directory must not be redirected: {path}")
    return directory


def child_environment(tools: dict[str, str], sdk: SDK | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME") if k in os.environ}
    bins = [str(Path(p).parent) for p in tools.values()]
    if sdk is not None:
        bins.insert(0, str(sdk.venv / "bin"))
        env.update(
            ROCM_PATH=str(sdk.root),
            HIP_PATH=str(sdk.root),
            LD_LIBRARY_PATH=str(sdk.root / "lib"),
        )
    env.update(
        PATH=os.pathsep.join(dict.fromkeys([*bins, "/usr/bin", "/bin"])),
        LANG="C.UTF-8",
        LC_ALL="C.UTF-8",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PIP_CONFIG_FILE=os.devnull,
        ROCM_SDK_TARGET_FAMILY="gfx950",
    )
    return env


class Runner:
    def __init__(self, directory: Path, env: dict[str, str]) -> None:
        self.directory = directory
        self.env = env
        self.sequence = 0
        for name in ("tmp", "comgr"):
            (directory / name).mkdir()

    def run(self, argv: list[str], cwd: Path, timeout: int = 600) -> str:
        log = self.directory / f"command-{self.sequence:03d}.log"
        self.sequence += 1
        try:
            with log.open("x") as stream:
                subprocess.run(
                    argv,
                    cwd=cwd,
                    env=dict(
                        self.env,
                        TMPDIR=str(self.directory / "tmp"),
                        COMGR_CACHE_DIR=str(self.directory / "comgr"),
                    ),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
        except (OSError, subprocess.SubprocessError) as error:
            raise BuildError(f"Command failed; inspect {log}: {error}") from error
        return log.read_text()


def prerequisites() -> dict[str, str]:
    if (
        sys.version_info[:2] != (3, 12)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        raise BuildError("Use fbpython Python 3.12 on Linux x86-64 with venv/ensurepip")
    tools: dict[str, str] = {}
    for name in ("sl", "readelf", "patchelf", "c++"):
        if (path := shutil.which(name)) is not None:
            tools[name] = str(Path(path).absolute())
        else:
            raise BuildError(f"Missing host prerequisite: {name}")
    return tools


def prepare_sdk(
    cache: Path,
    runner: Runner,
    artifacts: list[Artifact],
    lock_hash: str,
    offline: bool,
) -> SDK:
    # Keep the original archive location; never read or reuse its legacy SDK venv.
    downloads = managed_directory(cache, cache / "sdk" / lock_hash)
    downloads.mkdir(parents=True, exist_ok=True)
    paths = {a.name: cached_artifact(downloads, a, offline) for a in artifacts}
    venv = runner.directory / "sdk-venv"
    runner.run([sys.executable, "-m", "venv", str(venv)], runner.directory)
    pip = [
        str(venv / "bin/python"),
        "-I",
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-index",
        "--no-deps",
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    wheels = [str(paths[n]) for n in ("setuptools", "packaging", *sorted(SDK_NAMES))]
    runner.run([*pip, *wheels], runner.directory)
    runner.run([*pip, "--no-build-isolation", str(paths["rocm"])], runner.directory)
    cli = str(venv / "bin/rocm-sdk")
    runner.run([cli, "init"], runner.directory)
    root = Path(runner.run([cli, "path", "--root"], runner.directory).strip()).resolve()
    if not root.is_relative_to(venv.resolve()) or root.name != "_rocm_sdk_devel":
        raise BuildError("SDK CLI returned a foreign development root")
    hipcc, ar = venv / "bin/hipcc", root / "lib/llvm/bin/llvm-ar"
    if not ar.is_file():
        ar = root / "llvm/bin/llvm-ar"
    for tool in (hipcc, ar):
        if not tool.resolve().is_relative_to(venv.resolve()) or not os.access(
            tool, os.X_OK
        ):
            raise BuildError(f"Missing or foreign SDK tool: {tool}")
    sdk = SDK(root, venv, hipcc, ar, "7.14.60850")
    runner.env = child_environment({"python": sys.executable}, sdk)
    compiler = runner.run([str(hipcc), "--version"], runner.directory)
    if (
        "HIP version: " + sdk.hip_version not in compiler
        or "AMD clang version " not in compiler
    ):
        raise BuildError("SDK compiler does not match pinned HIP 7.14.60850")
    return sdk


def compile_command(
    hipcc: Path, source: Path, output: Path, includes: list[Path], host: bool = False
) -> list[str]:
    flags = HOST_FLAGS if host else ()
    return [
        str(hipcc),
        *COMMON_FLAGS,
        *flags,
        *["-I" + str(p) for p in includes],
        "-c",
        str(source),
        "-o",
        str(output),
    ]


def link_flags(sdk: SDK) -> list[str]:
    return ["-L" + str(sdk.root / "lib"), "-Wl,--no-as-needed", "-lamdhip64"]


def native_commands(
    source: Path, bundle: Path, objects: Path, sdk: SDK, patchelf: str
) -> list[list[str]]:
    commands = []
    for name in (
        "registered_allreduce_relay_kernels.cu",
        "hipipc_relay.cc",
        "hipipc_probe.cu",
    ):
        host = name != "registered_allreduce_relay_kernels.cu"
        inputs = [source, source / HIPIPC] if host else [source / RELAY]
        file = source / (HIPIPC if host else RELAY) / name
        commands.append(
            compile_command(
                sdk.hipcc, file, objects / (Path(name).stem + ".o"), inputs, host
            )
        )
    archive = bundle / packager.KERNEL_ARCHIVE
    commands.insert(
        1,
        [
            str(sdk.ar),
            "rcs",
            str(archive),
            str(objects / "registered_allreduce_relay_kernels.o"),
        ],
    )
    objects_to_link = [str(objects / n) for n in ("hipipc_relay.o", "hipipc_probe.o")]
    commands.append(
        [
            str(sdk.hipcc),
            *objects_to_link,
            str(archive),
            "-shared",
            "-Wl,-soname," + packager.SONAME,
            *link_flags(sdk),
            "-o",
            str(bundle / packager.SONAME),
        ]
    )
    return [*commands, [patchelf, "--remove-rpath", str(bundle / packager.SONAME)]]


def build_native(source: Path, runner: Runner, sdk: SDK, tools: dict[str, str]) -> Path:
    bundle, objects = runner.directory / "bundle", runner.directory / "objects"
    bundle.mkdir()
    objects.mkdir()
    for header in packager.PUBLIC_HEADERS:
        dest = bundle / "include" / header
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / header, dest)
    runner.env = child_environment(tools, sdk)
    for command in native_commands(source, bundle, objects, sdk, tools["patchelf"]):
        runner.run(command, runner.directory)
    smoke = runner.directory / "public-header-smoke.cc"
    smoke.write_text(
        "".join(f'#include "{h}"\n' for h in packager.PUBLIC_HEADERS)
        + "static_assert(rcclx::relay::hipipc::kAbiVersion == 1);\n"
        + "static_assert(std::is_same_v<decltype(hipipc_relay_abi_version()), uint32_t>);\n"
        + "int main() { return hipipc_relay_abi_version(); }\n"
    )
    runner.run(
        [
            str(sdk.hipcc),
            *COMMON_FLAGS,
            *HOST_FLAGS,
            "-I" + str(bundle / "include"),
            str(smoke),
            "-L" + str(bundle),
            "-lhipipc_relay",
            *link_flags(sdk),
            "-Wl,-rpath-link," + str(sdk.root / "lib"),
            "-o",
            str(runner.directory / "public-header-smoke"),
        ],
        runner.directory,
    )
    return bundle


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, help="Dedicated downloads and retained runs"
    )
    parser.add_argument("--output-dir", type=Path, help="Must not exist")
    parser.add_argument("--base-version", default="0.1.0", help="X.Y.Z")
    parser.add_argument("--offline", action="store_true", help="Verified cache only")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.base_version):
        parser.error("base-version must be X.Y.Z")
    return args


def fresh_output(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise BuildError(f"Output already exists; choose a fresh --output-dir: {path}")
    if any(p.is_symlink() for p in path.parents) or ".." in path.parts:
        raise BuildError(f"Output must not be redirected: {path}")
    return path


def require_clean_inputs(root: Path, runner: Runner, sl: str) -> None:
    paths = ["fbcode/" + name for name in NATIVE_INPUTS]
    paths.append("fbcode/" + HIPIPC + "packaging")
    changed = runner.run(
        [
            sl,
            "--reason",
            "Verify wheel source provenance - sl help status",
            "status",
            *paths,
        ],
        root,
    )
    if changed.strip():
        raise BuildError(
            "Commit native and packaging inputs before building a source-versioned wheel"
        )


def execute(
    args: argparse.Namespace, cache: Path, runner: Runner, tools: dict[str, str]
) -> Path:
    output = fresh_output(args.output_dir or runner.directory / "wheels")
    output.mkdir(parents=True)
    root = Path(
        runner.run(
            [
                tools["sl"],
                "--reason",
                "Resolve wheel source root - sl help root",
                "root",
            ],
            Path(__file__).resolve().parent,
        ).strip()
    ).resolve()
    if (
        Path(__file__).resolve()
        != root / "fbcode" / HIPIPC / "packaging/build_rocm714_wheel.py"
    ):
        raise BuildError("Invoke the checked-in entry point from its Sapling checkout")
    head = runner.run(
        [
            tools["sl"],
            "--reason",
            "Record wheel source version - sl help log",
            "log",
            "-r",
            ".",
            "-T",
            "{node}",
        ],
        root,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise BuildError("Cannot capture full source HEAD")
    require_clean_inputs(root, runner, tools["sl"])
    source = runner.directory / "source"
    for name in NATIVE_INPUTS:
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "fbcode" / name, dest)
    artifacts, lock_hash = load_lock()
    sdk = prepare_sdk(cache, runner, artifacts, lock_hash, args.offline)
    bundle = build_native(source, runner, sdk, tools)
    version = f"{args.base_version}+abi1.{head[:10]}"
    options = {
        "native-dir": str(bundle),
        "output-dir": str(output),
        "version": version,
        "source-commit": head,
        "rocm-version": sdk.hip_version,
        "architecture": "gfx950",
    }
    cli_args = [
        value for name, value in options.items() for value in ("--" + name, value)
    ]
    runner.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("build_hipipc_relay_wheel.py").resolve()),
            *cli_args,
        ],
        runner.directory,
    )
    wheel = output / f"hipipc_relay-{version}-py3-none-linux_x86_64.whl"
    print(f"{wheel}\nSHA256 {digest(wheel)}")
    return wheel


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.output_dir is not None:
        fresh_output(args.output_dir)
    tools = prerequisites()
    user_cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = (args.cache_dir or user_cache / "hipipc-rocm714").expanduser().resolve()
    runs = managed_directory(cache, cache / "runs")
    runs.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="build-", dir=runs))
    execute(args, cache, Runner(directory, child_environment(tools)), tools)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (BuildError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
