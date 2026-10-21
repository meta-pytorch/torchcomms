#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Package a materialized HIPIPC bundle offline, without loading native code."""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PKG = "hipipc_relay"
SONAME = "libhipipc_relay.so"
KERNEL_ARCHIVE = "libhipipc_relay_kernel.a"
ABI_VERSION = 1
ABI_EXPORT = "hipipc_relay_abi_version"
PUBLIC_HEADERS = (
    "comms/rcclx/develop/meta/relay/hipipc/hipipc_relay.h",
    "comms/rcclx/develop/meta/relay/hipipc/hipipc_record.h",
    "comms/rcclx/develop/meta/relay/registered_allreduce_relay_args.h",
    "comms/rcclx/develop/meta/relay/registered_allreduce_relay_kernels.h",
)


def readelf(path: Path, *options: str) -> str:
    try:
        return subprocess.run(
            ["readelf", *options, "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"readelf verification failed: {error}") from error


def verify_elf(path: Path) -> None:
    dynamic = readelf(path, "-d")
    if re.findall(r"\(SONAME\)[^\n]*\[([^\]]+)\]", dynamic) != [SONAME]:
        raise SystemExit(f"ELF SONAME must be exactly {SONAME}")
    needed = re.findall(r"\(NEEDED\)[^\n]*\[([^\]]+)\]", dynamic)
    if any(
        part in name.lower()
        for name in needed
        for part in ("rccl", "nccl", "torch", "c10", "lazify", "amdhip64-lazy")
    ):
        raise SystemExit("standalone ELF has forbidden runtime dependencies")
    paths = re.findall(r"\((?:RPATH|RUNPATH)\)[^\n]*\[([^\]]+)\]", dynamic)
    if paths:
        raise SystemExit("standalone ELF must not have runtime paths")
    header = readelf(path, "-h")
    if not re.search(r"Class:\s+ELF64\b", header) or not re.search(
        r"Machine:\s+Advanced Micro Devices X86-64", header
    ):
        raise SystemExit("the wheel requires an ELF64 x86-64 library")
    exports = set()
    for line in readelf(path, "--wide", "--dyn-syms").splitlines():
        fields = line.split()
        if len(fields) < 8 or not re.fullmatch(r"\d+:", fields[0]):
            continue
        name = fields[7].split("@", 1)[0]
        if name.startswith("ncclRegistered"):
            raise SystemExit("standalone ELF must not reference RCCLX registered APIs")
        if (
            fields[3] == "FUNC"
            and fields[4] in ("GLOBAL", "WEAK")
            and fields[5] in ("DEFAULT", "PROTECTED")
            and fields[6] != "UND"
        ):
            exports.add(name)
    if ABI_EXPORT not in exports:
        raise SystemExit(f"ELF misses visible defined function: {ABI_EXPORT}")


def native_payload(root: Path) -> list[tuple[str, bytes]]:
    if (
        not root.is_absolute()
        or ".." in root.parts
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise SystemExit("native-dir must be an absolute materialized directory")
    names = [SONAME, KERNEL_ARCHIVE, *("include/" + h for h in PUBLIC_HEADERS)]
    files = []
    for name in names:
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise SystemExit(f"native payload must not contain symlinks: {path}")
        if not path.is_file():
            raise SystemExit(f"native payload file not found: {path}")
        data = path.read_bytes()
        if not data:
            raise SystemExit(f"native payload file is empty: {path}")
        files.append((f"{PKG}/{name}", data))
    payload = dict(files)
    if not payload[f"{PKG}/{KERNEL_ARCHIVE}"].startswith(b"!<arch>\n"):
        raise SystemExit("kernel archive must have regular ar magic !<arch>")
    header = "\n".join(
        payload[f"{PKG}/include/{h}"].decode("utf-8") for h in PUBLIC_HEADERS
    )
    header = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", header)
    if not re.search(r"\bkAbiVersion\s*=\s*1\s*;", header) or not re.search(
        rf"\buint32_t\s+{ABI_EXPORT}\s*\(\s*\)\s*;", header
    ):
        raise SystemExit(
            "public headers must declare kAbiVersion = 1 and the uint32_t ABI getter"
        )
    # Inspect the same snapshot bytes that the manifest and wheel will contain.
    with tempfile.TemporaryDirectory() as scratch:
        library = Path(scratch) / SONAME
        library.write_bytes(payload[f"{PKG}/{SONAME}"])
        verify_elf(library)
    return files


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    flags = "native-dir output-dir version source-commit rocm-version"
    for flag in flags.split():
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--architecture", required=True, choices=["gfx950"])
    args = parser.parse_args(argv)
    match = re.fullmatch(r"\d+\.\d+\.\d+\+abi1\.([0-9a-f]{10,40})", args.version)
    if match is None or not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_commit):
        parser.error(
            "version requires X.Y.Z+abi1.SOURCE_HASH and source-commit full 40-hex"
        )
    if not args.source_commit.lower().startswith(match[1]):
        parser.error("version source hash must match source-commit prefix")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}", args.rocm_version):
        parser.error("rocm-version must be a nonempty build identifier")
    if not all(Path(p).is_absolute() for p in (args.native_dir, args.output_dir)):
        parser.error("native-dir and output-dir must be absolute")
    return args


def wheel_files(
    args: argparse.Namespace, native: list[tuple[str, bytes]]
) -> list[tuple[str, bytes]]:
    info = f"{PKG}-{args.version}.dist-info"
    manifest = {
        "schema_version": 1,
        "abi_version": ABI_VERSION,
        "source_commit": args.source_commit.lower(),
        "rocm_version": args.rocm_version,
        "architecture": args.architecture,
        "payloads": [
            {
                "path": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
            for name, data in native
        ],
    }
    init = (
        '"""HIPIPC build resources; importing this package does not load HIP."""\n'
        "from pathlib import Path\n"
        f"__version__ = {args.version!r}\nABI_VERSION = {ABI_VERSION}\n\n"
        'def get_include() -> str:\n    return str(Path(__file__).resolve().parent / "include")\n\n'
        "def get_library_dir() -> str:\n    return str(Path(__file__).resolve().parent)\n"
    )
    files = dict(native)
    files[f"{PKG}/__init__.py"] = init.encode()
    files[f"{PKG}/manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    metadata = f"Metadata-Version: 2.1\nName: {PKG}\nVersion: {args.version}\n"
    files[f"{info}/METADATA"] = (metadata + "Requires-Python: >=3.12\n").encode()
    files[f"{info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: build_hipipc_relay_wheel.py\n"
        b"Root-Is-Purelib: false\nTag: py3-none-linux_x86_64\n"
    )
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, data in files.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        digest = encoded.rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    return [*files.items(), (f"{info}/RECORD", record.getvalue().encode())]


def write_wheel(dest: Path, files: list[tuple[str, bytes]]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=".hipipc_relay-", suffix=".tmp", dir=dest.parent
    )
    try:
        with os.fdopen(fd, "wb") as target:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, data in files:
                    archive.writestr(name, data)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, dest)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    native = native_payload(Path(args.native_dir))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / f"{PKG}-{args.version}-py3-none-linux_x86_64.whl"
    print(dest)
    write_wheel(dest, wheel_files(args, native))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
