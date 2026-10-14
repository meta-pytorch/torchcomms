#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Build a standalone offline hipipc_relay wheel from a built libhipipc_relay.so.

Standard library only. Copies no native sources. The wheel is a leaf package:
``hipipc_relay`` imports with no hip runtime, ships the prebuilt shared object
as data, and declares no torch/rccl dependencies. Tag: py3-none-linux_x86_64.

Usage:
  python3 build_hipipc_relay_wheel.py \
    --library /abs/path/libhipipc_relay.so \
    --output-dir /abs/scratch \
    --version 0.1.0+abi1.sha

Verifies the ELF SONAME/exports with readelf (no GPU runtime at build).
Writes a structured RECORD with METADATA/WHEEL and a csv hash manifest.
No uploads, no pip installs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


SONAME = "libhipipc_relay.so"
PKG = "hipipc_relay"
REQUIRED_EXPORTS = (
    "hipipc_relay_abi_version",
    "hipipc_relay_device_info",
    "hipipc_relay_descriptor_size",
    "hipipc_relay_create",
    "hipipc_relay_export",
    "hipipc_relay_import",
    "hipipc_relay_probe",
    "hipipc_relay_execute",
    "hipipc_relay_drain",
    "hipipc_relay_close_imports",
    "hipipc_relay_destroy",
    "hipipc_relay_last_error",
)


def _abspath(value: str, label: str) -> str:
    if not os.path.isabs(value):
        raise SystemExit(f"--{label} must be absolute, got {value!r}")
    return value


def _verify_elf(path: str) -> None:
    try:
        dyn = subprocess.run(
            ["readelf", "-d", path], capture_output=True, text=True, check=True
        )
        syms = subprocess.run(
            ["readelf", "--wide", "--dyn-syms", path],
            capture_output=True,
            text=True,
            check=True,
        )
        header = subprocess.run(
            ["readelf", "-h", path], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"readelf verification failed: {exc}")
    sonames = re.findall(r"\(SONAME\)\s+Library soname:\s+\[([^\]]+)\]", dyn.stdout)
    if sonames != [SONAME]:
        raise SystemExit(f"ELF SONAME must be exactly {SONAME}: {sonames}")
    if not re.search(
        r"Machine:\s+Advanced Micro Devices X86-64", header.stdout
    ) or not re.search(r"Class:\s+ELF64", header.stdout):
        raise SystemExit("the linux_x86_64 wheel requires a 64-bit x86-64 ELF")
    needed = re.findall(r"\(NEEDED\)[^\n]*\[([^\]]+)\]", dyn.stdout)
    forbidden = [
        name
        for name in needed
        if any(
            part in name.lower() for part in ("rccl", "nccl", "torch", "c10", "lazify")
        )
    ]
    if forbidden:
        raise SystemExit(
            f"standalone ELF has forbidden runtime dependencies: {forbidden}"
        )
    paths = re.findall(r"\((?:RPATH|RUNPATH)\)[^\n]*\[([^\]]+)\]", dyn.stdout)
    if any("fbsource" in value or "buck-out" in value for value in paths):
        raise SystemExit(
            "standalone ELF must not resolve runtime dependencies through build-tree paths"
        )
    exports = set()
    for line in syms.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[7].startswith("ncclRegistered"):
            raise SystemExit("standalone ELF must not resolve RCCLX registered APIs")
        if (
            len(fields) >= 8
            and fields[3] == "FUNC"
            and fields[4] in ("GLOBAL", "WEAK")
            and fields[5] in ("DEFAULT", "PROTECTED")
            and fields[6] != "UND"
        ):
            exports.add(fields[7].split("@", 1)[0])
    missing = sorted(set(REQUIRED_EXPORTS) - exports)
    if missing:
        raise SystemExit(f"ELF misses exports: {missing}")


def _hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="ABS path to built .so")
    parser.add_argument("--output-dir", required=True, help="ABS scratch dir")
    parser.add_argument("--version", required=True, help="e.g. 0.1.0+abi1.sha")
    args = parser.parse_args(argv)
    if not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:\+[a-z0-9]+(?:\.[a-z0-9]+)*)?", args.version
    ):
        raise SystemExit("version must be X.Y.Z with an optional +local.identifier")
    library = _abspath(args.library, "library")
    outdir = _abspath(args.output_dir, "output-dir")
    if not os.path.isfile(library):
        raise SystemExit(f"library not found: {library}")
    if os.path.isdir(library):
        raise SystemExit("library must be a file, not a directory")
    _verify_elf(library)
    os.makedirs(outdir, exist_ok=True)

    dist_info = f"{PKG}-{args.version}.dist-info"
    wheel_name = f"{PKG}-{args.version}-py3-none-linux_x86_64.whl"
    metadata = (
        f"Metadata-Version: 2.1\nName: {PKG}\nVersion: {args.version}\n"
        "Summary: Standalone TP2 HIPIPC relay runtime (data-only wheel)\n"
        "Requires-Python: >=3.12\n"
    )
    wheel = (
        "Wheel-Version: 1.0\nGenerator: build_hipipc_relay_wheel.py\n"
        "Root-Is-Purelib: false\nTag: py3-none-linux_x86_64\n"
    )
    init_py = '"""Leaf hipipc_relay package: no hip runtime at import."""\n'
    with open(library, "rb") as fh:
        so_bytes = fh.read()

    files: list[tuple[str, bytes]] = [
        (f"{PKG}/__init__.py", init_py.encode()),
        (f"{PKG}/{SONAME}", so_bytes),
        (f"{dist_info}/METADATA", metadata.encode()),
        (f"{dist_info}/WHEEL", wheel.encode()),
    ]
    record_rows: list[tuple[str, str, str]] = []
    for name, data in files:
        record_rows.append((name, _hash(data), str(len(data))))
    record_rows.append((f"{dist_info}/RECORD", "", ""))
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in record_rows:
        writer.writerow(row)
    files.append((f"{dist_info}/RECORD", buf.getvalue().encode()))

    with tempfile.TemporaryDirectory() as tmp:
        wheel_path = os.path.join(tmp, wheel_name)
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files:
                zf.writestr(name, data)
        dest = os.path.join(outdir, wheel_name)
        with open(wheel_path, "rb") as source, open(dest, "xb") as target:
            shutil.copyfileobj(source, target)
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
