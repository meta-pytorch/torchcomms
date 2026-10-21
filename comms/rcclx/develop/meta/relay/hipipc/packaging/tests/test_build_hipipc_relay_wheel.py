# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import io
import json
import subprocess
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from comms.rcclx.develop.meta.relay.hipipc.packaging import (
    build_hipipc_relay_wheel as builder,
)

HEAD = "0123456789abcdef0123456789abcdef01234567"
HEADERS = (
    "comms/rcclx/develop/meta/relay/hipipc/hipipc_relay.h",
    "comms/rcclx/develop/meta/relay/hipipc/hipipc_record.h",
    "comms/rcclx/develop/meta/relay/registered_allreduce_relay_args.h",
    "comms/rcclx/develop/meta/relay/registered_allreduce_relay_kernels.h",
)


class WheelBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)
        self.native = self.directory / "native"
        self.payload = {
            "libhipipc_relay.so": b"synthetic ELF",
            "libhipipc_relay_kernel.a": b"!<arch>\nfixture",
            **{"include/" + h: b"#pragma once\n" for h in HEADERS},
        }
        self.payload["include/" + HEADERS[0]] += (
            b'#include "hipipc_record.h"\nextern "C" uint32_t hipipc_relay_abi_version();\n'
        )
        self.payload["include/" + HEADERS[1]] += (
            b"namespace rcclx::relay::hipipc { inline constexpr uint32_t kAbiVersion = 1; }\n"
        )
        for name, data in self.payload.items():
            path = self.native / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        options = {
            "native-dir": str(self.native),
            "output-dir": str(self.directory),
            "version": "0.1.0+abi1." + HEAD[:10],
            "source-commit": HEAD,
            "rocm-version": "7.14.60850",
            "architecture": "gfx950",
        }
        self.argv = [v for k, v in options.items() for v in ("--" + k, v)]
        self.args = builder.parse_args(self.argv)
        self.dynamic = " (SONAME) Library soname: [libhipipc_relay.so]\n"
        self.header = " Class: ELF64\n Machine: Advanced Micro Devices X86-64\n"
        self.symbols = (
            " 1: 00000000 16 FUNC GLOBAL DEFAULT 12 hipipc_relay_abi_version\n"
        )

    def readelf(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.assertEqual(command[0], "readelf")
        self.assertEqual(command[-2], "--")
        self.assertEqual(kwargs["timeout"], 60)
        path = Path(command[-1])
        self.assertNotEqual(path.parent, self.native)
        self.assertEqual(path.read_bytes(), self.payload["libhipipc_relay.so"])
        output = {"-d": self.dynamic, "-h": self.header}.get(command[1], self.symbols)
        return subprocess.CompletedProcess(command, 0, stdout=output)

    def files(self) -> list[tuple[str, bytes]]:
        with patch.object(builder.subprocess, "run", side_effect=self.readelf):
            native = builder.native_payload(self.native)
        self.assertEqual(len(native), 6)
        return builder.wheel_files(self.args, native)

    def test_metadata_manifest_and_record(self) -> None:
        files = dict(self.files())
        info = f"hipipc_relay-{self.args.version}.dist-info"
        self.assertEqual(len(files), 11)
        self.assertIn("Tag: py3-none-linux_x86_64\n", files[info + "/WHEEL"].decode())
        self.assertIn(
            "Version: " + self.args.version, files[info + "/METADATA"].decode()
        )
        self.assertNotIn(b"Requires-Dist:", files[info + "/METADATA"])
        manifest = json.loads(files["hipipc_relay/manifest.json"])
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(manifest["source_commit"], HEAD)
        self.assertEqual(len(manifest["payloads"]), 6)
        for item in manifest["payloads"]:
            self.assertEqual(
                item["sha256"], hashlib.sha256(files[item["path"]]).hexdigest()
            )
        rows = list(csv.reader(io.StringIO(files[info + "/RECORD"].decode())))
        self.assertCountEqual([r[0] for r in rows], files)
        for name, digest, size in rows[:-1]:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest())
            self.assertEqual(digest, "sha256=" + encoded.rstrip(b"=").decode())
            self.assertEqual(int(size), len(files[name]))
        self.assertEqual(rows[-1], [info + "/RECORD", "", ""])

    def test_cpu_safe_installed_api(self) -> None:
        code = dict(self.files())["hipipc_relay/__init__.py"]
        module = types.ModuleType("hipipc_relay")
        root = self.directory / "installed/hipipc_relay"
        module.__file__ = str(root / "__init__.py")
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("runtime audit")
        ):
            exec(compile(code, module.__file__, "exec"), module.__dict__)
            self.assertEqual(module.__dict__["__version__"], self.args.version)
            self.assertEqual(module.__dict__["ABI_VERSION"], 1)
            self.assertEqual(module.__dict__["get_include"](), str(root / "include"))
            self.assertEqual(module.__dict__["get_library_dir"](), str(root))

    def test_missing_empty_symlink_payload_and_regular_archive(self) -> None:
        for name, data in self.payload.items():
            path = self.native / name
            for kind in ("missing", "empty", "symlink"):
                with self.subTest(name=name, kind=kind):
                    path.unlink()
                    if kind == "empty":
                        path.touch()
                    elif kind == "symlink":
                        path.symlink_to(self.directory / "absent")
                    with self.assertRaises(SystemExit):
                        builder.native_payload(self.native)
                    if path.exists() or path.is_symlink():
                        path.unlink()
                    path.write_bytes(data)
        (self.native / "libhipipc_relay_kernel.a").write_bytes(b"!<thin>\n")
        with self.assertRaisesRegex(SystemExit, "regular ar"):
            builder.native_payload(self.native)

    def test_header_and_cli(self) -> None:
        (self.native / "include" / HEADERS[0]).write_bytes(b"// kAbiVersion = 1;\n")
        with self.assertRaisesRegex(SystemExit, "public header"):
            builder.native_payload(self.native)
        declaration = self.payload["include/" + HEADERS[0]].replace(b"uint32_t", b"int")
        (self.native / "include" / HEADERS[0]).write_bytes(declaration)
        with self.assertRaisesRegex(SystemExit, "uint32_t ABI getter"):
            builder.native_payload(self.native)
        for flag, value in (
            ("--version", "0.1.0+abi1.aaaaaaaaaa"),
            ("--output-dir", "relative"),
        ):
            argv = self.argv.copy()
            argv[argv.index(flag) + 1] = value
            with (
                self.subTest(flag=flag),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                builder.parse_args(argv)

    def test_forbidden_dependencies_and_rpaths(self) -> None:
        original = self.dynamic
        cases = [
            ("NEEDED", n)
            for n in (
                "librccl.so",
                "libnccl.so",
                "libtorch.so",
                "libc10.so",
                "liblazify.so",
                "libamdhip64-lazy.so",
            )
        ]
        cases += [("RPATH", "/tmp/fbsource/lib"), ("RUNPATH", "/tmp/buck-out/lib")]
        cases += [("RPATH", "/tmp/user-sdk/lib"), ("RUNPATH", "$ORIGIN")]
        for tag, value in cases:
            with self.subTest(tag=tag, value=value):
                self.dynamic = original + f" ({tag}) Library: [{value}]\n"
                with (
                    patch.object(builder.subprocess, "run", side_effect=self.readelf),
                    self.assertRaises(SystemExit),
                ):
                    builder.native_payload(self.native)

    def test_abi_visibility_and_forbidden_references(self) -> None:
        attributes = (
            "FUNC GLOBAL HIDDEN 12",
            "FUNC GLOBAL DEFAULT UND",
            "FUNC LOCAL DEFAULT 12",
            "OBJECT GLOBAL DEFAULT 12",
        )
        for attr in attributes:
            with self.subTest(attr=attr):
                self.symbols = f" 1: 00000000 16 {attr} hipipc_relay_abi_version\n"
                with (
                    patch.object(builder.subprocess, "run", side_effect=self.readelf),
                    self.assertRaisesRegex(SystemExit, "visible defined"),
                ):
                    builder.native_payload(self.native)
        self.symbols = (
            " 1: 00000000 16 FUNC GLOBAL DEFAULT UND ncclRegisteredAllReduce\n"
        )
        with (
            patch.object(builder.subprocess, "run", side_effect=self.readelf),
            self.assertRaisesRegex(SystemExit, "registered APIs"),
        ):
            builder.native_payload(self.native)

    def test_elf_identity_and_readelf_failure(self) -> None:
        self.header = " Class: ELF32\n Machine: AArch64\n"
        with (
            patch.object(builder.subprocess, "run", side_effect=self.readelf),
            self.assertRaisesRegex(SystemExit, "ELF64"),
        ):
            builder.native_payload(self.native)
        with (
            patch.object(builder.subprocess, "run", side_effect=FileNotFoundError),
            self.assertRaisesRegex(SystemExit, "readelf verification"),
        ):
            builder.native_payload(self.native)

    def test_atomic_write_collision_and_failure(self) -> None:
        files = self.files()
        dest = self.directory / "fixture.whl"
        builder.write_wheel(dest, files)
        before = dest.read_bytes()
        with zipfile.ZipFile(dest) as archive:
            self.assertIsNone(archive.testzip())
        with self.assertRaises(FileExistsError):
            builder.write_wheel(dest, files)
        self.assertEqual(dest.read_bytes(), before)
        failed = self.directory / "failed.whl"
        with (
            patch.object(builder.os, "fsync", side_effect=OSError("fixture failure")),
            self.assertRaises(OSError),
        ):
            builder.write_wheel(failed, files)
        self.assertFalse(failed.exists())
        self.assertEqual(list(self.directory.glob(".hipipc_relay-*.tmp")), [])
