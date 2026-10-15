# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from comms.rcclx.develop.meta.relay.hipipc.packaging import (
        build_hipipc_relay_wheel as builder,
    )
except ModuleNotFoundError as error:
    package = "comms.rcclx.develop.meta.relay.hipipc.packaging"
    if error.name is None or not (
        error.name == package or package.startswith(error.name + ".")
    ):
        raise
    _SPEC = importlib.util.spec_from_file_location(
        "build_hipipc_relay_wheel",
        Path(__file__).parent.parent / "build_hipipc_relay_wheel.py",
    )
    assert _SPEC is not None and _SPEC.loader is not None
    builder = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(builder)


class HipipcRelayWheelTest(unittest.TestCase):
    def _readelf(self, soname="libhipipc_relay.so", undefined=False):
        index = "UND" if undefined else "12"
        symbols = "\n".join(
            f" 1: 0000000000001000 16 FUNC GLOBAL DEFAULT {index} {name}"
            for name in builder.REQUIRED_EXPORTS
        )
        return [
            SimpleNamespace(stdout=f" (SONAME) Library soname: [{soname}]\n"),
            SimpleNamespace(stdout=symbols),
            SimpleNamespace(
                stdout=" Class: ELF64\n Machine: Advanced Micro Devices X86-64\n"
            ),
        ]

    def test_exact_soname_and_defined_exports(self):
        with mock.patch.object(
            builder.subprocess, "run", side_effect=self._readelf()
        ) as run:
            builder._verify_elf("/temporary/libhipipc_relay.so")
        self.assertEqual(run.call_count, 3)
        self.assertIn("--wide", run.call_args_list[1].args[0])

    def test_similar_soname_is_rejected(self):
        with mock.patch.object(
            builder.subprocess,
            "run",
            side_effect=self._readelf("other_libhipipc_relay.so"),
        ):
            with self.assertRaisesRegex(SystemExit, "SONAME must be exactly"):
                builder._verify_elf("/temporary/libhipipc_relay.so")

    def test_undefined_api_references_are_not_exports(self):
        with mock.patch.object(
            builder.subprocess, "run", side_effect=self._readelf(undefined=True)
        ):
            with self.assertRaisesRegex(SystemExit, "misses exports"):
                builder._verify_elf("/temporary/libhipipc_relay.so")

    def test_nonfunction_and_hidden_api_symbols_are_rejected(self):
        for replacement in ("OBJECT GLOBAL DEFAULT", "FUNC GLOBAL HIDDEN"):
            results = self._readelf()
            results[1].stdout = results[1].stdout.replace(
                "FUNC GLOBAL DEFAULT", replacement
            )
            with (
                self.subTest(replacement=replacement),
                mock.patch.object(builder.subprocess, "run", side_effect=results),
            ):
                with self.assertRaisesRegex(SystemExit, "misses exports"):
                    builder._verify_elf("/temporary/libhipipc_relay.so")

    def test_runtime_dependency_and_build_path_rejection(self):
        for dependency in (
            "librccl.so.1",
            "libnccl.so",
            "libtorch.so",
            "libc10.so",
            "libtools_lazify_lazy_stub_binder.so",
        ):
            results = self._readelf()
            results[0].stdout += f" (NEEDED) Shared library: [{dependency}]\n"
            with (
                self.subTest(dependency=dependency),
                mock.patch.object(builder.subprocess, "run", side_effect=results),
            ):
                with self.assertRaisesRegex(
                    SystemExit, "forbidden runtime dependencies"
                ):
                    builder._verify_elf("/temporary/libhipipc_relay.so")
        results = self._readelf()
        results[
            0
        ].stdout += (
            " (RUNPATH) Library runpath: [/temporary/fbsource/buck-out/runtime]\n"
        )
        with mock.patch.object(builder.subprocess, "run", side_effect=results):
            with self.assertRaisesRegex(SystemExit, "build-tree paths"):
                builder._verify_elf("/temporary/libhipipc_relay.so")

    def test_registered_api_resolution_is_rejected(self):
        results = self._readelf()
        results[
            1
        ].stdout += (
            "\n 13: 0000000000000000 0 FUNC GLOBAL DEFAULT UND ncclRegisteredAllReduce"
        )
        with mock.patch.object(builder.subprocess, "run", side_effect=results):
            with self.assertRaisesRegex(SystemExit, "RCCLX registered APIs"):
                builder._verify_elf("/temporary/libhipipc_relay.so")

    def test_version_cannot_escape_output_directory(self):
        with self.assertRaisesRegex(SystemExit, "version must be"):
            builder.main(
                [
                    "--library",
                    "/temporary/library.so",
                    "--output-dir",
                    "/temporary/output",
                    "--version",
                    "../../escape",
                ]
            )

    def test_wheel_record_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            library = directory / "libhipipc_relay.so"
            library.write_bytes(b"test artifact payload")
            args = [
                "--library",
                str(library),
                "--output-dir",
                str(directory),
                "--version",
                "0.1.0+abi1.test",
            ]
            with (
                mock.patch.object(builder, "_verify_elf"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(builder.main(args), 0)
                with self.assertRaises(FileExistsError):
                    builder.main(args)
            wheels = list(directory.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                record = next(
                    name for name in archive.namelist() if name.endswith("/RECORD")
                )
                for name, digest, size in csv.reader(
                    io.StringIO(archive.read(record).decode())
                ):
                    if name == record:
                        self.assertEqual((digest, size), ("", ""))
                        continue
                    contents = archive.read(name)
                    expected = (
                        base64.urlsafe_b64encode(hashlib.sha256(contents).digest())
                        .rstrip(b"=")
                        .decode()
                    )
                    self.assertEqual(digest, "sha256=" + expected)
                    self.assertEqual(int(size), len(contents))
                metadata = archive.read(
                    "hipipc_relay-0.1.0+abi1.test.dist-info/METADATA"
                ).decode()
                self.assertNotIn("Requires-Dist:", metadata)


if __name__ == "__main__":
    unittest.main()
