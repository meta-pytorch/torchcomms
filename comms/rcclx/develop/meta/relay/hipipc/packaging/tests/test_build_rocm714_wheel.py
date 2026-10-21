# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from comms.rcclx.develop.meta.relay.hipipc.packaging import (
    build_rocm714_wheel as builder,
)


class DownloadFixture(io.BytesIO):
    def __init__(self, data: bytes, url: str) -> None:
        super().__init__(data)
        self.url = url

    def geturl(self) -> str:
        return self.url


class CompilerFixture(builder.Runner):
    def __init__(self, directory: Path) -> None:
        super().__init__(directory, {})
        self.commands: list[list[str]] = []

    def run(self, argv: list[str], cwd: Path, timeout: int = 600) -> str:
        del cwd, timeout
        self.commands.append(argv)
        return ""


class RocmWheelBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)
        self.artifacts, self.lock_hash = builder.load_lock()
        self.artifact = next(a for a in self.artifacts if a.name == "rocm-sdk-core")
        self.cache = self.directory / "cache"
        self.cache.mkdir()
        self.source = self.directory / "fixture"
        self.source.write_bytes(b"pinned artifact fixture")
        self.fixture = replace(self.artifact, sha256=builder.digest(self.source))

    def test_lock_identity_and_validation(self) -> None:
        self.assertEqual(len(self.artifacts), 7)
        self.assertEqual(
            self.lock_hash,
            "6aeb3ff1bc3bcd9da5d83932d8d68ea01eeae1124716f8268e3eb6564fee9be2",
        )
        lock = Path(builder.__file__).with_name("rocm714_sdk_lock.json")
        data = json.loads(lock.read_bytes())
        data["architecture"] = "gfx942"
        (self.directory / lock.name).write_text(json.dumps(data))
        with (
            patch.object(
                builder.importlib.resources, "files", return_value=self.directory
            ),
            self.assertRaises(builder.BuildError),
        ):
            builder.load_lock()

    def test_offline_missing_and_cached_hash_mismatch(self) -> None:
        path = self.cache / self.artifact.filename
        with patch.object(builder.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(builder.BuildError, "Offline cache"):
                builder.cached_artifact(self.cache, self.artifact, True)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(builder.BuildError, "SHA256"):
                builder.cached_artifact(self.cache, self.artifact, False)
        download.assert_not_called()
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_verified_download_and_offline_reuse(self) -> None:
        response = DownloadFixture(self.source.read_bytes(), self.fixture.url)
        with patch.object(
            builder.urllib.request, "urlopen", return_value=response
        ) as download:
            path = builder.cached_artifact(self.cache, self.fixture, False)
        download.assert_called_once_with(self.fixture.url, timeout=60)
        self.assertEqual(path.read_bytes(), self.source.read_bytes())
        with patch.object(builder.urllib.request, "urlopen") as download:
            self.assertEqual(
                builder.cached_artifact(self.cache, self.fixture, True), path
            )
        download.assert_not_called()
        self.assertEqual(list(self.cache.glob("*.partial")), [])

    def test_invalid_download_not_cached(self) -> None:
        for url in (self.artifact.url, "https://repo.amd.com.evil.example/file"):
            with self.subTest(url=url):
                response = DownloadFixture(b"corrupt", url)
                with (
                    patch.object(
                        builder.urllib.request, "urlopen", return_value=response
                    ),
                    self.assertRaises(builder.BuildError),
                ):
                    builder.cached_artifact(self.cache, self.artifact, False)
                self.assertFalse((self.cache / self.artifact.filename).exists())
                self.assertEqual(list(self.cache.glob("*.partial")), [])

    def test_existing_output_refused_before_external_work(self) -> None:
        for kind in ("empty", "nonempty", "dangling", "file"):
            output = self.directory / kind
            if kind == "dangling":
                output.symlink_to(self.directory / "absent")
            elif kind == "file":
                output.write_bytes(b"preserve")
            else:
                output.mkdir()
                if kind == "nonempty":
                    (output / "sentinel").write_bytes(b"preserve")
            with (
                self.subTest(kind=kind),
                patch.object(builder.subprocess, "run") as external,
                patch.object(builder.urllib.request, "urlopen") as download,
            ):
                with self.assertRaisesRegex(
                    builder.BuildError, "Output already exists"
                ):
                    builder.main(["--output-dir", str(output)])
            external.assert_not_called()
            download.assert_not_called()
            self.assertTrue(output.exists() or output.is_symlink())

    def test_typed_compile_fixture_and_header_smoke(self) -> None:
        sdk = builder.SDK(
            self.directory / "sdk",
            self.directory / "venv",
            self.directory / "hipcc",
            self.directory / "llvm-ar",
            "7.14.60850",
        )
        runner = CompilerFixture(self.directory)
        source = self.directory / "source"
        for header in builder.packager.PUBLIC_HEADERS:
            file = source / header
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"fixture header")
        builder.build_native(source, runner, sdk, {"patchelf": "/tools/patchelf"})
        compile_commands = [command for command in runner.commands if "-c" in command]
        self.assertEqual(
            {
                Path(command[command.index("-c") + 1]).name
                for command in compile_commands
            },
            {
                "registered_allreduce_relay_kernels.cu",
                "hipipc_relay.cc",
                "hipipc_probe.cu",
            },
        )
        smoke_command = next(
            command
            for command in runner.commands
            if str(self.directory / "public-header-smoke.cc") in command
        )
        for command in [*compile_commands, smoke_command]:
            self.assertIn("-D_GLIBCXX_USE_CXX11_ABI=1", command)
            self.assertIn("--offload-arch=gfx950", command)
        archive_command = next(
            command for command in runner.commands if command[0] == str(sdk.ar)
        )
        self.assertIn("rcs", archive_command)
        link_command = next(
            command for command in runner.commands if "-shared" in command
        )
        self.assertIn("-lamdhip64", link_command)
        path_command = next(
            command for command in runner.commands if command[0] == "/tools/patchelf"
        )
        self.assertIn("--remove-rpath", path_command)
        smoke = (self.directory / "public-header-smoke.cc").read_text()
        self.assertIn("rcclx::relay::hipipc::kAbiVersion == 1", smoke)
        self.assertIn("decltype(hipipc_relay_abi_version()), uint32_t", smoke)
        self.assertEqual(len(builder.NATIVE_INPUTS), 8)
        record = builder.HIPIPC + "hipipc_record.h"
        self.assertIn(record, builder.NATIVE_INPUTS)
        self.assertEqual(
            (self.directory / "bundle/include" / record).read_bytes(), b"fixture header"
        )
        self.assertFalse(any("graph_generation" in n for n in builder.NATIVE_INPUTS))

    def test_legacy_cache_missing_download_does_not_reuse_sdk(self) -> None:
        legacy = self.cache / "sdk" / self.lock_hash
        (legacy / "venv").mkdir(parents=True)
        runner = CompilerFixture(self.directory)
        with self.assertRaises(builder.BuildError) as failure:
            builder.prepare_sdk(
                self.cache, runner, self.artifacts, self.lock_hash, True
            )
        self.assertIn(str(legacy), str(failure.exception))
        self.assertEqual(runner.commands, [])

    def test_missing_host_prerequisite(self) -> None:
        with (
            patch.object(builder.sys, "version_info", (3, 12)),
            patch.object(builder.platform, "system", return_value="Linux"),
            patch.object(builder.platform, "machine", return_value="x86_64"),
            patch.object(builder.shutil, "which", return_value=None),
            self.assertRaisesRegex(builder.BuildError, "Missing host prerequisite: sl"),
        ):
            builder.prerequisites()

    def test_dirty_source_inputs_rejected(self) -> None:
        runner = CompilerFixture(self.directory)
        for status in ("M fbcode/hipipc_relay.cc\n", "? fbcode/packaging/new.py\n"):
            with (
                patch.object(runner, "run", return_value=status),
                self.assertRaisesRegex(builder.BuildError, "Commit native"),
            ):
                builder.require_clean_inputs(self.directory, runner, "/tools/sl")
        builder.require_clean_inputs(self.directory, runner, "/tools/sl")
        self.assertEqual(
            runner.commands[-1][-1], "fbcode/" + builder.HIPIPC + "packaging"
        )

    def test_environment_and_cli(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CXXFLAGS": "poison",
                "HIPCC_COMPILE_FLAGS_APPEND": "poison",
                "PYTHONPATH": "poison",
            },
        ):
            env = builder.child_environment({})
        self.assertFalse(
            {"CXXFLAGS", "HIPCC_COMPILE_FLAGS_APPEND", "PYTHONPATH"} & env.keys()
        )
        self.assertEqual(env["ROCM_SDK_TARGET_FAMILY"], "gfx950")
        self.assertEqual(builder.parse_args([]).base_version, "0.1.0")
        self.assertEqual(
            builder.fresh_output(self.directory / "fresh"), self.directory / "fresh"
        )
