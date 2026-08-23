# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Host-only unit tests for the NCU wrapper boilerplate (no GPU / ncu binary needed)."""

import os
import tempfile
import unittest
from unittest import mock

from comms.dsl.tests.ncu_common import (
    build_driver_shim,
    NCU_DEFAULT_METRICS,
    ncu_wrap_argv,
    resolve_ncu_bin,
)


class NcuWrapArgvTest(unittest.TestCase):
    def test_wraps_program_with_ncu_flags(self) -> None:
        argv = ncu_wrap_argv(
            ["mybin", "--x"],
            ncu_bin="/usr/bin/ncu",
            out_prefix="/tmp/o",
            metrics="m1,m2",
            launch_count=3,
        )
        self.assertEqual(argv[0], "/usr/bin/ncu")
        self.assertEqual(argv[-2:], ["mybin", "--x"])  # program appended last
        self.assertIn("all", argv[argv.index("--target-processes") + 1 :])
        self.assertEqual(argv[argv.index("--launch-count") + 1], "3")
        self.assertEqual(argv[argv.index("--metrics") + 1], "m1,m2")
        self.assertEqual(argv[argv.index("-o") + 1], "/tmp/o_%p")  # %p per-process

    def test_default_metrics_are_single_pass_launch(self) -> None:
        # Default metrics must be launch__ (no-replay) so the comm kernel does not deadlock.
        for m in NCU_DEFAULT_METRICS.split(","):
            self.assertTrue(m.startswith("launch__"), m)

    def test_kernel_regex_added_when_given(self) -> None:
        argv = ncu_wrap_argv(
            ["b"], ncu_bin="/ncu", out_prefix="/o", kernel_regex="_a2a_kernel"
        )
        self.assertIn("-k", argv)
        self.assertEqual(argv[argv.index("-k") + 1], "_a2a_kernel")

    def test_kernel_regex_omitted_by_default(self) -> None:
        self.assertNotIn("-k", ncu_wrap_argv(["b"], ncu_bin="/ncu", out_prefix="/o"))


class ResolveNcuBinTest(unittest.TestCase):
    def test_explicit_wins(self) -> None:
        self.assertEqual(resolve_ncu_bin("/custom/ncu"), "/custom/ncu")

    @mock.patch("comms.dsl.tests.ncu_common.os.path.exists", return_value=False)
    @mock.patch("comms.dsl.tests.ncu_common.shutil.which", return_value=None)
    def test_raises_when_absent(
        self, _which: mock.MagicMock, _exists: mock.MagicMock
    ) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_ncu_bin()

    @mock.patch("comms.dsl.tests.ncu_common.shutil.which", return_value="/path/ncu")
    def test_path_lookup(self, _which: mock.MagicMock) -> None:
        self.assertEqual(resolve_ncu_bin(), "/path/ncu")


class BuildDriverShimTest(unittest.TestCase):
    def test_absent_platform_dir_returns_empty(self) -> None:
        # Local dev: no platform driver dir -> no shim, driver already resolvable.
        self.assertEqual(build_driver_shim("/no/such/dir", shim_dir="/tmp/nope"), {})

    def test_present_dir_symlinks_libs_and_sets_env(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            for lib in ("libcuda.so", "libcuda.so.1", "libnvidia-ml.so"):
                open(os.path.join(src, lib), "w").close()
            shim = os.path.join(base, "cuda_driver_ncu")
            env = build_driver_shim(src, shim_dir=shim)
            self.assertTrue(env["LD_LIBRARY_PATH"].startswith(shim))
            self.assertIn(f"{shim}/libcuda.so", env["LD_PRELOAD"])
            self.assertEqual(env["TRITON_LIBCUDA_PATH"], f"{shim}/libcuda.so")
            self.assertTrue(os.path.islink(os.path.join(shim, "libcuda.so")))
            self.assertTrue(os.path.islink(os.path.join(shim, "libnvidia-ml.so")))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_unset_library_path_has_no_empty_component(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            open(os.path.join(src, "libcuda.so"), "w").close()
            shim = os.path.join(base, "cuda_driver_ncu")
            env = build_driver_shim(src, shim_dir=shim)
            self.assertEqual(shim, env["LD_LIBRARY_PATH"])

    @mock.patch.dict(os.environ, {"LD_PRELOAD": "/existing/injector.so"}, clear=True)
    def test_existing_preload_is_preserved(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            open(os.path.join(src, "libcuda.so"), "w").close()
            shim = os.path.join(base, "cuda_driver_ncu")
            env = build_driver_shim(src, shim_dir=shim)
            self.assertEqual(
                f"{shim}/libcuda.so:/existing/injector.so", env["LD_PRELOAD"]
            )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_optional_nvml_is_omitted(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            open(os.path.join(src, "libcuda.so"), "w").close()
            shim = os.path.join(base, "cuda_driver_ncu")
            env = build_driver_shim(src, shim_dir=shim)
            self.assertEqual(f"{shim}/libcuda.so", env["LD_PRELOAD"])

    def test_dangling_symlink_is_repaired(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            source = os.path.join(src, "libcuda.so")
            open(source, "w").close()
            shim = os.path.join(base, "cuda_driver_ncu")
            os.makedirs(shim)
            link = os.path.join(shim, "libcuda.so")
            os.symlink("/missing/libcuda.so", link)
            env = build_driver_shim(src, shim_dir=shim)
            self.assertEqual(source, os.readlink(link))
            self.assertEqual(link, env["TRITON_LIBCUDA_PATH"])

    def test_missing_libcuda_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            open(os.path.join(src, "libnvidia-ml.so"), "w").close()
            with self.assertRaisesRegex(FileNotFoundError, "libcuda"):
                build_driver_shim(src, shim_dir=os.path.join(base, "cuda_driver_ncu"))

    def test_versioned_only_lib_falls_back_in_preload(self) -> None:
        # Some driver installs ship only libcuda.so.1 (no unversioned name); LD_PRELOAD must
        # point at the versioned lib that exists, not a dangling unversioned path.
        with (
            tempfile.TemporaryDirectory() as src,
            tempfile.TemporaryDirectory() as base,
        ):
            for lib in ("libcuda.so.1", "libnvidia-ml.so.1"):
                open(os.path.join(src, lib), "w").close()
            shim = os.path.join(base, "cuda_driver_ncu")
            env = build_driver_shim(src, shim_dir=shim)
            self.assertIn(f"{shim}/libcuda.so.1", env["LD_PRELOAD"])
            self.assertEqual(env["TRITON_LIBCUDA_PATH"], f"{shim}/libcuda.so.1")
