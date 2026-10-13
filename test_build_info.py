# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-3 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import torchcomms_build_info


class TorchCommsBuildInfoTest(unittest.TestCase):
    def test_serialization_is_deterministic(self) -> None:
        information = {
            "schema_version": 1,
            "enabled_backends": ["nccl", "ncclx"],
            "torchcomms_revision": "a" * 40,
        }
        first = torchcomms_build_info.serialized_build_information(information)
        second = torchcomms_build_info.serialized_build_information(information)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), information)

    def test_explicit_source_identity_requires_exact_clean_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            with mock.patch.dict(
                os.environ,
                {"TORCHCOMMS_SOURCE_REVISION": "not-a-revision"},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "full lowercase"):
                    torchcomms_build_info.source_identity(root)

    def test_source_identity_records_unknown_dirty_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            (root / ".git").mkdir()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    torchcomms_build_info,
                    "_git_output",
                    return_value="a" * 40,
                ),
                mock.patch.object(
                    torchcomms_build_info,
                    "_git_dirty",
                    return_value=None,
                ),
            ):
                self.assertEqual(
                    torchcomms_build_info.source_identity(root),
                    ("a" * 40, None),
                )

    def test_implicit_source_identity_allows_failed_git_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            (root / ".git").mkdir()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    torchcomms_build_info,
                    "_git_output",
                    return_value=None,
                ),
                mock.patch.object(torchcomms_build_info, "_git_dirty") as dirty,
            ):
                self.assertEqual(
                    torchcomms_build_info.source_identity(root),
                    (None, None),
                )
                dirty.assert_not_called()

    def test_explicit_source_identity_rejects_failed_git_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            (root / ".git").mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {"TORCHCOMMS_SOURCE_REVISION": "a" * 40},
                    clear=True,
                ),
                mock.patch.object(
                    torchcomms_build_info,
                    "_git_output",
                    return_value=None,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Could not determine"):
                    torchcomms_build_info.source_identity(root)

    def test_explicit_source_identity_requires_known_dirty_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            (root / ".git").mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {"TORCHCOMMS_SOURCE_REVISION": "a" * 40},
                    clear=True,
                ),
                mock.patch.object(
                    torchcomms_build_info,
                    "_git_output",
                    return_value="a" * 40,
                ),
                mock.patch.object(
                    torchcomms_build_info,
                    "_git_dirty",
                    return_value=None,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "whether.*clean"):
                    torchcomms_build_info.source_identity(root)

    def test_dependency_prefix_requires_sha256(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TORCHCOMMS_DEPS_PREFIX_DIGEST": "not-a-digest"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "lowercase SHA-256"):
                torchcomms_build_info.dependency_prefix_digest()

    def test_source_tree_requires_sha256(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TORCHCOMMS_SOURCE_TREE_SHA256": "not-a-digest"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "lowercase SHA-256"):
                torchcomms_build_info.source_tree_sha256()

    def test_soname_parser(self) -> None:
        dynamic = (
            "0x000000000000000e (SONAME)             "
            "Library soname: [libobservatory.so.1]"
        )
        self.assertEqual(
            torchcomms_build_info.parse_soname(dynamic),
            "libobservatory.so.1",
        )

    def test_optional_soname_allows_missing_library_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = pathlib.Path(temporary_directory)
            (package_root / "libobservatory.so").symlink_to("libobservatory.so.missing")
            self.assertIsNone(
                torchcomms_build_info.observatory_soname(
                    package_root,
                    required=False,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Could not determine"):
                torchcomms_build_info.observatory_soname(
                    package_root,
                    required=True,
                )

    def test_build_information_records_explicit_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            with mock.patch.dict(
                os.environ,
                {
                    "USE_SYSTEM_LIBS": "1",
                    "TORCHCOMMS_SOURCE_TREE_SHA256": "b" * 64,
                },
                clear=True,
            ):
                information = torchcomms_build_info.build_information(
                    root=root,
                    package_root=root / "package",
                    package_version="0.3.0",
                    pytorch_version="2.12.0+cu131",
                    pytorch_cxx11_abi=True,
                    torchcomms_revision="a" * 40,
                    source_dirty=False,
                    use_ncclx=False,
                    bundle_observatory=True,
                    enabled_backends=["ncclx", "nccl"],
                )
        self.assertEqual(information["schema_version"], 1)
        self.assertEqual(information["torchcomms_revision"], "a" * 40)
        self.assertEqual(information["torchcomms_source_tree_sha256"], "b" * 64)
        self.assertEqual(information["enabled_backends"], ["nccl", "ncclx"])
        self.assertTrue(information["use_system_libs"])


if __name__ == "__main__":
    unittest.main()
