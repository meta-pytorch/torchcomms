# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torchcomms.mccl import _identity, _registration


_DIGEST = "1" * 64


class StringOnlyPath:
    def __init__(self, path: str) -> None:
        self.path = path

    def __str__(self) -> str:
        return self.path


def core_build_info() -> dict[str, object]:
    return {
        "schema_version": 1,
        "build_type": "Release",
        "compiler": "compiler",
        "cuda_toolkit": "cuda",
        "enabled_backends": ["ncclx"],
        "ncclx_identity": "ncclx",
        "package_version": "1.0",
        "python_soabi": "soabi",
        "python_version": "3.12",
        "pytorch_cxx11_abi": True,
        "pytorch_version": "2.0",
        "torch_cuda_arch_list": "9.0a",
        "torchcomms_bundle_observatory": True,
        "torchcomms_deps_prefix_digest": _DIGEST,
        "torchcomms_revision": "revision",
        "torchcomms_source_tree_sha256": _DIGEST,
        "use_ncclx": True,
    }


def mccl_build_info() -> dict[str, object]:
    return {
        "schema_version": 1,
        "build_type": "Release",
        "compiler": "compiler",
        "ctran_linkage": "folded-into-libmccl",
        "cuda_toolkit": "cuda",
        "ncclx_identity": "ncclx",
        "observatory_owner": "torchcomms",
        "prims_enabled": True,
        "python_soabi": "soabi",
        "python_version": "3.12",
        "pytorch_cxx11_abi": True,
        "pytorch_version": "2.0",
        "torch_cuda_arch_list": "9.0a",
        "torchcomms_core_build_info_digest": _DIGEST,
        "torchcomms_deps_prefix_digest": _DIGEST,
        "torchcomms_revision": "revision",
        "torchcomms_source_tree_sha256": _DIGEST,
        "torchcomms_version": "1.0",
    }


class BuildIdentityTest(unittest.TestCase):
    def test_distribution_file_accepts_importlib_simple_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "_build_info.json"
            path.write_text("{}\n")
            installed = MagicMock()
            installed.locate_file.return_value = StringOnlyPath(str(path))
            with patch.object(
                _identity.importlib.metadata,
                "distribution",
                return_value=installed,
            ):
                self.assertEqual(
                    _identity.distribution_file(
                        "torchcomms", "torchcomms/_build_info.json"
                    ),
                    path,
                )

    def test_c10d_registrar_forwards_dynamic_options(self) -> None:
        process_group = MagicMock()
        process_group.bound_device_id = torch.device("cuda:3")
        opts = MagicMock()
        opts.process_group = process_group
        opts.group_rank = 1
        opts.group_size = 4
        opts.group_id = "test_mccl_registration"
        opts.store = dist.HashStore()
        opts.timeout = timedelta(seconds=45)
        opts.enable_reconfigure = True
        backend_options = object()
        wrapped_backend = object()
        factory = MagicMock(return_value=wrapped_backend)

        with (
            patch.object(
                _registration,
                "require_c10d_torchcomms_factory",
                return_value=factory,
            ),
            patch.object(dist.Backend, "register_backend") as register_backend,
        ):
            load_backend = MagicMock()
            _registration.register_c10d_backend(load_backend)
            creator = register_backend.call_args.args[1]
            self.assertIs(creator(opts, backend_options), wrapped_backend)

        load_backend.assert_called_once_with()
        register_backend.assert_called_once_with(
            "mccl",
            creator,
            extended_api=True,
            devices=["cuda"],
        )
        factory.assert_called_once_with(
            "mccl",
            "cuda",
            group_rank=1,
            group_size=4,
            group_name="test_mccl_registration",
            store=opts.store,
            device_id=torch.device("cuda:3"),
            backend_options=backend_options,
            timeout=opts.timeout,
            enable_reconfigure=opts.enable_reconfigure,
        )

    def test_supported_c10d_factory_passes(self) -> None:
        def factory(*, timeout=None, enable_reconfigure=False):
            return timeout, enable_reconfigure

        with patch.object(c10d, "_create_torchcomms_backend", factory, create=True):
            self.assertIs(_identity.require_c10d_torchcomms_factory(), factory)

    def test_missing_c10d_factory_fails(self) -> None:
        with patch.object(c10d, "_create_torchcomms_backend", None, create=True):
            with self.assertRaisesRegex(
                _identity.BuildIdentityError,
                "c10d TorchComms backend support",
            ):
                _identity.require_c10d_torchcomms_factory()

    def test_dynamic_c10d_options_are_required(self) -> None:
        def factory(*, timeout=None):
            return timeout

        with patch.object(c10d, "_create_torchcomms_backend", factory, create=True):
            with self.assertRaisesRegex(
                _identity.BuildIdentityError,
                "enable_reconfigure",
            ):
                _identity.require_c10d_torchcomms_factory()

    def test_matching_records_pass(self) -> None:
        _identity.validate_core_and_mccl(core_build_info(), mccl_build_info(), _DIGEST)

    def test_source_identity_mismatch_fails(self) -> None:
        mccl = copy.deepcopy(mccl_build_info())
        mccl["torchcomms_source_tree_sha256"] = "2" * 64
        with self.assertRaisesRegex(
            _identity.BuildIdentityError,
            "torchcomms_source_tree_sha256",
        ):
            _identity.validate_core_and_mccl(core_build_info(), mccl, _DIGEST)

    def test_pytorch_abi_requires_json_boolean(self) -> None:
        for invalid in (1, "ON"):
            with self.subTest(invalid=invalid):
                core = copy.deepcopy(core_build_info())
                mccl = copy.deepcopy(mccl_build_info())
                core["pytorch_cxx11_abi"] = invalid
                mccl["pytorch_cxx11_abi"] = invalid
                with self.assertRaisesRegex(
                    _identity.BuildIdentityError,
                    "pytorch_cxx11_abi must be a boolean",
                ):
                    _identity.validate_core_and_mccl(core, mccl, _DIGEST)

    def test_prims_is_required(self) -> None:
        mccl = copy.deepcopy(mccl_build_info())
        mccl["prims_enabled"] = False
        with self.assertRaisesRegex(_identity.BuildIdentityError, "enable PRIMS"):
            _identity.validate_core_and_mccl(core_build_info(), mccl, _DIGEST)

    def test_ncclx_is_required(self) -> None:
        core = copy.deepcopy(core_build_info())
        core["use_ncclx"] = False
        with self.assertRaisesRegex(_identity.BuildIdentityError, "enable NCCLX"):
            _identity.validate_core_and_mccl(core, mccl_build_info(), _DIGEST)

    def test_digest_format_is_strict(self) -> None:
        self.assertEqual(
            _identity.require_sha256(_DIGEST, "test digest"),
            _DIGEST,
        )
        with self.assertRaisesRegex(_identity.BuildIdentityError, "lowercase SHA-256"):
            _identity.require_sha256("ABC", "test digest")
