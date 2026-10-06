# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Exact build-identity checks for the TorchComms MCCL plugin."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import re
import sysconfig
from pathlib import Path
from typing import Any, Callable, cast


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SHARED_BUILD_FIELDS = (
    "build_type",
    "compiler",
    "cuda_toolkit",
    "python_soabi",
    "python_version",
    "pytorch_cxx11_abi",
    "pytorch_version",
    "torch_cuda_arch_list",
    "torchcomms_deps_prefix_digest",
    "torchcomms_revision",
    "torchcomms_source_tree_sha256",
)


class BuildIdentityError(RuntimeError):
    """Raised when installed artifacts do not form one qualified build."""


def require_c10d_torchcomms_factory() -> Callable[..., Any]:
    """Return the c10d factory after checking the API required by MCCL."""
    import torch.distributed.distributed_c10d as c10d

    factory = getattr(c10d, "_create_torchcomms_backend", None)
    if not callable(factory):
        raise BuildIdentityError(
            "torchcomms-mccl requires a PyTorch build with c10d TorchComms "
            "backend support"
        )

    parameters = inspect.signature(factory).parameters
    has_keyword_sink = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    required = {"timeout", "enable_reconfigure"}
    missing = sorted(required.difference(parameters))
    if missing and not has_keyword_sink:
        raise BuildIdentityError(
            "torchcomms-mccl requires a PyTorch c10d TorchComms factory with "
            + ", ".join(missing)
        )
    return cast(Callable[..., Any], factory)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise BuildIdentityError(
            f"Failed to read build identity {path}: {error}"
        ) from error
    return digest.hexdigest()


def read_build_info(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BuildIdentityError(
            f"Failed to load {label} build identity: {error}"
        ) from error
    if not isinstance(value, dict):
        raise BuildIdentityError(f"{label} build identity must be a JSON object")
    return cast(dict[str, object], value)


def distribution_file(distribution: str, relative_path: str) -> Path:
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise BuildIdentityError(
            f"Required distribution is not installed: {distribution}"
        ) from error
    path = Path(installed.locate_file(relative_path))
    if not path.is_file():
        raise BuildIdentityError(
            f"{distribution} does not contain required build identity {relative_path}"
        )
    return path


def _required(record: dict[str, object], field: str, label: str) -> object:
    if field not in record:
        raise BuildIdentityError(f"{label} build identity is missing {field}")
    return record[field]


def _required_string(record: dict[str, object], field: str, label: str) -> str:
    value = _required(record, field, label)
    if not isinstance(value, str) or not value:
        raise BuildIdentityError(
            f"{label} build identity field {field} must be a non-empty string"
        )
    return value


def _required_bool(record: dict[str, object], field: str, label: str) -> bool:
    value = _required(record, field, label)
    if not isinstance(value, bool):
        raise BuildIdentityError(
            f"{label} build identity field {field} must be a boolean"
        )
    return value


def require_sha256(value: str, label: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise BuildIdentityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_core_and_mccl(  # noqa: C901
    core: dict[str, object], mccl: dict[str, object], core_digest: str
) -> None:
    if _required(core, "schema_version", "torchcomms") != 1:
        raise BuildIdentityError("Unsupported torchcomms build identity schema")
    if _required(mccl, "schema_version", "mccl") != 1:
        raise BuildIdentityError("Unsupported mccl build identity schema")

    for field in _SHARED_BUILD_FIELDS:
        core_value = _required(core, field, "torchcomms")
        mccl_value = _required(mccl, field, "mccl")
        if core_value != mccl_value:
            raise BuildIdentityError(
                f"torchcomms and mccl build identities disagree on {field}"
            )
    _required_bool(core, "pytorch_cxx11_abi", "torchcomms")
    _required_bool(mccl, "pytorch_cxx11_abi", "mccl")

    if _required(core, "use_ncclx", "torchcomms") is not True:
        raise BuildIdentityError("The qualified torchcomms build must enable NCCLX")
    if _required(core, "torchcomms_bundle_observatory", "torchcomms") is not True:
        raise BuildIdentityError(
            "The qualified torchcomms build must own the Observatory library"
        )
    enabled_backends = _required(core, "enabled_backends", "torchcomms")
    if not isinstance(enabled_backends, list) or "ncclx" not in enabled_backends:
        raise BuildIdentityError("The qualified torchcomms build must include NCCLX")
    if _required(mccl, "prims_enabled", "mccl") is not True:
        raise BuildIdentityError("The qualified mccl build must enable PRIMS")
    if _required_string(mccl, "ctran_linkage", "mccl") != "folded-into-libmccl":
        raise BuildIdentityError(
            "The qualified mccl build must fold CTran into libmccl"
        )
    if _required_string(mccl, "observatory_owner", "mccl") != "torchcomms":
        raise BuildIdentityError(
            "The qualified mccl build must use TorchComms Observatory"
        )
    if (
        _required_string(mccl, "torchcomms_core_build_info_digest", "mccl")
        != core_digest
    ):
        raise BuildIdentityError(
            "mccl was not built against the installed torchcomms core"
        )
    if _required_string(mccl, "torchcomms_version", "mccl") != _required_string(
        core, "package_version", "torchcomms"
    ):
        raise BuildIdentityError("mccl and torchcomms package versions do not match")
    if _required_string(mccl, "ncclx_identity", "mccl") != _required_string(
        core, "ncclx_identity", "torchcomms"
    ):
        raise BuildIdentityError("mccl and torchcomms NCCLX identities do not match")


def create_companion_build_info(
    *,
    package_version: str,
    torchcomms_source_revision: str,
    torchcomms_source_tree_sha256: str,
    mccl_source_revision: str,
    torchcomms_core_wheel_sha256: str,
    mccl_wheel_sha256: str,
    runtime_closure_decision_sha256: str | None = None,
) -> dict[str, object]:
    core_path = distribution_file("torchcomms", "torchcomms/_build_info.json")
    mccl_path = distribution_file("mccl", "mccl/_build_info.json")
    core = read_build_info(core_path, "torchcomms")
    mccl = read_build_info(mccl_path, "mccl")
    core_digest = sha256_file(core_path)
    mccl_digest = sha256_file(mccl_path)
    validate_core_and_mccl(core, mccl, core_digest)

    require_sha256(torchcomms_source_tree_sha256, "TorchComms source-tree digest")
    require_sha256(torchcomms_core_wheel_sha256, "torchcomms wheel digest")
    require_sha256(mccl_wheel_sha256, "mccl wheel digest")
    if runtime_closure_decision_sha256 is not None:
        require_sha256(
            runtime_closure_decision_sha256, "runtime-closure decision digest"
        )

    expected_values = {
        "torchcomms_revision": torchcomms_source_revision,
        "torchcomms_source_tree_sha256": torchcomms_source_tree_sha256,
    }
    for field, expected in expected_values.items():
        if _required_string(core, field, "torchcomms") != expected:
            raise BuildIdentityError(
                f"Installed torchcomms does not match requested {field}"
            )
    if _required_string(mccl, "mccl_revision", "mccl") != mccl_source_revision:
        raise BuildIdentityError("Installed mccl does not match requested revision")
    if (
        _required_string(mccl, "torchcomms_core_wheel_sha256", "mccl")
        != torchcomms_core_wheel_sha256
    ):
        raise BuildIdentityError("mccl records a different torchcomms wheel digest")

    versions = {
        "torch_version": importlib.metadata.version("torch"),
        "torchcomms_version": importlib.metadata.version("torchcomms"),
        "mccl_version": importlib.metadata.version("mccl"),
    }
    if versions["torch_version"] != _required_string(
        core, "pytorch_version", "torchcomms"
    ):
        raise BuildIdentityError(
            "Installed torch does not match torchcomms build identity"
        )
    if versions["torchcomms_version"] != _required_string(
        core, "package_version", "torchcomms"
    ):
        raise BuildIdentityError(
            "Installed torchcomms version does not match its build identity"
        )
    if versions["mccl_version"] != _required_string(mccl, "package_version", "mccl"):
        raise BuildIdentityError(
            "Installed mccl version does not match its build identity"
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "package_version": package_version,
        **versions,
        "python_soabi": _required_string(core, "python_soabi", "torchcomms"),
        "pytorch_cxx11_abi": _required_bool(core, "pytorch_cxx11_abi", "torchcomms"),
        "cuda_toolkit": _required_string(core, "cuda_toolkit", "torchcomms"),
        "torch_cuda_arch_list": _required_string(
            core, "torch_cuda_arch_list", "torchcomms"
        ),
        "torchcomms_revision": torchcomms_source_revision,
        "torchcomms_source_tree_sha256": torchcomms_source_tree_sha256,
        "torchcomms_core_build_info_digest": core_digest,
        "torchcomms_core_wheel_sha256": torchcomms_core_wheel_sha256,
        "mccl_revision": mccl_source_revision,
        "mccl_source_manifest_digest": _required_string(
            mccl, "mccl_source_manifest_digest", "mccl"
        ),
        "mccl_build_info_digest": mccl_digest,
        "mccl_wheel_sha256": mccl_wheel_sha256,
        "torchcomms_deps_prefix_digest": _required_string(
            core, "torchcomms_deps_prefix_digest", "torchcomms"
        ),
        "doca_identity": _required_string(mccl, "doca_identity", "mccl"),
        "prims_enabled": True,
        "use_ncclx": True,
    }
    if runtime_closure_decision_sha256 is not None:
        result["runtime_closure_decision_sha256"] = runtime_closure_decision_sha256
    return result


def validate_installed_build_info(companion_path: Path) -> None:
    companion = read_build_info(companion_path, "torchcomms-mccl")
    if _required(companion, "schema_version", "torchcomms-mccl") != 1:
        raise BuildIdentityError("Unsupported torchcomms-mccl build identity schema")

    core_path = distribution_file("torchcomms", "torchcomms/_build_info.json")
    mccl_path = distribution_file("mccl", "mccl/_build_info.json")
    core = read_build_info(core_path, "torchcomms")
    mccl = read_build_info(mccl_path, "mccl")
    core_digest = sha256_file(core_path)
    mccl_digest = sha256_file(mccl_path)
    validate_core_and_mccl(core, mccl, core_digest)

    expected_digests = {
        "torchcomms_core_build_info_digest": core_digest,
        "mccl_build_info_digest": mccl_digest,
    }
    for field, actual in expected_digests.items():
        if _required_string(companion, field, "torchcomms-mccl") != actual:
            raise BuildIdentityError(
                f"Installed artifacts do not match companion {field}"
            )

    version_fields = {
        "torch_version": "torch",
        "torchcomms_version": "torchcomms",
        "mccl_version": "mccl",
    }
    for field, distribution in version_fields.items():
        if _required_string(
            companion, field, "torchcomms-mccl"
        ) != importlib.metadata.version(distribution):
            raise BuildIdentityError(
                f"Installed {distribution} version does not match torchcomms-mccl"
            )

    if _required_string(companion, "python_soabi", "torchcomms-mccl") != str(
        sysconfig.get_config_var("SOABI")
    ):
        raise BuildIdentityError("Python ABI does not match torchcomms-mccl")

    import torch

    if torch.__version__ != _required_string(
        companion, "torch_version", "torchcomms-mccl"
    ):
        raise BuildIdentityError("Loaded torch does not match torchcomms-mccl")
    if bool(torch._C._GLIBCXX_USE_CXX11_ABI) != _required_bool(
        companion, "pytorch_cxx11_abi", "torchcomms-mccl"
    ):
        raise BuildIdentityError("PyTorch C++ ABI does not match torchcomms-mccl")
