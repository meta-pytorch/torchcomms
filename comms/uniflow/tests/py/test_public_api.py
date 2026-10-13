# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import uniflow
from uniflow import _core


class PublicApiTest(unittest.TestCase):
    @staticmethod
    def _stub_classes() -> dict[str, ast.ClassDef]:
        stub_path = Path(uniflow.__file__).with_name("_core.pyi")
        stub = ast.parse(stub_path.read_text())
        return {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}

    def test_top_level_exports_match_extension_and_stub(self) -> None:
        runtime_exports = {name for name in dir(_core) if not name.startswith("_")}
        top_level_exports = set(uniflow.__all__)
        stub_exports = set(self._stub_classes())

        self.assertEqual(top_level_exports, runtime_exports)
        self.assertEqual(top_level_exports, stub_exports)

    def test_public_class_members_match_stub(self) -> None:
        for name, stub_class in self._stub_classes().items():
            runtime_members = {
                member
                for member in dir(getattr(_core, name))
                if not member.startswith("_")
            }
            stub_members = {
                statement.name
                for statement in stub_class.body
                if isinstance(statement, ast.FunctionDef)
                and not statement.name.startswith("_")
            }
            stub_members.update(
                statement.target.id
                for statement in stub_class.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and not statement.target.id.startswith("_")
            )
            self.assertEqual(runtime_members, stub_members, name)

    def test_error_code_members_match_stub(self) -> None:
        err_code = self._stub_classes()["ErrCode"]
        stub_members = {
            statement.target.id
            for statement in err_code.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and not statement.target.id.startswith("_")
        }
        self.assertEqual(stub_members, set(_core.ErrCode.__members__))

    def test_gpu_backend_config_preserves_none(self) -> None:
        for config in (
            uniflow.UniflowAgentConfig(),
            uniflow.UniflowAgentConfig(gpu_backend=None),
        ):
            self.assertIsNone(config.gpu_backend)
            config.gpu_backend = "amd"
            self.assertEqual(config.gpu_backend, "amd")
            config.gpu_backend = None
            self.assertIsNone(config.gpu_backend)

    def test_gpu_backend_is_the_last_config_argument(self) -> None:
        config = uniflow.UniflowAgentConfig(
            -1, "cpu", "", 7, 250, False, "", None, None, None
        )
        self.assertEqual(config.device_id, -1)
        self.assertEqual(config.connect_retries, 7)
        self.assertEqual(config.connect_timeout_ms, 250)
        self.assertIsNone(config.gpu_backend)
        requested = uniflow.UniflowAgentConfig(
            -1, "cpu", "", 7, 250, False, "", None, None, None, "mtia"
        )
        self.assertEqual(requested.gpu_backend, "mtia")

    def test_gpu_backend_config_rejects_invalid_requests(self) -> None:
        config = uniflow.UniflowAgentConfig(gpu_backend="amd")
        for backend in ("", "cuda", "NVIDIA", "unknown"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "Invalid GPU backend"):
                    uniflow.UniflowAgentConfig(gpu_backend=backend)
                with self.assertRaisesRegex(ValueError, "Invalid GPU backend"):
                    config.gpu_backend = backend
                self.assertEqual(config.gpu_backend, "amd")

    def test_factory_rejects_invalid_backend_before_discovery(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid GPU backend"):
            uniflow.MultiTransportFactory(-1, gpu_backend="unknown")

    def test_package_has_type_marker(self) -> None:
        self.assertTrue(Path(uniflow.__file__).with_name("py.typed").is_file())
