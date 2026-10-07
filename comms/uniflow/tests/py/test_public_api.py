# Copyright (c) Meta Platforms, Inc. and affiliates.

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
            target.id
            for statement in err_code.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance((target := statement.target), ast.Name)
        }
        self.assertEqual(stub_members, set(_core.ErrCode.__members__))

    def test_cmake_prefix_is_packaged(self) -> None:
        prefix = Path(uniflow.cmake_prefix_path)
        configs = list(prefix.glob("lib*/cmake/uniflow/uniflowConfig.cmake"))
        if not configs:
            self.skipTest("C++ SDK is not installed")
        self.assertEqual(len(configs), 1)
