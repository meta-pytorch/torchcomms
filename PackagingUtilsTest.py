# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from comms.github import packaging_utils


class PackagingUtilsTest(unittest.TestCase):
    def test_core_dynamic_path_policy(self) -> None:
        for value in ("$ORIGIN", "${ORIGIN}/../torch/lib"):
            packaging_utils.validate_core_dynamic_search_paths(
                f"0x0 (RUNPATH) Library runpath: [{value}]", strict=False
            )
        for value in ("/tmp/build", "$ORIGIN]:/tmp/build", ""):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "unsafe dynamic search paths"),
            ):
                packaging_utils.validate_core_dynamic_search_paths(
                    f"0x0 (RUNPATH) Library runpath: [{value}]", strict=False
                )

    def test_core_dynamic_path_policy_rejects_malformed_or_ambiguous_tags(
        self,
    ) -> None:
        invalid = (
            "0x0 (RUNPATH) Library runpath: [$ORIGIN\n/tmp/build]",
            "0x0 (RPATH) Library rpath: [$ORIGIN]\n"
            "0x1 (RPATH) Library rpath: [$ORIGIN/lib]",
            "0x0 (RPATH) Library rpath: [$ORIGIN]\n"
            "0x1 (RUNPATH) Library runpath: [$ORIGIN]",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                packaging_utils.validate_core_dynamic_search_paths(value, strict=False)

    def test_strict_core_policy_rejects_any_dynamic_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "has a dynamic search path"):
            packaging_utils.validate_core_dynamic_search_paths(
                "0x0 (RPATH) Library rpath: [$ORIGIN]", strict=True
            )
