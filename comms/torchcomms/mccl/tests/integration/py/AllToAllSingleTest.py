#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

from comms.torchcomms.tests.integration.py.AllToAllSingleTest import (
    AllToAllSingleTest as BaseAllToAllSingleTest,
)


class AllToAllSingleTest(BaseAllToAllSingleTest):
    def test_all_tests(self) -> None:
        input_tensor = self._create_input_tensor(1, self.dtypes[0])
        output_tensor = self._create_output_tensor(2, self.dtypes[0])

        for async_op in (False, True):
            with self.subTest(async_op=async_op):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^all_to_all_single is not supported by MCCL$",
                ):
                    self.torchcomm.all_to_all_single(
                        output_tensor, input_tensor, async_op
                    )


del BaseAllToAllSingleTest
