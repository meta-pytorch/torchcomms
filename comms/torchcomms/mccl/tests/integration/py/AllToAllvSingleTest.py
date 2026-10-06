#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

from comms.torchcomms.tests.integration.py.AllToAllvSingleTest import (
    AllToAllvSingleTest as BaseAllToAllvSingleTest,
)


class AllToAllvSingleTest(BaseAllToAllvSingleTest):
    def test_all_tests(self) -> None:
        split_sizes = [1] * self.num_ranks
        input_tensor = self._create_input_tensor(split_sizes, self.dtypes[0])
        output_tensor = self._create_output_tensor(split_sizes, self.dtypes[0])

        for async_op in (False, True):
            with self.subTest(async_op=async_op):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^all_to_all_v_single is not supported by MCCL$",
                ):
                    self.torchcomm.all_to_all_v_single(
                        output_tensor,
                        input_tensor,
                        [],
                        [],
                        async_op,
                    )


del BaseAllToAllvSingleTest
