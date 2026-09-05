#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torchcomms


class TestAbortInfo(unittest.TestCase):
    def test_ibrc_proxy_timeout_is_exposed(self) -> None:
        self.assertEqual(
            torchcomms.AbortReason.IBRC_PROXY_TIMEOUT.name,
            "IBRC_PROXY_TIMEOUT",
        )

    def test_value_equality_and_repr(self) -> None:
        info = torchcomms.AbortInfo(
            torchcomms.AbortReason.INTERNAL_ERROR,
            "watchdog's context",
        )

        self.assertEqual(
            info,
            torchcomms.AbortInfo(
                torchcomms.AbortReason.INTERNAL_ERROR,
                "watchdog's context",
            ),
        )
        self.assertNotEqual(
            info,
            torchcomms.AbortInfo(torchcomms.AbortReason.TIMED_OUT, "timeout"),
        )
        self.assertEqual(
            repr(info),
            'AbortInfo(reason=internal_error, context="watchdog\'s context")',
        )
        self.assertEqual(
            hash(info),
            hash(
                torchcomms.AbortInfo(
                    torchcomms.AbortReason.INTERNAL_ERROR,
                    "watchdog's context",
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
