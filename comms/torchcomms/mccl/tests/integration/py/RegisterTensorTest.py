#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from comms.torchcomms.tests.integration.py.RegisterTensorTest import (
    RegisterTensorTest as BaseRegisterTensorTest,
)


class RegisterTensorTest(BaseRegisterTensorTest):
    pass


del BaseRegisterTensorTest

if __name__ == "__main__":
    unittest.main()
