#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

# pyre-fixme[5]: Global annotation for skip decorator.
skip_if_ncclx = unittest.skipIf(
    os.getenv("TEST_BACKEND") == "ncclx", "Skipping tests for NCCLX backend."
)
