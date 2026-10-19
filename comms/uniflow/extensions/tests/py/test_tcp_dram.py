# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from comms.uniflow.tests.py.transfer_test_utils import MultiTransportTransferTest
from uniflow._core import TransportType


class TcpDramIntegrationTest(MultiTransportTransferTest):
    def test_tcp_dram_put_get(self) -> None:
        self._put_get(TransportType.TCP, gpu_buffers=False)
