# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
from uniflow._core import (
    ErrCode,
    MultiTransport,
    MultiTransportFactory,
    RegisteredSegment,
    RemoteRegisteredSegment,
    Segment,
    TransferRequest,
    TransportType,
)


class MultiTransportTransferTest(unittest.TestCase):
    def _put_get(self, backend: TransportType, gpu_buffers: bool) -> None:
        expect_extensions = os.environ.get("UNIFLOW_EXPECT_EXTENSIONS") == "1"
        require_rdma = (
            torch.version.hip is None or os.environ.get("UNIFLOW_REQUIRE_RDMA") == "1"
        )
        supported = MultiTransportFactory.supported(backend)
        if not supported:
            if (
                backend == TransportType.RDMA
                and supported.error().code == ErrCode.ResourceExhausted
                and not require_rdma
            ):
                self.skipTest(str(supported.error()))
            if backend == TransportType.RDMA or expect_extensions:
                self.fail(str(supported.error()))
            self.skipTest(str(supported.error()))
        if gpu_buffers:
            self.assertTrue(torch.cuda.is_available(), "Requires GPU buffers")
            self.assertGreaterEqual(torch.cuda.device_count(), 2)

        p2p = backend == TransportType.NVLink
        factories = [
            MultiTransportFactory(
                device_id=device_id,
                nic_filter=""
                if backend == TransportType.RDMA
                else "=uniflow_test_no_rdma",
                enable_tcp=backend == TransportType.TCP,
                tcp_bind_host="" if p2p else "127.0.0.1",
            )
            for device_id in ((0, 1) if p2p else (-1, -1))
        ]
        source = torch.arange(
            16384, dtype=torch.int32, device="cuda:0" if gpu_buffers else "cpu"
        )
        target = torch.zeros_like(source, device="cuda:1" if gpu_buffers else "cpu")
        returned = torch.zeros_like(source)
        if gpu_buffers:
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)

        segments = [
            Segment.from_tensor(tensor) for tensor in (source, target, returned)
        ]
        registrations: list[RegisteredSegment] = [
            factory.register_segment(segment).unwrap()
            for factory, segment in zip(
                (factories[0], factories[1], factories[0]), segments
            )
        ]
        remote: RemoteRegisteredSegment = (
            factories[0].import_segment(registrations[1].export_id().unwrap()).unwrap()
        )
        transports: list[MultiTransport] = [
            factories[0].create_transport(factories[1].get_topology()).unwrap(),
            factories[1].create_transport(factories[0].get_topology()).unwrap(),
        ]
        try:
            bind_info = [transport.bind().unwrap() for transport in transports]
            with ThreadPoolExecutor(max_workers=2) as executor:
                connections = [
                    executor.submit(transport.connect, bind_info[1 - index])
                    for index, transport in enumerate(transports)
                ]
                for connection in connections:
                    connection.result(timeout=30).unwrap()

            offset = 64
            length = source.nbytes - 2 * offset
            expected = torch.zeros_like(source, device="cpu")
            edge = offset // source.element_size()
            expected[edge:-edge] = source.cpu()[edge:-edge]
            for operation, registration, destination in (
                (transports[0].put, registrations[0], target),
                (transports[0].get, registrations[2], returned),
            ):
                future = operation(
                    [
                        TransferRequest(
                            registration.span(offset, length),
                            remote.span(offset, length),
                        )
                    ]
                )
                self.assertTrue(future.wait_for(10000), "Transfer timed out")
                future.get().unwrap()
                if gpu_buffers:
                    torch.cuda.synchronize(0)
                    torch.cuda.synchronize(1)
                torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)

            counts = {
                transport.name: transports[0].transfer_count(transport)
                for transport in (
                    TransportType.NVLink,
                    TransportType.RDMA,
                    TransportType.TCP,
                )
            }
            self.assertEqual(counts[backend.name], 2)
            self.assertEqual(sum(counts.values()), 2)
            print(
                f"{backend.name} put/get verified: {counts}, {length} bytes", flush=True
            )
        finally:
            for transport in transports:
                transport.shutdown()
