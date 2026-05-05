#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
from torchcomms._transport import (
    MemoryType,
    MultiTransportFactory,
    Segment,
    TransferRequest,
)


class TransportTest(unittest.TestCase):
    def check_multi_gpu(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest(
                f"Test requires at least 2 CUDA devices, found {torch.cuda.device_count()}"
            )

    def setup_transport_pair(self, device0: int = 0, device1: int = 1):
        """Create two MultiTransportFactory instances, exchange topology,
        create transports, bind, and connect. Returns (factory0, factory1,
        transport0, transport1)."""
        factory0 = MultiTransportFactory(device_id=device0)
        factory1 = MultiTransportFactory(device_id=device1)

        topo0 = factory0.get_topology()
        topo1 = factory1.get_topology()

        transport0_result = factory0.create_transport(topo1)
        self.assertTrue(
            transport0_result.has_value(),
            transport0_result.error() if transport0_result.has_error() else "",
        )
        transport0 = transport0_result.value()

        transport1_result = factory1.create_transport(topo0)
        self.assertTrue(
            transport1_result.has_value(),
            transport1_result.error() if transport1_result.has_error() else "",
        )
        transport1 = transport1_result.value()

        bind0_result = transport0.bind()
        self.assertTrue(
            bind0_result.has_value(),
            bind0_result.error() if bind0_result.has_error() else "",
        )
        bind1_result = transport1.bind()
        self.assertTrue(
            bind1_result.has_value(),
            bind1_result.error() if bind1_result.has_error() else "",
        )

        connect0_result = transport0.connect(bind1_result.value())
        self.assertTrue(
            connect0_result.has_value(),
            connect0_result.error() if connect0_result.has_error() else "",
        )
        connect1_result = transport1.connect(bind0_result.value())
        self.assertTrue(
            connect1_result.has_value(),
            connect1_result.error() if connect1_result.has_error() else "",
        )

        return factory0, factory1, transport0, transport1

    def test_construct_factory(self) -> None:
        _ = MultiTransportFactory(device_id=0)

    def test_topology_exchange(self) -> None:
        self.check_multi_gpu()
        factory0 = MultiTransportFactory(device_id=0)
        factory1 = MultiTransportFactory(device_id=1)

        topo0 = factory0.get_topology()
        topo1 = factory1.get_topology()

        self.assertIsInstance(topo0, bytes)
        self.assertIsInstance(topo1, bytes)
        self.assertGreater(len(topo0), 0)
        self.assertGreater(len(topo1), 0)

    def test_create_transport_and_connect(self) -> None:
        self.check_multi_gpu()
        factory0, factory1, transport0, transport1 = self.setup_transport_pair()
        transport0.shutdown()
        transport1.shutdown()

    def run_put_transfer(self, device0: str, device1: str) -> None:
        """Test put: write data from device0 to device1."""
        self.check_multi_gpu()

        cuda_dev0 = 0 if device0 == "cpu" else int(device0.split(":")[1])
        cuda_dev1 = 0 if device1 == "cpu" else int(device1.split(":")[1])

        factory0, factory1, transport0, transport1 = self.setup_transport_pair(
            cuda_dev0, cuda_dev1
        )

        # Create source tensor with pattern, destination with zeros.
        src_tensor = torch.arange(1024, dtype=torch.uint8, device=device0)
        dst_tensor = torch.zeros(1024, dtype=torch.uint8, device=device1)

        src_mem_type = MemoryType.VRAM if device0 != "cpu" else MemoryType.DRAM
        dst_mem_type = MemoryType.VRAM if device1 != "cpu" else MemoryType.DRAM
        src_dev_id = int(device0.split(":")[1]) if device0 != "cpu" else -1
        dst_dev_id = int(device1.split(":")[1]) if device1 != "cpu" else -1

        # Register local source segment on factory0.
        src_seg = Segment(
            ptr=src_tensor.data_ptr(),
            length=src_tensor.nbytes,
            mem_type=src_mem_type,
            device_id=src_dev_id,
        )
        src_reg_result = factory0.register_segment(src_seg)
        self.assertTrue(
            src_reg_result.has_value(),
            src_reg_result.error() if src_reg_result.has_error() else "",
        )
        src_reg = src_reg_result.value()

        # Register destination segment on factory1 and export for remote access.
        dst_seg = Segment(
            ptr=dst_tensor.data_ptr(),
            length=dst_tensor.nbytes,
            mem_type=dst_mem_type,
            device_id=dst_dev_id,
        )
        dst_reg_result = factory1.register_segment(dst_seg)
        self.assertTrue(
            dst_reg_result.has_value(),
            dst_reg_result.error() if dst_reg_result.has_error() else "",
        )
        dst_reg = dst_reg_result.value()

        # Export destination's ID and import on factory0.
        export_result = dst_reg.export_id()
        self.assertTrue(
            export_result.has_value(),
            export_result.error() if export_result.has_error() else "",
        )
        remote_result = factory0.import_segment(export_result.value())
        self.assertTrue(
            remote_result.has_value(),
            remote_result.error() if remote_result.has_error() else "",
        )
        remote_seg = remote_result.value()

        # Build transfer request and execute put.
        local_span = src_reg.span(0, src_tensor.nbytes)
        remote_span = remote_seg.span(0, dst_tensor.nbytes)
        req = TransferRequest(local_span, remote_span)

        future = transport0.put([req])
        ready = future.wait_for(timeout_ms=10000)
        self.assertTrue(ready, "Put timed out")
        result = future.get()
        self.assertTrue(
            result.has_value(), result.error() if result.has_error() else ""
        )

        # Verify data was transferred correctly.
        self.assertTrue(
            torch.equal(src_tensor.cpu(), dst_tensor.cpu()),
            "Data mismatch after put transfer",
        )

        transport0.shutdown()
        transport1.shutdown()

    def test_put_gpu_to_gpu(self) -> None:
        self.run_put_transfer("cuda:0", "cuda:1")

    def test_put_gpu_to_gpu_same_device(self) -> None:
        self.run_put_transfer("cuda:0", "cuda:0")

    def run_get_transfer(self, device0: str, device1: str) -> None:
        """Test get: read data from device0 into device1."""
        self.check_multi_gpu()

        cuda_dev0 = 0 if device0 == "cpu" else int(device0.split(":")[1])
        cuda_dev1 = 0 if device1 == "cpu" else int(device1.split(":")[1])

        factory0, factory1, transport0, transport1 = self.setup_transport_pair(
            cuda_dev0, cuda_dev1
        )

        # Source has data, destination is zeroed.
        src_tensor = torch.arange(1024, dtype=torch.uint8, device=device0)
        dst_tensor = torch.zeros(1024, dtype=torch.uint8, device=device1)

        src_mem_type = MemoryType.VRAM if device0 != "cpu" else MemoryType.DRAM
        dst_mem_type = MemoryType.VRAM if device1 != "cpu" else MemoryType.DRAM
        src_dev_id = int(device0.split(":")[1]) if device0 != "cpu" else -1
        dst_dev_id = int(device1.split(":")[1]) if device1 != "cpu" else -1

        # Register source on factory0 and export for remote access.
        src_seg = Segment(
            ptr=src_tensor.data_ptr(),
            length=src_tensor.nbytes,
            mem_type=src_mem_type,
            device_id=src_dev_id,
        )
        src_reg_result = factory0.register_segment(src_seg)
        self.assertTrue(
            src_reg_result.has_value(),
            src_reg_result.error() if src_reg_result.has_error() else "",
        )
        src_reg = src_reg_result.value()

        export_result = src_reg.export_id()
        self.assertTrue(
            export_result.has_value(),
            export_result.error() if export_result.has_error() else "",
        )
        remote_result = factory1.import_segment(export_result.value())
        self.assertTrue(
            remote_result.has_value(),
            remote_result.error() if remote_result.has_error() else "",
        )
        remote_seg = remote_result.value()

        # Register destination locally on factory1.
        dst_seg = Segment(
            ptr=dst_tensor.data_ptr(),
            length=dst_tensor.nbytes,
            mem_type=dst_mem_type,
            device_id=dst_dev_id,
        )
        dst_reg_result = factory1.register_segment(dst_seg)
        self.assertTrue(
            dst_reg_result.has_value(),
            dst_reg_result.error() if dst_reg_result.has_error() else "",
        )
        dst_reg = dst_reg_result.value()

        # Build transfer request and execute get.
        local_span = dst_reg.span(0, dst_tensor.nbytes)
        remote_span = remote_seg.span(0, src_tensor.nbytes)
        req = TransferRequest(local_span, remote_span)

        future = transport1.get([req])
        ready = future.wait_for(timeout_ms=10000)
        self.assertTrue(ready, "Get timed out")
        result = future.get()
        self.assertTrue(
            result.has_value(), result.error() if result.has_error() else ""
        )

        # Verify data was transferred correctly.
        self.assertTrue(
            torch.equal(src_tensor.cpu(), dst_tensor.cpu()),
            "Data mismatch after get transfer",
        )

        transport0.shutdown()
        transport1.shutdown()

    def test_get_gpu_to_gpu(self) -> None:
        self.run_get_transfer("cuda:0", "cuda:1")

    def test_partial_put_transfer(self) -> None:
        """Test put with partial spans (offset + length)."""
        self.check_multi_gpu()

        factory0, factory1, transport0, transport1 = self.setup_transport_pair()

        src_tensor = torch.arange(1024, dtype=torch.uint8, device="cuda:0")
        dst_tensor = torch.zeros(1024, dtype=torch.uint8, device="cuda:1")

        src_seg = Segment(
            ptr=src_tensor.data_ptr(),
            length=src_tensor.nbytes,
            mem_type=MemoryType.VRAM,
            device_id=0,
        )
        src_reg = factory0.register_segment(src_seg).value()

        dst_seg = Segment(
            ptr=dst_tensor.data_ptr(),
            length=dst_tensor.nbytes,
            mem_type=MemoryType.VRAM,
            device_id=1,
        )
        dst_reg = factory1.register_segment(dst_seg).value()

        export_id = dst_reg.export_id().value()
        remote_seg = factory0.import_segment(export_id).value()

        # Transfer bytes 256-767 from source to start of destination.
        local_span = src_reg.span(256, 512)
        remote_span = remote_seg.span(0, 512)
        req = TransferRequest(local_span, remote_span)

        future = transport0.put([req])
        result = future.get()
        self.assertTrue(
            result.has_value(), result.error() if result.has_error() else ""
        )

        # First 512 bytes of dest should match bytes 256-767 of source.
        expected = src_tensor[256:768].cpu()
        actual = dst_tensor[:512].cpu()
        self.assertTrue(torch.equal(expected, actual))

        # Rest of dest should still be zeros.
        self.assertTrue(torch.all(dst_tensor[512:].cpu() == 0))

        transport0.shutdown()
        transport1.shutdown()


if __name__ == "__main__":
    unittest.main()
