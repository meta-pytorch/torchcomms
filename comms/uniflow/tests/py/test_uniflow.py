# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import ctypes
import unittest


def _has_gpu() -> bool:
    """Check if CUDA GPUs are available without importing torch."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


_HAS_GPU: bool = _has_gpu()


class TestTypesAndSegment(unittest.TestCase):
    def test_err_code_enum(self) -> None:
        from uniflow._core import ErrCode

        self.assertEqual(int(ErrCode.NotImplemented), 0)
        self.assertEqual(int(ErrCode.InvalidArgument), 3)
        self.assertEqual(int(ErrCode.Timeout), 8)

    def test_memory_type_enum(self) -> None:
        from uniflow._core import MemoryType

        self.assertEqual(int(MemoryType.DRAM), 0)
        self.assertEqual(int(MemoryType.VRAM), 1)
        self.assertEqual(int(MemoryType.NVME), 2)

    def test_transport_type_enum(self) -> None:
        from uniflow._core import TransportType

        self.assertEqual(int(TransportType.NVLink), 0)
        self.assertEqual(int(TransportType.RDMA), 1)
        self.assertEqual(int(TransportType.TCP), 2)

    def test_segment_creation(self) -> None:
        from uniflow._core import MemoryType, Segment

        buf = ctypes.create_string_buffer(256)
        ptr = ctypes.addressof(buf)
        seg = Segment(ptr=ptr, length=256, mem_type=MemoryType.VRAM, device_id=0)

        self.assertEqual(seg.data_ptr, ptr)
        self.assertEqual(seg.length, 256)
        self.assertEqual(seg.mem_type, MemoryType.VRAM)
        self.assertEqual(seg.device_id, 0)

    def test_segment_cpu(self) -> None:
        from uniflow._core import MemoryType, Segment

        buf = ctypes.create_string_buffer(64)
        ptr = ctypes.addressof(buf)
        seg = Segment(ptr=ptr, length=64, mem_type=MemoryType.DRAM, device_id=-1)

        self.assertEqual(seg.mem_type, MemoryType.DRAM)
        self.assertEqual(seg.device_id, -1)

    def test_uniflow_agent_config(self) -> None:
        from uniflow._core import UniflowAgentConfig

        config = UniflowAgentConfig(
            device_id=0,
            name="test_agent",
            connect_retries=5,
            connect_timeout_ms=2000,
        )
        self.assertEqual(config.device_id, 0)
        self.assertEqual(config.name, "test_agent")
        self.assertEqual(config.connect_retries, 5)
        self.assertEqual(config.connect_timeout_ms, 2000)

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_get_unique_id_with_server(self) -> None:
        from uniflow._core import UniflowAgent, UniflowAgentConfig

        config = UniflowAgentConfig(device_id=0, name="test", listen_address="*:0")
        agent = UniflowAgent(config)

        # get_unique_id should succeed
        result = agent.get_unique_id()
        self.assertTrue(result.has_value())

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_result_error(self) -> None:
        from uniflow._core import ErrCode, UniflowAgent, UniflowAgentConfig

        # Agent without listen_address — get_unique_id should fail gracefully
        # We can't easily create an agent without a server (constructor requires
        # listen_address or throws), so test connect to a bad address instead
        config = UniflowAgentConfig(device_id=0, name="test", listen_address="*:0")
        agent = UniflowAgent(config)
        result = agent.connect("127.0.0.1:1")  # nothing listening
        self.assertTrue(result.has_error())
        self.assertEqual(result.error().code, ErrCode.ConnectionFailed)

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_result_value_raises_on_error(self) -> None:
        from uniflow._core import UniflowAgent, UniflowAgentConfig

        config = UniflowAgentConfig(device_id=0, name="test", listen_address="*:0")
        agent = UniflowAgent(config)
        result = agent.connect("127.0.0.1:1")

        self.assertFalse(result)
        with self.assertRaises(RuntimeError):
            result.value()

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_result_error_raises_on_value(self) -> None:
        from uniflow._core import UniflowAgent, UniflowAgentConfig

        config = UniflowAgentConfig(device_id=0, name="test", listen_address="*:0")
        agent = UniflowAgent(config)
        result = agent.get_unique_id()

        self.assertTrue(result)
        with self.assertRaises(RuntimeError):
            result.error()
