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
    def test_public_surface_works_end_to_end(self) -> None:
        # Mirrors what real callers will do: a single `import uniflow`, then use
        # symbols straight off the package. No reaching into uniflow._core.
        import uniflow

        buf = ctypes.create_string_buffer(128)
        seg = uniflow.Segment(
            ptr=ctypes.addressof(buf),
            length=128,
            mem_type=uniflow.MemoryType.DRAM,
            device_id=-1,
        )
        self.assertEqual(seg.length, 128)
        self.assertEqual(seg.mem_type, uniflow.MemoryType.DRAM)
        self.assertEqual(int(uniflow.TransportType.NVLink), 0)

    def test_public_surface_matches_core(self) -> None:
        # Maintainer-facing check: every name advertised in __all__ is reachable
        # on the package and is the *same identity* as the underlying _core
        # symbol. Catches drift if a new pybind class is added without being
        # re-exported, or if __init__.py shadows a name with a wrapper.
        import uniflow
        import uniflow._core

        for name in uniflow.__all__:
            self.assertTrue(
                hasattr(uniflow, name), f"uniflow missing public export: {name}"
            )
            self.assertIs(getattr(uniflow, name), getattr(uniflow._core, name))

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

    def test_segment_from_tensor_cpu(self) -> None:
        import torch
        from uniflow._core import MemoryType, Segment

        t = torch.zeros(64, dtype=torch.float32)
        seg = Segment.from_tensor(t)

        self.assertEqual(seg.data_ptr, t.data_ptr())
        self.assertEqual(seg.length, 64 * 4)
        self.assertEqual(seg.mem_type, MemoryType.DRAM)
        self.assertEqual(seg.device_id, -1)

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_segment_from_tensor_cuda(self) -> None:
        import torch
        from uniflow._core import MemoryType, Segment

        t = torch.zeros(128, dtype=torch.float32, device="cuda:0")
        seg = Segment.from_tensor(t)

        self.assertEqual(seg.data_ptr, t.data_ptr())
        self.assertEqual(seg.length, 128 * 4)
        self.assertEqual(seg.mem_type, MemoryType.VRAM)
        self.assertEqual(seg.device_id, 0)

    def test_segment_from_tensor_rejects_non_tensor(self) -> None:
        from uniflow._core import Segment

        with self.assertRaises(TypeError):
            Segment.from_tensor("not a tensor")

    def test_segment_from_tensor_rejects_non_contiguous(self) -> None:
        import torch
        from uniflow._core import Segment

        t = torch.zeros(8, 8, dtype=torch.float32).t()
        self.assertFalse(t.is_contiguous())
        with self.assertRaises(RuntimeError):
            Segment.from_tensor(t)

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

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_result_unwrap_returns_value(self) -> None:
        from uniflow._core import UniflowAgent, UniflowAgentConfig

        config = UniflowAgentConfig(device_id=0, name="test", listen_address="*:0")
        agent = UniflowAgent(config)
        result = agent.get_unique_id()

        self.assertEqual(result.unwrap(), result.value())

    @unittest.skipUnless(_HAS_GPU, "Requires GPU")
    def test_result_unwrap_raises_on_error(self) -> None:
        from uniflow._core import UniflowAgent, UniflowAgentConfig

        config = UniflowAgentConfig(device_id=0, name="test", listen_address="*:0")
        agent = UniflowAgent(config)
        result = agent.connect("127.0.0.1:1")

        with self.assertRaises(RuntimeError):
            result.unwrap()
