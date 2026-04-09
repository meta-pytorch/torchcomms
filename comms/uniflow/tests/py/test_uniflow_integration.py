# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
End-to-end integration test for UniFlow.

Exercises the full path: agent creation → segment registration →
connection establishment → ctrl message exchange → get() data transfer →
data correctness verification.

Uses a VMM allocator with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR so
that NVLink segment registration via cuMemExportToShareableHandle works.
"""

import os
import threading
import unittest

# Must be set before any CUDA allocation.
# expandable_segments: use cuMemCreate (VMM) for allocations.
# TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC: create allocations with
# CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR so that NVLink transport
# can export/import via cuMemExportToShareableHandle.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC"] = "1"

import torch  # noqa: E402


_SKIP_NO_MULTI_GPU: bool = (
    not torch.cuda.is_available() or torch.cuda.device_count() < 2
)

_NUM_ELEMS: int = 1024 * 1024  # 4MB at float32


class TestUniflowIntegration(unittest.TestCase):
    """Full-path single-process integration test on real GPUs."""

    @unittest.skipIf(_SKIP_NO_MULTI_GPU, "Need at least 2 GPUs")
    def test_full_path(self) -> None:
        from uniflow._core import (
            MemoryType,
            Segment,
            TransferRequest,
            UniflowAgent,
            UniflowAgentConfig,
        )

        # Allocate GPU tensors using PyTorch's default allocator.
        # expandable_segments:True (set via env var) makes PyTorch use
        # cuMemCreate internally, which is required for NVLink registration.
        server_tensor: torch.Tensor = torch.arange(
            _NUM_ELEMS, dtype=torch.float32, device="cuda:0"
        )
        client_tensor: torch.Tensor = torch.zeros(
            _NUM_ELEMS, dtype=torch.float32, device="cuda:1"
        )

        # Debug: print tensor sizes and data_ptrs
        import sys

        print(
            f"server_tensor: data_ptr=0x{server_tensor.data_ptr():x}, "
            f"nbytes={server_tensor.nbytes}, "
            f"device={server_tensor.device}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"client_tensor: data_ptr=0x{client_tensor.data_ptr():x}, "
            f"nbytes={client_tensor.nbytes}, "
            f"device={client_tensor.device}",
            file=sys.stderr,
            flush=True,
        )

        # Shared state between threads
        server_id_holder: list[str | None] = [None]
        server_id_ready: threading.Event = threading.Event()
        errors: list[str] = []

        def server_fn() -> None:
            try:
                torch.cuda.set_device(0)

                config = UniflowAgentConfig(
                    device_id=0,
                    name="server",
                    listen_address="*:0",
                )
                agent = UniflowAgent(config)
                id_result = agent.get_unique_id()
                assert id_result.has_value(), id_result.error()
                server_id_holder[0] = id_result.value()
                server_id_ready.set()

                # Accept connection
                conn_result = agent.accept()
                assert conn_result.has_value(), conn_result.error()
                conn = conn_result.value()

                # Register GPU tensor
                seg = Segment(
                    ptr=server_tensor.data_ptr(),
                    length=server_tensor.nbytes,
                    mem_type=MemoryType.VRAM,
                    device_id=0,
                )
                reg_result = agent.register_segment(seg)
                assert reg_result.has_value(), reg_result.error()
                reg_seg = reg_result.value()

                # Send export ID to client
                export_result = reg_seg.export_id()
                assert export_result.has_value(), export_result.error()
                send_result = conn.send_ctrl_msg(export_result.value())
                assert send_result.has_value(), send_result.error()

                # Receive client's export ID and import
                recv_result = conn.recv_ctrl_msg()
                assert recv_result.has_value(), recv_result.error()
                import_result = agent.import_segment(recv_result.value())
                assert import_result.has_value(), import_result.error()

                # Wait for client to signal transfer is done
                done_result = conn.recv_ctrl_msg()
                assert done_result.has_value(), done_result.error()

                conn.shutdown()

            except Exception as e:
                errors.append(f"Server: {e}")

        def client_fn() -> None:
            try:
                torch.cuda.set_device(1)
                server_id_ready.wait(timeout=10)
                server_id: str | None = server_id_holder[0]
                assert server_id is not None

                config = UniflowAgentConfig(
                    device_id=1,
                    name="client",
                    listen_address="*:0",
                )
                agent = UniflowAgent(config)

                # Connect to server
                conn_result = agent.connect(server_id)
                assert conn_result.has_value(), conn_result.error()
                conn = conn_result.value()

                # Register GPU tensor
                seg = Segment(
                    ptr=client_tensor.data_ptr(),
                    length=client_tensor.nbytes,
                    mem_type=MemoryType.VRAM,
                    device_id=1,
                )
                reg_result = agent.register_segment(seg)
                assert reg_result.has_value(), reg_result.error()
                reg_seg = reg_result.value()

                # Send export ID to server
                export_result = reg_seg.export_id()
                assert export_result.has_value(), export_result.error()
                send_result = conn.send_ctrl_msg(export_result.value())
                assert send_result.has_value(), send_result.error()

                # Receive server's export ID and import
                recv_result = conn.recv_ctrl_msg()
                assert recv_result.has_value(), recv_result.error()
                import_result = agent.import_segment(recv_result.value())
                assert import_result.has_value(), import_result.error()
                server_remote_seg = import_result.value()

                # Get data from server's GPU into our GPU
                local_span = reg_seg.span(0, reg_seg.length)
                remote_span = server_remote_seg.span(0, server_remote_seg.length)
                future = conn.get(requests=[TransferRequest(local_span, remote_span)])
                get_result = future.get()
                assert get_result.has_value(), get_result.error()

                # Signal server that transfer is done
                done_result = conn.send_ctrl_msg(b"done")
                assert done_result.has_value(), done_result.error()

                conn.shutdown()

            except Exception as e:
                errors.append(f"Client: {e}")

        server_thread = threading.Thread(target=server_fn)
        client_thread = threading.Thread(target=client_fn)
        server_thread.start()
        client_thread.start()
        server_thread.join(timeout=30)
        client_thread.join(timeout=30)

        self.assertFalse(server_thread.is_alive(), "Server thread hung")
        self.assertFalse(client_thread.is_alive(), "Client thread hung")
        self.assertEqual(errors, [], f"Thread errors: {errors}")

        # Verify data correctness
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        expected: torch.Tensor = torch.arange(
            _NUM_ELEMS, dtype=torch.float32, device="cuda:1"
        )
        torch.testing.assert_close(client_tensor, expected)
