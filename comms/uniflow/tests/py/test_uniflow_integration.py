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

        # Create agents on main thread (avoids GIL contention during construction)
        srv_config = UniflowAgentConfig(
            device_id=0, name="server", listen_address="*:0"
        )
        cli_config = UniflowAgentConfig(
            device_id=1, name="client", listen_address="*:0"
        )
        srv_agent: UniflowAgent = UniflowAgent(srv_config)
        cli_agent: UniflowAgent = UniflowAgent(cli_config)
        srv_uid_result = srv_agent.get_unique_id()
        assert srv_uid_result.has_value(), f"get_unique_id: {srv_uid_result.error()}"
        srv_uid: str = srv_uid_result.value()

        errors: list[str] = []

        def server_fn() -> None:
            try:
                conn_result = srv_agent.accept()
                assert conn_result.has_value(), f"accept: {conn_result.error()}"
                conn = conn_result.value()

                seg = Segment(
                    ptr=server_tensor.data_ptr(),
                    length=server_tensor.nbytes,
                    mem_type=MemoryType.VRAM,
                    device_id=0,
                )
                reg_result = srv_agent.register_segment(seg)
                assert reg_result.has_value(), f"register: {reg_result.error()}"
                reg_seg = reg_result.value()

                # Export our segment ID and send to client via ctrl msg
                export_result = reg_seg.export_id()
                assert export_result.has_value(), f"export_id: {export_result.error()}"
                send_result = conn.send_ctrl_msg(export_result.value())
                assert send_result.has_value(), f"send export: {send_result.error()}"

                # Receive client's export ID and import
                recv_result = conn.recv_ctrl_msg()
                assert recv_result.has_value(), f"recv export: {recv_result.error()}"
                import_result = srv_agent.import_segment(recv_result.value())
                assert import_result.has_value(), f"import: {import_result.error()}"

                # Send data_ready notification via ctrl msg
                notif_result = conn.send_ctrl_msg(b"data_ready")
                assert notif_result.has_value(), f"send notif: {notif_result.error()}"

                # Wait for client's "done" message before shutting down
                done_result = conn.recv_ctrl_msg()
                assert done_result.has_value(), f"recv done: {done_result.error()}"
                assert done_result.value() == b"done"

                conn.shutdown()
            except Exception as e:
                errors.append(f"Server: {e}")

        def client_fn() -> None:
            try:
                conn_result = cli_agent.connect(srv_uid)
                assert conn_result.has_value(), f"connect: {conn_result.error()}"
                conn = conn_result.value()

                seg = Segment(
                    ptr=client_tensor.data_ptr(),
                    length=client_tensor.nbytes,
                    mem_type=MemoryType.VRAM,
                    device_id=1,
                )
                reg_result = cli_agent.register_segment(seg)
                assert reg_result.has_value(), f"register: {reg_result.error()}"
                reg_seg = reg_result.value()

                # Export our segment ID and send to server via ctrl msg
                export_result = reg_seg.export_id()
                assert export_result.has_value(), f"export_id: {export_result.error()}"
                send_result = conn.send_ctrl_msg(export_result.value())
                assert send_result.has_value(), f"send export: {send_result.error()}"

                # Receive server's export ID and import
                recv_result = conn.recv_ctrl_msg()
                assert recv_result.has_value(), f"recv export: {recv_result.error()}"
                import_result = cli_agent.import_segment(recv_result.value())
                assert import_result.has_value(), f"import: {import_result.error()}"
                server_remote_seg = import_result.value()

                # Pull data from server via get()
                local_span = reg_seg.span(0, reg_seg.length)
                remote_span = server_remote_seg.span(0, server_remote_seg.length)
                future = conn.get(requests=[TransferRequest(local_span, remote_span)])

                # UniflowFuture API coverage
                ready = future.wait_for(timeout_ms=10000)
                assert ready, "Transfer timed out"
                assert future.done()

                get_result = future.get()
                assert get_result.has_value(), f"get: {get_result.error()}"

                # Receive "data_ready" notification via ctrl msg
                notif_result = conn.recv_ctrl_msg()
                assert notif_result.has_value(), f"recv notif: {notif_result.error()}"
                assert notif_result.value() == b"data_ready"

                # Double-get returns cached result
                get_result2 = future.get()
                assert get_result2.has_value(), get_result2.error()

                # Signal server that transfer is done
                done_result = conn.send_ctrl_msg(b"done")
                assert done_result.has_value(), done_result.error()

                conn.shutdown()
            except Exception as e:
                errors.append(f"Client: {e}")

        st = threading.Thread(target=server_fn)
        ct = threading.Thread(target=client_fn)
        st.start()
        ct.start()
        st.join(timeout=30)
        ct.join(timeout=30)

        self.assertFalse(st.is_alive(), "Server thread hung")
        self.assertFalse(ct.is_alive(), "Client thread hung")
        self.assertEqual(errors, [], f"Thread errors: {errors}")

        # Verify data correctness
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        expected: torch.Tensor = torch.arange(
            _NUM_ELEMS, dtype=torch.float32, device="cuda:1"
        )
        torch.testing.assert_close(client_tensor, expected)
