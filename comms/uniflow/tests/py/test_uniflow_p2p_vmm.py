# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
Integration test for VMM segments over the AMD P2P (XGMI) transport.

PyTorch expandable segments allocated with TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC
are VMM chunks shareable as POSIX fds, and the P2P transport shares them as
such instead of through HIP IPC. AMD only: on NVIDIA the NVLink transport, not
P2P, carries intra-node VMM segments.
"""

import os
import resource
import struct
import threading
import unittest

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC"] = "1"

import torch  # noqa: E402


# Segment export ID layout (Segment.cpp) and P2P VMM handle payload layout
# (transport/p2p/P2pRegistrationHandle.cpp).
_EXPORT_ID_HEADER: struct.Struct = struct.Struct("<BBBBiQQ")
_EXPORT_ID_HANDLE: struct.Struct = struct.Struct("<BI")
_P2P_VMM_HEADER: struct.Struct = struct.Struct("<B16sQiiQQI")
_P2P_VMM_CHUNK_BYTES: int = 28
_P2P_POSIX_FD: int = 1

# A pad allocated first puts each data tensor at a non-zero offset inside the
# first chunk of its expandable segment; the data spans several chunks.
_PAD_BYTES: int = 4 << 20
_P2P_NUM_WORDS: int = 50 << 20  # 200 MiB at int32
# Exported chunk fds plus the transport's fd reserve.
_MIN_FREE_FDS: int = 512


def _transport_payloads(export_id: bytes) -> dict[int, bytes]:
    """Splits a segment export ID into its handle payloads by transport type."""
    num_handles = _EXPORT_ID_HEADER.unpack_from(export_id)[2]
    pos = _EXPORT_ID_HEADER.size
    payloads: dict[int, bytes] = {}
    for _ in range(num_handles):
        transport_type, size = _EXPORT_ID_HANDLE.unpack_from(export_id, pos)
        pos += _EXPORT_ID_HANDLE.size
        payloads[transport_type] = export_id[pos : pos + size]
        pos += size
    assert pos == len(export_id), "export ID has trailing bytes"
    return payloads


class TestUniflowP2pVmm(unittest.TestCase):
    """Single-process P2P test on two real GPUs with expandable segments."""

    @unittest.skipIf(
        torch.version.hip is None,
        "On NVIDIA the NVLink transport, not P2P, carries VMM segments",
    )
    def test_p2p_carries_expandable_segments(self) -> None:
        """AMD P2P moves expandable-segment tensors as VMM chunks, not HIP IPC."""
        from uniflow._core import (
            MemoryType,
            MultiTransport,
            MultiTransportFactory,
            RegisteredSegment,
            RemoteRegisteredSegment,
            Segment,
            TransferRequest,
            TransportType,
        )

        # AMD-only and scheduled on two GPUs, so fewer is a misrouted run, not
        # a reason to skip.
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        # A failed VMM export falls back to HIP IPC, which some ROCm 7.0
        # runtimes accept on VMM memory with an unusable handle. Rule out each
        # cause of a failed export before registering.
        soft_fd_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_fd_limit != resource.RLIM_INFINITY:
            free_fds = soft_fd_limit - len(os.listdir("/proc/self/fd"))
            self.assertGreater(free_fds, _MIN_FREE_FDS)

        devices = (0, 1)
        pads = [
            torch.zeros(_PAD_BYTES, dtype=torch.uint8, device=f"cuda:{d}")
            for d in devices
        ]
        # Device 0 starts with the first half of the pattern, device 1 with
        # the second.
        pattern = torch.arange(_P2P_NUM_WORDS, dtype=torch.int32)
        data = [pattern.to(f"cuda:{d}") for d in devices]
        half = _P2P_NUM_WORDS // 2
        data[0][half:].zero_()
        data[1][:half].zero_()

        segments = torch.cuda.memory_snapshot(include_traces=False)
        for d, tensor in zip(devices, data):
            ptr, nbytes = tensor.data_ptr(), tensor.nbytes
            owners = [
                s
                for s in segments
                if s["device"] == d
                and s["address"] <= ptr
                and ptr + nbytes <= s["address"] + s["total_size"]
            ]
            self.assertEqual(len(owners), 1)
            self.assertIs(owners[0]["is_expandable"], True)

        # An exact-match NIC filter naming no device keeps RDMA out, and a
        # loopback TCP host without enable_tcp keeps TCP out.
        factories = [
            MultiTransportFactory(
                d, nic_filter="=uniflow_p2p_vmm_test_nonic", tcp_bind_host="::1"
            )
            for d in devices
        ]
        registered: list[RegisteredSegment] = []
        export_ids: list[bytes] = []
        for d, factory, tensor in zip(devices, factories, data):
            reg = factory.register_segment(
                Segment(tensor.data_ptr(), tensor.nbytes, MemoryType.VRAM, d)
            )
            assert reg.has_value(), f"register {d}: {reg.error()}"
            registered.append(reg.value())
            eid = registered[-1].export_id()
            assert eid.has_value(), f"export_id {d}: {eid.error()}"
            export_id: bytes = eid.value()
            export_ids.append(export_id)

            # Prove the P2P handle is a VMM export before anything imports it
            # or registers the next segment. An IPC payload never has a valid
            # VMM length.
            p2p = _transport_payloads(export_id)[TransportType.NVLink.value]
            mode, _, _, _, _, offset, size, num_chunks = _P2P_VMM_HEADER.unpack_from(
                p2p
            )
            self.assertEqual(mode, _P2P_POSIX_FD)
            self.assertEqual(
                len(p2p), _P2P_VMM_HEADER.size + num_chunks * _P2P_VMM_CHUNK_BYTES
            )
            self.assertEqual(size, tensor.nbytes)
            self.assertGreater(offset, 0)
            self.assertGreater(num_chunks, 1)

        remotes: list[RemoteRegisteredSegment] = []
        transports: list[MultiTransport] = []
        infos: list[bytes] = []
        for factory, peer, export_id in zip(
            factories, reversed(factories), reversed(export_ids)
        ):
            imported = factory.import_segment(export_id)
            assert imported.has_value(), f"import: {imported.error()}"
            remotes.append(imported.value())
            created = factory.create_transport(peer.get_topology())
            assert created.has_value(), f"create_transport: {created.error()}"
            transports.append(created.value())
            bound = transports[-1].bind()
            assert bound.has_value(), f"bind: {bound.error()}"
            infos.append(bound.value())

        errors: list[str] = []

        def connect(transport: MultiTransport, info: bytes) -> None:
            connected = transport.connect(info)
            if not connected.has_value():
                errors.append(f"connect: {connected.error()}")

        threads = [
            threading.Thread(target=connect, args=(transport, info), daemon=True)
            for transport, info in zip(transports, reversed(infos))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads), "connect hung")
        self.assertEqual(errors, [])

        # Each side writes a quarter it holds into its peer and reads a quarter
        # it lacks, so both tensors must end up holding the whole pattern.
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        quarter = _P2P_NUM_WORDS // 4 * pattern.element_size()
        for side, is_put, index in (
            (0, True, 0),
            (0, False, 2),
            (1, True, 3),
            (1, False, 1),
        ):
            request = TransferRequest(
                registered[side].span(index * quarter, quarter),
                remotes[side].span(index * quarter, quarter),
            )
            transport = transports[side]
            future = (transport.put if is_put else transport.get)([request])
            self.assertTrue(future.wait_for(timeout_ms=30000), "transfer timed out")
            done = future.get()
            assert done.has_value(), f"transfer {side} {index}: {done.error()}"
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        # A transfer that ignored the segment offset would land in the pads.
        for tensor, pad in zip(data, pads):
            self.assertTrue(torch.equal(tensor.cpu(), pattern))
            self.assertEqual(int(pad.count_nonzero()), 0)
        for transport in transports:
            self.assertEqual(transport.transfer_count(TransportType.NVLink), 2)
            self.assertEqual(transport.transfer_count(TransportType.RDMA), 0)
            self.assertEqual(transport.transfer_count(TransportType.TCP), 0)
            transport.shutdown()
