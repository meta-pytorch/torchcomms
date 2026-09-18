# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import re
import unittest
from pathlib import Path


_PROGRESS_ONLY_FUNCTIONS = (
    "init_send_progress",
    "init_registered_send_progress",
    "init_recv_progress",
    "progress_send_once",
    "progress_registered_send_once",
    "progress_registered_send_drain_once",
    "poll_recv_data_ready",
    "progress_recv_once",
    "send_registered",
    "store_progress_state",
    "make_progress_geometry",
    "active_payload_offset",
    "reserve_progress_step",
    "validate_send_progress_stage",
    "validate_recv_progress_stage",
    "is_valid_progress_transition",
    "transition_progress_stage",
    "next_chunk",
    "try_prepare_send_slot",
)

_PROGRESS_ONLY_TYPES = ("ProgressChunk", "ProgressGeometry")

# Markers for "this transport reads NIC-written memory after a completion".
#
# Regexes, not substrings. `doca_gpu_dev_verbs_get` is a real RDMA READ -- the
# generic entry point in `device/doca_gpunetio_dev_verbs_onesided.cuh`, of which
# `_get_thread`/`_get_warp` are the exec-scope implementations, and the one
# ncclx GIN calls -- so it has to trip this. But `doca_gpu_dev_verbs_get_wqe_ptr`
# (used at every WQE build site), `_get_lane_id` and `_get_counter` are
# unrelated and must not. A plain substring cannot separate them, so require a
# call or template-argument list immediately after the generic name.
_NIC_READ_MARKERS = (
    r"DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_READ",
    r"doca_gpu_dev_verbs_get\s*[<(]",
    r"doca_gpu_dev_verbs_get_wait",
    r"doca_gpu_dev_verbs_get_thread",
    r"doca_gpu_dev_verbs_get_warp",
    r"wqe_prepare_read",
)


class P2pIbTransportBuildContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.package = Path(__file__).resolve().parent
        self.transport = self.package / "transport"

    def test_progress_definitions_have_a_narrow_owner(self) -> None:
        core = (self.transport / "P2pIbTransportDeviceImpl.cuh").read_text()
        progress = (self.transport / "P2pIbTransportProgressImpl.cuh").read_text()

        for function in _PROGRESS_ONLY_FUNCTIONS:
            definition = re.compile(rf"\b{function}\s*\([^;{{]*\)\s*\{{", re.DOTALL)
            self.assertIsNone(definition.search(core), function)
            self.assertIsNotNone(definition.search(progress), function)

        for type_name in _PROGRESS_ONLY_TYPES:
            self.assertNotIn(f"struct {type_name}", core)
            self.assertIn(f"struct {type_name}", progress)

    def test_stable_and_backend_headers_do_not_pull_progress(self) -> None:
        paths = (
            self.transport / "P2pIbTransportDeviceDecl.cuh",
            self.transport / "P2pIbTransportDevice.cuh",
            self.transport / "ibgda/P2pIbgdaTransportDevice.cuh",
            self.transport / "ibrc/P2pIbrcTransportDevice.cuh",
        )
        for path in paths:
            self.assertNotIn(
                "P2pIbTransportProgressImpl.cuh", path.read_text(), path.name
            )

    def test_progress_header_has_a_separate_build_owner(self) -> None:
        buck = (self.package / "BUCK").read_text()
        core_start = buck.index('name = "p2p_ib_transport_device_impl"')
        progress_start = buck.index(
            'name = "p2p_ib_transport_progress_impl"', core_start
        )
        contract_start = buck.index(
            'name = "p2p_ib_transport_build_contract_test"', progress_start
        )
        core_target = buck[core_start:progress_start]
        progress_target = buck[progress_start:contract_start]
        self.assertEqual(
            core_target.count('"transport/P2pIbTransportDeviceImpl.cuh"'),
            1,
        )
        self.assertNotIn("P2pIbTransportProgressImpl.cuh", core_target)
        self.assertEqual(
            progress_target.count('"transport/P2pIbTransportProgressImpl.cuh"'),
            1,
        )
        self.assertIn('":p2p_ib_transport_device_impl"', progress_target)

    def test_cta_cq_acquire_scope_implies_no_rdma_read(self) -> None:
        """Guards the CTA-scope CQ acquire fence.

        On Blackwell the SYS-scope acquire DOCA runs after a completion lowers
        to CCTL.IVALL, a whole-L1 invalidate; CTA lowers to a NOP. Dropping it
        is only sound while this transport never reads NIC-written memory
        through L1 after a completion -- today it issues no RDMA READ, atomic
        results go to a discard sink, and the signal is read via a system-scope
        atomic load that carries its own invalidate.

        Adding an RDMA READ breaks that silently: no crash, no wrong return
        code, just stale data under a timing window a short test will not hit.
        So fail loudly here instead.
        """
        ibgda = (self.transport / "ibgda/P2pIbgdaTransportDevice.cuh").read_text()
        scope = re.search(
            r"kCqAcquireScope\s*=\s*(DOCA_GPUNETIO_VERBS_SYNC_SCOPE_\w+)", ibgda
        )
        self.assertIsNotNone(
            scope,
            "kCqAcquireScope is not declared in P2pIbgdaTransportDevice.cuh. It was "
            "renamed, moved or deleted; this guard cannot tell whether the CTA fence "
            "is live, so it must not pass by default. Re-point the guard at wherever "
            "the acquire scope is now decided.",
        )
        assert scope is not None  # for the type checker
        if scope.group(1) != "DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA":
            self.skipTest(
                f"CQ acquire scope is {scope.group(1)}; invariant not required"
            )
        for marker in _NIC_READ_MARKERS:
            self.assertIsNone(
                re.search(marker, ibgda),
                f"{marker} matches an RDMA-read path here. Reading NIC-written "
                "memory is incompatible with kCqAcquireScope == "
                "DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA. Either set kCqAcquireScope "
                "back to DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS or give the new read "
                "path its own system-scope acquire. See "
                "third-party/nvidia-doca/patches/README.md.",
            )


if __name__ == "__main__":
    unittest.main()
