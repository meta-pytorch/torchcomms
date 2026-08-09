# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import re
import unittest
from pathlib import Path


_PROGRESS_ONLY_FUNCTIONS = (
    "init_send_progress",
    "init_recv_progress",
    "progress_send_once",
    "poll_recv_data_ready",
    "progress_recv_once",
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

    def test_header_ownership_adds_no_production_target(self) -> None:
        buck = (self.package / "BUCK").read_text()
        target_start = buck.index('name = "p2p_ib_transport_device_impl"')
        contract_start = buck.index(
            'name = "p2p_ib_transport_build_contract_test"', target_start
        )
        implementation_target = buck[target_start:contract_start]
        self.assertEqual(
            implementation_target.count('"transport/P2pIbTransportDeviceImpl.cuh"'),
            1,
        )
        self.assertEqual(
            implementation_target.count('"transport/P2pIbTransportProgressImpl.cuh"'),
            1,
        )


if __name__ == "__main__":
    unittest.main()
