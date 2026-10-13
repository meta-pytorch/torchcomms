#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Test MCCL AllReduce under CUDA graph capture and replay.

All tests use dynamic regime (enable_reconfigure=True).
The comm is initialized via reconfigure() before each test.

Test categories mirror the C++ AllReduceCudaGraphTest:
- Happy path: SingleElement, OneMB, MultiSizeGraph
- Concurrent: GraphReplayWithConcurrentEager
- Fault tolerance: SingleElementFT, OneMBFT, MultiSizeGraphFT
- Recovery: ReconfigureAndRecapture
"""

import os
import time
import unittest
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

import torch
from torch.distributed import TCPStore
from torchcomms import new_comm, ReduceOp, TorchComm

ONE_MB_FLOATS = 1024 * 1024 // 4
OP_TIMEOUT = timedelta(seconds=1)
TIMED_TIMEOUT = timedelta(seconds=3)
TIMED_TIMEOUT_LOWER_BOUND = timedelta(seconds=2)
TIMED_TIMEOUT_UPPER_BOUND = timedelta(seconds=5)


class AllReduceCudaGraphTest(unittest.TestCase):
    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"

    # Shared store across tests (created once per process)
    _shared_store: Optional[TCPStore] = None
    # UUID counter for reconfigure calls (must be unique across tests)
    _uuid_counter = 0

    def setUp(self) -> None:
        self.rank = int(
            os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        )
        self.world_size = int(
            os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        )

        if AllReduceCudaGraphTest._shared_store is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = int(os.environ.get("MASTER_PORT", "29500"))
            AllReduceCudaGraphTest._shared_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
                timeout=timedelta(seconds=30),
            )
        self.store = AllReduceCudaGraphTest._shared_store

        device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self.device)

        self.expected = float(self.world_size)

    def tearDown(self) -> None:
        """Store barrier so all ranks finish each test before proceeding."""
        test_id = self._test_id()
        self.store.set(f"barrier_{test_id}_{self.rank}", "1")
        for r in range(self.world_size):
            self.store.wait([f"barrier_{test_id}_{r}"])

    def _next_uuid(self) -> int:
        AllReduceCudaGraphTest._uuid_counter += 1
        return AllReduceCudaGraphTest._uuid_counter

    def _create_comm(self, name: str) -> TorchComm:
        """Create a dynamic-regime comm and reconfigure with all ranks."""
        comm = new_comm(
            "mccl",
            self.device,
            name=name,
            enable_reconfigure=True,
            hints={"watchdog_timeout_ms": "10000"},
            abort_process_on_timeout_or_error=False,
        )

        uuid = self._next_uuid()
        key_prefix = f"{name}_{uuid}"

        my_handle = comm.get_init_handle()
        self.store.set(f"{key_prefix}_handle_{self.rank}", my_handle)
        init_handles = []
        for r in range(self.world_size):
            handle = self.store.get(f"{key_prefix}_handle_{r}").decode("utf-8")
            init_handles.append(handle)
        work = comm.reconfigure(
            uuid=uuid,
            init_handles=init_handles,
            timeout=timedelta(seconds=30),
        )
        work.wait_blocking()
        return comm

    def _warmup(self, comm: TorchComm, tensors: List[torch.Tensor]) -> None:
        """Eager warmup to establish IB connections."""
        for t in tensors:
            comm.all_reduce(t, ReduceOp.SUM, async_op=True).wait()
        torch.cuda.synchronize(self.device)

    def _capture_graph(
        self,
        comm: TorchComm,
        tensors: List[torch.Tensor],
        timeout: timedelta = OP_TIMEOUT,
    ) -> Tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
        """Capture allreduces into a CUDA graph. Returns (graph, stream)."""
        for t in tensors:
            t.fill_(1.0)
        torch.cuda.synchronize(self.device)

        stream = torch.cuda.Stream(device=self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(g, stream=stream):
                for t in tensors:
                    work = comm.all_reduce(
                        t, ReduceOp.SUM, async_op=True, timeout=timeout
                    )
                    work.wait()
        return g, stream

    def _replay_and_validate(
        self,
        g: torch.cuda.CUDAGraph,
        tensors: List[torch.Tensor],
        sizes: List[int],
        expected: float,
    ) -> None:
        """Fill tensors with 1.0, replay graph, validate results."""
        for t in tensors:
            t.fill_(1.0)
        torch.cuda.synchronize(self.device)
        g.replay()
        torch.cuda.synchronize(self.device)
        for j, t in enumerate(tensors):
            self.assertTrue(
                torch.allclose(t, torch.full_like(t, expected)),
                f"Tensor {j} (size={sizes[j]}) failed: "
                f"got {t.flatten()[:4].tolist()}, expected {expected}",
            )

    def _run_graph_allreduce(self, sizes: List[int], num_replays: int) -> None:
        """Capture allreduces, replay num_replays times, validate."""
        comm = self._create_comm(f"graph_{self._test_id()}")
        tensors = [
            torch.ones(s, dtype=torch.float32, device=self.device) for s in sizes
        ]
        self._warmup(comm, tensors)
        g, stream = self._capture_graph(comm, tensors)

        for _ in range(num_replays):
            self._replay_and_validate(g, tensors, sizes, self.expected)

        del g
        comm.finalize()

    def _run_graph_allreduce_ft(self, sizes: List[int], num_replays: int) -> None:
        """Like _run_graph_allreduce but last rank drops before final replay.

        Surviving ranks replay the graph with one rank missing. The allreduce
        inside the graph times out (OP_TIMEOUT), causing the comm to enter
        aborted state. This exercises the real GPU-side fault detection path.
        """
        self._run_graph_allreduce_with_missing_rank(
            name="graph_ft",
            sizes=sizes,
            num_successful_replays=num_replays - 1,
        )

    def _run_graph_allreduce_with_missing_rank(
        self,
        name: str,
        sizes: List[int],
        num_successful_replays: int,
        timeout: timedelta = OP_TIMEOUT,
        final_replay_assertion: Optional[
            Callable[[torch.cuda.CUDAGraph, TorchComm], None]
        ] = None,
        wait_for_survivors: bool = False,
    ) -> None:
        """Capture allreduces, then replay once with the last rank missing."""
        test_id = self._test_id()
        comm = self._create_comm(f"{name}_{test_id}")
        tensors = [
            torch.ones(s, dtype=torch.float32, device=self.device) for s in sizes
        ]
        self._warmup(comm, tensors)
        g, stream = self._capture_graph(comm, tensors, timeout=timeout)

        for _ in range(num_successful_replays):
            self._replay_and_validate(g, tensors, sizes, self.expected)
            self.assertFalse(comm.is_aborted())

        drop_key = f"{name}_dropped_{test_id}"
        done_keys = [f"{name}_done_{test_id}_{r}" for r in range(self.world_size - 1)]

        if self.rank == self.world_size - 1:
            self.store.set(drop_key, "1")
            if wait_for_survivors:
                self.store.wait(done_keys)
            del g
            comm.finalize()
            return

        self.store.wait([drop_key])
        try:
            for t in tensors:
                t.fill_(1.0)
            torch.cuda.synchronize(self.device)
            if final_replay_assertion is None:
                g.replay()
                torch.cuda.synchronize(self.device)
                self.assertTrue(comm.is_aborted())
            else:
                final_replay_assertion(g, comm)
        finally:
            if wait_for_survivors:
                self.store.set(f"{name}_done_{test_id}_{self.rank}", "1")
            del g
            comm.finalize()

    def _assert_timed_out_with_requested_duration(
        self, g: torch.cuda.CUDAGraph, comm: TorchComm
    ) -> None:
        """Replay graph and verify it times out near TIMED_TIMEOUT."""
        start = time.monotonic()
        g.replay()
        torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - start

        elapsed_ms = int(elapsed * 1000)
        timeout_ms = int(TIMED_TIMEOUT.total_seconds() * 1000)
        print(
            f"Rank {self.rank}: graph allReduce failure "
            f"elapsed={elapsed_ms}ms timeout={timeout_ms}ms",
            flush=True,
        )
        self.assertTrue(comm.is_aborted())
        self.assertGreater(
            elapsed,
            TIMED_TIMEOUT_LOWER_BOUND.total_seconds(),
            "Replay aborted too quickly to confirm the per-call graph "
            f"timeout was used: elapsed={elapsed_ms}ms, "
            f"timeout={timeout_ms}ms",
        )
        self.assertLess(
            elapsed,
            TIMED_TIMEOUT_UPPER_BOUND.total_seconds(),
            "Replay aborted later than the per-call graph timeout window; "
            f"elapsed={elapsed_ms}ms, timeout={timeout_ms}ms",
        )

    def _run_graph_allreduce_timeout_duration(self) -> None:
        """Measure captured allreduce timeout duration on the ctdirect path."""
        self._run_graph_allreduce_with_missing_rank(
            name="graph_timeout",
            sizes=[1],
            num_successful_replays=1,
            timeout=TIMED_TIMEOUT,
            final_replay_assertion=self._assert_timed_out_with_requested_duration,
            wait_for_survivors=True,
        )

    def _test_id(self) -> str:
        return self.id().split(".")[-1]

    # --- Happy-path graph tests ---

    def test_a_single_element(self) -> None:
        """ctdirect path, 100 replays."""
        self._run_graph_allreduce(sizes=[1], num_replays=100)

    def test_b_one_mb(self) -> None:
        """ctring path (1MB), 100 replays."""
        self._run_graph_allreduce(sizes=[ONE_MB_FLOATS], num_replays=100)

    def test_c_multi_size_graph(self) -> None:
        """Multiple sizes in one graph. Mirrors C++ MultiSizeGraph."""
        self._run_graph_allreduce(sizes=[1, 32, ONE_MB_FLOATS, 8, 256], num_replays=3)

    # --- Concurrent eager + graph ---

    def test_d_graph_replay_with_concurrent_eager(self) -> None:
        """Graph replay interleaved with eager allreduce.

        Verifies pool resources (GpeKernelSync, KernelFlagItem) owned by
        the graph are not reclaimed by concurrent eager operations.
        """
        comm = self._create_comm(f"concurrent_{self._test_id()}")
        graph_tensors = [
            torch.ones(ONE_MB_FLOATS, dtype=torch.float32, device=self.device)
        ]
        eager_tensor = torch.ones(
            ONE_MB_FLOATS, dtype=torch.float32, device=self.device
        )

        self._warmup(comm, graph_tensors + [eager_tensor])
        g, stream = self._capture_graph(comm, graph_tensors)

        for round_idx in range(3):
            # Graph replay
            self._replay_and_validate(g, graph_tensors, [ONE_MB_FLOATS], self.expected)

            # Eager allreduce on default stream (exercises pool reclaim)
            eager_tensor.fill_(1.0)
            comm.all_reduce(eager_tensor, ReduceOp.SUM, async_op=True).wait()
            torch.cuda.synchronize(self.device)
            self.assertTrue(
                torch.allclose(
                    eager_tensor, torch.full_like(eager_tensor, self.expected)
                ),
                f"Eager round {round_idx} failed",
            )

        del g
        comm.finalize()

    # --- Fault-tolerant variants (last rank drops before final replay) ---

    def test_ft_single_element(self) -> None:
        """ctdirect with fault tolerance."""
        self._run_graph_allreduce_ft(sizes=[1], num_replays=3)

    def test_ft_one_mb(self) -> None:
        """ctring (1MB) with fault tolerance."""
        self._run_graph_allreduce_ft(sizes=[ONE_MB_FLOATS], num_replays=3)

    def test_ft_multi_size_graph(self) -> None:
        """Multi-size graph with fault tolerance."""
        self._run_graph_allreduce_ft(
            sizes=[1, 32, ONE_MB_FLOATS, 8, 256], num_replays=3
        )

    def test_e_timeout_duration_single_element(self) -> None:
        """ctdirect CUDA graph replay times out near the requested timeout."""
        self._run_graph_allreduce_timeout_duration()

    # --- Recovery: reconfigure and recapture ---

    def test_reconfigure_and_recapture(self) -> None:
        """Full lifecycle: capture -> replay -> drop -> reconfigure -> re-capture.

        Uses multiple sizes to exercise both ctdirect and ctring paths
        through the full reconfigure lifecycle.
        """
        test_id = self._test_id()
        comm = self._create_comm(f"reconfig_{test_id}")

        sizes = [1, 32, ONE_MB_FLOATS, 8, 256]
        tensors = [
            torch.ones(s, dtype=torch.float32, device=self.device) for s in sizes
        ]
        self._warmup(comm, tensors)

        # Phase A: capture and replay
        g1, stream = self._capture_graph(comm, tensors)
        for _ in range(3):
            self._replay_and_validate(g1, tensors, sizes, self.expected)
            self.assertFalse(comm.is_aborted())

        # Phase B: last rank drops, surviving ranks replay with missing rank.
        # tearDown() barrier ensures it waits for surviving ranks.
        if self.rank == self.world_size - 1:
            self.store.set(f"reconfig_dropped_{test_id}", "1")
            del g1
            comm.finalize()
            return

        # Surviving ranks: replay graph — allreduce times out on missing rank
        self.store.wait([f"reconfig_dropped_{test_id}"])
        for t in tensors:
            t.fill_(1.0)
        torch.cuda.synchronize(self.device)
        g1.replay()
        torch.cuda.synchronize(self.device)
        self.assertTrue(comm.is_aborted())
        del g1

        # Phase C: reconfigure with surviving ranks
        surviving_ranks = list(range(self.world_size - 1))
        new_world_size = len(surviving_ranks)
        uuid = self._next_uuid()
        key_prefix = f"reconfig_{test_id}_{uuid}"

        my_handle = comm.get_init_handle()
        self.store.set(f"{key_prefix}_handle_{self.rank}", my_handle)
        all_handles = []
        for r in surviving_ranks:
            handle = self.store.get(f"{key_prefix}_handle_{r}").decode("utf-8")
            all_handles.append(handle)

        work = comm.reconfigure(
            uuid=uuid,
            init_handles=all_handles,
            timeout=timedelta(seconds=30),
        )
        work.wait_blocking()
        self.assertFalse(comm.is_aborted())
        self.assertEqual(comm.get_size(), new_world_size)

        # Phase D: re-capture on reconfigured comm
        self._warmup(comm, tensors)
        g2, stream = self._capture_graph(comm, tensors)
        new_expected = float(new_world_size)
        for _ in range(3):
            self._replay_and_validate(g2, tensors, sizes, new_expected)
            self.assertFalse(comm.is_aborted())

        del g2
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
