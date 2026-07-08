# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import datetime
import os
import unittest
from typing import cast

import torch
import torch.distributed as dist
from comms.ncclx.pg import ProcessGroupNCCLX, register_ncclx_backend


def get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class AllGatherPTest(unittest.TestCase):
    """Test persistent allgather (allgather_p) collective operations using NCCLX."""

    def setUp(self):
        # These env vars are also set by the BUCK configuration; setdefault
        # keeps them as a fallback for standalone/manual runs.
        os.environ.setdefault("NCCL_CTRAN_ENABLE", "1")
        os.environ.setdefault("NCCL_ALLGATHER_P_ALGO", "ctpipeline")
        os.environ.setdefault("NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET", "1")

        register_ncclx_backend(devices=["cuda"])

        self.local_rank = get_env_int("LOCAL_RANK", 0)
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        dist.init_process_group(
            backend="ncclx",
            timeout=datetime.timedelta(seconds=60),
            device_id=self.device,
        )

        pg = dist.distributed_c10d._get_default_group()
        self.ncclx: ProcessGroupNCCLX = cast(
            ProcessGroupNCCLX, pg._get_backend(self.device)
        )
        self.pool = torch.cuda.MemPool(self.ncclx.mem_allocator)
        self.ncclx.register_mem_pool(self.pool)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def tearDown(self):
        self.ncclx.deregister_mem_pool(self.pool)
        dist.barrier()
        dist.destroy_process_group()

    def _run_allgather_p(self, buffer_size: int) -> None:
        """Run a single persistent allgather and verify correctness."""
        with torch.cuda.use_mem_pool(self.pool):
            op_tensor = torch.empty(
                [buffer_size], dtype=torch.uint8, device=self.device
            )
        # Zero-fill so non-gathered regions act as a canary for partial writes
        op_tensor.fill_(0)

        p_req = self.ncclx._allgather_p_init(op_tensor)

        rank_numel = op_tensor.numel() // self.world_size
        self.assertEqual(
            op_tensor.numel() % self.world_size,
            0,
            "Buffer size must be divisible by world size",
        )

        # Fill this rank's slice with its rank value
        ip_tensor = op_tensor[rank_numel * self.rank : rank_numel * (self.rank + 1)]
        ip_tensor.fill_(self.rank)

        work = self.ncclx._allgather_p(op_tensor, ip_tensor, p_req)
        work.wait()
        torch.cuda.synchronize()

        # Verify: each rank's slice should contain that rank's value
        for r in range(self.world_size):
            chunk = op_tensor[rank_numel * r : rank_numel * (r + 1)]
            expected = torch.full_like(chunk, r)
            torch.testing.assert_close(
                chunk,
                expected,
                msg=f"Rank {self.rank}: slice for rank {r} has wrong values",
            )

    def test_allgather_p_basic(self):
        """Test persistent allgather with various buffer sizes."""
        buffer_sizes = [
            self.world_size,  # 1 byte per rank (minimum)
            self.world_size * 1000,  # non-power-of-2 to catch alignment assumptions
            self.world_size * 1024,  # 1 KiB per rank
            self.world_size * 1024 * 1024,  # 1 MiB per rank
        ]
        for buffer_size in buffer_sizes:
            with self.subTest(buffer_size=buffer_size):
                self._run_allgather_p(buffer_size)

    def test_allgather_p_multiple_iterations(self):
        """Test persistent allgather across multiple iterations to verify reuse."""
        buffer_size = self.world_size * 1024 * 1024  # 1 MiB per rank

        with torch.cuda.use_mem_pool(self.pool):
            op_tensor = torch.empty(
                [buffer_size], dtype=torch.uint8, device=self.device
            )
        # Zero-fill so non-gathered regions act as a canary for partial writes
        op_tensor.fill_(0)

        p_req = self.ncclx._allgather_p_init(op_tensor)
        rank_numel = op_tensor.numel() // self.world_size

        num_iterations = 3
        for iteration in range(num_iterations):
            with self.subTest(iteration=iteration):
                # Use iteration-dependent fill value to verify each round
                fill_val = (self.rank + iteration) % 256
                ip_tensor = op_tensor[
                    rank_numel * self.rank : rank_numel * (self.rank + 1)
                ]
                ip_tensor.fill_(fill_val)

                work = self.ncclx._allgather_p(op_tensor, ip_tensor, p_req)
                work.wait()
                torch.cuda.synchronize()

                for r in range(self.world_size):
                    expected_val = (r + iteration) % 256
                    chunk = op_tensor[rank_numel * r : rank_numel * (r + 1)]
                    expected = torch.full_like(chunk, expected_val)
                    torch.testing.assert_close(
                        chunk,
                        expected,
                        msg=f"Rank {self.rank}, iteration {iteration}: "
                        f"slice for rank {r} expected {expected_val}",
                    )


if __name__ == "__main__":
    unittest.main()
