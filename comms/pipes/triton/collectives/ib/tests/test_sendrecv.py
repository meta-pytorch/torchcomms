# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Correctness tests for Triton pipelined sendrecv.

Tests bidirectional sendrecv between 2 GPUs using torchcomms window API.
"""

import os
import socket

# Enable GIN (GPU-Initiated Networking) for device-side window operations
os.environ.setdefault("NCCL_GIN_ENABLE", "1")
os.environ.setdefault("NCCL_GIN_TYPE", "-1")
# Disable P2P to force IB transport even on single-node
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# Large QP depth for parallel kernel (many WQEs per step)
os.environ.setdefault("NCCL_GIN_GDAKI_QP_DEPTH", "1024")

import torch
import torch.multiprocessing as mp
import torchcomms
from comms.pipes.triton.collectives.ib.sendrecv_op import SendRecvOp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Test configurations: (name, msg_bytes, section_bytes, pipeline_depth, num_blocks)
TEST_CONFIGS = [
    ("medium_64KB", 64 * 1024, 32 * 1024, 2, 2),
    ("medium_256KB", 256 * 1024, 64 * 1024, 4, 4),
    ("large_1MB", 1024 * 1024, 256 * 1024, 4, 8),
    ("large_4MB", 4 * 1024 * 1024, 1024 * 1024, 4, 16),
    ("large_16MB", 16 * 1024 * 1024, 4 * 1024 * 1024, 4, 32),
]


def run_test_worker(local_rank, master_port):
    """Worker function for each GPU process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = "2"

    peer_rank = 1 - local_rank
    torch.cuda.set_device(local_rank)

    comm = torchcomms.new_comm(
        "ncclx",
        torch.device(f"cuda:{local_rank}"),
        name="sendrecv_test",
    )

    passed = 0
    failed = 0

    for name, msg_bytes, section_bytes, pd, nblocks in TEST_CONFIGS:
        try:
            total_elements = msg_bytes // 4  # float32

            # Create known data pattern: rank-specific values
            src = torch.full(
                (total_elements,),
                float(local_rank + 1),
                dtype=torch.float32,
                device=f"cuda:{local_rank}",
            )
            dst = torch.zeros(
                total_elements,
                dtype=torch.float32,
                device=f"cuda:{local_rank}",
            )

            # Create SendRecvOp
            op = SendRecvOp(comm, pd, section_bytes, nblocks)

            # Execute sendrecv
            op(src, dst, peer_rank)
            torch.cuda.synchronize()
            comm.barrier(False)

            # Verify: dst should contain peer's data
            expected_value = float(peer_rank + 1)
            expected = torch.full_like(dst, expected_value)

            if torch.allclose(dst, expected):
                if local_rank == 0:
                    print(f"  PASS: {name} ({msg_bytes} bytes)")
                passed += 1
            else:
                mismatch = (dst != expected).sum().item()
                if local_rank == 0:
                    print(
                        f"  FAIL: {name} — {mismatch}/{total_elements} elements wrong"
                    )
                    print(
                        f"    Expected: {expected_value}, "
                        f"Got first 10: {dst[:10].tolist()}"
                    )
                failed += 1

            op.teardown()
            del op, src, dst
            torch.cuda.empty_cache()

        except Exception as e:
            if local_rank == 0:
                print(f"  ERROR: {name} — {e}")
            failed += 1
            comm.barrier(False)

    if local_rank == 0:
        print(
            f"\nResults: {passed} passed, {failed} failed "
            f"out of {len(TEST_CONFIGS)} tests"
        )

    comm.finalize()


if __name__ == "__main__":
    port = find_free_port()
    print("Running Triton SendRecv Correctness Tests")
    print("=" * 50)
    mp.spawn(run_test_worker, args=(port,), nprocs=2, join=True)
