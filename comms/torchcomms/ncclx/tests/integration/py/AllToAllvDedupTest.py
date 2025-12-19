#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest
from dataclasses import dataclass
from typing import List

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)


@dataclass
class TestParams:
    """Data class to hold test parameters for AllToAllv dedup dispatch tests."""

    num_tokens: int
    token_numel: int
    topk: int
    num_experts: int
    dtype: torch.dtype


class AllToAllvDedupDispatchTest(unittest.TestCase):
    # Define test parameters to be combined in the test suite
    num_tokens_list = [4]
    token_numel_list = [16]
    topk_list = [2]
    num_experts_list = [64]
    dtype_list = [torch.int]

    # Global variables used for each test run
    # - store test parameters for each run_test call
    test_params: TestParams | None = None
    # - store global top-k ids across ranks
    all_ranks_topk_ids: List[torch.Tensor] | None = None

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        os.environ["NCCL_CTRAN_ENABLE"] = "1"

        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.num_local_ranks = self.num_ranks
        self.num_nodes = self.num_ranks // self.num_local_ranks
        self.device = self.torchcomm.get_device()

        # Get the NCCLX backend for NCCLX-specific APIs
        self.ncclx_backend = self.torchcomm.unsafe_get_backend()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def _reset_run_test(
        self,
        num_tokens: int,
        token_numel: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> None:
        """Reset test parameters and global top-k ids for each run_test call."""
        self.all_ranks_topk_ids = None
        self.test_params = TestParams(
            num_tokens=num_tokens,
            token_numel=token_numel,
            topk=top_k,
            num_experts=num_experts,
            dtype=dtype,
        )

    def _expert_to_rank(self, expert_id: int, num_ranks: int) -> int:
        """Map a global expert ID to the owning rank."""
        assert self.test_params is not None, "test_params must be set"
        num_experts = self.test_params.num_experts

        # experts are evenly distributed across ranks, in the order
        # of expert 0,1->rank 0, expert 2,3->rank 1, etc, if num_expert_per_rank is 2
        num_expert_per_rank = num_experts // self.num_ranks
        return expert_id // num_expert_per_rank

    def _count_num_recv_tokens(self) -> int:
        """Count total tokens this rank will receive from all peers based on expert ownership."""
        assert self.all_ranks_topk_ids is not None, "all_ranks_topk_ids must be set"
        all_ranks_topk_ids = self.all_ranks_topk_ids
        num_recv_tokens = torch.zeros(
            self.num_ranks, dtype=torch.int32, device=self.device
        )
        for rank_id in range(self.num_ranks):
            rank_topk_ids = all_ranks_topk_ids[rank_id]
            for expert_id in rank_topk_ids.flatten():
                if self._expert_to_rank(expert_id.item(), self.num_ranks) == self.rank:
                    num_recv_tokens[rank_id] += 1
        print(
            f"TEST: myRank {self.rank} num_recv_tokens from all ranks: {num_recv_tokens}"
        )
        return int(num_recv_tokens.sum().item())

    def _get_indice_map_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Return a compacted index map from mask where True positions are 0..N-1 and False are -1."""
        """N is the number of True positions."""
        nz = torch.nonzero(mask, as_tuple=False).flatten()
        indice_map = torch.full(
            (mask.size(0),), -1, device=mask.device, dtype=torch.int
        )
        indice_map[nz] = torch.arange(nz.numel(), device=mask.device, dtype=torch.int)
        return indice_map

    def _get_id_list_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Return indices of True positions in the boolean mask."""
        nz = torch.nonzero(mask, as_tuple=False).flatten()
        return nz

    def _get_block_tensor(
        self,
        tensor: torch.Tensor,
        send_rank: int,
        token_id: int,
        token_numel: int,
    ) -> torch.Tensor:
        """In-place assign tensor block with unique values based on sender rank and block ID."""
        for idx in range(token_numel):
            tensor[idx] = send_rank * 10000 + token_id * 1000 + idx
        return tensor

    def _set_global_topk_ids(self, expert_range: torch.Tensor) -> None:
        """Generate local random top-k expert IDs per token and all-gather them across ranks."""
        assert self.test_params is not None, "test_params must be set"
        num_tokens = self.test_params.num_tokens
        top_k = self.test_params.topk
        topk_ids = torch.stack(
            [
                expert_range[
                    torch.randperm(expert_range.size(0), device=self.device)[:top_k]
                ]
                for _ in range(num_tokens)
            ]
        )

        topk_ids = topk_ids.contiguous()
        all_ranks_topk_ids = [torch.empty_like(topk_ids) for _ in range(self.num_ranks)]
        self.torchcomm.all_gather(all_ranks_topk_ids, topk_ids, async_op=False)

        # Save as class member field
        self.all_ranks_topk_ids = all_ranks_topk_ids

    def _prepare_send_indices(self) -> torch.Tensor:
        """Generate send_indices for each node based on expert assignments in all_ranks_topk_ids."""
        assert self.all_ranks_topk_ids is not None, "all_ranks_topk_ids must be set"
        all_ranks_topk_ids = self.all_ranks_topk_ids

        assert self.test_params is not None, "test_params must be set"
        num_experts = self.test_params.num_experts

        send_indices = []
        num_expert_per_node = num_experts // self.num_nodes
        expert_nodes = all_ranks_topk_ids[self.rank] // num_expert_per_node
        for node in range(self.num_nodes):
            # Create a mask for tokens that have at least one expert assigned to node
            mask = (expert_nodes == node).any(dim=1)  # Shape: [num_tokens]
            indice_map = self._get_indice_map_from_mask(mask)
            send_indices.append(indice_map)
        return torch.stack(send_indices).contiguous()

    def _prepare_forward_indices(self) -> torch.Tensor:
        """Generate forward_indices for tokens received from each node."""
        assert self.all_ranks_topk_ids is not None, "all_ranks_topk_ids must be set"
        all_ranks_topk_ids = self.all_ranks_topk_ids

        assert self.test_params is not None, "test_params must be set"
        num_experts = self.test_params.num_experts

        forward_indices = []
        my_local_rank = self.rank % self.num_local_ranks
        my_node = self.rank // self.num_local_ranks
        num_expert_per_rank = num_experts // self.num_ranks
        # Iterate over ranks in the cross-node rail
        for send_rank in range(my_local_rank, self.num_ranks, self.num_local_ranks):
            node_forward_indices = []
            # Iterate over local ranks on the same node
            for recv_rank in range(
                my_node * self.num_local_ranks,
                (my_node + 1) * self.num_local_ranks,
            ):
                # Create a mask for tokens that have at least one expert assigned to recv_rank
                mask = (
                    (all_ranks_topk_ids[send_rank] // num_expert_per_rank) == recv_rank
                ).any(dim=1)
                indice_map = self._get_indice_map_from_mask(mask)
                node_forward_indices.append(indice_map)
            forward_indices.append(node_forward_indices)

        return torch.stack(
            [torch.stack(node_indices) for node_indices in forward_indices]
        ).contiguous()

    def _prepare_recv_indices(self) -> torch.Tensor:
        """Generate recv_indices for each local expert."""
        assert self.all_ranks_topk_ids is not None, "all_ranks_topk_ids must be set"
        all_ranks_topk_ids = self.all_ranks_topk_ids

        assert self.test_params is not None, "test_params must be set"
        num_experts = self.test_params.num_experts

        recv_indices = []
        num_expert_per_rank = num_experts // self.num_ranks
        # Iterate over all local experts
        for expert_id in range(
            num_expert_per_rank * self.rank, num_expert_per_rank * (self.rank + 1)
        ):
            expert_recv_indices = []
            # Iterate over all ranks
            for send_rank in range(self.num_ranks):
                # Create a mask for tokens that have at least one expert assigned to local expert
                mask = (all_ranks_topk_ids[send_rank] == expert_id).any(dim=1)
                indice_map = self._get_indice_map_from_mask(mask)
                expert_recv_indices.append(indice_map)
            recv_indices.append(expert_recv_indices)
        return torch.cat(
            [torch.stack(node_indices) for node_indices in recv_indices]
        ).contiguous()

    def _prepare_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate input tensor, output tensors, and recv_token_ids tensor
        Shapes:
          - input_tensor: [num_tokens * token_numel]
          - output_tensor: [num_recv_tokens * token_numel]
          - recv_token_ids: [num_recv_tokens]
        """
        # Count number of tokens to be received from each rank
        num_recv_tokens = self._count_num_recv_tokens()

        assert self.test_params is not None, "test_params must be set"
        num_tokens = self.test_params.num_tokens
        token_numel = self.test_params.token_numel
        dtype = self.test_params.dtype

        input_tensor = torch.arange(
            num_tokens * token_numel, dtype=dtype, device=self.device
        ).reshape(num_tokens, token_numel)
        for token_id in range(num_tokens):
            self._get_block_tensor(
                input_tensor[token_id], self.rank, token_id, token_numel
            )

        output_tensor = torch.full(
            (num_recv_tokens * token_numel,), -1, dtype=dtype, device=self.device
        )
        recv_token_ids = torch.full(
            (num_recv_tokens,), -1, dtype=torch.int32, device=self.device
        )
        return input_tensor.contiguous(), output_tensor, recv_token_ids

    def _prepare_expected_recv_token_ids(self) -> list[list[torch.Tensor]]:
        """Generate recv_token_ids for each local expert and each send rank."""
        assert self.all_ranks_topk_ids is not None, "all_ranks_topk_ids must be set"
        all_ranks_topk_ids = self.all_ranks_topk_ids

        assert self.test_params is not None, "test_params must be set"
        num_experts = self.test_params.num_experts

        expected_recv_token_ids = []
        num_expert_per_rank = num_experts // self.num_ranks
        # Iterate over all local experts and all ranks
        for expert_id in range(
            num_expert_per_rank * self.rank, num_expert_per_rank * (self.rank + 1)
        ):
            expert_token_ids = []
            for send_rank in range(self.num_ranks):
                # Create a mask for tokens that have at least one expert assigned to local expert
                mask = (all_ranks_topk_ids[send_rank] == expert_id).any(dim=1)
                rank_token_ids = self._get_id_list_from_mask(mask)
                expert_token_ids.append(rank_token_ids)
            expected_recv_token_ids.append(expert_token_ids)
        return expected_recv_token_ids

    def _prepare_expected_output_tensor(
        self, expected_recv_token_ids: list[list[torch.Tensor]]
    ) -> list[list[list[torch.Tensor]]]:
        """Generate expected output tensor for each local expert, each send rank and each token"""

        assert self.test_params is not None, "test_params must be set"
        token_numel = self.test_params.token_numel
        dtype = self.test_params.dtype

        expected_tensors = []
        for expert_token_ids in expected_recv_token_ids:
            expert_blocks = []
            for send_rank, rank_token_ids in enumerate(expert_token_ids):
                rank_blocks = []
                for _, token_id in enumerate(rank_token_ids):
                    block = torch.empty((token_numel,), dtype=dtype, device=self.device)
                    self._get_block_tensor(block, send_rank, token_id, token_numel)
                    rank_blocks.append(block)
                expert_blocks.append(rank_blocks)
            expected_tensors.append(expert_blocks)
        return expected_tensors

    def _check_exec_output(
        self, output_tensor: torch.Tensor, recv_token_ids: torch.Tensor
    ) -> None:
        """Check output tensor and recv_token_ids against expected values."""
        assert self.test_params is not None, "test_params must be set"
        token_numel = self.test_params.token_numel

        expected_recv_token_ids = self._prepare_expected_recv_token_ids()
        expected_recv_token_ids_flat = torch.cat(
            [
                torch.cat(expert_token_ids)
                for expert_token_ids in expected_recv_token_ids
            ]
        )

        # Sanity check length equal
        self.assertEqual(
            len(expected_recv_token_ids_flat),
            len(recv_token_ids),
            f"Rank {self.rank}: recv_token_ids length mismatch.\n"
            f"Expected: {len(expected_recv_token_ids_flat)}\n"
            f"Got: {len(recv_token_ids)}",
        )

        # Assert that recv_token_ids matches expected values
        self.assertTrue(
            torch.equal(recv_token_ids, expected_recv_token_ids_flat),
            f"Rank {self.rank}: recv_token_ids mismatch.\n"
            f"Expected: {expected_recv_token_ids_flat}\n"
            f"Got: {recv_token_ids}",
        )

        expected_output_tensors = self._prepare_expected_output_tensor(
            expected_recv_token_ids
        )

        # Assert that output_tensor matches expected values
        global_recv_indice = 0
        for expert_id, expert_blocks in enumerate(expected_output_tensors):
            for send_rank, rank_blocks in enumerate(expert_blocks):
                for recv_indice, expected in enumerate(rank_blocks):
                    # Get the corresponding block from output_tensor
                    output_start = global_recv_indice * token_numel
                    output_end = output_start + token_numel
                    output = output_tensor[output_start:output_end]
                    token_id = expected_recv_token_ids[expert_id][send_rank][
                        recv_indice
                    ]

                    self.assertTrue(
                        torch.equal(output, expected),
                        f"Rank {self.rank}: output_tensor block mismatch at {recv_indice=} ({expert_id=}, {send_rank=}, {token_id=}) at {global_recv_indice=}.\n"
                        f"Expected: {expected}\n"
                        f"Got: {output}",
                    )
                    global_recv_indice += 1

    def _run_test(self):
        """Run AllToAllv dedup dispatch test with given parameters."""

        print(f"TEST: {self.test_params=}")
        num_tokens = self.test_params.num_tokens
        token_numel = self.test_params.token_numel
        num_experts = self.test_params.num_experts
        top_k = self.test_params.topk
        dtype = self.test_params.dtype

        # Generate test specific expert range and generate global topk_ids mapping for all ranks
        expert_range = torch.arange(
            num_experts // self.num_nodes * 2, device=self.device
        )
        if self.rank == 0:
            print(f"TEST: {expert_range=}")
        self._set_global_topk_ids(expert_range)
        if self.rank == 0:
            print(f"TEST: {self.all_ranks_topk_ids=}")

        # Generate indices as exec input arguments:
        # - send_indices: for each node, determine which unique local tokens to send
        send_indices = self._prepare_send_indices()
        # - forward_indices: for tokens received from each node, determine which local tokens to forward
        forward_indices = self._prepare_forward_indices()
        # - recv_indices: for each local expert, determine which tokens from each send rank to receive
        recv_indices = self._prepare_recv_indices()

        # Allocate input and output tensors
        input_tensor, output_tensor, recv_token_ids = self._prepare_tensors()

        p_req = self.ncclx_backend.alltoallv_dedup_init(
            num_tokens,
            token_numel,
            top_k,
            num_experts // self.num_ranks,
            dtype,
            async_op=False,
        )
        self.assertIsNotNone(p_req)

        print(f"TEST: myRank {self.rank} send_indices: {send_indices.tolist()}")
        print(f"TEST: myRank {self.rank} forward_indices: {forward_indices.tolist()}")
        print(f"TEST: myRank {self.rank} recv_indices: {recv_indices.tolist()}")
        print(f"TEST: myRank {self.rank} input_tensor: {input_tensor.tolist()}")

        work = self.ncclx_backend.alltoallv_dedup_exec(
            output_tensor,
            recv_token_ids,
            input_tensor,
            send_indices,
            forward_indices,
            recv_indices,
            p_req,
        )
        work.wait()

        print(f"TEST: myRank {self.rank} output_tensor: {output_tensor.tolist()}")
        print(f"TEST: myRank {self.rank} recv_token_ids: {recv_token_ids.tolist()}")

        self._check_exec_output(output_tensor, recv_token_ids)

        # TODO: add combine once filled the implementation

    def test_dispatch_combine(self):
        """Run dispatch-combine round-trip test with all parameter combinations."""
        # Nested loops for all parameter combinations
        for num_tokens, token_numel, num_experts, top_k, dtype in itertools.product(
            self.num_tokens_list,
            self.token_numel_list,
            self.num_experts_list,
            self.topk_list,
            self.dtype_list,
        ):
            # Create a descriptive test name for better test output
            test_name = f"nTokens_{num_tokens}_nExperts_{num_experts}_topK_{top_k}_{get_dtype_name(dtype)}"
            print(f"Running dispatch-combine test with parameters: {test_name}")
            self._reset_run_test(num_tokens, token_numel, num_experts, top_k, dtype)

            # Run sync alltoallv_dedup with combined parameters
            self._run_test()


if __name__ == "__main__":
    unittest.main()
