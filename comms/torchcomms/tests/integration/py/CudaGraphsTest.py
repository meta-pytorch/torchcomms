# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import os
import unittest

import torch
import torchcomms


class TestCudaGraphs(unittest.TestCase):
    # pyre-ignore[56]
    @unittest.skipIf(
        os.getenv("TEST_BACKEND") != "ncclx", "Skipping AllReduce NCCLX-only tests"
    )
    def test_sync(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.accelerator.current_accelerator()

        comm = torchcomms.new_comm(backend, device, name="my_comm")
        try:
            for async_op in [False]:
                with self.subTest(async_op=async_op):
                    graph = torch.cuda.CUDAGraph()

                    inp = torch.ones(10, 10, device=device)
                    expected = inp * comm.get_size()

                    graph_inp = inp.clone()

                    with torch.cuda.graph(graph):
                        work = comm.all_reduce(
                            graph_inp, torchcomms.ReduceOp.SUM, async_op=async_op
                        )
                        if async_op:
                            work.wait()

                    for _ in range(3):
                        graph_inp.copy_(inp)
                        graph.replay()
                        torch.testing.assert_close(graph_inp, expected)

        finally:
            torch.accelerator.synchronize()
            graph.reset()
            comm.finalize()


if __name__ == "__main__":
    unittest.main()
