#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Model-shaped TorchComm registered all-reduce example for RCCLX."""

import torch
from torchcomms import new_comm, ReduceOp


ROWS = 64
INNER = 8192
WIDTH = 8192


def assert_value(tensor: torch.Tensor, value: float, what: str) -> None:
    expected = torch.full_like(tensor, value)
    torch.testing.assert_close(tensor, expected, rtol=0, atol=0, msg=what)


def main() -> None:
    device = torch.device("hip")
    comm = new_comm("rcclx", device, name="registered_all_reduce_example")
    rank = comm.get_rank()
    world_size = comm.get_size()

    if world_size != 4:
        raise RuntimeError(
            f"RegisteredAllReduce v1 requires TP4; launched {world_size} ranks"
        )

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("RegisteredAllReduce requires a CUDA/HIP device")
    target_device = torch.device(f"cuda:{rank % device_count}")
    torch.cuda.set_device(target_device)

    # This is the 1 MiB [64, 8192] BF16 buffer used by the 42B decode path.
    # The producer writes directly into the registered address, as attention and
    # MoE projections do before invoking AITER CustomAllreduce.
    activation = torch.empty((ROWS, INNER), dtype=torch.bfloat16, device=target_device)
    weight = torch.full(
        (INNER, WIDTH),
        1.0 / INNER,
        dtype=torch.bfloat16,
        device=target_device,
    )
    registered_input = torch.empty(
        (ROWS, WIDTH), dtype=torch.bfloat16, device=target_device
    )
    persistent_output = torch.empty_like(registered_input)
    first_output = torch.empty_like(registered_input)
    second_output = torch.empty_like(registered_input)

    request = comm.registered_all_reduce(
        registered_input,
        persistent_output,
        capacity_bytes=registered_input.nbytes,
    )
    execution_stream = torch.cuda.Stream(device=target_device)
    execution_stream.wait_stream(torch.cuda.current_stream(target_device))
    graph: torch.cuda.CUDAGraph | None = None
    try:
        # Queue two producer-plus-collective calls on the request's one stream
        # without an intermediate host synchronization. The snapshots prove
        # that the next call's start handshake protects the prior call's
        # scratch before its producer overwrites the registered input.
        with torch.cuda.stream(execution_stream):
            activation.fill_(float(rank + 1))
            torch.mm(activation, weight, out=registered_input)
            request.all_reduce(
                registered_input,
                op=ReduceOp.SUM,
                out=persistent_output,
                registered_input=True,
            )
            first_output.copy_(persistent_output)

            activation.fill_(float(2 * (rank + 1)))
            torch.mm(activation, weight, out=registered_input)
            request.all_reduce(
                registered_input,
                op=ReduceOp.SUM,
                out=persistent_output,
                registered_input=True,
            )
            second_output.copy_(persistent_output)

        execution_stream.synchronize()
        rank_sum = float(world_size * (world_size + 1) // 2)
        assert_value(first_output, rank_sum, "first back-to-back output")
        assert_value(second_output, 2 * rank_sum, "second back-to-back output")

        # Capture the 42B pattern on that same stream: a producer writes
        # directly into the registered input, followed immediately by
        # registered all-reduce into a persistent output. Replay reads changed
        # producer operands without changing either registered address.
        with torch.cuda.stream(execution_stream):
            torch.mm(activation, weight, out=registered_input)
        execution_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=execution_stream):
            torch.mm(activation, weight, out=registered_input)
            request.all_reduce(
                registered_input,
                op=ReduceOp.SUM,
                out=persistent_output,
                registered_input=True,
            )

        activation.fill_(float(3 * (rank + 1)))
        execution_stream.wait_stream(torch.cuda.current_stream(target_device))
        with torch.cuda.stream(execution_stream):
            graph.replay()
        execution_stream.synchronize()
        assert_value(persistent_output, 3 * rank_sum, "first graph replay output")

        # Leave the final replay queued. Graph teardown happens first, then
        # close() synchronizes the pinned execution stream and proves final-call
        # global quiescence before releasing peer mappings.
        activation.fill_(float(4 * (rank + 1)))
        execution_stream.wait_stream(torch.cuda.current_stream(target_device))
        with torch.cuda.stream(execution_stream):
            graph.replay()
        graph.reset()
        graph = None
        request.close()
        assert_value(persistent_output, 4 * rank_sum, "final graph replay output")

        if rank == 0:
            print(
                "registered all-reduce validated back-to-back eager calls "
                "and model-shaped graph replay"
            )
    finally:
        if graph is not None:
            graph.reset()
        if not request.closed:
            request.close()
        comm.finalize()


if __name__ == "__main__":
    main()
