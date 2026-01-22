# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Functional collectives implementation for torchcomms."""

from datetime import timedelta

import torch

# Import TorchComm, TorchCommWindow, and BatchSendRecv
from torchcomms._comms import BatchSendRecv, ReduceOp, TorchComm, TorchCommWindow

from torchcomms.functional.registry import finalize_registration, register_collective


# Only create library if not already registered
try:
    lib: torch.library.Library | None = torch.library.Library("torchcomms", "DEF")
except RuntimeError:
    # Already registered (e.g., in forked process)
    lib = None

if lib is not None:
    # Maps tensor data_ptr to work handle for async collectives
    _TENSOR_TO_WORK: dict[int, object] = {}

    def _register_tensor_work(tensor: torch.Tensor, work: object) -> None:
        """Register a work handle for a tensor (only during eager execution)."""
        if not torch.compiler.is_compiling():
            _TENSOR_TO_WORK[tensor.data_ptr()] = work
        return None

    def _get_tensor_work(tensor: torch.Tensor) -> object | None:
        """Get and remove the work handle for a tensor."""
        if not torch.compiler.is_compiling():
            return _TENSOR_TO_WORK.pop(tensor.data_ptr(), None)
        return None

    # === INPLACE VERSION: torchcomm_wait_tensors_ ===
    # Marks all tensors as mutated to create data dependencies.
    # Returns the input tensors (like native PyTorch in-place ops) to enable
    # proper mutation tracking in functionalization.
    lib.define(
        "torchcomm_wait_tensors_(Tensor(a!)[] inputs) -> Tensor(a!)[]",
        tags=[torch.Tag.pt2_compliant_tag],
    )

    @torch.library.impl(lib, "torchcomm_wait_tensors_", "Meta")
    def _wait_tensors_inplace_meta(
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Return the input tensors (they are the same tensors, just aliased)
        return inputs

    @torch.library.impl(lib, "torchcomm_wait_tensors_", "CompositeExplicitAutograd")
    def _wait_tensors_inplace_eager(
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Wait on the first tensor's work handle (all tensors share the same work)
        if inputs:
            work = _get_tensor_work(inputs[0])
            if work is not None:
                work.wait()  # type: ignore[attr-defined]
        # Return the input tensors
        return inputs

    # === FUNCTIONAL VERSION: torchcomm_wait_tensors ===
    # Takes tensors, waits, returns them (creates data dependency via return value).
    lib.define(
        "torchcomm_wait_tensors(Tensor[] inputs) -> Tensor[]",
        tags=[torch.Tag.pt2_compliant_tag],
    )

    @torch.library.impl(lib, "torchcomm_wait_tensors", "Meta")
    def _wait_tensors_functional_meta(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        # Return empty_like with requires_grad preserved for autograd tracking
        return [torch.empty_like(t, requires_grad=t.requires_grad) for t in inputs]

    @torch.library.impl(lib, "torchcomm_wait_tensors", "CompositeExplicitAutograd")
    def _wait_tensors_functional_eager(
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Wait first, then clone to get tensors with completed data
        if inputs:
            work = _get_tensor_work(inputs[0])
            if work is not None:
                work.wait()  # type: ignore[attr-defined]
        # Clone AFTER wait so clones have the completed data
        return inputs
        # return [t.clone() for t in inputs]

    # === FUNCTIONALIZE IMPL FOR INPLACE WAIT ===
    # Register py_functionalize_impl to swap inplace for functional
    # and wrap with with_effects for proper effect token tracking
    def _wait_tensors_functionalize_impl(
        ctx, inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

        # Unwrap tensors to get raw tensors
        unwrapped_args = ctx.unwrap_tensors(inputs)

        # Get the functional op
        functional_op = torch.ops.torchcomms.torchcomm_wait_tensors

        # Check if we have access to the mode's tokens for effects
        assert (
            isinstance(ctx, PythonFunctionalizeAPI)
            and hasattr(ctx, "mode")
            and ctx.mode is not None
            and hasattr(ctx.mode, "_tokens")
        )

        from torch._higher_order_ops.effects import handle_effects, has_effects

        assert has_effects(functional_op.default)

        # Use handle_effects with our mode's tokens
        with ctx.redispatch_to_next():
            return handle_effects(
                ctx.mode._allow_token_discovery,
                ctx.mode._tokens,
                functional_op.default,
                (unwrapped_args,),  # Pass the list as a single argument
                {},
            )

    # Register the py_functionalize_impl
    torch.ops.torchcomms.torchcomm_wait_tensors_.default.py_functionalize_impl(
        _wait_tensors_functionalize_impl
    )

    def _get_tensor_meta(
        window: TorchCommWindow,  # actually a FakeScriptObject
        rank: int,
    ) -> torch.Tensor:
        # Get buffer shape, dtype, and device from the window object
        # These are registered as constant methods -- dynamo can trace through them
        return torch.empty(
            window.shape,
            dtype=window.dtype,
            device=window.device,
        )

    # Collective Registrations
    from torch._library.opaque_object import MemberType, register_opaque_type

    from torchcomms.functional.param_parsing import ParamKind, ParamSpec

    # Register TorchComm as opaque type with constant methods
    register_opaque_type(
        TorchComm,
        typ="reference",
        members={
            "get_rank": MemberType.USE_REAL,
            "get_size": MemberType.USE_REAL,
            "get_name": MemberType.USE_REAL,
            "get_device": MemberType.USE_REAL,
        },
    )

    # Register TorchCommWindow as opaque type with constant methods
    register_opaque_type(
        TorchCommWindow,
        typ="reference",
        members={
            "get_size": MemberType.USE_REAL,
            "dtype": MemberType.USE_REAL,
            "shape": MemberType.USE_REAL,
            "device": MemberType.USE_REAL,
        },
    )

    register_collective(
        TorchComm,
        TorchComm.all_reduce,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        TorchComm,
        TorchComm.reduce,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor),
            ParamSpec("root", ParamKind.EXTRA, int),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        TorchComm,
        TorchComm.broadcast,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor),
            ParamSpec("root", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        TorchComm,
        TorchComm.barrier,
        param_specs=[
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        TorchComm,
        TorchComm.all_gather,
        param_specs=[
            ParamSpec("tensor_list", ParamKind.INPUT, list[torch.Tensor], mutable=True, write_only=True),
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        TorchCommWindow,
        TorchCommWindow.put,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("dst_rank", ParamKind.EXTRA, int),
            ParamSpec("target_disp", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ],
    )

    register_collective(
        TorchCommWindow,
        TorchCommWindow.signal,
        param_specs=[
            ParamSpec("dst_rank", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ],
    )

    register_collective(
        TorchCommWindow,
        TorchCommWindow.wait_signal,
        param_specs=[
            ParamSpec("peer_rank", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ],
    )

    register_collective(
        TorchCommWindow,
        TorchCommWindow.map_remote_tensor,
        param_specs=[
            ParamSpec("result", ParamKind.OUTPUT, torch.Tensor),
            ParamSpec("rank", ParamKind.EXTRA, int),
        ],
        meta_fn=_get_tensor_meta,
    )

    register_collective(
        TorchComm,
        TorchComm.send,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("dst", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        TorchComm,
        TorchComm.recv,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("src", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # all_gather_single: gather tensor from all ranks into a single output tensor
    # (equivalent to all_gather_into_tensor in c10d)
    register_collective(
        TorchComm,
        TorchComm.all_gather_single,
        param_specs=[
            ParamSpec("output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # all_gather_v: variable-size all_gather
    register_collective(
        TorchComm,
        TorchComm.all_gather_v,
        param_specs=[
            ParamSpec("tensor_list", ParamKind.INPUT, list[torch.Tensor], mutable=True, write_only=True),
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # reduce_scatter_single: reduce then scatter to a single output tensor per rank
    # (equivalent to reduce_scatter_tensor in c10d)
    register_collective(
        TorchComm,
        TorchComm.reduce_scatter_single,
        param_specs=[
            ParamSpec("output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # reduce_scatter_v: variable-size reduce_scatter
    register_collective(
        TorchComm,
        TorchComm.reduce_scatter_v,
        param_specs=[
            ParamSpec("output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("input_list", ParamKind.INPUT, list[torch.Tensor], mutable=False),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # all_to_all_single: all-to-all with single input/output tensors
    register_collective(
        TorchComm,
        TorchComm.all_to_all_single,
        param_specs=[
            ParamSpec("output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # all_to_all_v_single: variable-size all-to-all with single input/output tensors
    register_collective(
        TorchComm,
        TorchComm.all_to_all_v_single,
        param_specs=[
            ParamSpec("output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("output_split_sizes", ParamKind.EXTRA, list[int]),
            ParamSpec("input_split_sizes", ParamKind.EXTRA, list[int]),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # all_to_all: all-to-all with tensor lists
    register_collective(
        TorchComm,
        TorchComm.all_to_all,
        param_specs=[
            ParamSpec(
                "output_tensor_list", ParamKind.INPUT, list[torch.Tensor], mutable=True, write_only=True
            ),
            ParamSpec(
                "input_tensor_list", ParamKind.INPUT, list[torch.Tensor], mutable=False
            ),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # reduce_scatter: reduce then scatter with tensor lists
    register_collective(
        TorchComm,
        TorchComm.reduce_scatter,
        param_specs=[
            ParamSpec("output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("input_list", ParamKind.INPUT, list[torch.Tensor], mutable=False),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # scatter: scatter from root to all ranks
    register_collective(
        TorchComm,
        TorchComm.scatter,
        param_specs=[
            ParamSpec("output_tensor", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec(
                "input_tensor_list", ParamKind.INPUT, list[torch.Tensor], mutable=False
            ),
            ParamSpec("root", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    # gather: gather from all ranks to root
    register_collective(
        TorchComm,
        TorchComm.gather,
        param_specs=[
            ParamSpec(
                "output_tensor_list", ParamKind.INPUT, list[torch.Tensor], mutable=True, write_only=True
            ),
            ParamSpec("input_tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("root", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )

    register_collective(
        BatchSendRecv,
        BatchSendRecv.send,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("dst", ParamKind.EXTRA, int),
        ],
    )

    register_collective(
        BatchSendRecv,
        BatchSendRecv.recv,
        param_specs=[
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True),
            ParamSpec("src", ParamKind.EXTRA, int),
        ],
    )

    """
    we don't currently support async ops that don't take mutable inputs since there is no way
    to thread the data dependencies through the work object.

    e.g.,

    batch.send(t0, peer0)
    batch.recv(t1, peer1)

    batch.issue()

    print(t1)

    ---> there is no way to ensure that batch.issue isn't reordered after the print, since it doesn't take a
         dependency or create a new value (like window.get_tensor).

    register_collective(
        BatchSendRecv,
        BatchSendRecv.issue,
        param_specs=[
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
    )
    """

    finalize_registration(lib)
