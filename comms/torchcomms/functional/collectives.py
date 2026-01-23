# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Functional collectives implementation for torchcomms."""

import threading
from datetime import timedelta

import torch

# Import TorchComm, TorchCommWindow, and BatchSendRecv
from torchcomms._comms import BatchSendRecv, ReduceOp, TorchComm, TorchCommWindow
from torchcomms.functional.registry import finalize_registration, register_collective


# === Coalescing Context Tracking ===
# Thread-local storage for the active CoalescingManager.
# This enables torch.compile fullgraph support by tracking tensors at the Python
# level during both eager execution and tracing.
# We store a reference to the manager so tensors are associated directly with it.
_coalescing_context = threading.local()


def _start_coalescing_context(manager: "CoalescingManager") -> None:  # noqa: F821
    """Start tracking coalesced tensors with the given manager.

    Args:
        manager: The CoalescingManager to register tensors with.

    Raises:
        RuntimeError: If a coalescing context is already active (no nesting allowed).
    """
    if getattr(_coalescing_context, "manager", None) is not None:
        raise RuntimeError(
            "Nested coalescing blocks are not supported. "
            "A coalescing context is already active."
        )
    _coalescing_context.manager = manager


def _end_coalescing_context() -> None:
    """End coalescing tracking."""
    _coalescing_context.manager = None


def _is_coalescing_active() -> bool:
    """Check if we're in a coalescing context."""
    return getattr(_coalescing_context, "manager", None) is not None


def _get_coalescing_manager() -> "CoalescingManager | None":  # noqa: F821
    """Get the active CoalescingManager, or None if not in a coalescing context."""
    return getattr(_coalescing_context, "manager", None)


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
        """Register a work handle for a tensor (only during eager execution).

        Also tracks tensors during coalescing mode for torch.compile fullgraph support.
        """
        # Track tensor if we're in coalescing mode (works during tracing too)
        # Append directly to the manager's tensors list
        manager = _get_coalescing_manager()
        if manager is not None:
            manager.tensors.append(tensor)

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

    # === AUTOGRAD FOR WAIT_TENSORS ===
    # Wait is just synchronization - gradients pass through unchanged.
    def _wait_tensors_setup_context(ctx, inputs, output):
        # No context needed - wait is identity for gradients
        pass

    def _wait_tensors_backward(ctx, *grad_outputs):
        # Identity backward - return gradients unchanged
        # grad_outputs contains the gradient list as passed by supports_tensorlist wrapper
        # For wait_tensors(Tensor[] inputs), grad_outputs = ([grad1, grad2, ...],)
        # We return it as-is since it's already the right structure
        return grad_outputs

    # Register autograd for the functional wait_tensors
    torch.library.register_autograd(
        "torchcomms::torchcomm_wait_tensors",
        _wait_tensors_backward,
        setup_context=_wait_tensors_setup_context,
        lib=lib,
    )

    # Register autograd kernel for the inplace wait_tensors_ using make_autograd_impl
    from torch._library import autograd as library_autograd

    inplace_wait_info = library_autograd.Info(
        _backward_fn=_wait_tensors_backward,
        _setup_context_fn=_wait_tensors_setup_context,
    )

    inplace_wait_autograd_kernel = library_autograd.make_autograd_impl(
        torch.ops.torchcomms.torchcomm_wait_tensors_.default, inplace_wait_info
    )
    lib.impl(
        "torchcomm_wait_tensors_",
        inplace_wait_autograd_kernel,
        "Autograd",
        with_keyset=True,
    )

    def _end_coalescing_meta(
        comm: TorchComm,
        async_op: bool,
    ) -> torch.Tensor:
        return torch.empty(0, device=comm.get_device(), dtype=torch.int)

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

    # ==========================================================================
    # Autograd support for collectives using torch.library.register_autograd
    # ==========================================================================
    #
    # Each collective can define:
    # - setup_context_fn(ctx, inputs, output): Save what's needed for backward
    # - backward_fn(ctx, grad_output, ...): Compute gradients
    #
    # For functional ops, backward_fn receives:
    # - ctx: Context object with saved values
    # - grad_output: Gradient of the output tensor
    # And returns a tuple with one gradient per input (None for non-tensor inputs)

    def _all_reduce_setup_context(ctx, inputs, output):
        """Save context for all_reduce backward.

        For functional version (all tensors mutable, so all included):
        - inputs: (comm, tensor, op, async_op, hints, timeout)
        - output: The reduced tensor
        """
        ctx.comm = inputs[0]
        ctx.reduce_op = inputs[2]

    def _all_reduce_backward(ctx, grad_output):
        """Backward for all_reduce.

        all_reduce is its own gradient - if y = all_reduce(x, sum),
        then grad_x = all_reduce(grad_y, sum).

        Returns only tensor gradients - the wrapper expands to full tuple.
        """
        # Perform all_reduce on the gradient
        grad_input = grad_output.contiguous()
        work = ctx.comm.all_reduce(grad_input, ctx.reduce_op, async_op=True)
        return _maybe_wrap_tensor(grad_input, work)

    def _all_gather_single_setup_context(ctx, inputs, output):
        """Save context for all_gather_single backward.

        For functional version (all input params included):
        - inputs: (comm, output, input, async_op, hints, timeout)
        - output: The gathered output tensor (returned from functional op)
        """
        ctx.comm = inputs[0]
        ctx.group_size = inputs[0].get_size()

    def _all_gather_single_backward(ctx, grad_output):
        """Backward for all_gather_single.

        all_gather backward is reduce_scatter.

        Returns only input tensor gradients - the wrapper expands to full tuple.
        Non-mutable tensor input: input -> returns grad_input
        """
        grad_output = grad_output.contiguous()

        # Create output for reduce_scatter
        out_size = list(grad_output.size())
        out_size[0] //= ctx.group_size
        grad_input = grad_output.new_empty(out_size)

        work = ctx.comm.reduce_scatter_single(
            grad_input, grad_output, ReduceOp.SUM, async_op=True
        )
        return _maybe_wrap_tensor(grad_input, work)

    def _reduce_scatter_single_setup_context(ctx, inputs, output):
        """Save context for reduce_scatter_single backward.

        For functional version (all input params included):
        - inputs: (comm, output, input, op, async_op, hints, timeout)
        - output: The scattered output tensor (returned from functional op)
        """
        ctx.comm = inputs[0]
        ctx.group_size = ctx.group_size = inputs[0].get_size()

    def _reduce_scatter_single_backward(ctx, grad_output):
        """Backward for reduce_scatter_single.

        reduce_scatter backward is all_gather.

        Returns only input tensor gradients - the wrapper expands to full tuple.
        Non-mutable tensor input: input -> returns grad_input
        """
        grad_output = grad_output.contiguous()

        # Create output for all_gather
        out_size = list(grad_output.size())
        out_size[0] *= ctx.group_size
        grad_input = grad_output.new_empty(out_size)

        work = ctx.comm.all_gather_single(grad_input, grad_output, async_op=True)
        return _maybe_wrap_tensor(grad_input, work)

    def _all_to_all_single_setup_context(ctx, inputs, output):
        """Save context for all_to_all_single backward.

        For functional version (all input params included):
        - inputs: (comm, output, input, async_op, hints, timeout)
        - output: The all_to_all output tensor (returned from functional op)
        """
        ctx.comm = inputs[0]

    def _all_to_all_single_backward(ctx, grad_output):
        """Backward for all_to_all_single.

        all_to_all is its own inverse - just call all_to_all on the gradient.

        Returns only input tensor gradients - the wrapper expands to full tuple.
        Non-mutable tensor input: input -> returns grad_input
        """
        grad_output = grad_output.contiguous()

        # Create output for all_to_all (same shape as input grad)
        grad_input = grad_output.new_empty(grad_output.size())

        work = ctx.comm.all_to_all_single(grad_input, grad_output, async_op=True)
        return _maybe_wrap_tensor(grad_input, work)

    def _scatter_setup_context(ctx, inputs, output):
        """Save context for scatter backward.

        For functional version:
        - inputs: (comm, output_tensor, input_tensor_list, root, async_op, hints, timeout)
        - output: The scattered output tensor
        """
        ctx.comm = inputs[0]
        ctx.root = inputs[3]
        ctx.group_size = inputs[0].get_size()

    def _scatter_backward(ctx, grad_output):
        """Backward for scatter.

        scatter backward is gather - gradients flow from all ranks back to root.

        Returns gradients for input_tensor_list (only meaningful on root rank).
        """
        grad_output = grad_output.contiguous()

        # Create output list for gather (only root needs actual tensors)
        rank = ctx.comm.get_rank()
        if rank == ctx.root:
            grad_input_list = [
                grad_output.new_empty(grad_output.size()) for _ in range(ctx.group_size)
            ]
        else:
            grad_input_list = []

        work = ctx.comm.gather(grad_input_list, grad_output, ctx.root, async_op=True)
        # Return wrapped list for root, empty list for others
        if rank == ctx.root:
            return [_maybe_wrap_tensor(t, work) for t in grad_input_list]
        return grad_input_list

    def _gather_setup_context(ctx, inputs, output):
        """Save context for gather backward.

        For functional version:
        - inputs: (comm, output_tensor_list, input_tensor, root, async_op, hints, timeout)
        - output: The gathered output tensor list
        """
        ctx.comm = inputs[0]
        ctx.root = inputs[3]
        ctx.input_shape = inputs[2].shape

    def _gather_backward(ctx, grad_outputs):
        """Backward for gather.

        gather backward is scatter - gradients flow from root to all ranks.

        Returns gradient for input_tensor (the only non-mutable tensor input).
        """
        rank = ctx.comm.get_rank()

        # grad_list contains gradients for each tensor in output_tensor_list
        # Only root has meaningful gradients
        if rank == ctx.root:
            grad_input_list = [g.contiguous() for g in grad_outputs]
        else:
            grad_input_list = []

        assert (
            len(grad_outputs) > 0
        ), "gather outputs cannot be empty if you want to run backward!"

        # Create output tensor for scatter
        grad_input = torch.empty(
            ctx.input_shape,
            dtype=grad_outputs[0].dtype,
            device=ctx.comm.get_device(),
        )

        work = ctx.comm.scatter(grad_input, grad_input_list, ctx.root, async_op=True)
        return _maybe_wrap_tensor(grad_input, work)

    def _reduce_scatter_setup_context(ctx, inputs, output):
        """Save context for reduce_scatter backward.

        For functional version:
        - inputs: (comm, output, input_list, op, async_op, hints, timeout)
        - output: The scattered output tensor
        """
        ctx.comm = inputs[0]
        ctx.input_shapes = [t.shape for t in inputs[2]]

    def _reduce_scatter_backward(ctx, grad_output):
        """Backward for reduce_scatter.

        reduce_scatter: output_r = sum over all ranks of input_list[r]
        For local loss on rank r: only input_list[r] on this rank contributes
        to this rank's output, so gradient only flows to input_list[rank].

        Returns gradients for input_list (list of tensors).
        """
        grad_output = grad_output.contiguous()
        rank = ctx.comm.get_rank()

        # Only the rank-th chunk gets the gradient, rest are zeros
        grad_input_list = []
        for i, shape in enumerate(ctx.input_shapes):
            if i == rank:
                grad_input_list.append(grad_output)
            else:
                grad_input_list.append(
                    torch.zeros(
                        shape, dtype=grad_output.dtype, device=grad_output.device
                    )
                )

        return grad_input_list

    def _all_gather_setup_context(ctx, inputs, output):
        """Save context for all_gather backward.

        For functional version:
        - inputs: (comm, tensor_list, tensor, async_op, hints, timeout)
        - output: The gathered tensor list
        """
        ctx.comm = inputs[0]
        ctx.rank = inputs[0].get_rank()

    def _all_gather_backward(ctx, grad_outputs):
        """Backward for all_gather.

        The input tensor from this rank appears at position rank in the output tensor_list.
        So the gradient for the input is just the gradient at that position.

        Returns gradient for tensor (the non-mutable input).
        """
        # This rank's input was placed at output[rank], so return that gradient
        return grad_outputs[ctx.rank]

    def _broadcast_setup_context(ctx, inputs, output):
        """Save context for broadcast backward.

        For functional version:
        - inputs: (comm, tensor, root, async_op, hints, timeout)
        - output: The broadcasted tensor
        """
        pass  # No context needed for identity backward

    def _broadcast_backward(ctx, grad_output):
        """Backward for broadcast.

        For the local autograd case, broadcast is treated as identity -
        the gradient flows through unchanged.

        Returns gradient for tensor.
        """
        return grad_output

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
        backward_fn=_all_reduce_backward,
        setup_context_fn=_all_reduce_setup_context,
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
        backward_fn=_broadcast_backward,
        setup_context_fn=_broadcast_setup_context,
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
            ParamSpec(
                "tensor_list",
                ParamKind.INPUT,
                list[torch.Tensor],
                mutable=True,
                write_only=True,
            ),
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
        backward_fn=_all_gather_backward,
        setup_context_fn=_all_gather_setup_context,
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
            ParamSpec(
                "tensor", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
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
            ParamSpec(
                "output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
        backward_fn=_all_gather_single_backward,
        setup_context_fn=_all_gather_single_setup_context,
    )

    # all_gather_v: variable-size all_gather
    register_collective(
        TorchComm,
        TorchComm.all_gather_v,
        param_specs=[
            ParamSpec(
                "tensor_list",
                ParamKind.INPUT,
                list[torch.Tensor],
                mutable=True,
                write_only=True,
            ),
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
            ParamSpec(
                "output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
        backward_fn=_reduce_scatter_single_backward,
        setup_context_fn=_reduce_scatter_single_setup_context,
    )

    # reduce_scatter_v: variable-size reduce_scatter
    register_collective(
        TorchComm,
        TorchComm.reduce_scatter_v,
        param_specs=[
            ParamSpec(
                "output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
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
            ParamSpec(
                "output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
            ParamSpec("input", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
        backward_fn=_all_to_all_single_backward,
        setup_context_fn=_all_to_all_single_setup_context,
    )

    # all_to_all_v_single: variable-size all-to-all with single input/output tensors
    register_collective(
        TorchComm,
        TorchComm.all_to_all_v_single,
        param_specs=[
            ParamSpec(
                "output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
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
                "output_tensor_list",
                ParamKind.INPUT,
                list[torch.Tensor],
                mutable=True,
                write_only=True,
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
            ParamSpec(
                "output", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
            ParamSpec("input_list", ParamKind.INPUT, list[torch.Tensor], mutable=False),
            ParamSpec("op", ParamKind.EXTRA, ReduceOp),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
        backward_fn=_reduce_scatter_backward,
        setup_context_fn=_reduce_scatter_setup_context,
    )

    # scatter: scatter from root to all ranks
    register_collective(
        TorchComm,
        TorchComm.scatter,
        param_specs=[
            ParamSpec(
                "output_tensor",
                ParamKind.INPUT,
                torch.Tensor,
                mutable=True,
                write_only=True,
            ),
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
        backward_fn=_scatter_backward,
        setup_context_fn=_scatter_setup_context,
    )

    # gather: gather from all ranks to root
    register_collective(
        TorchComm,
        TorchComm.gather,
        param_specs=[
            ParamSpec(
                "output_tensor_list",
                ParamKind.INPUT,
                list[torch.Tensor],
                mutable=True,
                write_only=True,
            ),
            ParamSpec("input_tensor", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("root", ParamKind.EXTRA, int),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
            ParamSpec(
                "hints", ParamKind.EXTRA, dict[str, str] | None, default_value=None
            ),
            ParamSpec("timeout", ParamKind.EXTRA, timedelta | None, default_value=None),
        ],
        backward_fn=_gather_backward,
        setup_context_fn=_gather_setup_context,
    )

    register_collective(
        TorchComm,
        TorchComm.start_coalescing,
        param_specs=[],
    )

    register_collective(
        TorchComm,
        TorchComm.end_coalescing,
        param_specs=[
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=True),
        ],
        meta_fn=_end_coalescing_meta,
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
            ParamSpec(
                "tensor", ParamKind.INPUT, torch.Tensor, mutable=True, write_only=True
            ),
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
