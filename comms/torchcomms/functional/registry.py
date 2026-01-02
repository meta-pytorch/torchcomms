# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Collective operation registration system for torchcomms."""

import logging
from typing import Any, Callable, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor, in_kernel_invocation_manager
from torchcomms.functional.async_tensor import (
    _wrap_result_with_registered_work,
    FakeWork,
)
from torchcomms.functional.param_parsing import (
    _TYPE_NAME_TO_CLASS,
    CollectiveParamSchema,
    ParamSpec,
)


logger = logging.getLogger(__name__)

_REGISTERED_COLLECTIVES: dict[str, dict[str, Any]] = {}


def _pack_result(results: list) -> Any:
    """Pack a list of results into the appropriate return value.

    Returns:
        None if empty, single item if one element, tuple if multiple.
    """
    if len(results) == 0:
        return None
    elif len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def register_collective(
    target_class: type,
    method: Callable,
    param_specs: list[ParamSpec],
    meta_fn: Optional[Callable] = None,
):
    """Register a collective operation with torch.compile support.

    Args:
        target_class: The class this method belongs to (e.g., TorchComm, TorchCommWindow).
        method: The method to register as a collective op.
        param_specs: List of ParamSpec describing the op's parameters.
        meta_fn: Optional meta/abstract implementation for tracing.
    """
    name = method.__name__
    # Derive op prefix from class name (e.g., TorchComm -> torchcomm_)
    torch_op_prefix = target_class.__name__.lower() + "_"
    op_name = f"{torch_op_prefix}{name}"

    # Create the param schema - this handles all type processing and categorization
    param_schema = CollectiveParamSchema.from_raw_specs(target_class, param_specs)

    collective_info = {
        "name": name,
        "meta_fn": meta_fn,
        "method": method,
        "param_schema": param_schema,
        "target_class": target_class,
    }
    _REGISTERED_COLLECTIVES[op_name] = collective_info

    logger.info(
        f"Registered collective: {op_name} (method={name}, "
        f"inputs={[p.name for p in param_schema.input_params]}, "
        f"outputs={[p.name for p in param_schema.output_params]}, "
        f"target_class={target_class.__name__}, "
    )

    return method


# Track functional ops created for autograd (for effectful registration)
_FUNCTIONAL_OP_NAMES: list[str] = []

def _generate_lib_ops(lib: Any) -> None:
    """Generate torch.library op definitions for all registered collectives.

    For ops with mutable inputs, registers both:
    - Inplace version: op_name_ with Tensor(a!) and -> ()
    - Functional version: op_name with Tensor and -> Tensor

    The functional version clones inputs, calls inplace, and returns clones.
    This enables the reinplace pass to convert back when safe.
    """
    from torchcomms.functional import collectives

    for base_op_name, info in _REGISTERED_COLLECTIVES.items():
        schema = info["param_schema"]
        method = info["method"]
        meta_fn = info["meta_fn"]

        # Check if this op has mutable inputs
        has_mutable_inputs = len(schema.mutable_params) > 0

        if has_mutable_inputs:
            # Register both inplace (op_) and functional (op) versions
            inplace_op_name = f"{base_op_name}_"
            functional_op_name = base_op_name

            # === INPLACE VERSION (op_) ===
            # Use inplace_return_type to return aliased tensors (e.g., Tensor(a!))
            # This matches PyTorch native inplace ops and enables mutation tracking
            inplace_signature = (
                f"{inplace_op_name}({schema.signature}) -> {schema.inplace_return_type}"
            )
            logger.info(f"Defining inplace lib op: {inplace_signature}")
            lib.define(inplace_signature, tags=[torch.Tag.pt2_compliant_tag])

            # Inplace meta kernel - returns the input tensors (aliased)
            # This matches PyTorch native inplace ops (e.g., add_ returns self)
            def _create_inplace_meta_kernel(captured_schema: CollectiveParamSchema):
                def _inplace_meta(*args, **kwargs):
                    # Parse args to get mutable outputs (the tensors being mutated)
                    parsed = captured_schema.parse_lib_args(args)
                    mutable_outputs = parsed.get_mutable_outputs()

                    # Return the input tensors (they are mutated in place)
                    return _pack_result(mutable_outputs)

                return _inplace_meta

            inplace_meta_kernel = _create_inplace_meta_kernel(schema)
            torch.library.impl(lib, inplace_op_name, "Meta")(inplace_meta_kernel)

            # Inplace eager kernel - calls method, mutates in-place, returns mutated tensors
            def _create_inplace_eager_kernel(
                captured_method: Callable,
                captured_schema: CollectiveParamSchema,
            ):
                def _inplace_eager(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    async_op = parsed.get_value("async_op") or False

                    result_or_work = captured_method(*parsed.to_values())

                    # Get the mutable outputs (the tensors that were mutated)
                    mutable_outputs = parsed.get_mutable_outputs()

                    # Register work handle for async ops
                    if async_op:
                        work = result_or_work
                        mutable_indices = parsed.get_mutable_tensor_indices()
                        if mutable_indices:
                            anchor = parsed.values[mutable_indices[0]]
                            if isinstance(anchor, list):
                                anchor = anchor[0]
                            collectives._register_tensor_work(anchor, work)

                    # Return the mutated tensors (like native PyTorch inplace ops)
                    return _pack_result(mutable_outputs)

                return _inplace_eager

            inplace_eager_kernel = _create_inplace_eager_kernel(method, schema)
            torch.library.impl(lib, inplace_op_name, "CompositeExplicitAutograd")(
                inplace_eager_kernel
            )

            # === FUNCTIONAL VERSION (op) ===
            functional_signature = f"{functional_op_name}({schema.functional_signature}) -> {schema.functional_return_type}"
            logger.info(f"Defining functional lib op: {functional_signature}")
            lib.define(functional_signature, tags=[torch.Tag.pt2_compliant_tag])

            # Track functional op for effectful registration
            _FUNCTIONAL_OP_NAMES.append(functional_op_name)

            # Functional meta kernel - returns empty_like of mutable inputs
            # IMPORTANT: Must preserve requires_grad for autograd to work
            def _create_functional_meta_kernel(captured_schema: CollectiveParamSchema):
                def _functional_meta(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    # Get mutable outputs preserving structure (list vs single tensor)
                    mutable_outputs = parsed.get_mutable_outputs()

                    # Check if any input requires grad - propagate to outputs
                    any_requires_grad = parsed.has_requires_grad()

                    results = []
                    for out in mutable_outputs:
                        if isinstance(out, (list, tuple)):
                            # For tensor lists, return a list (not tuple)
                            # This is required for Tensor[] return type
                            # Propagate requires_grad from inputs or from each tensor
                            results.append(
                                [
                                    torch.empty_like(
                                        t,
                                        requires_grad=(
                                            any_requires_grad or t.requires_grad
                                        ),
                                    )
                                    for t in out
                                ]
                            )
                        else:
                            results.append(
                                torch.empty_like(
                                    out,
                                    requires_grad=(
                                        any_requires_grad or out.requires_grad
                                    ),
                                )
                            )

                    return _pack_result(results)

                return _functional_meta

            functional_meta_kernel = _create_functional_meta_kernel(schema)
            torch.library.impl(lib, functional_op_name, "Meta")(functional_meta_kernel)

            # Functional eager kernel - clones, calls inplace, returns clones
            def _create_functional_eager_kernel(
                captured_schema: CollectiveParamSchema,
                captured_inplace_op_name: str,
            ):
                def _functional_eager(*args):
                    parsed = captured_schema.parse_lib_args(args)

                    # Clone mutable inputs (or empty_like for write-only outputs)
                    cloned_values = parsed.to_values()
                    mutable_indices = parsed.get_mutable_tensor_indices()
                    clones = []

                    for idx in mutable_indices:
                        orig = parsed.values[idx]
                        spec = parsed.all_params[idx]
                        if spec.write_only:
                            # Write-only buffer, empty_like is fine
                            if isinstance(orig, list):
                                clone = [torch.empty_like(t) for t in orig]
                            else:
                                clone = torch.empty_like(orig)
                        else:
                            # Read+write, need actual clone to preserve input data
                            if isinstance(orig, list):
                                clone = [t.clone() for t in orig]
                            else:
                                clone = orig.clone()
                        cloned_values[idx] = clone
                        clones.append(clone)

                    # Call inplace op on clones (skip index 0 which is the object)
                    getattr(torch.ops.torchcomms, captured_inplace_op_name)(
                        *cloned_values
                    )

                    # Transfer work from original to clone for async ops
                    async_op = parsed.get_value("async_op") or False
                    if async_op and mutable_indices:
                        # Work was registered on the clone by the inplace op
                        # No need to transfer - the clone already has the work
                        pass

                    return _pack_result(clones)

                return _functional_eager

            functional_eager_kernel = _create_functional_eager_kernel(
                schema, inplace_op_name
            )
            torch.library.impl(lib, functional_op_name, "CompositeExplicitAutograd")(
                functional_eager_kernel
            )

            # === FUNCTIONALIZE IMPL FOR INPLACE OP ===
            # Register py_functionalize_impl to convert inplace -> functional
            # and wrap with with_effects for proper effect token tracking.
            # Also update the original tensors to reference the new results (inplace semantics).
            def _create_functionalize_impl(
                captured_functional_op_name: str,
                captured_schema: CollectiveParamSchema,
            ):
                def _functionalize_impl(ctx, *args):
                    from torch._subclasses.functional_tensor import (
                        FunctionalTensor,
                        PythonFunctionalizeAPI,
                    )

                    # Get the mutable tensor indices in all_params (obj + inputs + extras)
                    mutable_indices = captured_schema.mutable_indices

                    # Unwrap args to get raw tensors
                    unwrapped_args = ctx.unwrap_tensors(args)

                    # Get the functional op
                    functional_op = getattr(
                        torch.ops.torchcomms, captured_functional_op_name
                    )

                    # Check if we have access to the mode's tokens for effects
                    assert (
                        isinstance(ctx, PythonFunctionalizeAPI)
                        and hasattr(ctx, "mode")
                        and ctx.mode is not None
                        and hasattr(ctx.mode, "_tokens")
                    )

                    from torch._higher_order_ops.effects import (
                        handle_effects,
                        has_effects,
                    )

                    assert has_effects(functional_op.default)

                    with ctx.redispatch_to_next():
                        result = handle_effects(
                            ctx.mode._allow_token_discovery,
                            ctx.mode._tokens,
                            functional_op.default,
                            tuple(unwrapped_args),
                            {},
                        )

                    # Unwrap the result - handle_effects may return FunctionalTensors
                    # but ctx.replace expects unwrapped tensors
                    unwrapped_result = ctx.unwrap_tensors(result)

                    # Update original mutable tensors to reference the new results.
                    # This maintains inplace semantics - the original tensor now wraps
                    # the new result from the functional op.
                    if not isinstance(unwrapped_result, (list, tuple)):
                        unwrapped_result = (unwrapped_result,)

                    for i, idx in enumerate(mutable_indices):
                        if i < len(unwrapped_result) and idx < len(args):
                            orig_wrapped = args[idx]
                            new_unwrapped = unwrapped_result[i]
                            if isinstance(
                                orig_wrapped, FunctionalTensor
                            ) and isinstance(new_unwrapped, torch.Tensor):
                                ctx.replace(orig_wrapped, new_unwrapped)
                            elif isinstance(orig_wrapped, (list, tuple)) and isinstance(
                                new_unwrapped, (list, tuple)
                            ):
                                for ow, nu in zip(orig_wrapped, new_unwrapped):
                                    if isinstance(ow, FunctionalTensor) and isinstance(
                                        nu, torch.Tensor
                                    ):
                                        ctx.replace(ow, nu)

                    return result
                    """
                    # Return the original wrapped tensors (which now reference the new results)
                    mutable_outputs = [args[idx] for idx in mutable_indices]
                    return _pack_result(mutable_outputs)
                    """

                return _functionalize_impl

            functionalize_impl = _create_functionalize_impl(functional_op_name, schema)

            # Get the inplace op object and register py_functionalize_impl
            inplace_op = getattr(torch.ops.torchcomms, inplace_op_name)
            inplace_op.default.py_functionalize_impl(functionalize_impl)

            logger.info(
                f"Registered py_functionalize_impl: {inplace_op_name} -> {functional_op_name}"
            )

        else:
            # No mutable inputs - just register the regular op
            full_signature = (
                f"{base_op_name}({schema.signature}) -> {schema.return_type}"
            )
            logger.info(f"Defining lib op: {full_signature}")
            lib.define(full_signature, tags=[torch.Tag.pt2_compliant_tag])

            # Meta kernel
            def _create_meta_kernel(
                meta_func: Optional[Callable],
                captured_schema: CollectiveParamSchema,
            ):
                def _meta_kernel(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    result_tensors = []

                    if meta_func:
                        custom_result = meta_func(*args)
                        if custom_result is not None:
                            if not isinstance(custom_result, (list, tuple)):
                                result_tensors = [custom_result]
                            else:
                                result_tensors = list(custom_result)
                    elif captured_schema.has_tensor_outputs:
                        raise RuntimeError(
                            "Operation has OUTPUT params but no meta_func provided"
                        )

                    if captured_schema.needs_async_dummy_return:
                        async_op = parsed.get_value("async_op") or False
                        if async_op:
                            # Use CPU device for the dummy tensor since it's just a placeholder
                            # for work tracking and doesn't need to match input device
                            dummy = torch.empty(0, device="cpu", dtype=torch.int)
                            result_tensors.append(dummy)
                        else:
                            result_tensors.append(None)

                    return _pack_result(result_tensors)

                return _meta_kernel

            meta_kernel = _create_meta_kernel(meta_fn, schema)
            torch.library.impl(lib, base_op_name, "Meta")(meta_kernel)

            # Eager kernel
            def _create_eager_kernel(
                captured_method: Callable,
                captured_schema: CollectiveParamSchema,
            ):
                def _eager_kernel(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    async_op = parsed.get_value("async_op") or False
                    result_or_work = captured_method(*parsed.to_values())

                    result_tensors = []
                    if captured_schema.has_tensor_outputs:
                        if async_op:
                            raise RuntimeError(
                                "Async operations with OUTPUT params not yet supported."
                            )
                        if isinstance(result_or_work, (list, tuple)):
                            result_tensors.extend(result_or_work)
                        elif result_or_work is not None:
                            result_tensors.append(result_or_work)

                    if async_op:
                        work = result_or_work
                        if captured_schema.needs_async_dummy_return:
                            dummy = torch.empty(0, device="cpu", dtype=torch.int)
                            collectives._register_tensor_work(dummy, work)
                            result_tensors.append(dummy)

                    return _pack_result(result_tensors)

                return _eager_kernel

            eager_kernel = _create_eager_kernel(method, schema)
            torch.library.impl(lib, base_op_name, "CompositeExplicitAutograd")(
                eager_kernel
            )

            # For ops without tensor inputs (like barrier), register a FakeTensorMode
            # py_impl to bypass the default FakeTensor dispatch which fails to find
            # a device from tensor inputs.
            if (
                schema.num_output_tensors > 0 and schema.num_input_tensors == 0
            ) or schema.needs_async_dummy_return:

                def _create_fake_impl(captured_meta_kernel, captured_schema):
                    def _fake_impl(fake_mode, func, *args, **kwargs):
                        # First arg is the opaque object - get device directly from it
                        obj = args[0]
                        with in_kernel_invocation_manager(fake_mode):
                            result = captured_meta_kernel(*args, **kwargs)

                        if result is None:
                            return None
                        if isinstance(result, torch.Tensor):
                            # FakeTensor expects a meta tensor - convert if needed
                            if result.device.type != "meta":
                                meta_result = torch.empty_like(result, device="meta")
                            else:
                                meta_result = result
                            # For async dummy tensors, always use CPU device
                            # For other tensors, get device from the opaque object
                            if captured_schema.needs_async_dummy_return:
                                target_device = torch.device("cpu")
                            elif result.device.type != "meta":
                                target_device = result.device
                            else:
                                raise RuntimeError("Unable to determine target device")
                            return FakeTensor(fake_mode, meta_result, target_device)
                        return result

                    return _fake_impl

                torch_op = getattr(torch.ops.torchcomms, base_op_name)
                fake_impl = _create_fake_impl(meta_kernel, schema)
                from torch._subclasses.fake_impls import register_op_impl

                register_op_impl(torch_op.default)(fake_impl)
                logger.info(
                    f"Registered FakeTensorMode py_impl for {base_op_name} (no tensor inputs)"
                )

        logger.info(f"Registered ops for {base_op_name}")


def _register_lowerings() -> None:
    """Register inductor lowerings for all registered collectives."""
    try:
        from torch._inductor import ir
        from torch._inductor.lowering import register_lowering
    except ImportError:
        logger.info("torch._inductor not available, skipping lowering registration")
        return

    for op_name, collective_info in _REGISTERED_COLLECTIVES.items():
        schema = collective_info["param_schema"]

        try:
            torch_op = getattr(torch.ops.torchcomms, op_name)

            def _create_lowering(
                op_name: str,
                captured_torch_op,
                lowering_schema: CollectiveParamSchema,
            ):
                num_output_tensors = lowering_schema.num_output_tensors
                needs_async_dummy = lowering_schema.needs_async_dummy_return

                def _lowering(*args):
                    from torch._inductor.virtualized import V

                    logger.info(f"Lowering torchcomms.{op_name} with {len(args)} args")

                    # Parse args using schema
                    parsed = lowering_schema.parse_lib_args(args)

                    # Get flattened inputs with mutable mask
                    flat_inputs, flat_mutable_mask = (
                        parsed.get_tensor_inputs_flat_with_mutable_mask()
                    )

                    # Get device from first tensor input
                    device = None
                    tensor_inputs = parsed.get_tensor_inputs()
                    if tensor_inputs:
                        first_input = tensor_inputs[0]
                        if isinstance(first_input, (list, tuple)):
                            if first_input:
                                device = first_input[0].get_device()
                        else:
                            device = first_input.get_device()

                    # Realize all inputs and mark mutable ones as mutated
                    for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                        inp.realize()
                        if is_mutable:
                            V.graph.mark_buffer_mutated(inp.get_name())

                    # Process kernel args - pass opaque object and remaining args separately
                    # to avoid process_kernel trying to extract shapes from the opaque object
                    with V.graph.fake_mode:
                        (
                            example_output,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                            unbacked_bindings,
                        ) = ir._CollectiveKernel.process_kernel(
                            captured_torch_op.default, *args
                        )
                    assert not unbacked_bindings

                    # Check if the op returns an async dummy tensor
                    # example_output will be a tensor if async_op=True and needs_async_dummy=True
                    has_async_dummy_output = (
                        needs_async_dummy
                        and example_output is not None
                        and isinstance(example_output, torch.Tensor)
                    )

                    if num_output_tensors == 0 and not has_async_dummy_output:
                        # No outputs - just create collective for side effect
                        layout = ir.NoneLayout(device=device)
                        packed = ir._CollectiveKernel(
                            layout,
                            captured_torch_op.default,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                        )

                        # Set up mutation outputs for mutable inputs
                        for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                            if is_mutable:
                                packed.mutation_outputs.append(
                                    ir.MutationOutput(
                                        ir.NoneLayout(device=device), inp, packed
                                    )
                                )
                                packed.alias_names.append(inp.get_name())

                        return None

                    elif num_output_tensors == 1 or has_async_dummy_output:
                        # Single output - use its layout directly
                        if isinstance(example_output, (list, tuple)):
                            ex_out = example_output[0]
                        else:
                            ex_out = example_output

                        layout = ir._CollectiveKernel.tensor_to_layout(ex_out)
                        packed = ir._CollectiveKernel(
                            layout,
                            captured_torch_op.default,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                        )

                        # Set up mutation outputs for mutable inputs
                        for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                            if is_mutable:
                                packed.mutation_outputs.append(
                                    ir.MutationOutput(
                                        ir.NoneLayout(device=device), inp, packed
                                    )
                                )
                                packed.alias_names.append(inp.get_name())

                        return ir.TensorBox.create(packed)

                    else:
                        # Multiple outputs - use MultiOutputLayout
                        layout = ir.MultiOutputLayout(device=device)
                        packed = ir._CollectiveKernel(
                            layout,
                            captured_torch_op.default,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                        )

                        # Set up mutation outputs for mutable inputs
                        for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                            if is_mutable:
                                packed.mutation_outputs.append(
                                    ir.MutationOutput(
                                        ir.NoneLayout(device=device), inp, packed
                                    )
                                )
                                packed.alias_names.append(inp.get_name())

                        # Create MultiOutput for each output tensor
                        packed.outputs = []
                        result_tensors = []

                        if isinstance(example_output, (list, tuple)):
                            for i, ex_out in enumerate(example_output):
                                if hasattr(ex_out, "shape"):
                                    out = ir.MultiOutput(
                                        ir._CollectiveKernel.tensor_to_layout(ex_out),
                                        packed,
                                        [(list, i)],
                                    )
                                    packed.outputs.append(out)
                                    result_tensors.append(ir.TensorBox.create(out))

                        return tuple(result_tensors)

                return _lowering

            lowering_fn = _create_lowering(op_name, torch_op, schema)
            register_lowering(torch_op)(lowering_fn)

            logger.info(f"Registered lowering for {op_name}")
        except AttributeError as e:
            logger.warning(f"Failed to register lowering for {op_name}: {e}")

_EFFECTFUL_HANDLES: list = []  # Store handles to prevent GC


def _register_effectful_ops() -> None:
    """Register torchcomms ops as effectful with ORDERED effect type.

    Both functional and inplace ops with global side effects (collective communication)
    need effect ordering to ensure correct execution order across ranks.
    The with_effects HOP wraps these ops during tracing.

    We register BOTH versions so that:
    - If we trace inplace ops, they get wrapped with with_effects
    - Functionalization then converts inplace to functional inside the wrapper
    - The effect ordering is preserved throughout
    """
    global _EFFECTFUL_HANDLES
    try:
        from torch._higher_order_ops.effects import _get_effect, _register_effectful_op
        from torch._library.effects import EffectType
    except ImportError:
        logger.info("Effect system not available, skipping effectful registration")
        return

    # Collect all op names to register (both functional and inplace versions)
    ops_to_register = []

    for base_op_name in _REGISTERED_COLLECTIVES.keys():
        ops_to_register.append(base_op_name)

    ops_to_register.append("torchcomm_wait_tensors")

    logger.info(
        f"_register_effectful_ops: Registering {len(ops_to_register)} ops: {ops_to_register}"
    )

    for op_name in ops_to_register:
        try:
            op_packet = getattr(torch.ops.torchcomms, op_name, None)
            if op_packet is None:
                logger.warning(f"Op torchcomms.{op_name} not found, skipping")
                continue
            # Get the default overload
            if hasattr(op_packet, "default"):
                torch_op = op_packet.default
                handle = _register_effectful_op(torch_op, EffectType.ORDERED)
                _EFFECTFUL_HANDLES.append(handle)  # Keep handle alive
                # Verify registration worked
                effect = _get_effect(torch_op)
                logger.info(f"Registered torchcomms.{op_name}: effect={effect}")
        except Exception as e:
            logger.warning(f"Failed to register torchcomms.{op_name} as effectful: {e}")


def finalize_registration(lib: Any) -> None:
    """Finalize registration of all collectives."""
    from torchcomms.functional.dynamo import (
        _patch_dynamo_for_opaque_methods,
        register_with_dynamo,
    )
    from torchcomms.functional.inductor_lowering import register_torchcomms_lowerings

    logger.info(
        f"Finalizing registration for {len(_REGISTERED_COLLECTIVES)} collectives"
    )
    _generate_lib_ops(lib)
    _register_effectful_ops()
    _patch_dynamo_for_opaque_methods(
        _REGISTERED_COLLECTIVES,
        _TYPE_NAME_TO_CLASS,
    )
    register_torchcomms_lowerings()
    register_with_dynamo()

    logger.info("Collective registration finalized")
