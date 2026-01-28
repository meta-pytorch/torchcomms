# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

"""
FX graph passes for torchcomms.

This module provides compiler passes that can be applied to FX graphs
to optimize torchcomms operations.

NOTE: when inductor is enabled, these passes are applied automatically,
so they should be skipped to avoid redundant work.

- reinplacement_pass: converts functional ops back to in-place ops where possible
- strip_with_effects_pass: removes with_effects HOP wrappers from torchcomms ops

NOTE: when used in conjunction, the reinplacement pass should be applied first.
"""

import logging
import operator

import torch
from torch._higher_order_ops.effects import with_effects
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops
from torch._inductor.fx_utils import FakeTensorUpdater

logger = logging.getLogger(__name__)


def reinplacement_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    logger.info("starting reinplacement pass")
    logger.debug("graph before reinplacement pass: %s", gm.graph)

    from torch._guards import detect_fake_mode
    from torch._inductor.virtualized import V

    fake_tensor_updater = FakeTensorUpdater(gm.graph)

    fake_mode = detect_fake_mode(
        [node.meta.get("val") for node in gm.graph.nodes if "val" in node.meta]
    )

    if fake_mode is not None:
        with V.set_fake_mode(fake_mode):
            reinplace_inplaceable_ops(fake_tensor_updater, gm.graph)
            fake_tensor_updater.incremental_update()
    else:
        logger.warning("No fake mode detected, skipping reinplacement pass")

    gm.recompile()
    logger.info("finished reinplacement pass")
    logger.debug("graph after reinplacement pass: %s", gm.graph)
    return gm


def _replace_in_output_args(output_node, old_node, new_node):
    """Replace old_node with new_node in output args, handling nested structures."""

    def replace_in_structure(obj):
        if obj is old_node:
            return new_node
        elif isinstance(obj, (list, tuple)):
            result = [replace_in_structure(item) for item in obj]
            return type(obj)(result)
        elif isinstance(obj, dict):
            return {k: replace_in_structure(v) for k, v in obj.items()}
        else:
            return obj

    output_node.args = replace_in_structure(output_node.args)


def strip_with_effects_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    graph = gm.graph

    logger.info("starting strip_with_effects pass for torchcomms ops")
    logger.debug("graph before strip_with_effects pass: %s", graph)

    nodes_to_erase = []
    replacements_made = 0

    # iterate in reverse order so downstream with_effects nodes (which consume tokens
    # from upstream ones) are processed first, allowing less strenuous token chain cleanup
    for node in reversed(list(graph.nodes)):
        if node.op != "call_function" or node.target is not with_effects:
            continue

        # with_effects(token, op, *args)
        wrapped_op = node.args[1]

        if not str(wrapped_op).startswith("torchcomms."):
            continue

        logger.debug("Found with_effects wrapping torchcomms op: %s", wrapped_op)

        actual_args = node.args[2:]
        actual_kwargs = node.kwargs

        with graph.inserting_before(node):
            direct_call = graph.call_function(wrapped_op, actual_args, actual_kwargs)

        for user in list(node.users):
            if user.op != "call_function" or user.target != operator.getitem:
                continue

            getitem_index = user.args[1]
            if getitem_index > 0:
                direct_call.meta = user.meta.copy()
                user.replace_all_uses_with(direct_call)
            else:
                for token_user in list(user.users):
                    if token_user.op == "output":
                        _replace_in_output_args(token_user, user, direct_call)
            nodes_to_erase.append(user)

        nodes_to_erase.append(node)
        replacements_made += 1

    remaining = list(nodes_to_erase)
    while remaining:
        made_progress = False
        still_remaining = []
        for node in remaining:
            if len(node.users) == 0:
                graph.erase_node(node)
                made_progress = True
            else:
                still_remaining.append(node)
        remaining = still_remaining
        if not made_progress:
            for node in remaining:
                logger.warning(
                    f"Could not erase node {node}, still has users: {list(node.users)}"
                )
            break

    gm.recompile()

    logger.info(
        f"finished strip_with_effects pass: removed {replacements_made} with_effects wrappers"
    )
    logger.debug("graph after strip_with_effects pass: %s", gm.graph)

    return gm
