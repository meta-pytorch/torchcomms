# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
Triton kernels for microbenchmarking raw ``put_block`` RDMA throughput.

These kernels isolate put_block performance from local-copy overhead by
operating directly on pre-filled staging buffers. Three experiments are
provided, each adding a different dimension of overhead:

Kernel 1 — ``put_bw_fireforget_kernel``
    Fire-and-forget: no flow control, no ring-buffer wrapping. Each step
    puts from a unique staging offset. Measures the absolute ceiling of
    put_block throughput.

Kernel 2 — ``put_bw_pipelined_kernel``
    Pipelined with flow control: ring-buffer staging with SLOT_FREE
    signaling from the receiver. Measures put_block throughput under
    realistic back-pressure.

Kernel 3 — ``put_bw_multiblock_kernel``
    Multi-block pipelined: N send blocks each put their own tile of each
    section, sharing a single QP. Measures throughput scaling and QP
    contention effects.

All kernels are bidirectional (each rank sends and receives simultaneously)
and use monotonic signal/counter tracking identical to sendrecv.
"""

from torch.utils._triton import has_triton

if has_triton():
    import triton
    import triton.language as tl
    from torchcomms.triton.fb import (
        fence,
        put_block,
        requires_torchcomms,
        signal_block,
        wait_counter,
        wait_signal_from,
    )

    from .sendrecv_cooperative_kernel import _increment_iteration_kernel  # noqa: F401

    # ------------------------------------------------------------------
    # Kernel 1: Fire-and-forget put_block throughput
    # ------------------------------------------------------------------

    @requires_torchcomms
    @triton.jit
    def put_bw_fireforget_kernel(
        # torchcomms handles
        win,
        send_buf,
        # staging pointer (pre-filled, >= total message size)
        send_staging_ptr,
        # iteration tracking (monotonic, GPU-resident)
        iteration_ptr,
        # sizes
        total_elements,
        section_elements,
        elem_size_bytes,
        # peer
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fire-and-forget put_block bandwidth kernel.

        Measures the absolute ceiling of put_block throughput with zero flow
        control overhead. The staging buffer is sized to hold the entire
        message, so each step writes to a unique offset — no ring-buffer
        wrapping and no SLOT_FREE signaling from the receiver.

        Grid: 2 blocks (pid=0 send, pid=1 recv). Bidirectional.

        Signal / counter layout
        ~~~~~~~~~~~~~~~~~~~~~~~
        - signal_id=0 (DATA_READY): put_block increments by 1 each step.
        - signal_id=1 (ALL_DONE): receiver signals sender after all data
          arrived, ensuring end-to-end delivery is measured.
        - counter_id=0 (NIC_DONE): put_block increments by 1 each step.
        """
        pid = tl.program_id(axis=0)
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        base = iteration * total_steps

        if pid == 0:
            # -- Send block --
            # Post all puts without waiting — true fire-and-forget.
            # No per-step wait_counter: each step uses a unique staging
            # offset (no data hazard), and QP depth (1024) >> max steps.
            # This keeps multiple WQEs in flight to saturate the NIC.
            for step in range(total_steps):
                staging_byte_offset = step * section_elements * elem_size_bytes

                # Handle last section which may be smaller.
                remaining_elements = total_elements - step * section_elements
                actual_section_elements = tl.where(
                    section_elements < remaining_elements,
                    section_elements,
                    remaining_elements,
                )
                section_bytes = actual_section_elements * elem_size_bytes

                put_block(
                    win,
                    staging_byte_offset,  # dst_offset in peer's recv buffer
                    send_buf,
                    staging_byte_offset,  # src_offset in our staging
                    peer_rank,
                    section_bytes,
                    0,  # signal_id = DATA_READY
                    0,  # counter_id = NIC_DONE
                )

            # Wait for receiver to confirm all data arrived (end-to-end).
            # wait_counter only means NIC finished reading from staging —
            # NOT that data was delivered to the remote.
            # ALL_DONE increments by 1 per iteration (not per step), so
            # use iteration+1, NOT base+1 (base = iteration * total_steps
            # would grow too fast and deadlock on iteration >= 1).
            wait_signal_from(win, peer_rank, 1, iteration + 1)
        else:
            # -- Recv block --
            # Wait for all data to arrive.
            wait_signal_from(win, peer_rank, 0, base + total_steps)
            # Signal sender that all data was delivered.
            fence(win)
            signal_block(win, peer_rank, 1, 1)  # ALL_DONE

    # ------------------------------------------------------------------
    # Kernel 2: Pipelined put_block with flow control
    # ------------------------------------------------------------------

    @requires_torchcomms
    @triton.jit
    def put_bw_pipelined_kernel(
        # torchcomms handles
        win,
        send_buf,
        # staging pointer (pipeline_depth * section_size)
        send_staging_ptr,
        # iteration tracking (monotonic, GPU-resident)
        iteration_ptr,
        # sizes
        total_elements,
        section_elements,
        elem_size_bytes,
        # pipeline config
        pipeline_depth,
        # peer
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Pipelined put_block bandwidth kernel with flow control.

        Measures put_block throughput under realistic ring-buffer back-pressure.
        The staging buffer holds ``pipeline_depth`` sections, and the receiver
        signals SLOT_FREE after consuming each section.

        Grid: 2 blocks (pid=0 send, pid=1 recv). Bidirectional.

        Signal / counter layout
        ~~~~~~~~~~~~~~~~~~~~~~~
        - signal_id=0 (DATA_READY): put_block increments by 1 each step.
        - signal_id=1 (SLOT_FREE): receiver's signal_block increments by 1
          each step after draining.
        - counter_id=0 (NIC_DONE): put_block increments by 1 each step.
        """
        pid = tl.program_id(axis=0)
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        base = iteration * total_steps

        if pid == 0:
            # -- Send block --
            for step in range(total_steps):
                slot = step % pipeline_depth

                # Wait for NIC + receiver to free the slot before reuse.
                if step >= pipeline_depth:
                    wait_counter(win, 0, base + step - pipeline_depth + 1)
                    wait_signal_from(
                        win, peer_rank, 1, base + step - pipeline_depth + 1
                    )

                staging_byte_offset = slot * section_elements * elem_size_bytes

                # Handle last section which may be smaller.
                remaining_elements = total_elements - step * section_elements
                actual_section_elements = tl.where(
                    section_elements < remaining_elements,
                    section_elements,
                    remaining_elements,
                )
                section_bytes = actual_section_elements * elem_size_bytes

                put_block(
                    win,
                    staging_byte_offset,  # dst_offset in peer's recv staging
                    send_buf,
                    staging_byte_offset,  # src_offset in our staging
                    peer_rank,
                    section_bytes,
                    0,  # signal_id = DATA_READY
                    0,  # counter_id = NIC_DONE
                )
        else:
            # -- Recv block --
            for step in range(total_steps):
                # Wait for data to arrive for this step.
                wait_signal_from(win, peer_rank, 0, base + step + 1)
                fence(win)
                # Signal sender that the slot is free for reuse.
                signal_block(win, peer_rank, 1, 1)  # SLOT_FREE += 1

    # ------------------------------------------------------------------
    # Kernel 3: Multi-block pipelined put_block (QP contention)
    # ------------------------------------------------------------------

    @requires_torchcomms
    @triton.jit
    def put_bw_multiblock_kernel(
        # torchcomms handles
        win,
        send_buf,
        # staging pointer (pipeline_depth * section_size)
        send_staging_ptr,
        # iteration tracking (monotonic, GPU-resident)
        iteration_ptr,
        # sizes
        total_elements,
        section_elements,
        elem_size_bytes,
        # pipeline config
        pipeline_depth,
        num_blocks,  # send blocks = recv blocks
        # peer
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Multi-block pipelined put_block bandwidth kernel.

        Measures put_block throughput with N send blocks sharing a single QP,
        exposing contention effects. Each send block puts its own tile of each
        section independently — no inter-block coordination.

        Grid: ``2 * num_blocks`` (pids ``[0, num_blocks)`` send,
        pids ``[num_blocks, 2*num_blocks)`` recv). Bidirectional.

        Signal / counter layout (per block)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - signal_id=pid (DATA_READY): put_block increments by 1 each step.
        - signal_id=num_blocks+pid (SLOT_FREE): receiver's signal_block
          increments by 1 each step.
        - counter_id=pid (NIC_DONE): put_block increments by 1 each step.
        """
        pid = tl.program_id(axis=0)
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        base = iteration * total_steps

        if pid < num_blocks:
            # -- Send block --
            tile_elements = section_elements // num_blocks
            my_element_offset = pid * tile_elements
            # Last block absorbs remainder.
            if pid == num_blocks - 1:
                tile_elements = section_elements - my_element_offset

            for step in range(total_steps):
                slot = step % pipeline_depth

                # Wait for NIC + receiver to free the slot before reuse.
                if step >= pipeline_depth:
                    wait_counter(win, pid, base + step - pipeline_depth + 1)
                    wait_signal_from(
                        win,
                        peer_rank,
                        num_blocks + pid,
                        base + step - pipeline_depth + 1,
                    )

                staging_byte_offset = (
                    slot * section_elements + my_element_offset
                ) * elem_size_bytes

                # Handle last section which may be smaller.
                remaining_elements = total_elements - step * section_elements
                actual_section_elements = tl.where(
                    section_elements < remaining_elements,
                    section_elements,
                    remaining_elements,
                )
                # My tile within this (possibly smaller) section.
                actual_tile_elements = actual_section_elements - my_element_offset
                if actual_tile_elements > tile_elements:
                    actual_tile_elements = tile_elements
                if actual_tile_elements < 0:
                    actual_tile_elements = 0

                tile_bytes = actual_tile_elements * elem_size_bytes

                if tile_bytes > 0:
                    put_block(
                        win,
                        staging_byte_offset,  # dst_offset in peer's recv staging
                        send_buf,
                        staging_byte_offset,  # src_offset in our staging
                        peer_rank,
                        tile_bytes,
                        pid,  # signal_id = DATA_READY for my tile
                        pid,  # counter_id = NIC_DONE for my tile
                    )
                else:
                    # No data for this tile in the last (partial) section.
                    # Still signal DATA_READY so the receiver doesn't deadlock.
                    signal_block(win, peer_rank, pid, 1)
        else:
            # -- Recv block --
            recv_pid = pid - num_blocks

            for step in range(total_steps):
                # Wait for my tile's data to arrive.
                wait_signal_from(win, peer_rank, recv_pid, base + step + 1)
                fence(win)
                # Signal sender that my tile's slot is free.
                signal_block(win, peer_rank, num_blocks + recv_pid, 1)
