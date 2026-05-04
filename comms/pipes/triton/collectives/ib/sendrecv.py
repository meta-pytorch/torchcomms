# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
Block-scope pipelined send/recv kernel for GPU-to-GPU communication.

Each block independently RDMA-puts its own tile using block-scope APIs,
eliminating inter-block coordination overhead.

Architecture
------------
- Block i owns tile i of each section (section_elements // num_blocks).
- Block i copies its tile from src to send_staging, then calls put_block
  for its tile alone — no atomic coordination, no "last block" logic.
- Block i on the receiver waits for its tile's DATA_READY signal,
  copies its tile from recv_staging to dst, then signals SLOT_FREE.

This design posts N WQEs per pipeline step (one per block) instead of 1,
keeping the NIC pipeline full and enabling NIC-level parallelism.

Signal layout (per direction, per pipeline slot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let N = num_blocks, P = pipeline_depth.

Each (block, slot) pair has independent signals/counters, enabling true
pipeline parallelism — the sender can have P puts in-flight per block.

- signal_id ``pid * P + slot``             : DATA_READY (sender → receiver)
- signal_id ``N * P + pid * P + slot``     : SLOT_FREE  (receiver → sender)
- counter_id ``pid * P + slot``            : NIC_DONE   (local NIC completion)

Total signals: ``2 * N * P``.
Total counters: ``N * P``.

Monotonic signals across iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each per-slot signal is incremented once per use of that slot. Slot s is
used at steps s, s+P, s+2P, ..., giving ``steps_per_slot = ceil(total_steps/P)``
uses per iteration. The monotonic base is ``iteration * steps_per_slot``.
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
        wait_counter_block,
        wait_signal_from_block,
    )

    @triton.jit
    def _threadfence_system():
        """Emit fence.acq_rel.sys — ensures all prior stores are visible to
        all devices (including NIC DMA readers) before subsequent operations."""
        tl.inline_asm_elementwise(
            "fence.acq_rel.sys;",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _send_block_fn(
        win,
        send_buf,
        src_ptr,
        send_staging_ptr,
        iteration_ptr,
        total_elements,
        section_elements,
        elem_size_bytes,
        pipeline_depth,
        num_blocks,
        pid,
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Block-scope send: each block copies and RDMA-puts its own tile.

        Per-block signal/counter IDs (monotonically increasing values):
        - DATA_READY signal ``pid``: data arrived at receiver
        - SLOT_FREE signal ``N + pid``: receiver freed window slot
        - NIC_DONE counter ``pid``: NIC finished reading staging
        """
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        iter_base = iteration * total_steps

        # Per-block signal/counter IDs
        data_signal = pid
        slot_free_signal = num_blocks + pid
        nic_counter = pid

        # Each block owns a fixed tile within the section.
        tile_elements = section_elements // num_blocks
        my_element_offset = pid * tile_elements
        # Last block absorbs remainder.
        if pid == num_blocks - 1:
            tile_elements = section_elements - my_element_offset

        for step in range(total_steps):
            slot = step % pipeline_depth

            # -- Wait for NIC to finish reading staging[slot] --
            if step >= pipeline_depth:
                nic_expected = iter_base + step - pipeline_depth + 1
                wait_counter_block(win, nic_counter, nic_expected)

            # -- Copy my tile: src → send_staging --
            src_start = step * section_elements + my_element_offset
            staging_start = slot * section_elements + my_element_offset

            remaining = total_elements - step * section_elements - my_element_offset
            copy_elements = tl.where(
                tile_elements < remaining,
                tile_elements,
                remaining,
            )

            if copy_elements > 0:
                for i in range(0, copy_elements, BLOCK_SIZE):
                    offs = i + tl.arange(0, BLOCK_SIZE)
                    mask = offs < copy_elements
                    data = tl.load(src_ptr + src_start + offs, mask=mask)
                    tl.store(
                        send_staging_ptr + staging_start + offs,
                        data,
                        mask=mask,
                    )

            # Ensure staging writes are visible to the NIC (PCIe DMA reader).
            _threadfence_system()

            # -- Wait for receiver to free window[slot] --
            if step >= pipeline_depth:
                slot_expected = iter_base + step - pipeline_depth + 1
                wait_signal_from_block(
                    win,
                    peer_rank,
                    slot_free_signal,
                    slot_expected,
                )

            # -- RDMA put my tile --
            remaining_elements = total_elements - step * section_elements
            actual_section_elements = tl.where(
                section_elements < remaining_elements,
                section_elements,
                remaining_elements,
            )
            actual_tile_elements = actual_section_elements - my_element_offset
            if actual_tile_elements > tile_elements:
                actual_tile_elements = tile_elements
            if actual_tile_elements < 0:
                actual_tile_elements = 0

            actual_tile_bytes = actual_tile_elements.to(tl.int64) * elem_size_bytes
            staging_byte_offset = (
                tl.cast(slot * section_elements + my_element_offset, tl.int64)
                * elem_size_bytes
            )

            if actual_tile_bytes > 0:
                put_block(
                    win,
                    staging_byte_offset,
                    send_buf,
                    staging_byte_offset,
                    peer_rank,
                    actual_tile_bytes,
                    data_signal,
                    nic_counter,
                )
            else:
                signal_block(win, peer_rank, data_signal, 1)

        # -- Drain: SLOT_FREE only (implies NIC_DONE) --
        drain_val = iter_base + total_steps
        wait_signal_from_block(win, peer_rank, slot_free_signal, drain_val)

        # Increment iteration counter (only sender block 0, thread 0).
        # Device-side update keeps this CUDA graph replayable.
        if pid == 0:
            old_val = tl.load(iteration_ptr)
            tl.store(iteration_ptr, old_val + 1)

    @triton.jit
    def _recv_block_fn(
        win,
        dst_ptr,
        recv_staging_ptr,
        iteration_ptr,
        total_elements,
        section_elements,
        elem_size_bytes,
        pipeline_depth,
        num_blocks,
        pid,
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Block-scope recv: each block waits for and drains its own tile.

        Per-block signal IDs (monotonically increasing values):
        - DATA_READY signal ``pid``: data arrived from sender
        - SLOT_FREE signal ``N + pid``: window slot freed for sender
        """
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        iter_base = iteration * total_steps

        # Per-block signal IDs
        data_signal = pid
        slot_free_signal = num_blocks + pid

        tile_elements = section_elements // num_blocks
        my_element_offset = pid * tile_elements
        if pid == num_blocks - 1:
            tile_elements = section_elements - my_element_offset

        for step in range(total_steps):
            slot = step % pipeline_depth
            expected = iter_base + step + 1

            # -- Wait for data to arrive --
            wait_signal_from_block(win, peer_rank, data_signal, expected)
            # Ensure RDMA-written data in recv_staging is visible to GPU.
            _threadfence_system()

            # -- Copy my tile: recv_staging → dst --
            staging_start = slot * section_elements + my_element_offset
            dst_start = step * section_elements + my_element_offset

            remaining = total_elements - step * section_elements - my_element_offset
            copy_elements = tl.where(
                tile_elements < remaining,
                tile_elements,
                remaining,
            )

            if copy_elements > 0:
                for i in range(0, copy_elements, BLOCK_SIZE):
                    offs = i + tl.arange(0, BLOCK_SIZE)
                    mask = offs < copy_elements
                    data = tl.load(
                        recv_staging_ptr + staging_start + offs,
                        mask=mask,
                    )
                    tl.store(dst_ptr + dst_start + offs, data, mask=mask)

            # -- Signal sender: this slot is free --
            signal_block(win, peer_rank, slot_free_signal, 1)

    @requires_torchcomms
    @triton.jit
    def sendrecv_kernel(
        # torchcomms handles
        win,
        send_buf,
        # data pointers
        src_ptr,
        dst_ptr,
        send_staging_ptr,
        recv_staging_ptr,
        # iteration tracking (monotonic, GPU-resident)
        iteration_ptr,
        # sizes (in elements, not bytes)
        total_elements,
        section_elements,
        elem_size_bytes,
        # config
        pipeline_depth,
        num_blocks,  # must be equal for send and recv
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Block-scope pipelined sendrecv kernel.

        Launched with ``2 * num_blocks`` program instances.
        PIDs ``[0, num_blocks)`` execute the send path.
        PIDs ``[num_blocks, 2 * num_blocks)`` execute the recv path.

        The kernel loops ``iterations`` times internally, avoiding
        per-iteration kernel launch overhead. CUDA graph replayable:
        iteration_ptr is device-resident and updated by a separate
        increment kernel after this kernel completes.
        """
        pid = tl.program_id(axis=0)

        if pid < num_blocks:
            _send_block_fn(
                win,
                send_buf,
                src_ptr,
                send_staging_ptr,
                iteration_ptr,
                total_elements,
                section_elements,
                elem_size_bytes,
                pipeline_depth,
                num_blocks,
                pid,
                peer_rank,
                BLOCK_SIZE,
            )
        else:
            recv_pid = pid - num_blocks
            _recv_block_fn(
                win,
                dst_ptr,
                recv_staging_ptr,
                iteration_ptr,
                total_elements,
                section_elements,
                elem_size_bytes,
                pipeline_depth,
                num_blocks,
                recv_pid,
                peer_rank,
                BLOCK_SIZE,
            )
