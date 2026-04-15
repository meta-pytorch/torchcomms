# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
Triton-based pipelined send/recv kernel for GPU-to-GPU communication.

This module implements a copy-based sendrecv protocol using the torchcomms
window API (NCCLX GIN backend). The kernel is designed for InfiniBand RDMA
and uses a pipelined ring-buffer staging approach to overlap local copies
with network transfers.

Architecture
------------
The total message (src tensor) is divided into **sections**, the unit of
pipelining. A staging buffer acts as a ring buffer with ``pipeline_depth``
slots, each holding one section. Step ``s`` uses slot ``s % pipeline_depth``.

Data flow (IB path)::

    src -> send_staging[slot] -> (RDMA put via put_block) -> recv_staging[slot] -> dst
            local copy (GPU)      NIC DMA transfer            local copy (GPU)

Multi-block coordination
~~~~~~~~~~~~~~~~~~~~~~~~
- N send blocks cooperatively copy src -> send_staging. After all send blocks
  finish, the **last** block (detected via ``tl.atomic_add`` on a coordination
  counter) calls ``put_block`` for the entire section.
- M recv blocks cooperatively copy recv_staging -> dst. After all recv blocks
  finish, the **last** block calls ``signal_block`` to notify the sender that
  the staging slot is free for reuse.

Signal design (cumulative values, monotonic across iterations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let ``I`` = iteration index, ``S`` = total_steps per iteration.
``base = I * S``.

- ``signal_id=0`` (DATA_READY): ``put_block`` increments by 1 each step.
  Receiver waits for ``>= base + step + 1``.
- ``signal_id=1`` (SLOT_FREE): receiver's ``signal_block`` increments by 1
  each step after draining. Sender waits for ``>= base + step - P + 1``.
- ``counter_id=0`` (NIC_DONE): local counter incremented by 1 each step when
  the NIC finishes reading send_staging.
  Sender waits for ``>= base + step - P + 1``.

Coordination counters (per-slot, reset to 0 each iteration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``send_coord[slot]`` / ``recv_coord[slot]`` are reset to 0 before each call
via a GIN-safe Triton fill kernel. At step ``s`` (slot ``s % P``), this slot
has been used ``s // P + 1`` times in the current call.
Expected value = ``num_blocks * (s // P + 1)``.
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

    @triton.jit
    def _fill_kernel(ptr, value, N, BLOCK_SIZE: tl.constexpr):
        """Fill the first N elements of ptr with value (GIN-safe)."""
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        tl.store(ptr + offs, value, mask=mask)

    @triton.jit
    def _increment_iteration_kernel(iteration_ptr):
        """Increment the iteration counter by 1 (GIN-safe)."""
        old_val = tl.load(iteration_ptr)
        tl.store(iteration_ptr, old_val + 1)

    @triton.jit
    def _send_fn(
        win,
        send_buf,
        src_ptr,
        send_staging_ptr,
        send_coord_ptr,
        iteration_ptr,
        total_elements,
        section_elements,
        elem_size_bytes,
        pipeline_depth,
        num_send_blocks,
        pid,
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Send-side logic executed by each send block.

        Each send block copies its assigned portion of the current section from
        ``src_ptr`` into ``send_staging_ptr``. The last block to finish (detected
        via atomic increment on ``send_coord_ptr``) issues the RDMA put to the
        peer's recv staging buffer.
        """
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        base = iteration * total_steps

        # Each block handles a contiguous portion of the section.
        block_elements = section_elements // num_send_blocks
        my_element_offset = pid * block_elements
        # Last block absorbs any remainder elements.
        if pid == num_send_blocks - 1:
            block_elements = section_elements - my_element_offset

        for step in range(total_steps):
            slot = step % pipeline_depth

            # -- Wait for slot reuse (NIC must have finished reading this slot) --
            if step >= pipeline_depth:
                wait_counter(win, 0, base + step - pipeline_depth + 1)

            # -- Phase 1: Local copy src -> send_staging (all send blocks) --
            src_start = step * section_elements + my_element_offset
            staging_start = slot * section_elements + my_element_offset

            # Handle the last section which may be smaller than section_elements.
            remaining = total_elements - step * section_elements - my_element_offset
            copy_elements = tl.where(
                block_elements < remaining, block_elements, remaining
            )

            if copy_elements > 0:
                for i in range(0, copy_elements, BLOCK_SIZE):
                    offs = i + tl.arange(0, BLOCK_SIZE)
                    mask = offs < copy_elements
                    data = tl.load(src_ptr + src_start + offs, mask=mask)
                    tl.store(send_staging_ptr + staging_start + offs, data, mask=mask)

            # -- Phase 2: Coordinate; last block issues RDMA put --
            # fence(win)
            old_count = tl.atomic_add(send_coord_ptr + slot, 1)
            expected = num_send_blocks * (step // pipeline_depth + 1)

            if old_count + 1 == expected:
                # Last block: wait for receiver to free recv_staging[slot].
                if step >= pipeline_depth:
                    wait_signal_from(
                        win, peer_rank, 1, base + step - pipeline_depth + 1
                    )

                # Compute actual bytes for this section (last section may be
                # smaller).
                remaining_elements = total_elements - step * section_elements
                actual_section_elements = tl.where(
                    section_elements < remaining_elements,
                    section_elements,
                    remaining_elements,
                )
                actual_section_bytes = actual_section_elements * elem_size_bytes

                staging_byte_offset = slot * section_elements * elem_size_bytes

                # RDMA put: send_staging[slot] -> peer's recv_staging[slot].
                put_block(
                    win,
                    staging_byte_offset,  # dst_offset (bytes)
                    send_buf,
                    staging_byte_offset,  # src_offset (bytes)
                    peer_rank,
                    actual_section_bytes,
                    0,  # signal_id = DATA_READY
                    0,  # counter_id = NIC_DONE
                )

    @triton.jit
    def _recv_fn(
        win,
        dst_ptr,
        recv_staging_ptr,
        recv_coord_ptr,
        iteration_ptr,
        total_elements,
        section_elements,
        elem_size_bytes,
        pipeline_depth,
        num_recv_blocks,
        pid,
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Recv-side logic executed by each recv block.

        Each recv block copies its assigned portion of the current section from
        ``recv_staging_ptr`` into ``dst_ptr``. The last block to finish (detected
        via atomic increment on ``recv_coord_ptr``) signals the sender that the
        staging slot is free for reuse.
        """
        iteration = tl.load(iteration_ptr)
        total_steps = tl.cdiv(total_elements, section_elements)
        base = iteration * total_steps

        block_elements = section_elements // num_recv_blocks
        my_element_offset = pid * block_elements
        if pid == num_recv_blocks - 1:
            block_elements = section_elements - my_element_offset

        for step in range(total_steps):
            slot = step % pipeline_depth

            # -- Wait for data to arrive from sender --
            wait_signal_from(win, peer_rank, 0, base + step + 1)

            # -- Local copy: recv_staging[slot] -> dst (all recv blocks) --
            staging_start = slot * section_elements + my_element_offset
            dst_start = step * section_elements + my_element_offset

            remaining = total_elements - step * section_elements - my_element_offset
            copy_elements = tl.where(
                block_elements < remaining, block_elements, remaining
            )

            if copy_elements > 0:
                for i in range(0, copy_elements, BLOCK_SIZE):
                    offs = i + tl.arange(0, BLOCK_SIZE)
                    mask = offs < copy_elements
                    data = tl.load(recv_staging_ptr + staging_start + offs, mask=mask)
                    tl.store(dst_ptr + dst_start + offs, data, mask=mask)

            # -- Coordinate; last block signals SLOT_FREE --
            fence(win)
            old_count = tl.atomic_add(recv_coord_ptr + slot, 1)
            expected = num_recv_blocks * (step // pipeline_depth + 1)

            if old_count + 1 == expected:
                signal_block(win, peer_rank, 1, 1)  # SLOT_FREE += 1

    @requires_torchcomms
    @triton.jit
    def sendrecv_cooperative_kernel(
        # torchcomms handles
        win,
        send_buf,
        # data pointers
        src_ptr,
        dst_ptr,
        send_staging_ptr,
        recv_staging_ptr,
        # coordination counters
        send_coord_ptr,
        recv_coord_ptr,
        # iteration tracking (monotonic, GPU-resident)
        iteration_ptr,
        # sizes (in elements, not bytes)
        total_elements,
        section_elements,
        elem_size_bytes,
        # config
        pipeline_depth,
        num_send_blocks,
        num_recv_blocks,
        peer_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Pipelined send/recv kernel for GPU-to-GPU communication via RDMA.

        Launched with ``num_send_blocks + num_recv_blocks`` program instances.
        Program IDs ``[0, num_send_blocks)`` execute the send path; program IDs
        ``[num_send_blocks, num_send_blocks + num_recv_blocks)`` execute the
        recv path.

        Signals and counters are monotonic across iterations. The iteration
        counter (``iteration_ptr``) tracks how many times the kernel has been
        invoked, and wait thresholds are offset by ``iteration * total_steps``.
        """
        pid = tl.program_id(axis=0)

        if pid < num_send_blocks:
            _send_fn(
                win,
                send_buf,
                src_ptr,
                send_staging_ptr,
                send_coord_ptr,
                iteration_ptr,
                total_elements,
                section_elements,
                elem_size_bytes,
                pipeline_depth,
                num_send_blocks,
                pid,
                peer_rank,
                BLOCK_SIZE,
            )
        else:
            recv_pid = pid - num_send_blocks
            _recv_fn(
                win,
                dst_ptr,
                recv_staging_ptr,
                recv_coord_ptr,
                iteration_ptr,
                total_elements,
                section_elements,
                elem_size_bytes,
                pipeline_depth,
                num_recv_blocks,
                recv_pid,
                peer_rank,
                BLOCK_SIZE,
            )
