# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Python `Buffer` wrapper for the pipes MoE expert-parallel runtime.

The C++ runtime backing `Buffer` lives in `comms.prims.collectives.moe_ep._cpp`. For
Phase 1 (intranode, NVLink only) we only implement the dispatch / combine
methods exercised by `tests/test_intranode.py`; LL / internode methods are
stubbed and raise `NotImplementedError` until the follow-up diffs land.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
import torch.distributed as dist

# pyre-ignore[21]: cpp_python_extension at runtime
from comms.prims.collectives.moe_ep import _cpp  # @manual
from comms.prims.collectives.moe_ep.moe_ep.utils import (
    check_nvlink_connections,
    EventHandle,
    EventOverlap,
)

logger: logging.Logger = logging.getLogger(__name__)

Config = _cpp.Config  # type: ignore[misc]


_PEER_ACCESS_PREWARMED: bool = False


def _prewarm_peer_access(num_ranks: int) -> None:
    """Pre-enable peer access between every (cur_dev, peer) pair so PyTorch's
    HIP allocator's later `hipDeviceEnablePeerAccess` returns
    `hipErrorPeerAccessAlreadyEnabled` cleanly (PyTorch's caching allocator
    silently swallows that error). Idempotent — runs at most once per process.
    """
    global _PEER_ACCESS_PREWARMED
    if _PEER_ACCESS_PREWARMED:
        return
    if not torch.cuda.is_available():
        return
    cur = torch.cuda.current_device()
    n_devices = torch.cuda.device_count()
    n = min(num_ranks, n_devices)
    for peer in range(n):
        if peer == cur:
            continue
        if not torch.cuda.can_device_access_peer(cur, peer):
            continue
        try:
            torch.cuda._lazy_init()
            # Trigger BOTH directions: peer→cur (.to copy) and cur→peer
            # (an explicit CUDAGuard + write). Both directions must be
            # enabled before the dispatch kernel issues cross-rank atomics
            # and DMA transfers; PyTorch's caching allocator otherwise
            # tries to enable them lazily during the first kernel and
            # blows up with `hipErrorPeerAccessAlreadyEnabled` when NCCL
            # has already set them up.
            with torch.cuda.device(cur):
                _ = torch.zeros(1, device=f"cuda:{peer}").to(f"cuda:{cur}")
                _ = torch.zeros(1, device=f"cuda:{cur}").to(f"cuda:{peer}")
        except Exception as e:  # noqa: BLE001
            logger.debug("prewarm peer (%d <-> %d) failed: %s", cur, peer, e)
    _PEER_ACCESS_PREWARMED = True


class Buffer:
    """
    Core expert-parallel (EP) communication buffer for Mixture of Experts.

    Supports:
    - **High-throughput intranode** all-to-all (dispatch + combine, NVLink) — D3
    - **Low-latency** all-to-all (dispatch + combine, RDMA) — D4
    - **High-throughput internode** all-to-all (dispatch + combine, RDMA + NVLink) — D5

    Mirrors the `Buffer` API surface that the canonical `test_intranode.py`
    / `test_low_latency.py` / `test_internode.py` scripts in
    `comms/prims/collectives/moe_ep/tests/` exercise.
    """

    num_sms: int = 20

    rank: int
    group: dist.ProcessGroup | None
    group_size: int
    _internode_ready: bool = False

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        use_fabric: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
        # `comm: mpi4py.MPI.Comm | None` — accept Any to avoid hard mpi4py
        # dependency at import time.
        comm: object | None = None,
    ) -> None:
        if allow_mnnvl:
            raise NotImplementedError("allow_mnnvl=True is a v1 non-goal")
        if use_fabric:
            raise NotImplementedError("use_fabric=True is a v1 non-goal")
        if enable_shrink:
            raise NotImplementedError(
                "enable_shrink=True is a v1 non-goal (shrink-test family)"
            )

        check_nvlink_connections(group) if group is not None else None

        all_gather_object: Callable[[object], list[object]]
        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            def _all_gather_object_group(obj: object) -> list[object]:
                # pyre-ignore[9]: None pre-fill, overwritten by all_gather_object
                object_list: list[object] = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
                return object_list

            all_gather_object = _all_gather_object_group
        elif comm is not None:
            # mpi4py path
            self.rank = comm.Get_rank()  # pyre-ignore[16]
            # pyre-ignore[8]: mpi4py comm fallback (not a ProcessGroup)
            self.group = comm
            self.group_size = comm.Get_size()  # pyre-ignore[16]

            def _all_gather_object_mpi(obj: object) -> list[object]:
                return list(comm.allgather(obj))  # pyre-ignore[16]

            all_gather_object = _all_gather_object_mpi
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")

        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        # Cached layout from the most recent non-cached dispatch, reused on
        # subsequent cached calls (handle != None). See `dispatch()` for the
        # rationale.
        self._last_num_tokens_per_rank: torch.Tensor | None = None
        self._last_is_token_in_rank: torch.Tensor | None = None
        self._last_num_tokens_per_expert: torch.Tensor | None = None
        self.runtime = _cpp.Buffer(  # type: ignore[attr-defined]
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            explicitly_destroy,
            enable_shrink,
            use_fabric,
        )

        # On AMD, PyTorch's distributed `_object_to_tensor` path fails on the
        # first `dist.all_gather_object(...)` after our `_cpp.Buffer`
        # ctor with `hipErrorPeerAccessAlreadyEnabled`. The error originates
        # in `torch.ByteTensor(byte_storage).to(device)` because PyTorch's
        # caching allocator tries to enable peer access between the rank's
        # GPU and other GPUs that NCCL/RCCL has already set up. Pre-warm
        # peer access between every pair *before* the first collective so
        # PyTorch sees the access already exists and short-circuits — this
        # is benign on NVIDIA (idempotent) and avoids the AMD failure.
        _prewarm_peer_access(self.group_size)

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)

        # NVSHMEM unique-id exchange — RDMA / LL only (Phase 2/3). For
        # Phase 1 (intranode-only), get_num_rdma_ranks() returns 1 and
        # low_latency_mode is False, so we skip this branch entirely.
        root_unique_id: object | None = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            if num_qps_per_rank <= 0:
                raise ValueError("num_qps_per_rank must be > 0 for RDMA mode")

            # The pipes IBGDA transport doesn't use NVSHMEM env vars; the
            # equivalent settings live on `MultipeerIbgdaTransportConfig`,
            # set inside the C++ runtime when constructing the IBGDA
            # transport in Phase 2/3 (D4 / D5).

            if (low_latency_mode and self.rank == 0) or (
                not low_latency_mode and self.runtime.get_rdma_rank() == 0
            ):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            nvshmem_unique_ids = all_gather_object(root_unique_id)
            root_unique_id = nvshmem_unique_ids[
                0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)
            ]

        # Finalize — give the C++ runtime the gathered metadata.
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        if not self.runtime.is_available():
            raise RuntimeError("Buffer.sync() failed: runtime not available")

    def destroy(self) -> None:
        """Destroy the C++ runtime. Requires `explicitly_destroy=True`."""
        if not self.explicitly_destroy:
            raise RuntimeError(
                "destroy() requires `explicitly_destroy=True` at construction"
            )
        self.runtime.destroy()
        # pyre-ignore[8]: deliberately invalidate
        self.runtime = None

    @staticmethod
    def is_sm90_compiled() -> bool:
        # pyre-ignore[16]
        return _cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        if new_num_sms % 2 != 0:
            raise ValueError("num_sms must be even (sender/receiver block pairs)")
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        # pyre-ignore[16]
        return _cpp.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> object:  # Config
        # Tuned dispatch configs per EP rank count.
        config_map: dict[int, object] = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 32, 288, 8, 128),
            32: Config(Buffer.num_sms, 32, 288, 8, 128),
            48: Config(Buffer.num_sms, 32, 288, 8, 128),
            64: Config(Buffer.num_sms, 32, 288, 8, 128),
            96: Config(Buffer.num_sms, 20, 480, 12, 128),
            128: Config(Buffer.num_sms, 20, 560, 12, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        if num_ranks not in config_map:
            raise ValueError(f"Unsupported number of EP ranks: {num_ranks}")
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> object:  # Config
        config_map: dict[int, object] = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            48: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 8, 128),
            72: Config(Buffer.num_sms, 1, 288, 8, 128),
            96: Config(Buffer.num_sms, 1, 480, 8, 128),
            128: Config(Buffer.num_sms, 1, 560, 8, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        if num_ranks not in config_map:
            raise ValueError(f"Unsupported number of EP ranks: {num_ranks}")
        return config_map[num_ranks]

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        EventOverlap,
    ]:
        """Compute the per-rank / per-expert layout from `topk_idx`."""
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.runtime.get_dispatch_layout(
            topk_idx,
            num_experts,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

    def dispatch(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        handle: tuple[Any, ...] | None = None,
        num_tokens_per_rank: torch.Tensor | None = None,
        num_tokens_per_rdma_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        num_tokens_per_expert: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: object | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[Any, ...]:
        """Dispatch tokens to per-expert ranks. Phase 1: intranode (NVLink)."""
        # Route to internode_dispatch when crossing nodes (Phase 3).
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(
                x,
                handle,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )
        # Intranode call — implemented in C++ runtime via PyBindings.
        # C++ returns a 7-tuple:
        #   (recv_x_obj, recv_x_scales, recv_topk_idx, recv_topk_weights,
        #    num_recv_tokens_per_expert_list, handle_dict, event)
        # where `recv_x_obj` is already (data, scales) for FP8 and a Tensor
        # otherwise, and `handle_dict` is a py::dict with the cached
        # buffers. Reshape to the 6-tuple layout so callers (and
        # `test_intranode.py`) can unpack:
        #   (recv_x, recv_topk_idx, recv_topk_weights,
        #    num_recv_tokens_per_expert_list, handle, event)
        # with `handle = (rank_prefix_matrix, channel_prefix_matrix,
        #                 recv_channel_prefix_matrix, recv_src_idx,
        #                 is_token_in_rank, send_head)`.
        #
        # Cached-dispatch path (handle is not None): the C++ runtime currently
        # always re-runs notify_dispatch (the cached_notify_dispatch fast path
        # is disabled until kernel-side debug). To make the call valid we
        # need num_tokens_per_rank / is_token_in_rank / num_tokens_per_expert
        # — when the caller doesn't pass them (the cached test path), we
        # reuse the values from the most recent non-cached call. They're
        # invariant across cached calls on the same input `x`.
        # When the caller passes a 6-tuple handle (from a prior dispatch's
        # `out_handle`), repack into the dict shape that C++ expects so
        # `py::isinstance<py::dict>` returns True and the cached path is
        # taken. Otherwise stash kwargs for future cached calls.
        handle_for_cpp = handle
        if handle is not None:
            if isinstance(handle, tuple) and len(handle) == 6:
                (
                    rank_prefix_matrix,
                    channel_prefix_matrix,
                    recv_channel_prefix_matrix,
                    recv_src_idx,
                    h_is_token_in_rank,
                    send_head,
                ) = handle
                handle_for_cpp = {
                    "rank_prefix_matrix": rank_prefix_matrix,
                    "channel_prefix_matrix": channel_prefix_matrix,
                    "recv_channel_prefix_matrix": recv_channel_prefix_matrix,
                    "recv_src_idx": recv_src_idx,
                    "send_head": send_head,
                    "num_recv_tokens": int(recv_src_idx.size(0)),
                }
                if is_token_in_rank is None:
                    is_token_in_rank = h_is_token_in_rank
            if num_tokens_per_rank is None:
                num_tokens_per_rank = self._last_num_tokens_per_rank
            if is_token_in_rank is None:
                is_token_in_rank = self._last_is_token_in_rank
            if num_tokens_per_expert is None:
                num_tokens_per_expert = self._last_num_tokens_per_expert
        else:
            # Stash for the next cached call.
            self._last_num_tokens_per_rank = num_tokens_per_rank
            self._last_is_token_in_rank = is_token_in_rank
            self._last_num_tokens_per_expert = num_tokens_per_expert
        (
            recv_x_obj,
            _recv_x_scales,  # already merged into `recv_x_obj` for FP8
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle_dict,
            event,
        ) = self.runtime.intranode_dispatch(
            x,
            handle_for_cpp,
            num_tokens_per_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            topk_idx,
            topk_weights,
            expert_alignment,
            num_worst_tokens,
            config or Buffer.get_dispatch_config(self.group_size),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )
        out_handle = (
            handle_dict["rank_prefix_matrix"],
            handle_dict["channel_prefix_matrix"],
            handle_dict["recv_channel_prefix_matrix"],
            handle_dict["recv_src_idx"],
            is_token_in_rank,
            handle_dict["send_head"],
        )
        return (
            recv_x_obj,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            out_handle,
            EventOverlap(event),
        )

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple[Any, ...],
        topk_weights: torch.Tensor | None = None,
        bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        config: object | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, EventOverlap]:
        """Combine reduces dispatched tokens. Phase 1: intranode (NVLink)."""
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(
                x,
                handle,
                topk_weights,
                bias,
                config,
                previous_event,
                async_finish,
                allocate_on_comm_stream,
            )
        bias_0, bias_1 = self._unpack_bias(bias)
        # Unpack the 6-tuple handle returned by `dispatch` and
        # re-pack into the dict-shaped handle our C++ `intranode_combine`
        # consumes (only the cached tensors it actually reads need to be
        # present; `is_token_in_rank` is dispatch-side bookkeeping).
        #
        # CRITICAL: combine uses `recv_channel_prefix_matrix`
        # (position 2) NOT `channel_prefix_matrix` (position 1).
        # The channel_prefix_matrix (pos 1) describes the SENDER's per-channel
        # distribution; recv_channel_prefix_matrix (pos 2) describes the
        # RECEIVER's. Combine needs the receiver's view.
        (
            rank_prefix_matrix,
            _channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            _is_token_in_rank,
            send_head,
        ) = handle
        handle_dict = {
            "rank_prefix_matrix": rank_prefix_matrix,
            "channel_prefix_matrix": recv_channel_prefix_matrix,
            "recv_channel_prefix_matrix": recv_channel_prefix_matrix,
            "recv_src_idx": recv_src_idx,
            "send_head": send_head,
        }
        combined_x, combined_topk_weights, event = self.runtime.intranode_combine(
            x,
            topk_weights,
            bias_0,
            bias_1,
            handle_dict,
            config or Buffer.get_combine_config(self.group_size),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )
        return combined_x, combined_topk_weights, EventOverlap(event)

    @staticmethod
    def _unpack_bias(
        bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if bias is None:
            return None, None
        if isinstance(bias, torch.Tensor):
            return bias, None
        if isinstance(bias, tuple) and len(bias) == 2:
            return bias[0], bias[1]
        raise TypeError(
            "bias must be Tensor, (Tensor, Tensor), or None — got "
            f"{type(bias).__name__}"
        )

    def internode_dispatch(self, *args: object, **kwargs: object) -> tuple[Any, ...]:
        """Phase 3 (D5) — internode dispatch (RDMA + NVLink)."""
        raise NotImplementedError(
            "internode_dispatch lands in D5 of the design plan stack"
        )

    def internode_combine(self, *args: object, **kwargs: object) -> tuple[Any, ...]:
        """Phase 3 (D5) — internode combine (RDMA + NVLink)."""
        raise NotImplementedError(
            "internode_combine lands in D5 of the design plan stack"
        )

    def low_latency_dispatch(self, *args: object, **kwargs: object) -> tuple[Any, ...]:
        """Phase 2 (D4) — low-latency dispatch (IBGDA)."""
        raise NotImplementedError(
            "low_latency_dispatch lands in D4 of the design plan stack"
        )

    def low_latency_combine(self, *args: object, **kwargs: object) -> tuple[Any, ...]:
        """Phase 2 (D4) — low-latency combine (IBGDA)."""
        raise NotImplementedError(
            "low_latency_combine lands in D4 of the design plan stack"
        )

    def clean_low_latency_buffer(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ) -> None:
        """Phase 2 (D4) — zero the LL buffer regions."""
        raise NotImplementedError(
            "clean_low_latency_buffer lands in D4 of the design plan stack"
        )

    def get_next_low_latency_combine_buffer(
        self, handle: tuple[Any, ...]
    ) -> torch.Tensor:
        """Phase 2 (D4) — `zero_copy=True` combine path."""
        raise NotImplementedError("get_next_low_latency_combine_buffer lands in D4")
