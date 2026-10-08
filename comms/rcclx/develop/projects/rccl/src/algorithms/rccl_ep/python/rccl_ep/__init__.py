# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# The Python surface: a ctypes binding to the rccl_ep C ABI.
#
# See LICENSE.txt for license information

import ctypes
import os
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist

__version__ = "0.1.0"

# The public API uses int64 topk indices.
__all__ = ["ElasticBuffer", "EventOverlap", "EPHandle", "GroupedDispatch", "topk_idx_t"]

topk_idx_t = torch.int64

# How long an uncached dispatch waits for its peers' counts before giving up with a
# diagnostic instead of hanging. Generous: the wait absorbs cross-rank skew.
_WAIT_TIMEOUT_MS = int(os.environ.get("RCCL_EP_WAIT_TIMEOUT_MS", "600000"))
# CTA budget for the grouped epilogue, a purely local copy; 0 keeps the dispatch's budget,
# which is sized for the push rather than for this kernel.
_EPILOGUE_SMS = int(os.environ.get("RCCL_EP_EPILOGUE_SMS", "0"))

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        # An EP-capable librccl, loaded RTLD_GLOBAL before the extension so
        # that it and torch resolve to the same one. Without it the loader
        # binds our DT_NEEDED to whichever librccl arrived first, typically
        # torch's, which predates the device API this path needs.
        rccl = os.environ.get("RCCL_EP_LIBRCCL")
        if rccl and os.path.exists(rccl):
            ctypes.CDLL(rccl, mode=ctypes.RTLD_GLOBAL)

        default = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "librccl_ep.so"
        )
        # `or default`: an empty RCCL_EP_LIB would otherwise reach ctypes.CDLL(""),
        # which opens the main executable and fails with "undefined symbol: ep_create".
        path = os.environ.get("RCCL_EP_LIB") or default
        if path == default and not os.path.exists(default):
            # Built with ENABLE_RCCL_EP_IN_LIBRCCL: the entry points are in
            # librccl and no standalone library exists. Which layout you have
            # is a property of how RCCL was built, not anything visible here,
            # so the package handles both rather than making the caller say.
            # Named explicitly, because the bare dlopen failure would report a
            # missing librccl_ep.so to someone who deliberately did not build one.
            if not rccl:
                raise RuntimeError(
                    f"no {default}, and RCCL_EP_LIBRCCL is unset. Either install "
                    "the standalone library, or point RCCL_EP_LIBRCCL at an RCCL "
                    "built with ENABLE_RCCL_EP_IN_LIBRCCL -- see README.md."
                )
            path = rccl
        # RTLD_LOCAL for the extension itself: it exports the same symbol
        # names as librccl, and there is no reason to put those in the global
        # namespace. RTLD_DEEPBIND is not an option -- it would also redirect
        # the C++ runtime and HIP symbols shared with torch, and segfaults.
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        P, I = ctypes.c_size_t, ctypes.c_int
        lib.ep_create.restype = ctypes.c_void_p
        # The unique id is an opaque binary blob, not a string: raw pointers
        # rather than c_char_p, which is for NUL-terminated data.
        lib.ep_create.argtypes = [I, I, ctypes.c_void_p, I, I, I]
        lib.ep_configure.argtypes = [ctypes.c_void_p, I, I]
        lib.ep_window_bytes.restype = ctypes.c_size_t
        lib.ep_window_bytes.argtypes = [ctypes.c_void_p]
        lib.ep_destroy.argtypes = [ctypes.c_void_p]
        # Every data-path entry takes the caller's stream as a trailing argument.
        lib.ep_barrier.argtypes = [ctypes.c_void_p, P]
        lib.ep_plan.argtypes = [ctypes.c_void_p, P, I, P, P, P, P]
        lib.ep_dispatch.restype = I
        lib.ep_dispatch.argtypes = [ctypes.c_void_p, P, P, P, P, I, P, P, I, I, P, P, P, P, P, P]
        lib.ep_dispatch_v2.restype = I
        lib.ep_dispatch_v2.argtypes = [ctypes.c_void_p, P, P, P, P, I, P, P, I, I, P, P, P, P, P,
                                       P, P, P]
        lib.ep_plan_notify_v2.restype = I
        lib.ep_plan_notify_v2.argtypes = [ctypes.c_void_p, P, I, P, P, P, P, P, P, P, P, P]
        lib.ep_plan_notify_v3.restype = I
        lib.ep_plan_notify_v3.argtypes = [ctypes.c_void_p, P, P, I, P, P, P, P, P, P, P, P, P]
        lib.ep_dispatch_grouped.restype = I
        lib.ep_dispatch_grouped.argtypes = [ctypes.c_void_p, P, P, P, I, P, P, P, I, I, P]
        lib.ep_grouped_epilogue.restype = I
        lib.ep_grouped_epilogue.argtypes = [ctypes.c_void_p, P, P, P, I, P, P, P, P, P, P, P,
                                            P, I, P]
        lib.ep_dispatch_payload_v2.restype = I
        lib.ep_dispatch_payload_v2.argtypes = [ctypes.c_void_p, P, I, P, P, P, I, P, P]
        lib.ep_dispatch_payload.restype = I
        lib.ep_dispatch_payload.argtypes = [ctypes.c_void_p, P, I, P, P, I, P, P]
        lib.ep_plan_notify.restype = I
        lib.ep_plan_notify.argtypes = [ctypes.c_void_p, P, I, P, P, P, P, P, P]
        lib.ep_dispatch_v3.restype = I
        lib.ep_dispatch_v3.argtypes = [ctypes.c_void_p, P, P, P, P, I, P, P, I, I, P, P, P, P, P, P]
        lib.ep_wait_counts.restype = I
        lib.ep_wait_counts.argtypes = [ctypes.c_void_p, ctypes.c_void_p, I]
        lib.ep_wait_counts_v2.restype = I
        lib.ep_wait_counts_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, I, I]
        lib.ep_recv_counts.argtypes = [ctypes.c_void_p, P, P]
        lib.ep_expert_counts.argtypes = [ctypes.c_void_p, P, I, P, P]
        lib.ep_expand_build.restype = I
        lib.ep_expand_build.argtypes = [ctypes.c_void_p, P, I, I, P, P, P, P, P, P]
        lib.ep_expand_scatter.argtypes = [ctypes.c_void_p, I, P, P, P, P, I, P, P,
                                          I, I, P, I, I, P, P, I, P]
        lib.ep_combine.restype = I
        lib.ep_combine.argtypes = [ctypes.c_void_p, P, P, P, P, P, I, P, I, P, P, I, P, P, I, P]
        lib.ep_combine_v4.restype = I
        lib.ep_combine_v4.argtypes = [ctypes.c_void_p, P, P, P, P, P, I, P, P, I, P, I, P, P, I,
                                      P, P, I, I, P]
        lib.ep_combine_finish.restype = I
        lib.ep_combine_finish.argtypes = [ctypes.c_void_p, P]
        lib.ep_dispatch_defer_next.restype = I
        lib.ep_dispatch_defer_next.argtypes = [ctypes.c_void_p]
        lib.ep_dispatch_finish.restype = I
        lib.ep_dispatch_finish.argtypes = [ctypes.c_void_p, P]
        lib.ep_combine_v3.restype = I
        lib.ep_combine_v3.argtypes = [ctypes.c_void_p, P, P, P, P, P, I, P, P, I, P, I, P, P, I,
                                      P, P, I, P]
        lib.ep_combine_v2.restype = I
        lib.ep_combine_v2.argtypes = [ctypes.c_void_p, P, P, P, P, P, P, I, P, I, P, P, I, P, P,
                                      I, P]
        lib.ep_create_v2.restype = ctypes.c_void_p
        lib.ep_create_v2.argtypes = [I, I, ctypes.c_void_p, I, I, I, I]
        lib.ep_inter_counts.restype = I
        lib.ep_inter_counts.argtypes = [ctypes.c_void_p, P, ctypes.c_void_p, P]
        lib.ep_inter_bind.restype = I
        lib.ep_inter_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, P, I]
        lib.ep_configure_v2.restype = I
        lib.ep_configure_v2.argtypes = [ctypes.c_void_p, I, I, I]
        lib.ep_hybrid_enable.restype = I
        lib.ep_hybrid_enable.argtypes = [ctypes.c_void_p, I]
        lib.ep_hplan.restype = I
        lib.ep_hplan.argtypes = [ctypes.c_void_p, P, I, P, P, P, P, P, P, P]
        lib.ep_hcounts.restype = I
        lib.ep_hcounts.argtypes = [ctypes.c_void_p, P, P, P, I, ctypes.c_void_p, ctypes.c_void_p, P]
        lib.ep_hbind.restype = I
        lib.ep_hbind.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4 + [P] * 5 + [I]
        lib.ep_get_unique_id.argtypes = [ctypes.c_void_p]
        lib.ep_unique_id_size.restype = I
        _LIB = lib
    return _LIB


def _ptr(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.data_ptr()


def _bias(t: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    """Normalise a combine bias to what the kernel actually reads.

    combine.h loads bias0/bias1 as a dense bf16 [num_tokens, hidden] at 16-byte
    stride. A column slice of a wider buffer has the right shape and the wrong
    stride, and a float32 bias has the right stride and half the reach; neither
    raises anywhere, they just read from the wrong offsets. Copy the first and
    refuse the second, rather than converting silently -- a down-cast to bf16
    would change the result the strict-order accumulation exists to make exact.
    """
    if t is None:
        return None
    if t.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
    return t.contiguous()


def _h2d_i32(vals, device) -> torch.Tensor:
    """Copy a small host list to an int32 device tensor without synchronizing the stream.

    torch.tensor(vals, device=cuda) is a blocking copy that would wait for any enqueued
    dispatch. device="cpu" is explicit because callers may set a cuda default device."""
    return torch.tensor(vals, dtype=torch.int32, device="cpu").pin_memory().to(device, non_blocking=True)


def _align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


class EventOverlap:
    """Completion handle returned by the data path.

    rccl_ep's data path is enqueued on the caller's current stream, so anything
    that later runs on that stream is already ordered after it and no event is
    needed. The type is kept because callers hold it, pass it as
    `previous_event` and call `current_stream_wait`; all three stay meaningful,
    they just never block.
    """

    def __init__(self, event=None):
        self.event = event

    def current_stream_wait(self):
        if self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class EPHandle:
    """Routing metadata produced by dispatch and consumed by combine.

    Attribute names are part of the public API, because callers read them
    directly.
    """

    def __init__(self):
        self.topk_idx = None                            # as passed, or a copy
        self.topk_idx_i32 = None
        self.slot = None                                # routing plan
        self.send_list = None                           # dense slot -> token
        self.sendc = None
        self.num_tokens = 0
        self.num_recv = 0
        self.recv_src_metadata = None                   # [n, 2 + num_topk]
        self.recv_topk_idx = None                       # local ids, [n, num_topk]
        self.dst_buffer_slot_idx = None
        self.psum_num_recv_tokens_per_scaleup_rank = None
        self.psum_num_recv_tokens_per_expert = None
        self.num_recv_tokens_per_expert_list = None
        self.expert_counts = None
        self.expert_offsets = None
        self.row_map = None                             # expanded layout only
        self.expert_alignment = 1
        self.expanded = False
        self.grouped = False                            # from dispatch_grouped()
        self.recv_pairs_per_rank = None                 # uncached: see _wait_counts
        self.max_tokens_per_rank = None
        self.num_expanded_rows = 0
        # Non-expanded dispatch keeps just the source-index column; the full
        # recv_src_metadata is built from it on first access (see below).
        self.recv_src = None                            # [n] int32, contiguous
        # Internode only: this plan's [sendc] and [recvc] on the host, from the count
        # exchange; every dispatch or combine on the handle binds them (_inter_bind).
        self.inter_sendc = None
        self.inter_recvc = None
        # Hybrid only: the node plan, per-node record counts and the forwarder's fwd_meta.
        # _psum_src is the per-source receive prefix for the payload replay; in hybrid,
        # psum_num_recv_tokens_per_scaleup_rank is per scale-up rank instead.
        self.node_pos = self.node_list = self.node_cnt = None
        self.hyb_nsend = self.hyb_nrecv = None
        self.fwd_meta = None
        self._psum_src = None
        self._src_rows_per_rank = 0
        self._num_topk = 0

    @property
    def recv_src_metadata(self):
        """[n, 2 + num_topk]: src_token_global_idx, source rank, then the expand
        map (-1 without one). Built lazily for non-expanded dispatch, whose only
        in-library reader, combine(), takes recv_src directly."""
        if self._recv_src_metadata is None and self.recv_src is not None:
            meta = torch.empty((self.recv_src.numel(), 2 + self._num_topk),
                               dtype=torch.int32, device=self.recv_src.device)
            meta[:, 0] = self.recv_src
            meta[:, 1] = self.recv_src // self._src_rows_per_rank
            meta[:, 2:] = -1
            self._recv_src_metadata = meta
        return self._recv_src_metadata

    @recv_src_metadata.setter
    def recv_src_metadata(self, value):
        self._recv_src_metadata = value

    @property
    def psum_num_recv_tokens_per_expert(self):
        """Inclusive prefix of the per-expert counts. dispatch_grouped() builds it on
        first access, so a caller that needs only the counts keeps a launch off the host
        path after the count wait."""
        if self._psum_expert is None and self.grouped and self.expert_counts is not None:
            self._psum_expert = torch.cumsum(self.expert_counts, 0, dtype=torch.int32)
        return self._psum_expert

    @psum_num_recv_tokens_per_expert.setter
    def psum_num_recv_tokens_per_expert(self, value):
        self._psum_expert = value


class GroupedDispatch(NamedTuple):
    """dispatch_grouped()'s result, in grouped-by-expert layout: expert-major, then
    ascending receive index (source rank, then token order within a source). rows is
    the sum of the per-expert counts."""
    tokens: torch.Tensor             # [rows, hidden] bf16, contiguous
    row_topk_ids: torch.Tensor       # [rows, num_topk] int64, the row's token's local ids
    row_topk_weights: torch.Tensor   # [rows, num_topk] fp32, the row's token's weights
    expert_rows: torch.Tensor        # [experts_per_rank, num_recv] int64, row or -1
    topk_ids: torch.Tensor           # [num_recv, num_topk] int64, local ids or -1
    topk_weights: torch.Tensor       # [num_recv, num_topk] fp32
    handle: "EPHandle"


class ElasticBuffer:
    """Expert-parallel dispatch/combine buffer.

    Implemented: BF16 and FP8 dispatch, expanded (grouped-by-expert) dispatch
    with alignment and zero padding, cached replay, both combine reduction
    recipes, one or two bias tensors, and deterministic ordering -- which is
    inherent here, since routing uses no atomics and no arrival-order
    dependence, so repeated runs are byte-identical.

    Scale-up by default: every peer must be reachable by load and store, so a
    communicator spanning more than one node is rejected at construction.
    RCCL_EP_INTERNODE=1 reaches peers on other nodes by RCCL send/recv instead,
    in a direct mode: one flat (1, num_ranks) domain with no per-node
    pre-reduction, whatever allow_hybrid_mode says. Nodes are grouped by
    NCCL_HOSTID, else the host name, and each node's ranks must be contiguous.
    dispatch_grouped() is intranode only.
    """

    def __init__(self, group, num_max_tokens_per_rank, hidden,
                 num_experts=None, num_topk=None,
                 deterministic=False, allow_hybrid_mode=False,
                 allow_multiple_reduction=False,
                 prefer_overlap_with_compute=False, sl_idx=0,
                 num_allocated_qps=0, explicitly_destroy=False,
                 num_gpu_timeout_secs=100, num_cpu_timeout_secs=100,
                 grouped_combine_only=False, **kwargs):
        # Standard callers pass True for both, so refusing them outright would
        # leave the standard invocation unable to run at all.
        #
        # hybrid mode: on a single node there is no scale-out leg, so
        # num_scaleout_ranks is 1 and every hybrid branch degenerates to the
        # pure scale-up path we implement. Accepting it is therefore honest
        # HERE and only here -- across nodes it would silently do the wrong
        # thing, so the single-node property is verified rather than assumed.
        self.internode = os.environ.get("RCCL_EP_INTERNODE", "0") == "1"
        world = dist.get_world_size(group)
        node_size = world
        if allow_hybrid_mode or self.internode:
            import socket
            # NCCL_HOSTID, when set, is what RCCL itself groups ranks into nodes by, so the
            # LSA team ep_configure checks against is built from the same key.
            hosts = [None] * world
            dist.all_gather_object(hosts, os.environ.get("NCCL_HOSTID") or socket.gethostname(),
                                   group)
            if len(set(hosts)) != 1:
                if not self.internode:
                    raise NotImplementedError(
                        f"allow_hybrid_mode needs a scale-out leg, which this path does "
                        f"not provide, and this communicator spans {len(set(hosts))} "
                        f"nodes. It is accepted only on a single node, where it "
                        f"degenerates to the scale-up path; RCCL_EP_INTERNODE=1 selects "
                        f"the internode path.")
                node_size = hosts.count(hosts[0])
                if (world % node_size != 0 or len(set(hosts)) != world // node_size
                        or any(hosts[r] != hosts[r - r % node_size] for r in range(world))):
                    raise ValueError(
                        f"the internode path needs equal nodes, each holding contiguous "
                        f"ranks; this communicator's nodes are {hosts}")
        self.internode = node_size < world
        self.num_node_ranks = node_size
        # Two-level internode scheme, enabled by allow_hybrid_mode.
        self.hybrid = self.internode and bool(allow_hybrid_mode)
        # If set, the window's combine region holds one row per slot, so combine() must be
        # grouped (non-expanded handles, or allow_multiple_reduction).
        self.grouped_combine_only = bool(grouped_combine_only)
        # multiple reduction: the expanded combine has to apply the per-rank
        # grouped reduction before sending, instead of shipping one row per
        # (token, expert). Implemented; see combine().
        self.allow_multiple_reduction = bool(allow_multiple_reduction)
        self.allow_hybrid_mode = bool(allow_hybrid_mode)

        self.group = group
        self.rank_idx = dist.get_rank(group)
        self.num_ranks = dist.get_world_size(group)
        self.scaleup_rank_idx = self.rank_idx
        self.scaleout_rank_idx = 0  # hybrid: set below, once the node size is known
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.deterministic = deterministic
        self.num_allocated_qps = 0        # LSA path uses no QPs
        self.explicitly_destroy = explicitly_destroy
        self._configured = None
        # Host buffer ep_wait_counts_v2 fills, reused across calls:
        # [total, counts[epr], recv_pairs[num_ranks], max_tokens].
        self._wait_buf = None
        # Internode: the pinned [total, counts[epr]] an uncached dispatch waits on.
        self._inter_host = None

        lib = _lib()
        n = lib.ep_unique_id_size()
        buf = ctypes.create_string_buffer(n)
        # Unchecked, a failure here broadcasts the zero-filled buffer and every
        # rank hangs inside ncclCommInitRank on an id that is not one.
        if self.rank_idx == 0 and lib.ep_get_unique_id(buf) != 0:
            raise RuntimeError("ep_get_unique_id failed")
        # The bootstrap group may be gloo (CPU-only) or nccl (GPU-only), and
        # neither accepts the other's tensors, so follow the backend.
        t = torch.frombuffer(bytearray(buf.raw), dtype=torch.uint8).clone()
        t = t.cuda() if dist.get_backend(group) != "gloo" else t.cpu()
        # `group_src=0` names rank 0 *within* group, matching the rank_idx == 0
        # source check above. `src=0` is a global rank: torch raises on an EP
        # subgroup that excludes global rank 0, and on one that holds it at a
        # non-zero group index the zero-filled buffer wins the broadcast and
        # every rank calls ncclCommInitRank on a non-id -- the hang above.
        dist.broadcast(t, group_src=0, group=group)
        t = t.cpu()
        raw = bytes(bytearray(t.tolist()))

        if self.internode:
            self._h = lib.ep_create_v2(self.rank_idx, self.num_ranks, raw,
                                       num_max_tokens_per_rank, hidden,
                                       torch.cuda.current_device(), node_size)
        else:
            self._h = lib.ep_create(self.rank_idx, self.num_ranks, raw,
                                    num_max_tokens_per_rank, hidden,
                                    torch.cuda.current_device())
        if not self._h:
            raise RuntimeError("ep_create failed")
        if self.hybrid:
            if lib.ep_hybrid_enable(self._h, 1 if self.allow_multiple_reduction else 0) != 0:
                raise RuntimeError("ep_hybrid_enable failed")
            self.scaleout_rank_idx = self.rank_idx // node_size
            self.scaleup_rank_idx = self.rank_idx % node_size
        if self.internode and self.rank_idx % node_size == 0:
            print(f"[rccl_ep] internode ({'hybrid' if self.hybrid else 'direct'}): "
                  f"{world // node_size} nodes x {node_size} ranks, rank {self.rank_idx}; "
                  f"remote peers via RCCL send/recv", flush=True)

    # ---- informational -------------------------------------------------
    def get_logical_domain_size(self, *a, **k):
        if self.hybrid:
            return (self.num_ranks // self.num_node_ranks, self.num_node_ranks)
        return (1, self.num_ranks)          # (scaleout, scaleup)

    def get_theoretical_num_sms(self, *a, **k):
        # Measured on gfx950: peer-copy bandwidth is flat at ~6.0-6.7 GB/s per
        # CU up to 64 CUs, and dispatch reaches 81% of the achievable ceiling
        # at 128 total CTAs, which is where it stops improving. Constants
        # fitted on Hopper would under-provision here by roughly 2x.
        #
        # This is a TOTAL CTA budget across the grid, which is how num_sms is
        # defined here and how ep_dispatch divides it up.
        return 128

    def get_theoretical_num_qps(self, *a, **k):
        return 0

    def capture(self):
        ev = torch.cuda.Event()
        ev.record()
        return EventOverlap(ev)

    def barrier(self):
        # Before the first dispatch the handle is unconfigured, so the C side
        # returns -1 without launching anything. Swallowing that would let the
        # caller believe the ranks had rendezvoused.
        if _lib().ep_barrier(self._h, torch.cuda.current_stream().cuda_stream) != 0:
            raise RuntimeError("ep_barrier failed")

    def destroy(self):
        if getattr(self, "_h", None):
            _lib().ep_destroy(self._h)
            self._h = None

    # ---- internals -------------------------------------------------------
    def _configure(self, num_experts, num_topk):
        key = (num_experts, num_topk)
        if self._configured == key:
            return
        # A failed ep_configure has already released the old window on the C
        # side, so keeping the stale key here would short-circuit a retry at the
        # previous shape and leave dispatch running against no window at all.
        self._configured = None
        if _lib().ep_configure_v2(self._h, num_experts, num_topk,
                                  1 if self.grouped_combine_only else 0) != 0:
            raise RuntimeError(f"ep_configure({num_experts}, {num_topk}) failed")
        self._configured = key
        self.num_experts, self.num_topk = num_experts, num_topk

    def _inter_counts(self, h, stream):
        """Internode: exchange the plan's send counts (ep_inter_counts, one host sync) and
        keep [sendc, recvc] on the handle, where every later call on it binds them."""
        R = self.num_ranks
        out = (ctypes.c_int32 * (2 * R))()
        if _lib().ep_inter_counts(self._h, h.sendc.data_ptr(), ctypes.addressof(out), stream) != 0:
            raise RuntimeError("ep_inter_counts failed")
        h.inter_sendc = (ctypes.c_int32 * R)(*out[:R])
        h.inter_recvc = (ctypes.c_int32 * R)(*out[R:])

    def _hybrid_plan(self, h, ti32, num_tokens, stream):
        """Hybrid: build the rank and node plans and exchange counts (one host sync) into h.

        The exchange also carries per-expert counts, so hyb_expert_cnt is on the host before
        the payload moves and dispatch does not wait for it."""
        R, SU, dev = self.num_ranks, self.num_node_ranks, ti32.device
        SO = R // SU
        h.slot = torch.empty((R, num_tokens), dtype=torch.int32, device=dev)
        h.send_list = torch.empty((R, num_tokens), dtype=torch.int32, device=dev)
        h.sendc = torch.empty((R,), dtype=torch.int32, device=dev)
        h.node_pos = torch.empty((SO, num_tokens), dtype=torch.int32, device=dev)
        h.node_list = torch.empty((SO, num_tokens), dtype=torch.int32, device=dev)
        h.node_cnt = torch.empty((SO,), dtype=torch.int32, device=dev)
        lib = _lib()
        if lib.ep_hplan(self._h, ti32.data_ptr(), num_tokens, h.slot.data_ptr(),
                        h.send_list.data_ptr(), h.sendc.data_ptr(), h.node_pos.data_ptr(),
                        h.node_list.data_ptr(), h.node_cnt.data_ptr(), stream) != 0:
            raise RuntimeError("ep_hplan failed")
        out = (ctypes.c_int32 * (2 * R + 2 * SO))()
        out_expert = (ctypes.c_int32 * self.num_local_experts)()
        if lib.ep_hcounts(self._h, h.sendc.data_ptr(), h.node_cnt.data_ptr(), ti32.data_ptr(),
                          num_tokens, ctypes.addressof(out), ctypes.addressof(out_expert),
                          stream) != 0:
            raise RuntimeError("ep_hcounts failed")
        h.hyb_expert_cnt = list(out_expert)
        h.inter_sendc = (ctypes.c_int32 * R)(*out[:R])
        h.inter_recvc = (ctypes.c_int32 * R)(*out[R:2 * R])
        h.hyb_nsend = (ctypes.c_int32 * SO)(*out[2 * R:2 * R + SO])
        h.hyb_nrecv = (ctypes.c_int32 * SO)(*out[2 * R + SO:])
        h.fwd_meta = torch.empty((max(SO - 1, 1), self.num_max_tokens_per_rank, 1 + self.num_topk),
                                 dtype=torch.int32, device=dev)

    def _psum_by_scaleup(self, recvc):
        """psum_num_recv_tokens_per_scaleup_rank from per-source counts: hybrid sums each
        local index over the nodes (the scale-up ranks), otherwise it is per source."""
        if self.hybrid:
            SU = self.num_node_ranks
            per = [sum(recvc[v::SU]) for v in range(SU)]
        else:
            per = list(recvc)
        return _h2d_i32(per, torch.device("cuda", torch.cuda.current_device())).cumsum(0, dtype=torch.int32)

    def _inter_bind(self, h):
        """Internode: bind the handle's counts and plan for the dispatch or combine that
        follows -- the transport posts receives of exactly these sizes."""
        if not self.internode:
            return
        if self.hybrid:
            if _lib().ep_hbind(self._h, ctypes.addressof(h.inter_sendc), ctypes.addressof(h.inter_recvc),
                               ctypes.addressof(h.hyb_nsend), ctypes.addressof(h.hyb_nrecv),
                               h.send_list.data_ptr(), h.slot.data_ptr(), h.node_list.data_ptr(),
                               h.node_pos.data_ptr(), h.fwd_meta.data_ptr(), h.num_tokens) != 0:
                raise RuntimeError("ep_hbind failed")
            return
        if _lib().ep_inter_bind(self._h, ctypes.addressof(h.inter_sendc),
                                ctypes.addressof(h.inter_recvc), h.send_list.data_ptr(),
                                h.num_tokens) != 0:
            raise RuntimeError("ep_inter_bind failed")

    @property
    def num_local_experts(self):
        return self.num_experts // self.num_ranks

    def _expert_metadata(self, h, counts, expert_alignment):
        """Fill in the two different per-expert prefix sums the handle reports.

        They are genuinely different quantities: the plain handle reports a
        prefix over ALIGNED counts (there is no expanded tensor, so the numbers
        only describe how a consumer would lay one out), while the expanded
        handle reports the end of each expert's REAL rows inside the tensor that
        was actually produced.
        """
        cl = counts.tolist()
        aligned = [_align(c, expert_alignment) for c in cl]
        h.num_recv_tokens_per_expert_list = aligned
        h.expert_counts = counts
        if h.expanded:
            psum, acc = [], 0
            for c, a in zip(cl, aligned):
                psum.append(acc + c)
                acc += a
        else:
            psum, acc = [], 0
            for a in aligned:
                acc += a
                psum.append(acc)
        h.psum_num_recv_tokens_per_expert = torch.tensor(
            psum, dtype=torch.int32, device="cuda")

    # ---- data path -------------------------------------------------------
    def dispatch(self, x, topk_idx=None, topk_weights=None,
                 num_max_tokens_per_rank=None, num_experts=None,
                 num_sms=0, num_qps=0, expert_alignment=1,
                 async_with_compute_stream=0, allocate_on_comm_stream=0,
                 do_handle_copy=1, do_cpu_sync=1,
                 do_expand=False, use_tma_aligned_col_major_sf=False,
                 do_zero_padding=False, handle=None,
                 cumulative_local_expert_recv_stats=None,
                 previous_event=None, payload_only=False, return_recv_hook=False, **kwargs):
        lib = _lib()
        use_fp8 = isinstance(x, tuple)
        x, x_sf = x if use_fp8 else (x, None)
        cached = handle is not None
        if payload_only:
            return self._dispatch_payload(x, handle, use_fp8, num_sms, previous_event,
                                          return_recv_hook)
        if return_recv_hook:
            raise ValueError("return_recv_hook is for payload_only dispatch")

        # `async_with_compute_stream` and `allocate_on_comm_stream` are accepted
        # for API compatibility and ignored: the data path runs in order on the
        # caller's stream rather than overlapped on a private comm stream.
        if previous_event is not None:
            previous_event.current_stream_wait()
        stream = torch.cuda.current_stream().cuda_stream

        if cached:
            # A cached call replays the handle's plan verbatim, so everything the
            # plan encodes has to come from the handle. The routing in particular:
            # combine() reads it back as handle.topk_idx_i32, so re-deriving it
            # here from a caller-supplied tensor would let dispatch and combine
            # disagree about which rank owns a token, with no shape change and no
            # error to signal it.
            if topk_idx is not None and topk_idx is not handle.topk_idx:
                raise ValueError(
                    "cached dispatch replays the handle's routing; pass "
                    "topk_idx=None")
            topk_idx = handle.topk_idx
            num_experts = self.num_experts
            # Same reasoning for the alignment it was built with.
            expert_alignment = handle.expert_alignment
        if num_experts is None:
            num_experts = self.num_experts
            if num_experts is None:
                raise ValueError(
                    "num_experts must be passed to dispatch() or to the "
                    "ElasticBuffer constructor")

        num_tokens = x.shape[0]
        # The plan tensors are shaped (num_ranks, plan-time num_tokens) and the
        # dispatch kernel strides send_list by the num_tokens it is passed, so a
        # replay at a different token count reads the wrong rows -- and combine()
        # would then size its output from the handle's count, not this call's.
        # Refreshing handle.num_tokens would not help; the stride is already baked
        # into the plan, so the only sound answer is to refuse.
        if cached and num_tokens != handle.num_tokens:
            raise ValueError(
                f"cached dispatch has {num_tokens} tokens, but its handle was "
                f"built for {handle.num_tokens}")
        if num_tokens > self.num_max_tokens_per_rank:
            raise ValueError(
                f"num_tokens {num_tokens} exceeds num_max_tokens_per_rank "
                f"{self.num_max_tokens_per_rank}")

        num_topk = topk_idx.shape[1]
        self._configure(num_experts, num_topk)

        x = x.contiguous()
        x_sf = x_sf.contiguous().float() if x_sf is not None else None
        # Cached: the handle's own int32 copy, so dispatch pushes exactly the
        # routing combine() will reduce against. Also saves a cast per replay. An
        # uncached int64 plan-notify writes the copy itself (ep_plan_notify_v3).
        fuse_cast = (not cached and not do_expand and not self.internode
                     and topk_idx.dtype == torch.int64)
        if cached:
            ti32 = handle.topk_idx_i32
        elif fuse_cast:
            topk_idx = topk_idx.contiguous()
            ti32 = torch.empty(topk_idx.shape, dtype=torch.int32, device=x.device)
        else:
            ti32 = topk_idx.to(torch.int32).contiguous()
        tw = (topk_weights if topk_weights is not None
              else torch.zeros(topk_idx.shape, dtype=torch.float32,
                               device=x.device)).to(torch.float32).contiguous()

        # Routing plan. Reused verbatim on a cached call, which is what makes the
        # replay bit-identical rather than merely equivalent.
        if cached:
            h, slot, sendc = handle, handle.slot, handle.sendc
            send_list = handle.send_list
        else:
            h = EPHandle()
            h.num_tokens = num_tokens
            h.expert_alignment = expert_alignment
            # do_handle_copy is observable: callers compare data_ptr identity.
            h.topk_idx = topk_idx.clone() if do_handle_copy else topk_idx
            h.topk_idx_i32 = ti32
            slot = torch.empty((self.num_ranks, num_tokens), dtype=torch.int32, device=x.device)
            # Dense slot -> token map, so the send kernel skips no iterations.
            send_list = torch.empty((self.num_ranks, num_tokens), dtype=torch.int32, device=x.device)
            sendc = torch.empty((self.num_ranks,), dtype=torch.int32, device=x.device)
            if self.hybrid:
                self._hybrid_plan(h, ti32, num_tokens, stream)
                slot, send_list, sendc = h.slot, h.send_list, h.sendc
            elif do_expand or self.internode:
                # Internode takes this plan too: the notify handshake is intranode only,
                # so the counts are exchanged over the global communicator instead.
                if lib.ep_plan(self._h, ti32.data_ptr(), num_tokens, slot.data_ptr(),
                               send_list.data_ptr(), sendc.data_ptr(), stream) != 0:
                    raise RuntimeError("ep_plan failed")
                if self.internode:
                    h.sendc = sendc
                    self._inter_counts(h, stream)
            else:
                # The plan also exchanges this dispatch's counts with every peer, in
                # place of the dispatch's first barrier, so the host can learn the
                # receive total before the payload moves (see ep_plan_notify).
                nt_counts = torch.empty((self.num_local_experts,), dtype=torch.int32,
                                        device=x.device)
                nt_psum_rank = torch.empty((self.num_ranks,), dtype=torch.int32,
                                           device=x.device)
                if fuse_cast:
                    rc = lib.ep_plan_notify_v3(self._h, topk_idx.data_ptr(), ti32.data_ptr(),
                                               num_tokens, slot.data_ptr(), send_list.data_ptr(),
                                               sendc.data_ptr(), nt_counts.data_ptr(),
                                               nt_psum_rank.data_ptr(), 0, 0, 0, stream)
                else:
                    rc = lib.ep_plan_notify(self._h, ti32.data_ptr(), num_tokens, slot.data_ptr(),
                                            send_list.data_ptr(), sendc.data_ptr(),
                                            nt_counts.data_ptr(), nt_psum_rank.data_ptr(),
                                            stream)
                if rc != 0:
                    raise RuntimeError("ep_plan_notify failed")
            h.slot, h.sendc, h.send_list = slot, sendc, send_list

        cap = self.num_ranks * self.num_max_tokens_per_rank
        if self.internode:
            # Size outputs to the host-known receive total: a slice of a window-capacity
            # buffer would keep the whole buffer alive while the caller holds it.
            cap = max(1, handle.num_recv if cached else sum(h.inter_recvc))
        hidden_sf = (self.hidden + 127) // 128
        rx = (torch.empty((cap, self.hidden), dtype=torch.float8_e4m3fn, device=x.device)
              if use_fp8 else
              torch.empty((cap, self.hidden), dtype=torch.bfloat16, device=x.device))
        rsf = torch.empty((cap, hidden_sf), dtype=torch.float32, device=x.device) if use_fp8 else None
        rtk = torch.empty((cap, num_topk), dtype=torch.int32, device=x.device)
        rtw = torch.empty((cap, num_topk), dtype=torch.float32, device=x.device)
        rsrc = torch.empty((cap,), dtype=torch.int32, device=x.device)

        epr = self.num_local_experts
        if not do_expand:
            # The library calls never block the host. An uncached call then waits once
            # for the counts its plan exchanged; a cached replay already has them.
            if cached:
                self._inter_bind(h)
                if lib.ep_dispatch_v2(self._h, x.data_ptr(), _ptr(x_sf), ti32.data_ptr(),
                                      tw.data_ptr(), num_tokens, send_list.data_ptr(),
                                      sendc.data_ptr(), 1 if use_fp8 else 0, num_sms,
                                      rx.data_ptr(), _ptr(rsf), rtk.data_ptr(),
                                      rtw.data_ptr(), rsrc.data_ptr(), 0, 0, stream) != 0:
                    raise RuntimeError("ep_dispatch_v2 failed")
                n = handle.num_recv
            elif self.internode:
                # Barrier path, offsets from the exchanged counts. Hybrid already has the
                # total and per-expert counts on the host; direct mode reads them back.
                self._inter_bind(h)
                nt_counts = torch.empty((epr,), dtype=torch.int32, device=x.device)
                # Reused: this call waits for it before returning. device= explicitly, since
                # callers may set a cuda default device.
                if self._inter_host is None or self._inter_host.numel() != 1 + epr:
                    self._inter_host = torch.empty((1 + epr,), dtype=torch.int32, device="cpu",
                                                   pin_memory=True)
                host = self._inter_host
                if lib.ep_dispatch_v2(self._h, x.data_ptr(), _ptr(x_sf), ti32.data_ptr(),
                                      tw.data_ptr(), num_tokens, send_list.data_ptr(),
                                      sendc.data_ptr(), 1 if use_fp8 else 0, num_sms,
                                      rx.data_ptr(), _ptr(rsf), rtk.data_ptr(),
                                      rtw.data_ptr(), rsrc.data_ptr(), nt_counts.data_ptr(),
                                      0 if self.hybrid else host.data_ptr(), stream) != 0:
                    raise RuntimeError("ep_dispatch_v2 failed")
                h._psum_src = _h2d_i32(list(h.inter_recvc), x.device).cumsum(0, dtype=torch.int32)
                h.psum_num_recv_tokens_per_scaleup_rank = self._psum_by_scaleup(list(h.inter_recvc))
                h.expert_counts = nt_counts
                dev_aligned = (nt_counts if expert_alignment <= 1 else
                               (nt_counts + expert_alignment - 1)
                               // expert_alignment * expert_alignment)
                h.psum_num_recv_tokens_per_expert = torch.cumsum(
                    dev_aligned, 0, dtype=torch.int32)
                if cumulative_local_expert_recv_stats is not None:
                    cumulative_local_expert_recv_stats += nt_counts
                if self.hybrid:
                    n, cl = sum(h.inter_recvc), list(h.hyb_expert_cnt)
                else:
                    torch.cuda.current_stream().synchronize()
                    hl = host.tolist()
                    n, cl = hl[0], hl[1:]
            else:
                if lib.ep_dispatch_v3(self._h, x.data_ptr(), _ptr(x_sf), ti32.data_ptr(),
                                      tw.data_ptr(), num_tokens, send_list.data_ptr(),
                                      sendc.data_ptr(), 1 if use_fp8 else 0, num_sms,
                                      rx.data_ptr(), _ptr(rsf), rtk.data_ptr(),
                                      rtw.data_ptr(), rsrc.data_ptr(), stream) != 0:
                    raise RuntimeError("ep_dispatch_v3 failed")
                # Enqueue all device-only work before the host wait, so the GPU runs
                # it during the wait.
                h.psum_num_recv_tokens_per_scaleup_rank = nt_psum_rank
                h.expert_counts = nt_counts
                dev_aligned = (nt_counts if expert_alignment <= 1 else
                               (nt_counts + expert_alignment - 1)
                               // expert_alignment * expert_alignment)
                h.psum_num_recv_tokens_per_expert = torch.cumsum(
                    dev_aligned, 0, dtype=torch.int32)
                if cumulative_local_expert_recv_stats is not None:
                    cumulative_local_expert_recv_stats += nt_counts
                n, cl = self._wait_counts(h, epr)

            rx, rsf = rx[:n], (rsf[:n] if use_fp8 else None)
            rtk, rtw, rsrc = rtk[:n], rtw[:n], rsrc[:n]
            h.num_recv, h.recv_topk_idx = n, rtk
            if not cached:
                h.dst_buffer_slot_idx = slot
                h.expanded = False
                h.num_recv_tokens_per_expert_list = [
                    _align(c, expert_alignment) for c in cl]
                h.recv_src = rsrc
                h._src_rows_per_rank = self.num_max_tokens_per_rank
                h._num_topk = num_topk
                h.recv_src_metadata = None
            # A cached replay keeps the handle's counts and metadata: the plan is
            # identical, so they would not change.
            elif cumulative_local_expert_recv_stats is not None:
                cumulative_local_expert_recv_stats += h.expert_counts
            out_x = (rx, rsf) if use_fp8 else rx
            return out_x, rtk.to(topk_idx_t), rtw, h, EventOverlap()

        self._inter_bind(h)
        n = lib.ep_dispatch(self._h, x.data_ptr(), _ptr(x_sf), ti32.data_ptr(), tw.data_ptr(),
                            num_tokens, send_list.data_ptr(), sendc.data_ptr(),
                            1 if use_fp8 else 0, num_sms,
                            rx.data_ptr(), _ptr(rsf), rtk.data_ptr(), rtw.data_ptr(),
                            rsrc.data_ptr(), stream)
        if n < 0:
            raise RuntimeError("ep_dispatch failed")

        rx, rsf = rx[:n], (rsf[:n] if use_fp8 else None)
        rtk, rtw, rsrc = rtk[:n], rtw[:n], rsrc[:n]
        h.num_recv, h.recv_topk_idx = n, rtk

        # Per-source-rank prefix sum.
        rc = torch.empty((self.num_ranks,), dtype=torch.int32, device=x.device)
        if lib.ep_recv_counts(self._h, rc.data_ptr(), stream) != 0:
            raise RuntimeError("ep_recv_counts failed")
        h._psum_src = torch.cumsum(rc, 0).to(torch.int32)
        h.psum_num_recv_tokens_per_scaleup_rank = (self._psum_by_scaleup(rc.tolist()) if self.hybrid
                                                   else h._psum_src)
        h.dst_buffer_slot_idx = slot

        counts = torch.zeros((epr,), dtype=torch.int32, device=x.device)

        # ---- expanded layout ------------------------------------------------
        offsets = torch.empty((epr,), dtype=torch.int32, device=x.device)
        psum = torch.empty((epr,), dtype=torch.int32, device=x.device)
        hist = torch.empty((max(n, 1), epr), dtype=torch.int32, device=x.device)
        row_map = torch.empty((n, num_topk), dtype=torch.int32, device=x.device)
        rows = lib.ep_expand_build(self._h, rtk.data_ptr(), n, expert_alignment,
                                   counts.data_ptr(), offsets.data_ptr(), psum.data_ptr(),
                                   hist.data_ptr(), row_map.data_ptr(), stream)
        if rows < 0:
            raise RuntimeError("ep_expand_build failed")

        ex = (torch.empty((rows, self.hidden), dtype=torch.float8_e4m3fn, device=x.device)
              if use_fp8 else
              torch.empty((rows, self.hidden), dtype=torch.bfloat16, device=x.device))
        ew = torch.empty((rows,), dtype=torch.float32, device=x.device)
        esf, sf_rs, sf_cs = None, 0, 0
        if use_fp8:
            if use_tma_aligned_col_major_sf:
                # No TMA on AMD, but the column-major layout is still what a
                # downstream GEMM wants, so it is produced rather than ignored:
                # a [hidden_sf, rows] allocation viewed transposed.
                esf = torch.empty((hidden_sf, rows), dtype=torch.float32, device=x.device).t()
                sf_rs, sf_cs = 1, rows
            else:
                esf = torch.empty((rows, hidden_sf), dtype=torch.float32, device=x.device)
                sf_rs, sf_cs = hidden_sf, 1

        if lib.ep_expand_scatter(self._h, 1 if use_fp8 else 0,
                                 rx.data_ptr(), _ptr(rsf), rtw.data_ptr(),
                                 row_map.data_ptr(), n,
                                 ex.data_ptr(), _ptr(esf), sf_rs, sf_cs, ew.data_ptr(),
                                 1 if do_zero_padding else 0, expert_alignment,
                                 counts.data_ptr(), offsets.data_ptr(),
                                 num_sms, stream) != 0:
            raise RuntimeError("ep_expand_scatter failed")

        h.expanded = True
        h.row_map, h.expert_offsets, h.num_expanded_rows = row_map, offsets, rows
        self._expert_metadata(h, counts, expert_alignment)
        meta = torch.empty((n, 2 + num_topk), dtype=torch.int32, device=x.device)
        meta[:, 0] = rsrc
        meta[:, 1] = rsrc // self.num_max_tokens_per_rank
        meta[:, 2:] = row_map
        h.recv_src_metadata = meta
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats += counts
        out_x = (ex, esf) if use_fp8 else ex
        # Expanded dispatch returns no per-row top-k indices: a row IS an
        # (expert, token) pair, so the index would be a constant per run.
        return out_x, None, ew, h, EventOverlap()

    def dispatch_grouped(self, x, topk_idx=None, topk_weights=None, num_experts=None,
                         num_sms=0, handle=None, previous_event=None):
        """Dispatch straight into the grouped-by-expert layout; see GroupedDispatch.

        bf16 only, expert_alignment 1. An uncached call waits once on the host for the
        counts; a cached call (handle from dispatch_grouped) does not. The handle also
        serves combine() and cached dispatch().
        """
        lib = _lib()
        if self.internode:
            raise NotImplementedError("dispatch_grouped is intranode only: its plan comes from "
                                      "the notify handshake, which does not cross nodes")
        if isinstance(x, tuple):
            raise ValueError("grouped dispatch is bf16 only")
        if previous_event is not None:
            previous_event.current_stream_wait()
        stream = torch.cuda.current_stream().cuda_stream
        cached = handle is not None
        if cached:
            if not getattr(handle, "grouped", False):
                raise ValueError("a cached grouped dispatch needs a handle from dispatch_grouped")
            if topk_idx is not None and topk_idx is not handle.topk_idx:
                raise ValueError("cached dispatch replays the handle's routing; pass topk_idx=None")
            topk_idx = handle.topk_idx
            num_experts = self.num_experts
        if num_experts is None:
            num_experts = self.num_experts
            if num_experts is None:
                raise ValueError("num_experts must be passed to dispatch_grouped() or the constructor")
        num_tokens = x.shape[0]
        if cached and num_tokens != handle.num_tokens:
            raise ValueError(f"cached dispatch has {num_tokens} tokens, but its handle was "
                             f"built for {handle.num_tokens}")
        if num_tokens > self.num_max_tokens_per_rank:
            raise ValueError(f"num_tokens {num_tokens} exceeds num_max_tokens_per_rank "
                             f"{self.num_max_tokens_per_rank}")
        num_topk = topk_idx.shape[1]
        self._configure(num_experts, num_topk)
        x = x.contiguous()
        dev, R, epr = x.device, self.num_ranks, self.num_local_experts
        # Uncached int64 ids: the plan kernel writes the int32 copy (ep_plan_notify_v3).
        fuse_cast = not cached and topk_idx.dtype == torch.int64
        if cached:
            ti32 = handle.topk_idx_i32
        elif fuse_cast:
            topk_idx = topk_idx.contiguous()
        else:
            ti32 = topk_idx.to(torch.int32).contiguous()
        tw = (topk_weights if topk_weights is not None
              else torch.zeros(topk_idx.shape, dtype=torch.float32, device=dev)
              ).to(torch.float32).contiguous()

        if cached:
            h = handle
        else:
            h = EPHandle()
            h.num_tokens, h.expert_alignment = num_tokens, 1
            # One allocation for every per-call plan array, to keep allocator
            # overhead off the host path.
            RT, TK = R * num_tokens, num_tokens * num_topk
            plan = torch.empty(2 * RT + TK + R + R * epr + 2 * epr + R + (TK if fuse_cast else 0),
                               dtype=torch.int32, device=dev)
            if fuse_cast:
                ti32 = plan[plan.numel() - TK:].view(num_tokens, num_topk)
            h.topk_idx, h.topk_idx_i32 = topk_idx, ti32
            h.slot = plan[:RT].view(R, num_tokens)
            h.send_list = plan[RT:2 * RT].view(R, num_tokens)
            h.g_rank = plan[2 * RT:2 * RT + TK].view(num_tokens, num_topk)
            small = plan[2 * RT + TK:]
            h.sendc = small[:R]
            h.g_srcpref = small[R:R + R * epr]
            h.g_ebase = small[R + R * epr:R + R * epr + epr]
            h.expert_counts = small[R + R * epr + epr:R + R * epr + 2 * epr]
            h.psum_num_recv_tokens_per_scaleup_rank = small[R + R * epr + 2 * epr:R + R * epr + 2 * epr + R]
            plan_args = (num_tokens, h.slot.data_ptr(), h.send_list.data_ptr(), h.sendc.data_ptr(),
                         h.expert_counts.data_ptr(),
                         h.psum_num_recv_tokens_per_scaleup_rank.data_ptr(),
                         h.g_rank.data_ptr(), h.g_srcpref.data_ptr(), h.g_ebase.data_ptr(), stream)
            rc = (lib.ep_plan_notify_v3(self._h, topk_idx.data_ptr(), ti32.data_ptr(), *plan_args)
                  if fuse_cast else
                  lib.ep_plan_notify_v2(self._h, ti32.data_ptr(), *plan_args))
            if rc != 0:
                raise RuntimeError("ep_plan_notify failed")
        if lib.ep_dispatch_grouped(self._h, x.data_ptr(), ti32.data_ptr(), tw.data_ptr(),
                                   num_tokens, h.send_list.data_ptr(), h.sendc.data_ptr(),
                                   h.g_rank.data_ptr(), num_sms, 0 if cached else 1,
                                   stream) != 0:
            raise RuntimeError("ep_dispatch_grouped failed")
        if not cached:
            h.num_recv, h.num_recv_tokens_per_expert_list = self._wait_counts(h, epr)
            h.expanded, h.grouped = False, True
            h.dst_buffer_slot_idx = h.slot
            h._src_rows_per_rank, h._num_topk = self.num_max_tokens_per_rank, num_topk
            h.recv_src_metadata = None

        n, K = h.num_recv, num_topk
        rows = sum(h.num_recv_tokens_per_expert_list)
        out_rows = torch.empty((rows, self.hidden), dtype=torch.bfloat16, device=dev)
        i64 = torch.empty(((rows + n) * K + epr * n,), dtype=torch.int64, device=dev)
        row_ids = i64[:rows * K].view(rows, K)
        ids64 = i64[rows * K:(rows + n) * K].view(n, K)
        emap = i64[(rows + n) * K:].view(epr, n)
        f32 = torch.empty(((rows + n) * K,), dtype=torch.float32, device=dev)
        row_w, w = f32[:rows * K].view(rows, K), f32[rows * K:].view(n, K)
        i32 = torch.empty((n * K + n,), dtype=torch.int32, device=dev)
        ids32, src = i32[:n * K].view(n, K), i32[n * K:]
        if lib.ep_grouped_epilogue(self._h, h.psum_num_recv_tokens_per_scaleup_rank.data_ptr(),
                                   h.g_srcpref.data_ptr(), h.g_ebase.data_ptr(), n,
                                   out_rows.data_ptr(), row_ids.data_ptr(), row_w.data_ptr(),
                                   emap.data_ptr(), ids32.data_ptr(), ids64.data_ptr(),
                                   w.data_ptr(), src.data_ptr(), _EPILOGUE_SMS or num_sms,
                                   stream) != 0:
            raise RuntimeError("ep_grouped_epilogue failed")
        if not cached:
            h.recv_topk_idx, h.recv_src = ids32, src
        return GroupedDispatch(out_rows, row_ids, row_w, emap, ids64, w, h)

    def _wait_counts(self, h, epr):
        """Wait for the count exchange; returns (receive total, per-expert counts).

        Also sets h.recv_pairs_per_rank ((token, expert) pairs each rank receives, the
        same on every rank) and h.max_tokens_per_rank (largest count any rank sent)."""
        R = self.num_ranks
        full = 2 + epr + R
        if self._wait_buf is None or len(self._wait_buf) != full:
            self._wait_buf = (ctypes.c_int32 * full)()
        if _lib().ep_wait_counts_v2(self._h, ctypes.addressof(self._wait_buf), full,
                                    _WAIT_TIMEOUT_MS) != 0:
            raise RuntimeError("ep_wait_counts timed out: an EP peer never reached "
                               "this dispatch's plan, or took a different path")
        b = self._wait_buf
        h.recv_pairs_per_rank = list(b[1 + epr:1 + epr + R])
        h.max_tokens_per_rank = b[1 + epr + R]
        return b[0], list(b[1:1 + epr])

    def _dispatch_payload(self, x, handle, use_fp8, num_sms, previous_event,
                          return_recv_hook=False):
        """Cached replay that moves only the rows: (recv_x, None, None, handle, event).

        No top-k ids, weights or source indices are pushed, and the handle is left
        unchanged, including the recv_topk_idx that combine() reads.

        return_recv_hook (hybrid only), as in combine(): return once the local push is
        queued and the cross-node transfer is in flight, with a sixth value, hook; recv_x
        is complete only after hook() runs on the current stream, and no other dispatch or
        combine on this buffer may come between.
        """
        if handle is None or handle.expanded or use_fp8:
            raise ValueError("payload_only needs a non-expanded bf16 cached dispatch")
        if return_recv_hook and not self.hybrid:
            raise ValueError("return_recv_hook needs a hybrid internode buffer")
        if previous_event is not None:
            previous_event.current_stream_wait()
        num_tokens = x.shape[0]
        if num_tokens != handle.num_tokens:
            raise ValueError(
                f"cached dispatch has {num_tokens} tokens, but its handle was "
                f"built for {handle.num_tokens}")
        x = x.contiguous()
        rx = torch.empty((handle.num_recv, self.hidden), dtype=torch.bfloat16, device=x.device)
        # The plan's own offsets, so the library skips the counts memset and the scan.
        self._inter_bind(handle)
        lib = _lib()
        if return_recv_hook and lib.ep_dispatch_defer_next(self._h) != 0:
            raise RuntimeError("ep_dispatch_defer_next failed")
        if lib.ep_dispatch_payload_v2(self._h, x.data_ptr(), num_tokens,
                                         handle.send_list.data_ptr(), handle.sendc.data_ptr(),
                                         (handle._psum_src if handle._psum_src is not None
                                          else handle.psum_num_recv_tokens_per_scaleup_rank).data_ptr(),
                                         num_sms, rx.data_ptr(),
                                         torch.cuda.current_stream().cuda_stream) != 0:
            raise RuntimeError("ep_dispatch_payload failed")
        if not return_recv_hook:
            return rx, None, None, handle, EventOverlap()

        def hook():
            if lib.ep_dispatch_finish(self._h, torch.cuda.current_stream().cuda_stream) != 0:
                raise RuntimeError("ep_dispatch_finish failed")

        return rx, None, None, handle, EventOverlap(), hook

    def combine(self, x, handle, topk_weights=None, bias=None,
                num_sms=0, num_qps=0, async_with_compute_stream=0,
                allocate_on_comm_stream=0, previous_event=None, expert_rows=None,
                row_scales=None, fma=True, return_recv_hook=False, **kwargs):
        """Reduce received rows back to their source tokens.

        expert_rows (non-expanded handle): int64 [experts_per_rank, num_recv]; entry
        [e, i] is the row of x holding received token i's copy for local expert e, or
        -1. Each token's rows are reduced in ascending expert order in fp32 with one
        rounding, bit-identical to a caller-side reduction in that order followed by a
        plain combine. topk_weights, if given, is then [rows of x, num_topk] with one
        nonzero per row.

        row_scales (with expert_rows, no topk_weights): fp32 [num_recv, num_topk]; each
        row is scaled by its token's weight for that expert as it is summed. fma=False
        rounds the product before the add instead of fusing it.

        return_recv_hook (hybrid with allow_multiple_reduction only): return once the
        cross-node transfer is queued, with a fourth value, hook. The output is complete only
        after hook() runs on the current stream; work queued before it overlaps the transfer,
        and no other dispatch or combine on this buffer may come between.
        """
        if handle is None:
            raise ValueError("combine requires the handle returned by dispatch")
        if return_recv_hook and not (self.hybrid and self.allow_multiple_reduction):
            raise ValueError(
                "return_recv_hook needs a hybrid internode buffer with allow_multiple_reduction"
            )
        if expert_rows is not None:
            epr = self.num_experts // self.num_ranks
            if handle.expanded:
                raise ValueError("expert_rows is for a non-expanded handle")
            if (expert_rows.dtype != torch.int64
                    or tuple(expert_rows.shape) != (epr, handle.num_recv)):
                raise ValueError(f"expert_rows must be int64 [{epr}, {handle.num_recv}], "
                                 f"got {expert_rows.dtype} {tuple(expert_rows.shape)}")
            if topk_weights is not None and (topk_weights.dim() != 2
                                             or topk_weights.size(0) != x.size(0)
                                             or topk_weights.size(1) != self.num_topk):
                raise ValueError("with expert_rows, topk_weights must be [rows of x, num_topk]")
            expert_rows = expert_rows.contiguous()
        if row_scales is not None:
            if expert_rows is None or topk_weights is not None:
                raise ValueError("row_scales needs expert_rows and no topk_weights")
            if (row_scales.dtype != torch.float32
                    or tuple(row_scales.shape) != (handle.num_recv, self.num_topk)):
                raise ValueError(f"row_scales must be fp32 [{handle.num_recv}, {self.num_topk}]")
            row_scales = row_scales.contiguous()
        lib = _lib()
        # See dispatch(): the comm-stream arguments are accepted and ignored.
        if previous_event is not None:
            previous_event.current_stream_wait()
        stream = torch.cuda.current_stream().cuda_stream
        num_tokens = handle.num_tokens
        x = x.contiguous()
        b0, b1 = (bias, None)
        if isinstance(bias, tuple):
            b0, b1 = bias
        b0, b1 = _bias(b0, "bias0"), _bias(b1, "bias1")
        in_w = topk_weights.contiguous().float() if topk_weights is not None else None
        # Weights travel back iff this pointer is non-null, and internode all ranks must
        # agree. A rank that received no rows has data_ptr() 0, so give it a placeholder.
        if in_w is not None and in_w.numel() == 0:
            in_w = torch.zeros(1, dtype=torch.float32, device=x.device)

        out = torch.empty((num_tokens, self.hidden), dtype=torch.bfloat16, device=x.device)
        out_w = (torch.empty((num_tokens, self.num_topk), dtype=torch.float32, device=x.device)
                 if in_w is not None else None)

        # What each rank writes into the window:
        #   not expanded                      -> 1, one reduced row per token
        #   expanded, multiple reduction off  -> 0, one row per (slot, k)
        #   expanded, multiple reduction on   -> 1, reduced from the expanded
        #                                        input inside the kernel
        grouped = 1 if (not handle.expanded or self.allow_multiple_reduction) else 0
        # The column slice is strided, so .contiguous() really copies. Bind it:
        # inline, the copy is dropped as soon as data_ptr() returns and the
        # allocator can hand the block out again before the kernel reads it.
        recv_src = (handle.recv_src if handle.recv_src is not None
                    else handle.recv_src_metadata[:, 0].contiguous())
        self._inter_bind(handle)
        rc = lib.ep_combine_v4(self._h, x.data_ptr(), _ptr(in_w),
                            _ptr(handle.row_map) if handle.expanded else 0, _ptr(expert_rows),
                            _ptr(row_scales), 1 if fma else 0,
                            handle.recv_topk_idx.data_ptr(),
                            recv_src.data_ptr(),
                            handle.num_recv,
                            handle.topk_idx_i32.data_ptr(), num_tokens,
                            _ptr(b0), _ptr(b1), grouped,
                            out.data_ptr(), _ptr(out_w), num_sms,
                            1 if return_recv_hook else 0, stream)
        if rc != 0:
            raise RuntimeError("ep_combine failed")
        if not return_recv_hook:
            return out, out_w, EventOverlap()

        # The reduce that ep_combine_finish launches reads the biases and the top-k ids through
        # pointers taken above, and b0/b1 may be copies that nothing else holds: the hook keeps
        # them alive until then.
        def hook(_reduce_reads=(b0, b1, handle.topk_idx_i32)):
            if lib.ep_combine_finish(self._h, torch.cuda.current_stream().cuda_stream) != 0:
                raise RuntimeError("ep_combine_finish failed")

        return out, out_w, EventOverlap(), hook
