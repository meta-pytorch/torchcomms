# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""DSL-agnostic a2a tuning base: the shared all_to_all adapter lifecycle.

Split out of :mod:`comms.dsl.tuning_base` so that module stays collective-agnostic
(Config / Key / Spec / grid / geometry guard) while this module owns the pieces that
are specific to the all_to_all collective but identical across the Triton and CuTe
backends: the materialized rank-local inputs and the ``BaseA2AAdapter`` lifecycle
(CLI args, shape enumeration, key building, input materialization, candidate
dispatch, NCCL reference / correctness gate, and serialization). A per-backend
subclass supplies only what genuinely differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .transport import nvl_rendezvous
from .tuning_base import (
    _CommTunerBase,
    _device_tag,
    BaseAdapterMixin,
    import_obj,
    resolve_variant,
    SIZE_LADDER,
)


@dataclass
class A2AInputs:
    """Materialized rank-local state for one a2a tuning run (shared by both backends)."""

    transport: Any
    input: torch.Tensor
    output: torch.Tensor
    rows: int


# pyre-ignore[11]: _CommTunerBase is `object` or the optional CommKernelTuningAdapter.
class BaseA2AAdapter(BaseAdapterMixin, _CommTunerBase):
    """DSL-agnostic adapter lifecycle for the all_to_all schedule, shared by both backends.

    The comms-owned tuner engine (``comm_tuning``) drives this; the generic engine owns
    orchestration / timing / selection, and this base owns everything identical across the
    Triton and CuTe backends: CLI args, shape enumeration, key building, input
    materialization (equal-split a2a), candidate dispatch, the NCCL baseline / reference /
    correctness gate, and serialization. A per-backend subclass supplies only what genuinely
    differs: its config/key/spec classes + core fields + identity-copy hooks + generated
    artifact names (class attributes), and the expert candidate grid, the static feasibility
    filter, the config render template, and the collective launch (methods).

    Key sharing: ``make_key`` here MUST match the runtime key builder so the tuned map looks
    up. Each backend pins its default (copy) hook; a transformed variant is a separate tuned
    map keyed by ``variant``.
    """

    # Per-backend class attributes the subclass sets (config_cls / key_cls / spec_cls /
    # core_config_fields / backend_name come from BaseAdapterMixin).
    transport_kind: str = "NvlTransport"
    default_produce: Any = None
    default_consume: Any = None
    spec_tag: str = ""
    generated_table: str = ""
    generated_file: str = ""
    generated_import: str = ""
    # The backend's host-side ``all_to_all`` entry, bound by the subclass as a
    # ``staticmethod`` so the shared ``run_collective`` below is backend-agnostic.
    host_all_to_all: Any = None

    # Per-call state (set in __init__).
    _max_sizes: int = 0
    _max_candidates: int = 0
    _variant: str = ""
    # None (expert default) | dict grid | (spec, key) -> list[config] callable.
    _candidates: Any = None
    _feasible: Any = None

    def __init__(
        self,
        *,
        produce=None,
        consume=None,
        variant: str = "",
        reference=None,
        candidates=None,
        feasible=None,
        shapes=None,
    ) -> None:
        # Parametrized by the A2A object (or constructed bare for the CLI path). Hooks default
        # to the backend's identity copy hooks; the variant is validated once here
        # (explicit-required for a non-default hook).
        self._produce = produce if produce is not None else self.default_produce
        self._consume = consume if consume is not None else self.default_consume
        self._variant = resolve_variant(
            self._produce,
            self._consume,
            variant,
            default_produce=self.default_produce,
            default_consume=self.default_consume,
        )
        self._reference = reference
        self._candidates = candidates
        self._feasible = feasible
        self._shapes = shapes
        self._max_sizes = 0
        self._max_candidates = 0

    # --- tuner CLI knobs (engine calls add_cli_args/configure) -------------
    def add_cli_args(self, parser: Any) -> None:
        parser.add_argument(
            "--max-sizes",
            type=int,
            default=0,
            help="smoke: tune only the first N size bands (0 = all)",
        )
        parser.add_argument(
            "--max-candidates",
            type=int,
            default=0,
            help="smoke: sweep only the first M candidate configs per band (0 = all)",
        )
        # Tune a custom hook with NO adapter file: point at import paths + a variant.
        parser.add_argument(
            "--produce", default=None, help="custom produce hook as 'module:attr'"
        )
        parser.add_argument(
            "--consume", default=None, help="custom consume hook as 'module:attr'"
        )
        parser.add_argument(
            "--variant", default=None, help="tuned-table variant tag for a custom hook"
        )
        parser.add_argument(
            "--reference",
            default=None,
            help="correctness reference '(inputs, group)->Tensor' as 'module:attr' "
            "(default: identity all_to_all_single)",
        )

    def configure(self, args: Any) -> None:
        self._max_sizes = int(getattr(args, "max_sizes", 0) or 0)
        self._max_candidates = int(getattr(args, "max_candidates", 0) or 0)
        # CLI overrides (used by the generic tune entrypoint; absent in the A2A-object path).
        produce = getattr(args, "produce", None)
        consume = getattr(args, "consume", None)
        variant = getattr(args, "variant", None)
        reference = getattr(args, "reference", None)
        if produce:
            self._produce = import_obj(produce)
        if consume:
            self._consume = import_obj(consume)
        if produce or consume or variant:
            self._variant = resolve_variant(
                self._produce,
                self._consume,
                variant or self._variant,
                default_produce=self.default_produce,
                default_consume=self.default_consume,
            )
        if reference:
            self._reference = import_obj(reference)

    # --- what to tune -----------------------------------------------------
    def _spec_from_bytes(self, nbytes: int, world_size: int):
        elem = torch.bfloat16.itemsize
        numel = nbytes // elem
        numel -= numel % world_size
        return self.spec_cls(numel=numel, dtype=torch.bfloat16) if numel > 0 else None

    def enumerate_input_specs(self, world_size: int) -> list:
        # The user's production shapes (`shapes=` on the object, or the default ladder). Each
        # entry may be a spec (full control incl. dtype/rows) or an int = per-rank bytes
        # (bf16, 1D). The best launch config shifts across sizes, so each is tuned separately.
        if self._shapes is not None:
            specs = []
            for s in self._shapes:
                if isinstance(s, self.spec_cls):
                    specs.append(s)
                elif isinstance(s, int):
                    spec = self._spec_from_bytes(s, world_size)
                    if spec is not None:
                        specs.append(spec)
                else:
                    raise TypeError(
                        f"shapes entry must be {self.spec_cls.__name__} or "
                        f"int (per-rank bytes), got {type(s).__name__}"
                    )
        else:
            specs = [
                s
                for s in (self._spec_from_bytes(n, world_size) for n in SIZE_LADDER)
                if s is not None
            ]
        if self._max_sizes:
            specs = specs[: self._max_sizes]
        return specs

    def make_key(self, spec, world_size: int):
        # MUST match the runtime key builder so the tuned map looks up. Tuned on the target
        # GPU, so _device_tag() tags the entry with this host's hardware; _variant is "" for
        # the default copy hook and set by the A2A object for a custom hook.
        return self.make_key_from_spec(
            spec=spec,
            world_size=world_size,
            transport_kind=self.transport_kind,
            device=_device_tag(),
            variant=self._variant,
        )

    def enumerate_candidate_configs(self, spec, key) -> list:
        # Dispatch the candidate source: default expert banded grid / flat declarative grid
        # (dict) / callable (full control); then static-feasibility filter + size-cap.
        cand = self._candidates
        if cand is None:
            out = self.default_candidate_configs(spec, key)
        elif callable(cand):
            # pyre-ignore[6]: a callable candidate source returns the backend's config list.
            out = list(cand(spec, key))
        elif isinstance(cand, dict):
            out = self.expand_candidate_grid(cand)
        else:
            raise TypeError(
                "candidates must be None (expert default), a dict grid, or a callable "
                f"(spec, key) -> list[{self.config_cls.__name__}]; got {type(cand)!r}"
            )
        feasible = self._feasible or self.default_feasible
        out = [c for c in out if feasible(c, spec)]
        if self._max_candidates:
            out = out[: self._max_candidates]
        return out

    def enumerate_baselines(self, spec, key) -> list[str]:
        return ["nccl"]

    # --- materialize + run ------------------------------------------------
    def make_inputs(self, spec, *, rank: int, world_size: int, device: torch.device):
        assert spec.numel % world_size == 0
        chunk = spec.numel // world_size
        inp = torch.randn(spec.numel, dtype=spec.dtype, device=device)
        out = torch.empty_like(inp)
        group = dist.group.WORLD
        assert group is not None, "default process group not initialized"
        transport = nvl_rendezvous(
            group, device, per_peer_bytes=chunk * inp.element_size()
        )
        return A2AInputs(transport=transport, input=inp, output=out, rows=spec.rows)

    def run_candidate(self, inputs, config, group) -> torch.Tensor:
        self.run_collective(inputs, config)
        return inputs.output

    def run_baseline(self, inputs, baseline: str, group) -> torch.Tensor:
        if baseline != "nccl":
            raise ValueError(f"unknown baseline {baseline}")
        return self.run_reference(inputs, group)

    def run_reference(self, inputs, group) -> torch.Tensor:
        # Custom hooks (transpose/quantize/...) change the expected output, so the user
        # supplies a reference `(inputs, group) -> Tensor` capturing their semantics; the
        # default (identity copy hook) is a plain all_to_all_single.
        if self._reference is not None:
            return self._reference(inputs, group)
        ref = torch.empty_like(inputs.input)
        dist.all_to_all_single(ref, inputs.input, group=group)
        return ref

    def check_correctness(self, candidate_output, reference_output) -> dict[str, Any]:
        torch.testing.assert_close(candidate_output, reference_output, atol=0, rtol=0)
        # The engine records this dict verbatim and the selector keeps a candidate only when
        # correctness["status"] == "pass"; assert_close(atol=0, rtol=0) raises on any mismatch,
        # so reaching here means bit-exact -- the error is provably 0.0. Computing it via
        # .abs().max().item() would only force a needless GPU sync.
        return {
            "status": "pass",
            "max_abs_err": 0.0,
        }

    # --- serialization (parent -> child) ----------------------------------
    def spec_to_json(self, spec) -> tuple[str, dict[str, Any]]:
        return (self.spec_tag, spec.to_json_dict())

    def spec_from_json(self, spec_type: str, data: dict[str, Any]):
        if spec_type != self.spec_tag:
            raise ValueError(f"unknown spec type {spec_type}")
        return super().spec_from_json(spec_type, data)

    # --- generated artifact names -----------------------------------------
    def generated_table_name(self) -> str:
        return self.generated_table

    def generated_filename(self) -> str:
        return self.generated_file

    def generated_imports(self) -> str:
        return self.generated_import

    # --- per-backend hooks (subclass supplies these) ----------------------
    def default_candidate_configs(self, spec, key) -> list:
        """The shipped expert, size-banded candidate grid (the zero-effort default)."""
        raise NotImplementedError

    def default_feasible(self, config, spec) -> bool:
        """Cheap, shape-independent static feasibility pre-filter for user grids."""
        raise NotImplementedError

    def run_collective(self, inputs, config) -> None:
        """Launch this backend's all_to_all writing ``inputs.output`` (used by run_candidate).

        Backend-agnostic: dispatches to the subclass-bound ``host_all_to_all`` with the
        adapter's hooks; identical across Triton and CuTe, so it lives here."""
        self.host_all_to_all(
            inputs.transport,
            inputs.output,
            inputs.input,
            produce=self._produce,
            consume=self._consume,
            rows=inputs.rows,
            config=config,
        )
