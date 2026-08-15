# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""DSL-agnostic all_to_all entry: dispatch to the CuTe device backend.

A thin façade over the device backend so a caller selects the DSL with a
``backend=`` argument instead of importing a backend package directly. The backend
module is imported lazily, so a caller never pays the import cost of the DSL's
compiler stack until first use.

The backend exposes a stable surface -- staging ``all_to_all`` (caller output) and
zero-copy ``all_to_all_zc`` (returns the transport's output view) plus a tunable ``Config`` /
lookup ``Key`` / ``CommTuner`` adapter. The façade stays backend-parametrized so a second
DSL (Triton) can be re-registered later by adding it to ``_BACKENDS`` / ``_ADAPTER_CLS``
without touching callers; CuTe is the only shipped backend today.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

_BACKENDS: tuple[str, ...] = ("cute",)

# Backend -> the CommTuner adapter class the autotuner drives.
_ADAPTER_CLS: dict[str, str] = {
    "cute": "CuteA2ATuningAdapter",
}


def _check_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {_BACKENDS}")


def _backend_collectives(backend: str):
    _check_backend(backend)
    return import_module(f"comms.dsl.{backend}.a2a.host")


def all_to_all(
    transport,
    output,
    input,
    *,
    backend: str = "cute",
    **kwargs,
) -> None:
    """Equal-split all_to_all_single on the selected DSL backend.

    ``backend`` selects the DSL (``"cute"`` today); the remaining keyword arguments
    (``produce`` / ``consume`` / ``rows`` / ``config`` / ``variant``)
    forward to that backend's ``all_to_all``.

    Hook support: a non-identity ``produce`` / ``consume`` (elementwise) and the ``rows > 0``
    layout transpose both run on the fused CuTe copy staging schedule. The TMA and zero-copy
    (``direct`` / ``ce``) paths move raw bytes and apply neither."""
    _backend_collectives(backend).all_to_all(transport, output, input, **kwargs)


def all_to_all_zc(
    transport,
    input,
    *,
    backend: str = "cute",
    primitive: str = "direct",
    config: Any = None,
) -> Any:
    """Zero-copy all_to_all on the selected DSL backend: returns this rank's symmetric-memory
    output view (slot ``s`` = the chunk from sender ``s``), so the transport must be sized
    ``per_peer_bytes >= chunk * elem``. ``primitive`` picks ``"direct"`` (DirectWrite) or
    ``"ce"`` (copy-engine)."""
    return _backend_collectives(backend).all_to_all_zc(
        transport, input, primitive=primitive, config=config
    )


class A2A:
    """Backend-dispatching all_to_all collective object (callable + ``autotune``).

    ``backend=`` selects the DSL; the object then behaves like the per-backend collective
    -- ``__call__`` runs a tuned-table-looked-up all_to_all, and ``autotune`` drives the
    ``comm_tuning`` engine through the backend's adapter (the body of the tuning-job
    entrypoint). ``produce`` / ``consume`` default to ``None`` = the backend's identity
    copy hooks; a non-default hook MUST carry an explicit ``variant`` so its tuned configs
    do not collide with the default copy hook's."""

    def __init__(
        self,
        *,
        backend: str = "cute",
        produce: Callable[..., Any] | None = None,
        consume: Callable[..., Any] | None = None,
        variant: str = "",
        reference: Callable[..., Any] | None = None,
        candidates: Any = None,
        feasible: Any = None,
    ) -> None:
        _check_backend(backend)
        self.backend = backend
        self.produce = produce
        self.consume = consume
        self.variant = variant
        self.reference = reference
        self.candidates = candidates
        self.feasible = feasible
        # A non-default hook changes the access pattern, so it must carry an explicit variant
        # tag (else its tuned configs would silently reuse the default copy hook's entries).
        if (produce is not None or consume is not None) and not variant:
            raise ValueError(
                "a non-default produce/consume hook needs an explicit variant tag so its "
                "tuned configs don't collide with the default copy hook; pass variant=... "
                "(e.g. A2A(produce=my_op, variant='myq'))."
            )

    def _hook_kwargs(self) -> dict[str, Any]:
        # Forward a hook only when set, so the backend's own default identity hook applies.
        kw: dict[str, Any] = {}
        if self.produce is not None:
            kw["produce"] = self.produce
        if self.consume is not None:
            kw["consume"] = self.consume
        return kw

    def __call__(
        self,
        transport,
        output,
        input,
        *,
        rows: int = 0,
        config: Any = None,
    ) -> None:
        all_to_all(
            transport,
            output,
            input,
            backend=self.backend,
            rows=rows,
            config=config,
            variant=self.variant,
            **self._hook_kwargs(),
        )

    def autotune(self, *, shapes: Any = None) -> None:
        """Drive the ``comm_tuning`` engine through this backend's adapter (parent/child/
        select modes + ``--max-sizes`` etc. parsed from argv by ``run_tuning_cli``).

        The per-backend tuning adapter module (``comms.dsl.<backend>.a2a.adapter`` /
        :class:`A2ATuningAdapter`) ships in the tuning-adapter sub-diff layered on top of
        this one, so ``autotune`` is not callable until that module is present (a
        :exc:`ModuleNotFoundError` from the import below is expected before then)."""
        from comm_tuning.cli import run_tuning_cli  # pyre-ignore[21]

        adapter_mod = import_module(f"comms.dsl.{self.backend}.a2a.adapter")
        adapter_cls = getattr(adapter_mod, _ADAPTER_CLS[self.backend])
        kwargs: dict[str, Any] = {
            "variant": self.variant,
            "reference": self.reference,
            "candidates": self.candidates,
            "feasible": self.feasible,
            "shapes": shapes,
        }
        kwargs.update(self._hook_kwargs())
        run_tuning_cli(adapter_cls(**kwargs))
