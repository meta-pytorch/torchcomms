# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CuTe A2A tuned configs emitted by the comm_tuning select step (generated; do not hand-edit).

Placeholder: empty until an autotuner sweep emits the table, so ``get_a2a_config`` falls back
to ``DEFAULT_A2A_CONFIG`` (the analytic adaptive pick) for every key. The GB300 table is
populated in a later layer of this stack; refresh via ``mast_launch --delivery conda --tune``
and overwrite this file with the emitted table.
"""

from comms.dsl.cute.a2a.tuning import CuteA2AConfig, CuteA2AKey  # noqa: F401

TUNED_A2A_CONFIGS: dict[CuteA2AKey, CuteA2AConfig] = {}
