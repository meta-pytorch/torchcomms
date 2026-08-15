# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Runtime auto-load of the CuTe tuned table (the back half of the autotune round-trip).

The autotuner (``tune_a2a_cute`` -> ``comm_tuning`` select) emits
``generated/a2a_tuned_configs.py`` mapping each runtime key to the winning config; this
GPU-free test asserts the runtime lookup ``get_a2a_config`` returns that emitted config
(so a later ``all_to_all`` picks up the tuned config automatically, not the analytic default),
and that an unkeyed shape falls back to the default. The CuTe twin of the Triton round-trip.
"""

import unittest
from unittest.mock import patch

# Initialise the runtime lookup module before the generated table is imported below: the
# runtime table is populated by tuning's own (cycle-safe) generated import, so tuning must
# load first. A direct generated-first import would re-enter a half-initialised tuning module
# and leave its runtime table empty. (A plain ``import`` sorts ahead of the ``from`` imports,
# so this ordering is stable under the import formatter.)
import comms.dsl.cute.a2a.tuning  # noqa: F401
from comms.dsl.cute.a2a.generated.a2a_tuned_configs import TUNED_A2A_CONFIGS
from comms.dsl.cute.a2a.tuning import (
    CuteA2AConfig,
    CuteA2AKey,
    DEFAULT_A2A_CONFIG,
    get_a2a_config,
)


class CuteTunedRoundtripTest(unittest.TestCase):
    def test_tuned_entry_loads_at_runtime(self) -> None:
        # The shipped GB300 table now carries real tuned entries (see generated/a2a_tuned_configs.py).
        # To still prove the offline->generated->runtime LOOKUP mechanism (rather than accidentally
        # asserting on a pre-existing real entry), this uses a SYNTHETIC key whose numel (48) is
        # deliberately absent from the shipped table: the pre-injection lookup therefore MISSES to
        # the analytic default, and only the injected synthetic entry flips it -- so the assertions
        # exercise the lookup, not a table value that happened to already be present.
        key = CuteA2AKey(
            world_size=8,
            dtype="bfloat16",
            numel=48,
            rows=0,
            transport_kind="NvlTransport",
            device="GB300",
            backend="cute",
            variant="",
        )
        # Absent from the shipped table -> analytic default before injection.
        self.assertIs(get_a2a_config(key), DEFAULT_A2A_CONFIG)
        cfg = CuteA2AConfig(num_blocks=2, primitive="copy")
        with patch.dict(TUNED_A2A_CONFIGS, {key: cfg}, clear=False):
            got = get_a2a_config(key)
            self.assertIs(got, cfg)
            self.assertIsNot(got, DEFAULT_A2A_CONFIG)

    def test_untuned_key_falls_back_to_default(self) -> None:
        miss = CuteA2AKey(
            world_size=8,
            dtype="bfloat16",
            numel=1,
            rows=0,
            transport_kind="NvlTransport",
            device="GB300",
            backend="cute",
            variant="",
        )
        self.assertIs(get_a2a_config(miss), DEFAULT_A2A_CONFIG)
