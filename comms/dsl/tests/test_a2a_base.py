# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import unittest

from comms.dsl.a2a_base import BaseA2AAdapter
from comms.dsl.tuning_base import BaseInputSpec


class TestA2ABase(unittest.TestCase):
    def test_run_baseline_rejects_unknown_baseline(self):
        adapter = object.__new__(BaseA2AAdapter)
        with self.assertRaises(ValueError):
            adapter.run_baseline(inputs=None, baseline="gloo", group=None)

    def test_spec_from_json_rejects_unknown_spec_type(self):
        adapter = object.__new__(BaseA2AAdapter)
        adapter.spec_tag = "A2AInputSpec"
        with self.assertRaises(ValueError):
            adapter.spec_from_json("WrongSpec", {})

    def test_enumerate_input_specs_rejects_non_int_non_spec_shape(self):
        # A shapes entry that is neither a spec nor an int (per-rank bytes) must raise a
        # clear TypeError, not an opaque `nbytes // elem` failure deep in _spec_from_bytes.
        adapter = object.__new__(BaseA2AAdapter)
        adapter.spec_cls = BaseInputSpec
        adapter._shapes = [object()]
        adapter._max_sizes = 0
        with self.assertRaises(TypeError):
            adapter.enumerate_input_specs(world_size=8)
