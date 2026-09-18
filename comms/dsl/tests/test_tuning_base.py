# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import unittest
from unittest import mock

import torch
from comms.dsl.tuning_base import (
    _device_tag,
    BaseInputSpec,
    BaseTunableConfig,
    BaseTuningKey,
)


class TestTuningBase(unittest.TestCase):
    def test_key_no_backend_default_in_base(self):
        # The agnostic base stays backend-neutral: an empty backend is left as-is. A backend
        # default belongs on a per-backend key subclass; a missing backend is attributed to the
        # loading adapter only on the deserialization path (BaseAdapterMixin.key_from_json).
        k = BaseTuningKey(
            world_size=8,
            dtype="bfloat16",
            numel=1024,
            rows=0,
            transport_kind="NvlTransport",
            device="H100",
            backend="",
            variant="",
        )
        self.assertEqual(k.backend, "")

    def test_spec_json_roundtrip(self):
        spec = BaseInputSpec(numel=8, dtype=torch.bfloat16, rows=0)
        d = spec.to_json_dict()
        self.assertEqual(d["dtype"], "bfloat16")
        spec2 = BaseInputSpec.from_json_dict(d)
        self.assertEqual(spec, spec2)

    def test_config_hashable_as_dict_key(self):
        cfg1 = BaseTunableConfig()
        cfg2 = BaseTunableConfig()
        d = {cfg1: "v"}
        self.assertEqual(d[cfg2], "v")

    def test_device_tag_cuda_absent_returns_unknown(self):
        # CUDA unavailable raises RuntimeError -> the intended "unknown" fallback, no raise.
        with mock.patch(
            "torch.cuda.get_device_name", side_effect=RuntimeError("no CUDA")
        ):
            tag = _device_tag()
        self.assertEqual(tag, "unknown")

    def test_device_tag_cpu_device_returns_unknown(self):
        # A CPU device (e.g. a CPU input tensor's .device) makes torch raise ValueError;
        # the tag falls back to "unknown" rather than propagating (real make_a2a_key path).
        self.assertEqual(_device_tag(torch.device("cpu")), "unknown")

    def test_device_tag_known_models_match_exactly(self):
        cases = {
            "NVIDIA H100": "H100",
            "NVIDIA GB300": "GB300",
            "NVIDIA A100": "A100",
            "NVIDIA B200": "B200",
            "NVIDIA H200": "H200",
            # GH (Grace+Hopper) must tag distinctly, not collide with the inner H200.
            "NVIDIA GH200": "GH200",
        }
        for name, expected in cases.items():
            with mock.patch("torch.cuda.get_device_name", return_value=name):
                self.assertEqual(_device_tag(), expected)

    def test_device_tag_four_digit_model_not_truncated(self):
        # Regression: "RTX A6000" must NOT partial-match to "A600"; it falls through to the
        # readable cleaned-name path.
        with mock.patch("torch.cuda.get_device_name", return_value="NVIDIA RTX A6000"):
            self.assertEqual(_device_tag(), "NVIDIA_RTX_A6000")
