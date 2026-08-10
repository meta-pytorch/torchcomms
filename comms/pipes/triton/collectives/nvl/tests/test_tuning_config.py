# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Unit tests for ``tuning_config`` JSON lookup.

CPU-only — no GPU / multi-process required. Exercises:
  * ``_detect_hardware`` branch coverage via mocked
    ``torch.cuda.get_device_name``.
  * ``get_sendrecv_config`` lookup hit, miss-fallback, hardware
    filtering, msg-byte range filtering, and result caching.
  * ``_parse_stable`` / ``_parse_ws`` defaults + tile_cols derivation
    from ``tile_row_bytes / element_size``.
  * Empty JSON / missing-section fallback behaviour.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

from comms.pipes.triton.collectives.nvl import tuning_config


def _reset_caches() -> None:
    tuning_config._CONFIG_CACHE.clear()
    tuning_config._JSON_CACHE = None


def _write_json(d: dict[str, Any]) -> str:
    f = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".json", delete=False
    )
    json.dump(d, f)
    f.close()
    return f.name


class DetectHardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_caches()

    def test_h100(self) -> None:
        with (
            mock.patch.object(
                tuning_config.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                tuning_config.torch.cuda,
                "get_device_name",
                return_value="NVIDIA H100 80GB HBM3",
            ),
        ):
            self.assertEqual(tuning_config._detect_hardware(), "H100")

    def test_h200(self) -> None:
        with (
            mock.patch.object(
                tuning_config.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                tuning_config.torch.cuda, "get_device_name", return_value="NVIDIA H200"
            ),
        ):
            self.assertEqual(tuning_config._detect_hardware(), "H100")

    def test_b200(self) -> None:
        with (
            mock.patch.object(
                tuning_config.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                tuning_config.torch.cuda, "get_device_name", return_value="NVIDIA B200"
            ),
        ):
            self.assertEqual(tuning_config._detect_hardware(), "B200")

    def test_gb200_superchip(self) -> None:
        with (
            mock.patch.object(
                tuning_config.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                tuning_config.torch.cuda,
                "get_device_name",
                return_value="NVIDIA GB200",
            ),
        ):
            self.assertEqual(tuning_config._detect_hardware(), "GB200")

    def test_a100(self) -> None:
        with (
            mock.patch.object(
                tuning_config.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                tuning_config.torch.cuda,
                "get_device_name",
                return_value="NVIDIA A100-SXM4-80GB",
            ),
        ):
            self.assertEqual(tuning_config._detect_hardware(), "A100")

    def test_unknown_passthrough(self) -> None:
        with (
            mock.patch.object(
                tuning_config.torch.cuda, "is_available", return_value=True
            ),
            mock.patch.object(
                tuning_config.torch.cuda, "get_device_name", return_value="L40S"
            ),
        ):
            self.assertEqual(tuning_config._detect_hardware(), "L40S")

    def test_no_cuda(self) -> None:
        with mock.patch.object(
            tuning_config.torch.cuda, "is_available", return_value=False
        ):
            self.assertEqual(tuning_config._detect_hardware(), "unknown")


class GetSendrecvConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset_caches()
        self._json_path: str | None = None

    def tearDown(self) -> None:
        if self._json_path is not None and os.path.exists(self._json_path):
            os.unlink(self._json_path)
        os.environ.pop("TRITON_NVL_TUNING_JSON", None)
        _reset_caches()

    def _install_json(self, data: dict[str, Any]) -> None:
        self._json_path = _write_json(data)
        os.environ["TRITON_NVL_TUNING_JSON"] = self._json_path
        _reset_caches()

    def test_default_fallback_when_no_configs(self) -> None:
        self._install_json(
            {
                "default_stable": {
                    "tile_rows": 32,
                    "tile_row_bytes": 2048,
                    "signal_bytes": 262144,
                    "num_blocks": 16,
                    "num_warps": 8,
                },
                "configs": [],
            }
        )
        cfg = tuning_config.get_sendrecv_config(
            msg_bytes=1024 * 1024, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertEqual(cfg.stable.signal_bytes, 262144)
        self.assertEqual(cfg.stable.num_warps, 8)
        # tile_cols = tile_row_bytes / element_size
        self.assertEqual(cfg.stable.tile_cols, 2048 // 4)
        self.assertIsNone(cfg.ws)

    def test_default_fallback_with_default_ws(self) -> None:
        self._install_json(
            {
                "default_stable": {"signal_bytes": 262144, "num_warps": 8},
                "default_ws": {"sender_warps": 4, "receiver_warps": 4},
                "configs": [],
            }
        )
        cfg = tuning_config.get_sendrecv_config(
            msg_bytes=1024, element_size=2, num_peers=1, hardware="H100"
        )
        self.assertIsNotNone(cfg.ws)
        # pyre-ignore[16]: just asserted not-None
        self.assertEqual(cfg.ws.sender_warps, 4)
        self.assertEqual(cfg.ws.tile_cols, 2048 // 2)

    def test_lookup_hits_msg_range(self) -> None:
        self._install_json(
            {
                "default_stable": {"signal_bytes": 262144, "num_warps": 8},
                "configs": [
                    {
                        "hardware": "H100",
                        "num_peers": 1,
                        "msg_bytes_min": 0,
                        "msg_bytes_max": 1024 * 1024 - 1,
                        "stable": {"signal_bytes": 65536, "num_warps": 4},
                    },
                    {
                        "hardware": "H100",
                        "num_peers": 1,
                        "msg_bytes_min": 1024 * 1024,
                        "msg_bytes_max": 2**40,
                        "stable": {"signal_bytes": 524288, "num_warps": 16},
                    },
                ],
            }
        )
        small = tuning_config.get_sendrecv_config(
            msg_bytes=512 * 1024, element_size=4, num_peers=1, hardware="H100"
        )
        large = tuning_config.get_sendrecv_config(
            msg_bytes=64 * 1024 * 1024, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertEqual(small.stable.signal_bytes, 65536)
        self.assertEqual(small.stable.num_warps, 4)
        self.assertEqual(large.stable.signal_bytes, 524288)
        self.assertEqual(large.stable.num_warps, 16)

    def test_lookup_falls_back_when_hardware_mismatches(self) -> None:
        self._install_json(
            {
                "default_stable": {"signal_bytes": 262144, "num_warps": 8},
                "configs": [
                    {
                        "hardware": "GB200",
                        "stable": {"signal_bytes": 999, "num_warps": 99},
                    },
                ],
            }
        )
        cfg = tuning_config.get_sendrecv_config(
            msg_bytes=1024 * 1024, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertEqual(cfg.stable.signal_bytes, 262144)
        self.assertEqual(cfg.stable.num_warps, 8)

    def test_lookup_falls_back_when_num_peers_mismatches(self) -> None:
        self._install_json(
            {
                "default_stable": {"signal_bytes": 262144, "num_warps": 8},
                "configs": [
                    {
                        "hardware": "H100",
                        "num_peers": 7,
                        "stable": {"signal_bytes": 999},
                    },
                ],
            }
        )
        cfg = tuning_config.get_sendrecv_config(
            msg_bytes=1024 * 1024, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertEqual(cfg.stable.signal_bytes, 262144)

    def test_result_cached_per_key(self) -> None:
        self._install_json(
            {
                "default_stable": {"signal_bytes": 262144, "num_warps": 8},
                "configs": [],
            }
        )
        a = tuning_config.get_sendrecv_config(
            msg_bytes=1024, element_size=4, num_peers=1, hardware="H100"
        )
        b = tuning_config.get_sendrecv_config(
            msg_bytes=1024, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertIs(a, b)
        # Different msg_bytes -> different cache key, but same JSON-derived values.
        c = tuning_config.get_sendrecv_config(
            msg_bytes=2048, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertIsNot(a, c)
        self.assertEqual(a.stable, c.stable)

    def test_empty_default_stable_uses_hardcoded_defaults(self) -> None:
        self._install_json({"configs": []})
        cfg = tuning_config.get_sendrecv_config(
            msg_bytes=1024, element_size=4, num_peers=1, hardware="H100"
        )
        # Hardcoded defaults from _parse_stable: tile_rows=32, row_bytes=2048,
        # signal_bytes=262144, num_blocks=16, num_warps=8.
        self.assertEqual(cfg.stable.tile_rows, 32)
        self.assertEqual(cfg.stable.tile_cols, 2048 // 4)
        self.assertEqual(cfg.stable.signal_bytes, 262144)
        self.assertEqual(cfg.stable.num_blocks, 16)
        self.assertEqual(cfg.stable.num_warps, 8)

    def test_repo_default_json_loads(self) -> None:
        # Sanity check: the in-tree tuning_sendrecv.json is well-formed and
        # loads. This catches accidental commit of a broken JSON.
        os.environ.pop("TRITON_NVL_TUNING_JSON", None)
        _reset_caches()
        # Should not raise even without GPUs (hardware="H100" forces the path
        # without calling _detect_hardware).
        cfg = tuning_config.get_sendrecv_config(
            msg_bytes=1024 * 1024, element_size=4, num_peers=1, hardware="H100"
        )
        self.assertGreater(cfg.stable.signal_bytes, 0)
        self.assertGreater(cfg.stable.num_warps, 0)


def main() -> None:
    unittest.main(module=__name__, argv=["test_tuning_config", "-v"], exit=False)


if __name__ == "__main__":
    main()
