# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import time
import unittest

import colltrace as ct


class ColltraceTest(unittest.TestCase):
    def test_unregistered_comm_is_inert(self) -> None:
        self.assertEqual(ct.get_latest_coll_id("no_such_comm"), ct.INVALID_COLLID)
        self.assertEqual(ct.get_num_kernels(ct.INVALID_COLLID), -1)
        self.assertEqual(ct.get_coll_events("no_such_comm"), [])
        self.assertEqual(ct.new_coll_id("no_such_comm"), ct.INVALID_COLLID)

    def test_ids_advance_per_collective(self) -> None:
        desc = "colltrace_test_ids"
        self.assertTrue(ct.register_comm(desc))
        try:
            self.assertEqual(ct.get_latest_coll_id(desc), ct.INVALID_COLLID)
            first = ct.new_coll_id(desc)
            second = ct.new_coll_id(desc)
            self.assertNotEqual(first, ct.INVALID_COLLID)
            self.assertEqual(second, first + 1)
            self.assertEqual(ct.get_latest_coll_id(desc), second)
            # No kernels were launched, so no ring events and zero kernels.
            self.assertEqual(ct.get_num_kernels(first), 0)
            self.assertEqual(ct.get_coll_events(desc), [])
        finally:
            ct.deregister_comm(desc)
        self.assertEqual(ct.get_latest_coll_id(desc), ct.INVALID_COLLID)

    def test_register_twice_fails(self) -> None:
        desc = "colltrace_test_twice"
        self.assertTrue(ct.register_comm(desc))
        try:
            self.assertFalse(ct.register_comm(desc))
        finally:
            ct.deregister_comm(desc)

    def test_a_nic_reading_carries_its_fields(self) -> None:
        unix_ns = time.time_ns() - 5_000_000
        ct._put_nic_reading_for_test("mlx5_7", 1, {"tx_bytes": 42}, unix_ns)

        readings = [
            r
            for r in ct.get_nic_stats(counters=["tx_bytes", "rx_bytes"])
            if r["rail"] == "mlx5_7"
        ]

        self.assertEqual(len(readings), 1)
        reading = readings[0]
        self.assertEqual(reading["port"], 1)
        self.assertEqual(reading["unix_ns"], unix_ns)
        self.assertGreaterEqual(reading["age_ns"], 5_000_000)
        self.assertEqual(reading["counters"]["tx_bytes"], (42, "ok"))
        self.assertEqual(reading["counters"]["rx_bytes"], (None, "not_requested"))
        for field in (
            "sequence",
            "lag_ns",
            "sample_period_ns",
            "source",
            "counter_sources",
            "source_read_ns",
            "priority",
        ):
            self.assertIn(field, reading)
        self.assertNotIn("comm_desc", reading)

    def test_counters_pick_what_is_returned(self) -> None:
        ct._put_nic_reading_for_test(
            "mlx5_8", 1, {"tx_bytes": 1, "rx_bytes": 2}, time.time_ns()
        )

        reading = next(
            r for r in ct.get_nic_stats(counters=["rx_bytes"]) if r["rail"] == "mlx5_8"
        )

        self.assertEqual(reading["counters"], {"rx_bytes": (2, "ok")})

    def test_by_default_a_reading_returns_the_counters_it_was_read_for(
        self,
    ) -> None:
        ct._put_nic_reading_for_test("mlx5_10", 1, {"tx_bytes": 5}, time.time_ns())

        reading = next(r for r in ct.get_nic_stats() if r["rail"] == "mlx5_10")

        self.assertEqual(reading["counters"], {"tx_bytes": (5, "ok")})

    def test_a_rails_sequence_counts_its_readings(self) -> None:
        for value in (1, 2):
            ct._put_nic_reading_for_test(
                "mlx5_9", 1, {"tx_bytes": value}, time.time_ns()
            )

        reading = next(
            r for r in ct.get_nic_stats(counters=["tx_bytes"]) if r["rail"] == "mlx5_9"
        )

        self.assertEqual(reading["sequence"], 1)
        self.assertEqual(reading["counters"]["tx_bytes"], (2, "ok"))

    def test_an_unknown_counter_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            ct.get_nic_stats(counters=["not_a_counter"])

    def test_nic_stats_cannot_be_requested(self) -> None:
        self.assertFalse(hasattr(ct, "request_nic_stats"))

    def test_event_kind_values(self) -> None:
        self.assertNotEqual(ct.EventKind.COLL_START, ct.EventKind.COLL_END)


if __name__ == "__main__":
    unittest.main()
