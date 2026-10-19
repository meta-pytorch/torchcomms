# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

    def test_event_kind_values(self) -> None:
        self.assertNotEqual(ct.EventKind.COLL_START, ct.EventKind.COLL_END)


if __name__ == "__main__":
    unittest.main()
