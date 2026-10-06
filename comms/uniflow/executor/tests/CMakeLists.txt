# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(uniflow_event_base_test EventBaseTest.cpp)
uniflow_add_unit_test(
  uniflow_scoped_event_base_thread_test ScopedEventBaseThreadTest.cpp)
uniflow_add_unit_test(uniflow_stress_flood_test StressFloodTest.cpp)
