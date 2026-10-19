# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(uniflow_result_test ResultTest.cpp)
uniflow_add_unit_test(uniflow_segment_test SegmentTest.cpp)
if(NOT UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  uniflow_add_unit_test(uniflow_multi_transport_test MultiTransportTest.cpp)
endif()
uniflow_add_unit_test(uniflow_agent_test UniflowAgentTest.cpp)
