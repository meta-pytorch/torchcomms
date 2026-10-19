# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(
  uniflow_tcp_registration_handle_test TcpRegistrationHandleTest.cpp)
uniflow_add_unit_test(
  uniflow_tcp_wire_protocol_test TcpWireProtocolTest.cpp)
uniflow_add_unit_test(
  uniflow_tcp_transport_connect_test TcpTransportConnectTest.cpp)
uniflow_add_unit_test(
  uniflow_tcp_pinned_slab_pool_test TcpPinnedSlabPoolTest.cpp)
uniflow_add_unit_test(
  uniflow_tcp_transport_frame_test TcpTransportFrameTest.cpp)
uniflow_add_unit_test(
  uniflow_tcp_receive_pool_test TcpReceivePoolTest.cpp)
