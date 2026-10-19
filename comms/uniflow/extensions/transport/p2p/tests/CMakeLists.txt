# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_gpu_test(
  uniflow_p2p_registration_handle_test P2pRegistrationHandleTest.cpp unit)
uniflow_add_gpu_test(
  uniflow_p2p_transport_test P2pTransportTest.cpp unit)

add_subdirectory(integration)
