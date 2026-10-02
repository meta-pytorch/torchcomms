# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(
  uniflow_tcp_server_client_test TcpServerClientTest.cpp)
uniflow_add_unit_test(uniflow_tcp_conn_test TcpConnTest.cpp)
uniflow_add_unit_test(uniflow_tcp_async_accept_test TcpAsyncAcceptTest.cpp)
uniflow_add_unit_test(uniflow_tcp_async_connect_test TcpAsyncConnectTest.cpp)
uniflow_add_unit_test(
  uniflow_tcp_async_send_recv_test TcpAsyncSendRecvTest.cpp)
