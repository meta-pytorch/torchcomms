# Copyright (c) Meta Platforms, Inc. and affiliates.

find_package(GTest REQUIRED)

add_executable(uniflow_tcp_server_client_test
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpServerClientTest.cpp
)
target_compile_features(uniflow_tcp_server_client_test PRIVATE cxx_std_20)
target_include_directories(uniflow_tcp_server_client_test PRIVATE ${ROOT})
target_link_libraries(uniflow_tcp_server_client_test PRIVATE
    uniflow_controller
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
add_test(NAME TcpServerClientTest COMMAND uniflow_tcp_server_client_test)

add_executable(uniflow_tcp_conn_test
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpConnTest.cpp
)
target_compile_features(uniflow_tcp_conn_test PRIVATE cxx_std_20)
target_include_directories(uniflow_tcp_conn_test PRIVATE ${ROOT})
target_link_libraries(uniflow_tcp_conn_test PRIVATE
    uniflow_controller
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
add_test(NAME TcpConnTest COMMAND uniflow_tcp_conn_test)

add_executable(uniflow_tcp_async_accept_test
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpAsyncAcceptTest.cpp
)
target_compile_features(uniflow_tcp_async_accept_test PRIVATE cxx_std_20)
target_include_directories(uniflow_tcp_async_accept_test PRIVATE ${ROOT})
target_link_libraries(uniflow_tcp_async_accept_test PRIVATE
    uniflow_controller
    uniflow_executor
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
add_test(NAME TcpAsyncAcceptTest COMMAND uniflow_tcp_async_accept_test)

add_executable(uniflow_tcp_async_connect_test
    ${CMAKE_CURRENT_SOURCE_DIR}/TcpAsyncConnectTest.cpp
)
target_compile_features(uniflow_tcp_async_connect_test PRIVATE cxx_std_20)
target_include_directories(uniflow_tcp_async_connect_test PRIVATE ${ROOT})
target_link_libraries(uniflow_tcp_async_connect_test PRIVATE
    uniflow_controller
    uniflow_executor
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
add_test(NAME TcpAsyncConnectTest COMMAND uniflow_tcp_async_connect_test)
