# Copyright (c) Meta Platforms, Inc. and affiliates.

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  target_sources(uniflow PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/NvmlApiStub.cpp")
  target_link_libraries(uniflow PRIVATE uniflow_build_config)
endif()

set(UNIFLOW_EXTENSIONS_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${UNIFLOW_EXTENSIONS_BUILD_INCLUDE_DIR}/comms/uniflow")
file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}"
  "${UNIFLOW_EXTENSIONS_BUILD_INCLUDE_DIR}/comms/uniflow/extensions" SYMBOLIC)
target_include_directories(uniflow_build_config BEFORE INTERFACE
  $<BUILD_INTERFACE:${UNIFLOW_EXTENSIONS_BUILD_INCLUDE_DIR}>)

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  add_subdirectory(transport/p2p)
  target_sources(uniflow_implementation INTERFACE
    $<TARGET_OBJECTS:uniflow_p2p_transport>)
  target_link_libraries(uniflow_implementation INTERFACE uniflow_p2p_transport)
endif()

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP OR UNIFLOW_ENABLE_TCP)
  target_sources(uniflow PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/TransportHooks.cpp")
  target_link_libraries(uniflow PRIVATE uniflow_build_config)
endif()
