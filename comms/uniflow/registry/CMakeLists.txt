# Copyright (c) Meta Platforms, Inc. and affiliates.

if(DEFINED UNIFLOW_DEFAULT_GPU_BACKEND)
  set(_UNIFLOW_REQUESTED_DEFAULT_GPU_BACKEND "${UNIFLOW_DEFAULT_GPU_BACKEND}")
else()
  set(_UNIFLOW_REQUESTED_DEFAULT_GPU_BACKEND nvidia)
endif()
if(NOT _UNIFLOW_REQUESTED_DEFAULT_GPU_BACKEND MATCHES "^(nvidia|amd|mtia)$")
  message(FATAL_ERROR
    "UNIFLOW_DEFAULT_GPU_BACKEND must be nvidia, amd, or mtia.")
endif()
set(UNIFLOW_DEFAULT_GPU_BACKEND "${_UNIFLOW_REQUESTED_DEFAULT_GPU_BACKEND}"
  CACHE STRING "GPU vendor used when no explicit request or environment is set" FORCE)
set(UNIFLOW_DEFAULT_GPU_BACKEND "${_UNIFLOW_REQUESTED_DEFAULT_GPU_BACKEND}")
set_property(CACHE UNIFLOW_DEFAULT_GPU_BACKEND PROPERTY STRINGS nvidia amd mtia)
unset(_UNIFLOW_REQUESTED_DEFAULT_GPU_BACKEND)

add_library(uniflow_registry OBJECT
  RuntimeVersion.cpp
  TransportProviderRegistry.cpp
  PlatformRegistry.cpp
  GpuBackendSelector.cpp
)
target_link_libraries(uniflow_registry PRIVATE uniflow_build_config)
target_compile_definitions(uniflow_registry PRIVATE
  "UNIFLOW_DEFAULT_GPU_BACKEND=\"${UNIFLOW_DEFAULT_GPU_BACKEND}\""
)
