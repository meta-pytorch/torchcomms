# Copyright (c) Meta Platforms, Inc. and affiliates.

find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
find_library(NCCL_LIBRARY NAMES nccl)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
  add_library(NCCL::NCCL UNKNOWN IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
