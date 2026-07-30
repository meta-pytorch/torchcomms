# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(fault_tolerance OBJECT
    Abort.cc
)

target_compile_features(fault_tolerance PRIVATE cxx_std_20)
target_compile_options(fault_tolerance PRIVATE -fPIC)
set_target_properties(fault_tolerance PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(fault_tolerance PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
)
