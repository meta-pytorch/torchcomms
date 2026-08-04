# Copyright (c) Meta Platforms, Inc. and affiliates.

add_library(fault_tolerance OBJECT
    Abort.cc
)

option(COMMS_FAULT_TOLERANCE_ENABLE_CUDA "Enable CUDA-mapped Abort state" ON)
if (COMMS_FAULT_TOLERANCE_ENABLE_CUDA)
    find_package(CUDAToolkit)
endif()

target_compile_features(fault_tolerance PRIVATE cxx_std_20)
target_compile_options(fault_tolerance PRIVATE -fPIC)
set_target_properties(fault_tolerance PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(fault_tolerance PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
)
if (COMMS_FAULT_TOLERANCE_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(fault_tolerance PRIVATE
        COMMS_FAULT_TOLERANCE_WITH_CUDA=1
    )
    target_link_libraries(fault_tolerance PRIVATE CUDA::cudart)
endif()
