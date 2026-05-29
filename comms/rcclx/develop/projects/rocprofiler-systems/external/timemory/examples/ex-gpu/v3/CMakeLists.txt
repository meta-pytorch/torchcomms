project(timemory-GPU-Example-v3 LANGUAGES ${LANGS})

file(GLOB sources "${PROJECT_SOURCE_DIR}/*.cpp")
if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE CUDA)
elseif(CMAKE_HIP_COMPILER)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
endif()

add_executable(
    ex_kernel_instrument_v3
    ex_kernel_instrument.cpp gpu_device_timer.cpp gpu_op_tracker.cpp gpu_device_timer.hpp
    gpu_op_tracker.hpp gpu_common.hpp)

target_link_libraries(ex_kernel_instrument_v3 PRIVATE timemory-gpu-example)

install(
    TARGETS ex_kernel_instrument_v3
    DESTINATION bin
    OPTIONAL)
