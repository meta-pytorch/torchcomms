#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
project(PTL_GPU_Examples LANGUAGES C CXX)


#----------------------------------------------------------------------------
# Find PTL project
#
find_package(PTL REQUIRED)
set(EXTERNAL_INCLUDE_DIRS ${PTL_INCLUDE_DIRS})
set(EXTERNAL_LIBRARIES ${PTL_LIBRARIES})
include_directories(${EXTERNAL_INCLUDE_DIRS})
if(NOT DEFINED common_headers)
    get_filename_component(COMMON_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)
    include_directories(${COMMON_DIR})
endif()


#----------------------------------------------------------------------------
# Source files
#
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hh)

#----------------------------------------------------------------------------
# CUDA
#
if(PTL_USE_CUDA)
    #------------------------------------------------------------------------
    # Configuration for CUDA
    #
    include(CudaConfig)

    #------------------------------------------------------------------------
    # Add the executable, and link it to the Tasking libraries
    #
    cuda_add_library(sum STATIC sum.cc sum.h sum.hh sum.cu)
    target_link_libraries(sum ${EXTERNAL_CUDA_LIBRARIES})

    cuda_add_executable(gpu_tasking gpu_tasking.cc ${headers})
    target_link_libraries(gpu_tasking sum ${EXTERNAL_LIBRARIES})
    set_target_properties(gpu_tasking
        PROPERTIES COMPILE_DEFINITIONS USE_CUDA_TASKS)
    install(TARGETS gpu_tasking DESTINATION bin)

    if(PTL_USE_TBB)
        add_executable(gpu_tbb_tasking gpu_tasking.cc ${headers})
        set_target_properties(gpu_tbb_tasking
            PROPERTIES COMPILE_DEFINITIONS "USE_TBB_TASKS")
        target_link_libraries(gpu_tbb_tasking sum ${EXTERNAL_LIBRARIES})
        install(TARGETS gpu_tbb_tasking DESTINATION bin)
    endif(PTL_USE_TBB)

    #------------------------------------------------------------------------
    # Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
    #
endif()

