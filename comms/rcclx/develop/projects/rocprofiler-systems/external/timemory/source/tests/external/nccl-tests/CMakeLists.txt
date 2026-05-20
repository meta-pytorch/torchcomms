cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(NCCL-tests LANGUAGES CUDA)

add_library(nccl INTERFACE)
find_package(NCCL REQUIRED)
target_include_directories(nccl INTERFACE ${NCCL_INCLUDE_DIRS})
target_link_libraries(nccl INTERFACE ${NCCL_LIBRARIES})

file(GLOB nccl_test_headers ${PROJECT_SOURCE_DIR}/src/nccl_test/*.h)

add_library(
    nccl_test_common SHARED
    ${_EXCLUDE} ${PROJECT_SOURCE_DIR}/src/common.cu ${PROJECT_SOURCE_DIR}/src/common.h
    ${PROJECT_SOURCE_DIR}/src/nccl1_compat.h ${nccl_test_headers})
target_include_directories(nccl_test_common
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_link_libraries(nccl_test_common PUBLIC nccl)

add_library(nccl_test_all_gather SHARED ${_EXCLUDE}
                                        ${PROJECT_SOURCE_DIR}/src/all_gather.cu)
add_library(nccl_test_all_reduce SHARED ${_EXCLUDE}
                                        ${PROJECT_SOURCE_DIR}/src/all_reduce.cu)
add_library(nccl_test_alltoall SHARED ${_EXCLUDE} ${PROJECT_SOURCE_DIR}/src/alltoall.cu)
add_library(nccl_test_broadcast SHARED ${_EXCLUDE} ${PROJECT_SOURCE_DIR}/src/broadcast.cu)
add_library(nccl_test_reduce SHARED ${_EXCLUDE} ${PROJECT_SOURCE_DIR}/src/reduce.cu)
add_library(nccl_test_reduce_scatter SHARED ${_EXCLUDE}
                                            ${PROJECT_SOURCE_DIR}/src/reduce_scatter.cu)

target_link_libraries(nccl_test_all_gather PUBLIC nccl_test_common)
target_link_libraries(nccl_test_all_reduce PUBLIC nccl_test_common)
target_link_libraries(nccl_test_alltoall PUBLIC nccl_test_common)
target_link_libraries(nccl_test_broadcast PUBLIC nccl_test_common)
target_link_libraries(nccl_test_reduce PUBLIC nccl_test_common)
target_link_libraries(nccl_test_reduce_scatter PUBLIC nccl_test_common)

add_library(nccl_test_interface INTERFACE)
target_link_libraries(
    nccl_test_interface
    INTERFACE nccl_test_all_gather nccl_test_all_reduce nccl_test_alltoall
              nccl_test_broadcast nccl_test_reduce nccl_test_reduce_scatter)
