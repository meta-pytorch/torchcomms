if(NOT TIMEMORY_BUILD_GOOGLE_TEST)
    return()
endif()

set(CMAKE_CXX_VISIBILITY_PRESET "default")

set(_EXCLUDE)
if(NOT TIMEMORY_BUILD_TESTING
   OR NOT timemory_MAIN_PROJECT
   OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

if(TIMEMORY_SOCKET_TESTS)
    add_executable(socket-server ${_EXCLUDE} server.cpp)
    add_executable(socket-client ${_EXCLUDE} client.cpp)
    target_link_libraries(socket-server PRIVATE common-test-libs)
    target_link_libraries(socket-client PRIVATE common-test-libs)
endif()

add_library(extern-test-templates ${_EXCLUDE} SHARED extern_templates.cpp)
target_link_libraries(
    extern-test-templates PUBLIC common-test-libs timemory::timemory-plotting
                                 timemory::timemory-core timemory::timemory-dmp)

if(TIMEMORY_USE_GOTCHA)
    add_library(gotcha-tests-lib ${_EXCLUDE} SHARED gotcha_tests_lib.hpp
                                 gotcha_tests_lib.cpp)
    target_include_directories(gotcha-tests-lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(gotcha-tests-lib PUBLIC common-test-libs)
endif()

set(_LIB_TAG)
set(_LIB_TYPE)
if(BUILD_SHARED_LIBS)
    set(_LIB_TAG shared)
    set(_LIB_TYPE SHARED)
elseif(BUILD_STATIC_LIBS)
    set(_LIB_TAG static)
    set(_LIB_TYPE STATIC)
endif()

if(_LIB_TAG AND _LIB_TYPE)
    if(TIMEMORY_BUILD_C)
        add_library(custom-record-functions ${_EXCLUDE} ${_LIB_TYPE}
                                            custom_record_functions.cpp)
        target_link_libraries(
            custom-record-functions
            PUBLIC timemory::timemory-cxx-${_LIB_TAG} timemory::timemory-c-${_LIB_TAG}
                   timemory::timemory-plotting common-test-libs)
    endif()

    if(TIMEMORY_USE_DYNINST)
        add_library(custom-dynamic-instr ${_EXCLUDE} ${_LIB_TYPE}
                                         custom_dynamic_instr.cpp)
        target_link_libraries(
            custom-dynamic-instr PUBLIC timemory::timemory-cxx-${_LIB_TAG}
                                        timemory::timemory-plotting common-test-libs)
        set_target_properties(
            custom-dynamic-instr
            PROPERTIES OUTPUT_NAME custom_dynamic_instr
                       LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
                       ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
                       RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()
endif()

if(TIMEMORY_USE_NCCL AND TIMEMORY_BUILD_NCCLP_LIBRARY)
    timemory_checkout_git_submodule(
        RELATIVE_PATH source/tests/external/nccl-tests
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/nccl-tests.git
        REPO_BRANCH timemory)
    add_subdirectory(nccl-tests)
endif()
