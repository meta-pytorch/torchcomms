#
# Configures CUDA for timemory
#

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

function(timemory_add_cuda_arch ARCH FLAG)
    list(APPEND TIMEMORY_CUDA_ARCH_FLAGS ${FLAG})
    set(TIMEMORY_CUDA_ARCH_FLAGS
        ${TIMEMORY_CUDA_ARCH_FLAGS}
        PARENT_SCOPE)
    list(APPEND TIMEMORY_CUDA_ARCH_LIST ${ARCH})
    set(TIMEMORY_CUDA_ARCH_LIST
        ${TIMEMORY_CUDA_ARCH_LIST}
        PARENT_SCOPE)
endfunction()

# none
timemory_add_cuda_arch(auto 0)
# generic
timemory_add_cuda_arch(kepler 30)
timemory_add_cuda_arch(tesla 35)
timemory_add_cuda_arch(maxwell 50)
timemory_add_cuda_arch(pascal 60)
timemory_add_cuda_arch(volta 70)
timemory_add_cuda_arch(turing 75)
timemory_add_cuda_arch(ampere 80)
# specific
timemory_add_cuda_arch(kepler30 30)
timemory_add_cuda_arch(kepler32 32)
timemory_add_cuda_arch(kepler35 35)
timemory_add_cuda_arch(kepler37 37)
timemory_add_cuda_arch(maxwell50 50)
timemory_add_cuda_arch(maxwell52 52)
timemory_add_cuda_arch(maxwell53 53)
timemory_add_cuda_arch(pascal60 60)
timemory_add_cuda_arch(pascal61 61)
timemory_add_cuda_arch(volta70 70)
timemory_add_cuda_arch(volta72 72)
timemory_add_cuda_arch(turing75 75)
timemory_add_cuda_arch(ampere80 80)
timemory_add_cuda_arch(ampere86 86)

if("CUDA" IN_LIST LANGUAGES)

    find_package(CUDA REQUIRED)

    # copy our test to .cu so cmake compiles as CUDA
    configure_file(
        ${PROJECT_SOURCE_DIR}/cmake/Templates/cuda_compute_capability.cu
        ${PROJECT_BINARY_DIR}/compile-tests/cuda_compute_capability.cu COPYONLY)

    # run test again
    try_run(
        _RESULT _COMPILE_RESULT
        ${PROJECT_BINARY_DIR}/compile-tests/cuda-compute-capability-workdir
        ${PROJECT_BINARY_DIR}/compile-tests/cuda_compute_capability.cu
        COMPILE_DEFINITIONS -DSM_ONLY
        RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY)

    if(_COMPILE_RESULT AND _RESULT EQUAL 0)
        message(STATUS "Detected CUDA Compute Capability ${_CUDA_COMPUTE_CAPABILITY}")
        list(FIND TIMEMORY_CUDA_ARCH_FLAGS ${_CUDA_COMPUTE_CAPABILITY} FLAG_INDEX)
        list(GET TIMEMORY_CUDA_ARCH_LIST ${FLAG_INDEX} ARCHITECTURE)
    else()
        unset(_CUDA_COMPUTE_CAPABILITY)
    endif()

    timemory_target_compile_definitions(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
                                        ${PROJECT_USE_CUDA_OPTION})

    target_include_directories(
        ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
        INTERFACE ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set_target_properties(
        ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
        PROPERTIES INTERFACE_CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
                   INTERFACE_CUDA_STANDARD_REQUIRED ${CMAKE_CUDA_STANDARD_REQUIRED}
                   INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
                   INTERFACE_CUDA_SEPARABLE_COMPILATION ON)

    set(CUDA_AUTO_ARCH "auto")

    set(TIMEMORY_CUDA_ARCH
        "${CUDA_AUTO_ARCH}"
        CACHE STRING "CUDA architecture (options: ${TIMEMORY_CUDA_ARCH_LIST})")

    timemory_add_feature(TIMEMORY_CUDA_ARCH
                         "CUDA architecture (options: ${TIMEMORY_CUDA_ARCH_LIST})")

    set_property(CACHE TIMEMORY_CUDA_ARCH PROPERTY STRINGS ${TIMEMORY_CUDA_ARCH_LIST})

    set(_CUDA_ARCHES ${TIMEMORY_CUDA_ARCH})
    set(_CUDA_ARCH_NUMS)

    foreach(_ARCH ${_CUDA_ARCHES})
        if("${_ARCH}" STREQUAL "auto")
            continue()
        endif()
        if(NOT "${_ARCH}" IN_LIST TIMEMORY_CUDA_ARCH_LIST)
            message(
                WARNING
                    "CUDA architecture \"${_ARCH}\" not known. Options: ${TIMEMORY_CUDA_ARCH_LIST}"
                )
            if(NOT "auto" IN_LIST TIMEMORY_CUDA_ARCH)
                list(APPEND TIMEMORY_CUDA_ARCH "auto")
            endif()
            continue()
        endif()
        list(FIND TIMEMORY_CUDA_ARCH_LIST ${_ARCH} _INDEX)
        list(GET TIMEMORY_CUDA_ARCH_FLAGS ${_INDEX} _ARCH_NUM)
        list(APPEND _CUDA_ARCH_NUMS ${_ARCH_NUM})
    endforeach()

    if("auto" IN_LIST TIMEMORY_CUDA_ARCH)
        if(_CUDA_COMPUTE_CAPABILITY)
            list(APPEND _CUDA_ARCH_NUMS ${_CUDA_COMPUTE_CAPABILITY})
        endif()
        list(REMOVE_ITEM TIMEMORY_CUDA_ARCH "auto")
    endif()

    foreach(_ARCH_NUM ${_CUDA_ARCH_NUMS})
        if(DEFINED PROJECT_CUDA_USE_HALF_OPTION AND _ARCH_NUM LESS 60)
            set(${PROJECT_CUDA_USE_HALF_OPTION} OFF)
        endif()
        if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
            target_compile_options(
                ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
                INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${_ARCH_NUM}
                          -gencode=arch=compute_${_ARCH_NUM},code=sm_${_ARCH_NUM}
                          -gencode=arch=compute_${_ARCH_NUM},code=compute_${_ARCH_NUM}>)
        else()
            target_compile_options(
                ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
                INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--cuda-gpu-arch=sm_${_ARCH_NUM}>)
        endif()
    endforeach()

    list(APPEND CMAKE_CUDA_ARCHITECTURES ${_CUDA_ARCH_NUMS})
    list(SORT CMAKE_CUDA_ARCHITECTURES)

    timemory_add_feature(CMAKE_CUDA_ARCHITECTURES "CUDA architectures")

    if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
        if("${CUDA_VERSION}" VERSION_LESS 11.0)
            target_compile_options(
                ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:--expt-extended-lambda>>
                )
        else()
            target_compile_options(
                ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:--extended-lambda>>
                )
        endif()

        if(NOT WIN32)
            target_compile_options(
                ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda-compiler
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:-lineinfo>>)
        endif()

        # do not specify the --compiler-bindir in portable mode
        if(NOT WIN32 AND NOT TIMEMORY_BUILD_PORTABLE)
            # this complicated generator expression basically states: if compile language
            # is CUDA and CUDA compiler ID is NVIDIA and CMAKE_CUDA_HOST_COMPILER is
            # empty, then add the compiler flag --compiler-bindir=${CMAKE_CXX_COMPILER}.
            # According to NVCC docs, despite the flags suggesting it wants a directory,
            # you can specify the compiler executable to ensure that the correct compiler
            # is selected
            target_compile_options(
                ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda-compiler
                INTERFACE
                    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:$<IF:$<STREQUAL:CMAKE_CUDA_HOST_COMPILER,"">,--compiler-bindir=${CMAKE_CXX_COMPILER},>>>
                )
        endif()
    endif()

    add_user_flags(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda "CUDA")

    if(DEFINED PROJECT_CUDA_USE_HALF_OPTION AND ${PROJECT_CUDA_USE_HALF_OPTION})
        timemory_target_compile_definitions(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
                                            INTERFACE ${PROJECT_CUDA_USE_HALF_DEFINITION})
    endif()

    target_include_directories(
        ${PROJECT_CUDA_INTERFACE_PREFIX}-cuda
        INTERFACE ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    if(NOT CUDA_dl_LIBRARY)
        if(dl_LIBRARY)
            set(CUDA_dl_LIBRARY ${dl_LIBRARY})
        else()
            find_library(CUDA_dl_LIBRARY NAMES dl)
        endif()
    endif()

    if(NOT CUDA_rt_LIBRARY)
        if(rt_LIBRARY)
            set(CUDA_rt_LIBRARY ${rt_LIBRARY})
        else()
            find_library(CUDA_rt_LIBRARY NAMES rt)
        endif()
    endif()

    set(CUDA_extra_LIBS)
    if(CUDA_dl_LIBRARY)
        list(APPEND CUDA_extra_LIBS ${CUDA_dl_LIBRARY})
    endif()

    if(CUDA_rt_LIBRARY)
        list(APPEND CUDA_extra_LIBS ${CUDA_rt_LIBRARY})
    endif()

    target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart
                          INTERFACE ${CUDA_CUDART_LIBRARY} ${CUDA_extra_LIBS})

    target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-device
                          INTERFACE ${CUDA_cudadevrt_LIBRARY} ${CUDA_extra_LIBS})

    target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-static
                          INTERFACE ${CUDA_cudart_static_LIBRARY} ${CUDA_extra_LIBS})

else()
    message(
        FATAL_ERROR
            "CUDA is not supported! ${PROJECT_USE_CUDA_OPTION}=${${PROJECT_USE_CUDA_OPTION}}"
        )
endif()
