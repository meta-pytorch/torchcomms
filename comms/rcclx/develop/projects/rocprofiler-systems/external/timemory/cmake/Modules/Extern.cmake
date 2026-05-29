# ----------------------------------------------------------------------------------------#
# Extern initialization
#
file(GLOB extern_init_sources ${CMAKE_CURRENT_LIST_DIR}/init.cpp)
file(GLOB extern_native_sources ${CMAKE_CURRENT_LIST_DIR}/native_extern.cpp)
file(GLOB extern_cuda_sources ${CMAKE_CURRENT_LIST_DIR}/cuda_extern.cpp)
file(GLOB extern_auto_timer_sources ${CMAKE_CURRENT_LIST_DIR}/auto_timer_extern.cpp)

set(extern_init_link_libs_shared timemory-external-shared)
set(extern_cuda_link_libs_shared timemory-cuda # timemory-cupti
                                 timemory-cudart # timemory-cudart-device
    )
set(extern_native_link_libs_shared)
set(extern_auto_timer_link_libs_shared)

set(extern_init_link_libs_static timemory-external-static)
set(extern_cuda_link_libs_static timemory-cuda # timemory-cupti
                                 timemory-cudart-static # timemory-cudart-device
    )
set(extern_native_link_libs_static)
set(extern_auto_timer_link_libs_static)

set(extern_init_definitions TIMEMORY_EXTERN_INIT)
set(extern_cuda_definitions TIMEMORY_EXTERN_CUDA_TEMPLATES)
set(extern_native_definitions TIMEMORY_EXTERN_NATIVE_TEMPLATES)
set(extern_auto_timer_definitions TIMEMORY_EXTERN_AUTO_TIMER_TEMPLATES)

# message(STATUS "BUILD CXX SHARED: ${_BUILD_SHARED_CXX}") message(STATUS "BUILD CXX
# STATIC: ${_BUILD_STATIC_CXX}")

set(TIMEMORY_EXTERN_LIBRARIES extern-native extern-cuda extern-auto-timer)
foreach(_EXTERN_LIB init native cuda auto-timer)
    foreach(_TYPE shared static)
        # Windows makes all this crap too complicated
        if(WIN32
           OR ("${_TYPE}" STREQUAL "shared" AND NOT _BUILD_SHARED_CXX)
           OR ("${_TYPE}" STREQUAL "static" AND NOT _BUILD_STATIC_CXX))
            timemory_add_interface_library(timemory-extern-${_EXTERN_LIB}-${_TYPE})
            continue()
        endif()

        # determine linker language
        set(_LINK_LANGUAGE CXX)
        # don't build cuda
        if(NOT TIMEMORY_USE_CUDA AND "${_EXTERN_LIB}" STREQUAL "cuda")
            continue()
        elseif(TIMEMORY_USE_CUDA AND "${_EXTERN_LIB}" STREQUAL "cuda")
            set(_LINK_LANGUAGE CUDA)
        endif()

        string(REPLACE "-" "_" _NAME "${_EXTERN_LIB}")

        string(TOUPPER "${_TYPE}" _LIB_TYPE)
        # ------------------------------------------------------------------------------------#
        # build the extern libraries
        #
        timemory_build_library(
            PIC
            TYPE ${_LIB_TYPE}
            TARGET_NAME timemory-extern-${_EXTERN_LIB}-${_TYPE}
            OUTPUT_NAME timemory-extern-${_EXTERN_LIB}
            LANGUAGE CXX
            LINKER_LANGUAGE ${_LINK_LANGUAGE}
            OUTPUT_DIR ${PROJECT_BINARY_DIR}
            SOURCES ${extern_${_NAME}_sources}
            COMPILE_DEFINITIONS PUBLIC ${extern_${_NAME}_definitions}
            LINK_LIBRARIES timemory-headers ${extern_${_NAME}_link_libs_${_TYPE}} PRIVATE
                           timemory-compile-options timemory-arch)
        #
        # suffix when shared is empty and static is "-static
        #
        set(_SUFFIX)
        if("${_TYPE}" STREQUAL "static")
            set(_SUFFIX "-static")
        endif()

        if("${_EXTERN_LIB}" STREQUAL "init")
            if("${_TYPE}" STREQUAL "shared")
                # target_link_libraries(timemory-extern-init INTERFACE
                # timemory-extern-${_EXTERN_LIB}-shared)
            endif()
            # target_link_libraries(timemory-cxx-${_TYPE} PRIVATE
            # timemory-extern-init-${_TYPE})
        elseif("${_EXTERN_LIB}" STREQUAL "auto-timer")
            # target_link_libraries(timemory-extern-templates${_SUFFIX} INTERFACE
            # timemory-extern-${_EXTERN_LIB}-${_TYPE})
            # target_link_libraries(timemory-cxx-${_TYPE}               PRIVATE
            # timemory-extern-${_EXTERN_LIB}-${_TYPE})
        elseif(NOT "${_EXTERN_LIB}" STREQUAL "cuda")
            # target_link_libraries(timemory-extern-templates${_SUFFIX} INTERFACE
            # timemory-extern-${_EXTERN_LIB}-${_TYPE})
        endif()

    endforeach()

endforeach()
