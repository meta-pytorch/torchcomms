#
# Configures architecture options
#

set(_CpuArch_COMPONENTS)
if(TIMEMORY_BUILD_PORTABLE)
    set(CpuArch_FIND_DEFAULT ON)
    if(MSVC)
        set(_CpuArch_COMPONENTS OPTIONAL_COMPONENTS sse sse2 avx avx2)
    else()
        set(_CpuArch_COMPONENTS
            OPTIONAL_COMPONENTS
            sse
            sse2
            sse3
            ssse3
            sse4
            sse4_1
            sse4_2
            fma
            avx
            avx2
            altivec)
    endif()
endif()

find_package(CpuArch ${_CpuArch_COMPONENTS})

if(CpuArch_FOUND)

    if(VECTOR_INTERFACE_TARGET AND VECTOR_DEFINITION)
        set(_VEC_256 OFF)
        set(_VEC_512 OFF)
        foreach(_ARCH ${CpuArch_FEATURES})
            if("${_ARCH}" MATCHES ".*avx512.*" OR "${_ARCH}" MATCHES ".*AVX512.*")
                set(_VEC_512 ON)
            elseif("${_ARCH}" MATCHES ".*avx.*" OR "${_ARCH}" MATCHES ".*AVX.*")
                set(_VEC_256 ON)
            endif()
        endforeach()

        if(_VEC_512)
            set(TIMEMORY_VEC 512)
        elseif(_VEC_256)
            set(TIMEMORY_VEC 256)
        else()
            set(TIMEMORY_VEC 128)
        endif()
        timemory_message(STATUS "Compiling with vector width: ${TIMEMORY_VEC}")
        timemory_target_compile_definitions(${VECTOR_INTERFACE_TARGET} INTERFACE
                                            ${VECTOR_DEFINITION}=${TIMEMORY_VEC})
    endif()

    if(ARCH_INTERFACE_TARGET)
        if(CpuArch_FEATURES)
            foreach(_ARCH ${CpuArch_FEATURES})
                if(MSVC)
                    add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "/arch:${_ARCH}")
                else()
                    add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "-m${_ARCH}")
                endif()
            endforeach()

            # intel compiler
            if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
                list(LENGTH CpuArch_FEATURES NUM_FEATURES)
                math(EXPR LAST_FEATURE_IDX "${NUM_FEATURES}-1")
                list(GET CpuArch_FEATURES ${LAST_FEATURE_IDX} _ARCH)
                if(_ARCH)
                    add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "-ax${_ARCH}")
                endif()
                unset(_ARCH)
                unset(NUM_FEATURES)
                unset(LAST_FEATURE_IDX)
            endif()
        else()
            if(MSVC)
                add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "/arch")
            else()
                add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "-march=native")
            endif()
        endif()
    endif()

endif()
