# ------------------------------------------------------------------------------#
#
# Finds CPU architecture information for configuring architecture specific compiler flags,
# e.g. enabling AVX-512 instructions
#
# Example:
#
# find_package(CpuArch REQUIRED COMPONENTS          avx OPTIONAL_COMPONENTS fma avx2
# avx512f avx512pf avx512er avx512cd)
#
# add_library(${PROJECT_NAME}-arch INTERFACE) foreach(_ARCH ${CpuArch_FEATURES})
# target_compile_options(${PROJECT_NAME}-arch INTERFACE -m${_ARCH}) endforeach()
#
# ------------------------------------------------------------------------------#

include(CMakeParseArguments)

# always display find package message
unset(FIND_PACKAGE_MESSAGE_DETAILS_${CMAKE_FIND_PACKAGE_NAME} CACHE)

option(CpuArch_DEBUG "Enable verbose messages and failure if something not found" OFF)
mark_as_advanced(CpuArch_DEBUG)

# ----------------------------------------------------------------------------------------#
#
# For reference -- common cpu architecture flags
#
if(MSVC)
    set(CpuArch_AVAILABLE_COMPONENTS
        sse sse2 avx avx2 avx512
        CACHE STRING "Possible CPU flags (for reference, may be incomplete)")
else()
    set(CpuArch_AVAILABLE_COMPONENTS
        3dnowprefetch
        abm
        acpi
        adx
        aes
        altivec
        aperfmperf
        apic
        arat
        arch_perfmon
        avx
        avx1.0
        avx2
        avx512f
        avx512pf
        avx512er
        avx512cd
        bmi1
        bmi2
        bts
        cat_l3
        cdp_l3
        clflush
        clfsh
        clfsopt
        cmov
        constant_tsc
        cpuid
        cpuid_fault
        cqm
        cqm_llc
        cqm_mbm_local
        cqm_mbm_total
        cqm_occup_llc
        cx16
        cx8
        dca
        de
        ds
        ds_cpl
        dscpl
        dtes64
        dtherm
        dts
        epb
        ept
        erms
        est
        f16c
        flexpriority
        flush_l1d
        fma
        fpu
        fpu_csds
        fsgsbase
        fxsr
        hle
        ht
        htt
        ibpb
        ibrs
        ida
        intel_ppin
        intel_pt
        invpcid
        invpcid_single
        ipt
        l1df
        lahf_lm
        lm
        mca
        mce
        md_clear
        mdclear
        mmx
        mon
        monitor
        movbe
        mpx
        msr
        mtrr
        nonstop_tsc
        nopl
        nx
        osxsave
        pae
        pat
        pbe
        pcid
        pclmulqdq
        pdcm
        pdpe1gb
        pebs
        pge
        pln
        pni
        popcnt
        pse
        pse36
        pti
        pts
        rdrand
        rdseed
        rdt_a
        rdtscp
        rdwrfsgs
        rep_good
        rtm
        sdbg
        seglim64
        sep
        sgx
        sgxlc
        smap
        smep
        smx
        ss
        ssbd
        sse
        sse2
        sse3
        sse4_1
        sse4_2
        ssse3
        stibp
        syscall
        tm
        tm2
        tpr
        tpr_shadow
        tsc
        tsc_adjust
        tsc_deadline_timer
        tsc_thread_offset
        tsctmr
        tsxfa
        vme
        vmx
        vnmi
        vpid
        x2apic
        xsave
        xsaveopt
        xtopology
        xtp
        CACHE STRING "Possible CPU flags (for reference, may be incomplete)")
endif()

mark_as_advanced(CpuArch_AVAILABLE_COMPONENTS)

# ----------------------------------------------------------------------------------------#
# If no components specified, configure a default set
#
set(CpuArch_FIND_DEFAULT OFF)

# if target changed
if(DEFINED CpuArch_TARGET_LAST AND NOT "${CpuArch_TARGET_LAST}" STREQUAL
                                   "${CpuArch_TARGET}")
    # if features did not change when target changed
    if(DEFINED CpuArch_FEATURES_LAST AND "${CpuArch_FEATURES_LAST}" STREQUAL
                                         "${CpuArch_FEATURES}")
        unset(CpuArch_FEATURES CACHE)
    endif()
endif()

# if features are defined, set those as components
if(DEFINED CpuArch_FEATURES)
    set(CpuArch_FIND_COMPONENTS ${CpuArch_FEATURES})
    # if features were set explicitly, require them unless already set
    foreach(_COMP ${CpuArch_FIND_COMPONENTS})
        if(NOT DEFINED CpuArch_FIND_REQUIRED_${_COMP})
            set(CpuArch_FIND_REQUIRED_${_COMP} ON)
            set(CpuArch_FIND_REQUIRED ON)
        endif()
    endforeach()
endif()

# if no components are specified, configure default set
if(NOT CpuArch_FIND_COMPONENTS)
    set(CpuArch_FIND_DEFAULT ON)
    if(MSVC)
        set(CpuArch_FIND_COMPONENTS sse sse2 avx avx2 avx512)
    else()
        set(CpuArch_FIND_COMPONENTS
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
            avx512f
            avx512pf
            avx512er
            avx512cd
            altivec)
    endif()
    foreach(_COMP ${CpuArch_FIND_COMPONENTS})
        set(CpuArch_FIND_REQUIRED_${_COMP} ${CpuArch_FIND_REQUIRED})
    endforeach()
endif()

# ----------------------------------------------------------------------------------------#
#
# CpuArch_INPUT       Input /proc/cpuinfo file for Linux
#
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CpuArch_INPUT
        "/proc/cpuinfo"
        CACHE INTERNAL "File used to get CpuArch features")
    mark_as_advanced(CpuArch_INPUT)
endif()

# ----------------------------------------------------------------------------------------#
#
# Detects host architecture flags such as AVX instruction support, etc.
#
function(DETECT_HOST_FEATURES _CPU_FLAGS_VAR)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        file(READ "${CpuArch_INPUT}" _cpuinfo)
        string(REGEX REPLACE ".*flags[ \t]*:[ \t]+([^\n]+).*" "\\1" _cpu_flags
                             "${_cpuinfo}")
        if("${_cpu_flags}" STREQUAL "${_cpuinfo}")
            string(REPLACE "," " " _cpuinfo "${_cpuinfo}")
            string(
                REGEX
                REPLACE
                    ".*cpu[ \t]*:[ \t]+[a-zA-Z0-9_-]+[ \t]+([a-zA-Z0-9_-]+)[ \t]+supported.*"
                    "\\1"
                    _cpu_flags
                    "${_cpuinfo}")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        execute_process(
            COMMAND /usr/sbin/sysctl -n machdep.cpu.features
            OUTPUT_VARIABLE _cpu_flags
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(
            COMMAND /usr/sbin/sysctl -n machdep.cpu.leaf7_features
            OUTPUT_VARIABLE _cpu_leaf7_flags
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(_cpu_flags "${_cpu_flags} ${_cpu_leaf7_flags}")
        string(TOLOWER "${_cpu_flags}" _cpu_flags)
        string(REPLACE "." "_" _cpu_flags "${_cpu_flags}")
    endif()

    if(CpuArch_DEBUG)
        if("${_cpu_flags}" STREQUAL "${_cpuinfo}")
            message(FATAL_ERROR "CPU flags could not be found in:\n${_cpu_info}")
        else()
            message(STATUS "")
            message(STATUS "CPU FLAGS : ${_cpu_flags}")
            message(STATUS "")
        endif()
    endif()

    string(REPLACE " " ";" _cpu_flags "${_cpu_flags}")
    set(_TMP ${_cpu_flags})
    set(${_CPU_FLAGS_VAR}
        "${_TMP}"
        PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#
#
# Detects host architecture values: - architecture name - cpu vendor - cpu family - cpu
# model
#
function(DETECT_HOST_ARCHITECTURE _CPU_ARCH_VAR)

    cmake_parse_arguments(DETECT "VENDOR;FAMILY;MODEL" "" "" ${ARGN})

    set(_vendor_id)
    set(_cpu_family)
    set(_cpu_model)

    set(TARGET_ARCHITECTURE "generic")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        file(READ "${CpuArch_INPUT}" _cpuinfo)
        string(REGEX REPLACE ".*vendor_id[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1"
                             _vendor_id "${_cpuinfo}")
        string(REGEX REPLACE ".*cpu family[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1"
                             _cpu_family "${_cpuinfo}")
        string(REGEX REPLACE ".*model[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _cpu_model
                             "${_cpuinfo}")

        # if no matches anywhere, expect an IBM PowerPC based on experiences with Summit
        if("${_vendor_id}" STREQUAL "${_cpu_family}")
            string(REGEX REPLACE ".*cpu[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _cpu
                                 "${_cpuinfo}")
            string(REGEX REPLACE "([a-zA-Z]+).*" "\\1" _cpu_family "${_cpu}")
            string(REGEX REPLACE "[a-zA-Z]+([0-9]+).*" "\\1" _cpu_model "${_cpu}")
            if("${_cpu_family}" STREQUAL "POWER")
                set(_vendor_id "IBM")
            endif()
            string(TOLOWER "${_cpu}" TARGET_ARCHITECTURE)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        execute_process(
            COMMAND /usr/sbin/sysctl -n machdep.cpu.vendor
            OUTPUT_VARIABLE _vendor_id
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(
            COMMAND /usr/sbin/sysctl -n machdep.cpu.model
            OUTPUT_VARIABLE _cpu_model
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(
            COMMAND /usr/sbin/sysctl -n machdep.cpu.family
            OUTPUT_VARIABLE _cpu_family
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(_cpu_id
            "$ENV{PROCESSOR_IDENTIFIER}"
            CACHE INTERNAL "Cpu info")
        string(REGEX REPLACE ".*, ([A-Za-z]+)$" "\\1" _vendor_id "${_cpu_id}")
        string(REGEX REPLACE ".* Family ([0-9]+) .*" "\\1" _cpu_family "${_cpu_id}")
        string(REGEX REPLACE ".* Model ([0-9]+) .*" "\\1" _cpu_model "${_cpu_id}")
    endif()

    if(_vendor_id STREQUAL "GenuineIntel")
        #
        if(_cpu_family EQUAL 6)
            # taken from the Intel ORM
            # http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html
            # CPUID Signature Values of Of Recent Intel Microarchitectures 4E 5E       |
            # Skylake microarchitecture 3D 47 56    | Broadwell microarchitecture 3C 45 46
            # 3F | Haswell microarchitecture 3A 3E       | Ivy Bridge microarchitecture 2A
            # 2D       | Sandy Bridge microarchitecture 25 2C 2F    | Intel
            # microarchitecture Westmere 1A 1E 1F 2E | Intel microarchitecture Nehalem 17
            # 1D       | Enhanced Intel Core microarchitecture 0F          | Intel Core
            # microarchitecture
            #
            # Intel SDM Vol. 3C 35-1 / December 2016: 57          | Xeon Phi 3200, 5200,
            # 7200  [Knights Landing] 85          | Future Xeon Phi 8E 9E       | 7th gen.
            # Core              [Kaby Lake] 55          | Future Xeon [Skylake w/ AVX512]
            # 4E 5E       | 6th gen. Core / E3 v5      [Skylake w/o AVX512] 56          |
            # Xeon D-1500                [Broadwell] 4F          | Xeon E5 v4, E7 v4,
            # i7-69xx [Broadwell] 47          | 5th gen. Core / Xeon E3 v4 [Broadwell] 3D
            # | M-5xxx / 5th gen.          [Broadwell] 3F | Xeon E5 v3, E7 v3, i7-59xx
            # [Haswell-E] 3C 45 46    | 4th gen. Core, Xeon E3 v3  [Haswell] 3E          |
            # Xeon E5 v2, E7 v2, i7-49xx [Ivy Bridge-E] 3A | 3rd gen. Core, Xeon E3 v2
            # [Ivy Bridge] 2D          | Xeon E5, i7-39xx [Sandy Bridge] 2F          |
            # Xeon E7 2A          | Xeon E3, 2nd gen. Core [Sandy Bridge] 2E          |
            # Xeon 7500, 6500 series 25 2C       | Xeon 3600, 5600 series, Core i7, i5 and
            # i3
            #
            # Values from the Intel SDE: 5C | Goldmont 5A | Silvermont 57 | Knights
            # Landing 66 | Cannonlake 55 | Skylake Server 4E | Skylake Client 3C |
            # Broadwell (likely a bug in the SDE) 3C | Haswell

            function(issue_message _family _model _use)
                set(_msg "Your CPU (family ${_family}, model ${_model}) is not known.")
                set(_msg "${_msg} Auto-detection of optimization flags failed and will")
                set(_msg "${_msg} use the ${_use} settings.")
                message(WARNING "${_msg}")
                unset(_msg)
            endfunction()

            if(_cpu_model EQUAL 87) # 57
                set(TARGET_ARCHITECTURE "knl") # Knights Landing
            elseif(_cpu_model EQUAL 92)
                set(TARGET_ARCHITECTURE "goldmont")
            elseif(_cpu_model EQUAL 90 OR _cpu_model EQUAL 76)
                set(TARGET_ARCHITECTURE "silvermont")
            elseif(_cpu_model EQUAL 102)
                set(TARGET_ARCHITECTURE "cannonlake")
            elseif(_cpu_model EQUAL 125 OR _cpu_model EQUAL 126)
                set(TARGET_ARCHITECTURE "icelake")
            elseif(_cpu_model EQUAL 106 OR _cpu_model EQUAL 108)
                set(TARGET_ARCHITECTURE "icelake-server")
            elseif(_cpu_model EQUAL 142 OR _cpu_model EQUAL 158) # 8E, 9E
                set(TARGET_ARCHITECTURE "kaby-lake")
            elseif(_cpu_model EQUAL 85) # 55
                set(TARGET_ARCHITECTURE "skylake-avx512")
            elseif(
                _cpu_model EQUAL 78
                OR _cpu_model EQUAL 94
                OR _cpu_model EQUAL 165) # 4E, 5E
                set(TARGET_ARCHITECTURE "skylake")
            elseif(
                _cpu_model EQUAL 61
                OR _cpu_model EQUAL 71
                OR _cpu_model EQUAL 79
                OR _cpu_model EQUAL 86) # 3D, 47, 4F, 56
                set(TARGET_ARCHITECTURE "broadwell")
            elseif(
                _cpu_model EQUAL 60
                OR _cpu_model EQUAL 69
                OR _cpu_model EQUAL 70
                OR _cpu_model EQUAL 63)
                set(TARGET_ARCHITECTURE "haswell")
            elseif(_cpu_model EQUAL 58 OR _cpu_model EQUAL 62)
                set(TARGET_ARCHITECTURE "ivy-bridge")
            elseif(_cpu_model EQUAL 42 OR _cpu_model EQUAL 45)
                set(TARGET_ARCHITECTURE "sandy-bridge")
            elseif(
                _cpu_model EQUAL 37
                OR _cpu_model EQUAL 44
                OR _cpu_model EQUAL 47)
                set(TARGET_ARCHITECTURE "westmere")
            elseif(
                _cpu_model EQUAL 26
                OR _cpu_model EQUAL 30
                OR _cpu_model EQUAL 31
                OR _cpu_model EQUAL 46)
                set(TARGET_ARCHITECTURE "nehalem")
            elseif(_cpu_model EQUAL 23 OR _cpu_model EQUAL 29)
                set(TARGET_ARCHITECTURE "penryn")
            elseif(_cpu_model EQUAL 15)
                set(TARGET_ARCHITECTURE "merom")
            elseif(_cpu_model EQUAL 28)
                set(TARGET_ARCHITECTURE "atom")
            elseif(_cpu_model EQUAL 14)
                set(TARGET_ARCHITECTURE "core")
            elseif(_cpu_model LESS 14)
                issue_message(${_cpu_family} ${_cpu_model} "generic CPU w/ SSE2")
                set(TARGET_ARCHITECTURE "generic")
            else()
                message(WARNING)
                issue_message(${_cpu_family} ${_cpu_model} "65nm Core 2 CPU")
                set(TARGET_ARCHITECTURE "merom")
            endif()
        elseif(_cpu_family EQUAL 7) # Itanium (not supported)
            message(
                WARNING
                    "Your CPU (Itanium: family ${_cpu_family}, model ${_cpu_model}) is not supported."
                )
        elseif(_cpu_family EQUAL 15) # NetBurst
            list(APPEND _available_vector_units_list "sse" "sse2")
            if(_cpu_model GREATER 2) # Not sure whether this must be 3 or even 4 instead
                list(APPEND _available_vector_units_list "sse" "sse2" "sse3")
            endif(_cpu_model GREATER 2)
        endif()
        #
    elseif(_vendor_id STREQUAL "AuthenticAMD")
        #
        if(_cpu_family EQUAL 23)
            set(TARGET_ARCHITECTURE "zen")
        elseif(_cpu_family EQUAL 22) # 16h
            set(TARGET_ARCHITECTURE "AMD 16h")
        elseif(_cpu_family EQUAL 21) # 15h
            if(_cpu_model LESS 2)
                set(TARGET_ARCHITECTURE "bulldozer")
            else()
                set(TARGET_ARCHITECTURE "piledriver")
            endif()
        elseif(_cpu_family EQUAL 20) # 14h
            set(TARGET_ARCHITECTURE "AMD 14h")
        elseif(_cpu_family EQUAL 18) # 12h

        elseif(_cpu_family EQUAL 16) # 10h
            set(TARGET_ARCHITECTURE "barcelona")
        elseif(_cpu_family EQUAL 15)
            set(TARGET_ARCHITECTURE "k8")
            if(_cpu_model GREATER 64)
                set(TARGET_ARCHITECTURE "k8-sse3")
            endif(_cpu_model GREATER 64)
        endif()
        #
    endif()

    if(CpuArch_DEBUG)
        function(CpuArch_DEBUG_OUTPUT LABEL VARIABLE INFO)
            if("${${VARIABLE}}" STREQUAL "${INFO}" AND NOT "${INFO}" STREQUAL "")
                message(
                    FATAL_ERROR "${VARIABLE} (${LABEL}) could not be found in:\n${INFO}")
            else()
                message(STATUS "${LABEL} : ${${VARIABLE}}")
            endif()
        endfunction()

        message(STATUS "")
        cpuarch_debug_output("VENDOR ID   " _vendor_id "${_cpuinfo}")
        cpuarch_debug_output("CPU FAMILY  " _cpu_family "${_cpuinfo}")
        cpuarch_debug_output("CPU MODEL   " _cpu_model "${_cpuinfo}")
        cpuarch_debug_output("ARCHITECTURE" TARGET_ARCHITECTURE "${_cpuinfo}")
        message(STATUS "")
    endif()

    if(DETECT_VENDOR)
        set(${_CPU_ARCH_VAR}
            "${_vendor_id}"
            PARENT_SCOPE)
    elseif(DETECT_FAMILY)
        set(${_CPU_ARCH_VAR}
            "${_cpu_family}"
            PARENT_SCOPE)
    elseif(DETECT_MODEL)
        set(${_CPU_ARCH_VAR}
            "${_cpu_model}"
            PARENT_SCOPE)
    else()
        set(${_CPU_ARCH_VAR}
            "${TARGET_ARCHITECTURE}"
            PARENT_SCOPE)
    endif()

    # temporary variables
    set(CpuArch_VENDOR_ID
        "${_vendor_id}"
        PARENT_SCOPE)
    set(CpuArch_CPU_FAMILY
        "${_cpu_family}"
        PARENT_SCOPE)
    set(CpuArch_CPU_MODEL
        "${_cpu_model}"
        PARENT_SCOPE)

endfunction()

# ----------------------------------------------------------------------------------------#
#
# Find the request cpu features from the available flags and/or append the highest known
# flag if possible
#
function(GET_CPU_FEATURES _OUTVAR)

    # parse args
    cmake_parse_arguments(_CPU "" "TARGET" "VALID;CANDIDATES" ${ARGN})

    if("${_CPU_CANDIDATES}" STREQUAL "")
        message(STATUS "No CpuArch_COMPONENTS found")
        return()
    endif()

    if("${_CPU_VALID}" STREQUAL "")
        message(STATUS "No valid features")
        return()
    endif()

    if(MSVC)
        return()
    endif()

    string(REPLACE "_" "." _CPU_VALID "${_CPU_VALID}")
    foreach(_CANDIDATE ${_CPU_CANDIDATES})
        if("${_CANDIDATE}" IN_LIST _CPU_VALID)
            list(APPEND _FEATURES ${_CANDIDATE})
            list(REMOVE_DUPLICATES _FEATURES)
        endif()
        list(APPEND _MISSING ${_CANDIDATE})
        set(CpuArch_${_CANDIDATE}_FOUND
            OFF
            PARENT_SCOPE)
    endforeach()

    set(OFA_map_knl "MIC-AVX512")
    set(OFA_map_cannonlake "CORE-AVX512")
    set(OFA_map_skylake-avx512 "CORE-AVX512")
    set(OFA_map_skylake "CORE-AVX2")
    set(OFA_map_broadwell "CORE-AVX2")
    set(OFA_map_haswell "CORE-AVX2")
    set(OFA_map_ivybridge "CORE-AVX-I")
    set(OFA_map_sandybridge "AVX")
    set(OFA_map_westmere "SSE4.2")
    set(OFA_map_nehalem "SSE4.2")
    set(OFA_map_penryn "SSSE3")
    set(OFA_map_merom "SSSE3")
    set(OFA_map_core2 "SSE3")
    set(OFA_map_kaby-lake "AVX2")
    set(OFA_map_power9 "ALTIVEC")

    list(APPEND _FEATURES "${OFA_map_${_CPU_TARGET}}")
    if("${_FEATURES}" STREQUAL "")
        list(APPEND _FEATURES "HOST")
    endif()

    string(TOLOWER "${_FEATURES}" _FEATURES)
    if(NOT "${_FEATURES}" STREQUAL "")
        list(REMOVE_DUPLICATES _FEATURES)
    endif()
    string(REPLACE "_" "." _FEATURES "${_FEATURES}")

    set(_TMP ${_FEATURES})
    foreach(_VAR ${_TMP})
        list(APPEND _FOUND ${_VAR})
        list(REMOVE_ITEM _MISSING ${_VAR})
        set(CpuArch_${_VAR}_FOUND
            ON
            PARENT_SCOPE)
    endforeach()
    set(CpuArch_FOUND_COMPONENTS
        ${_FOUND}
        PARENT_SCOPE)
    set(CpuArch_MISSING_COMPONENTS
        ${_MISSING}
        PARENT_SCOPE)
    set(${_OUTVAR}
        "${_TMP}"
        PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#
#
detect_host_architecture(TARGET_ARCH)
set(CpuArch_TARGET
    "${TARGET_ARCH}"
    CACHE STRING "CPU architecture target")
set(CpuArch_TARGET_LAST
    "${CpuArch_TARGET}"
    CACHE INTERNAL "CPU architecture target" FORCE)

# get the target features
detect_host_features(TARGET_FEATURES)

# check the components against the available features
get_cpu_features(
    _FEATURES
    TARGET ${CpuArch_TARGET}
    VALID ${TARGET_FEATURES}
    CANDIDATES ${CpuArch_FIND_COMPONENTS})

set(CpuArch_FEATURES
    "${_FEATURES}"
    CACHE STRING "CPU architecture features")
set(CpuArch_FEATURES_LAST
    "${CpuArch_FEATURES}"
    CACHE INTERNAL "CPU architecture features" FORCE)

# if we populated w/ default set, don't report as missing and don't require them
if(CpuArch_FIND_DEFAULT)
    foreach(_MISSING ${CpuArch_MISSING_COMPONENTS})
        list(REMOVE_ITEM CpuArch_FIND_COMPONENTS ${_MISSING})
        set(CpuArch_FIND_REQUIRED_${_MISSING} OFF)
    endforeach()
endif()

# ----------------------------------------------------------------------------------------#
#
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    CpuArch
    FOUND_VAR CpuArch_FOUND
    REQUIRED_VARS CpuArch_TARGET
    HANDLE_COMPONENTS)
