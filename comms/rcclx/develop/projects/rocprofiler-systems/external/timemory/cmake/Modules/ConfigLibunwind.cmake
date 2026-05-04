#
#
# Builds Libunwind
#
#

set(TIMEMORY_LIBUNWIND_BUILD_COMMAND)

# finds an executable and fails if not found
macro(timemory_libunwind_find_exe VAR MSG)
    find_program(${VAR} NAMES ${ARGN})
    mark_as_advanced(${VAR})

    if(NOT ${VAR})
        message(FATAL_ERROR "Building libunwind submodule requires ${MSG}")
    endif()
endmacro()

timemory_libunwind_find_exe(AUTORECONF_EXE "autoreconf" autoreconf)
timemory_libunwind_find_exe(MAKE_EXE "make / gmake" make gmake)

# copy from source directory to binary directory
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/external/libunwind
            ${PROJECT_BINARY_DIR}/external/libunwind)

# update the SOVERSION of libunwind to avoid picking up system installs
file(READ ${PROJECT_BINARY_DIR}/external/libunwind/src/Makefile.am
     timemory_libunwind_src_makefile_am)
string(REGEX
       REPLACE "SOVERSION=([0-9]+):([0-9]+):([0-9]+)" "SOVERSION=99:0:0"
               timemory_libunwind_src_makefile_am "${timemory_libunwind_src_makefile_am}")
file(WRITE ${PROJECT_BINARY_DIR}/external/libunwind/src/Makefile.am
     "${timemory_libunwind_src_makefile_am}")

# glob the files copied over
file(GLOB_RECURSE timemory_libunwind_pre_build_files
     "${PROJECT_SOURCE_DIR}/external/libunwind/*")
string(REPLACE "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}"
               timemory_libunwind_pre_build_files "${timemory_libunwind_pre_build_files}")

function(timemory_libunwind_execute_process)
    execute_process(
        COMMAND ${ARGN}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external/libunwind
        OUTPUT_VARIABLE OUT
        ERROR_VARIABLE ERR
        RESULT_VARIABLE RET)

    if(NOT RET EQUAL 0)
        string(REPLACE ";" " " _CMD "${ARGN}")
        message(FATAL_ERROR "'${_CMD}' failed:\nOUTPUT:\n${OUT}\nERROR:\n${ERR}")
    endif()
endfunction()

function(timemory_libunwind_autoreconf)
    message(STATUS "[timemory] Generating libunwind configure...")
    timemory_libunwind_execute_process(${AUTORECONF_EXE} -i)
endfunction()

function(timemory_libunwind_configure)
    message(STATUS "[timemory] Configuring libunwind...")
    timemory_libunwind_execute_process(
        ${CMAKE_COMMAND}
        -E
        env
        CC=${CMAKE_C_COMPILER}
        CFLAGS=-fPIC\ -O3\ -Wno-unused-result\ -Wno-unused-but-set-variable\ -Wno-cpp
        CXX=${CMAKE_CXX_COMPILER}
        CXXFLAGS=-fPIC\ -O3\ -Wno-unused-result\ -Wno-unused-but-set-variable\ -Wno-cpp
        ./configure
        --enable-shared=yes
        --enable-static=no
        --prefix=${PROJECT_BINARY_DIR}/external/libunwind/install
        --libdir=${PROJECT_BINARY_DIR}/external/libunwind/install/lib)

    # remove installation if new build
    timemory_libunwind_execute_process(${MAKE_EXE} clean)
endfunction()

function(timemory_libunwind_build)
    message(STATUS "[timemory] Building libunwind...")
    timemory_libunwind_execute_process(${MAKE_EXE})

    # remove installation if new build
    file(REMOVE_RECURSE ${PROJECT_BINARY_DIR}/external/libunwind/src/install)
endfunction()

function(timemory_libunwind_install)
    message(STATUS "[timemory] Installing libunwind...")
    timemory_libunwind_execute_process(${MAKE_EXE} install)
endfunction()

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/libunwind/configure)
    timemory_libunwind_autoreconf()
    timemory_libunwind_configure()
    timemory_libunwind_build()
elseif(NOT EXISTS ${PROJECT_BINARY_DIR}/external/libunwind/Makefile)
    timemory_libunwind_configure()
    timemory_libunwind_build()
elseif(NOT EXISTS ${PROJECT_BINARY_DIR}/external/libunwind/src/.libs)
    timemory_libunwind_configure()
    timemory_libunwind_build()
endif()

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/libunwind/install)
    timemory_libunwind_install()
endif()

file(GLOB_RECURSE timemory_libunwind_post_build_files
     "${PROJECT_BINARY_DIR}/external/libunwind/*")

set(TIMEMORY_LIBUNWIND_BUILD_BYPRODUCTS ${timemory_libunwind_post_build_files})
list(REMOVE_ITEM TIMEMORY_LIBUNWIND_BUILD_BYPRODUCTS
     ${timemory_libunwind_pre_build_files})

add_custom_target(
    build-timemory-libunwind
    COMMAND ${AUTORECONF_EXE} -i
    COMMAND
        ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER}
        CFLAGS=-fPIC\ -O3\ -Wno-unused-result\ -Wno-unused-but-set-variable\ -Wno-cpp
        CXX=${CMAKE_CXX_COMPILER}
        CXXFLAGS=-fPIC\ -O3\ -Wno-unused-result\ -Wno-unused-but-set-variable\ -Wno-cpp
        ./configure --enable-shared=yes --enable-static=no
        --prefix=${PROJECT_BINARY_DIR}/external/libunwind/install
    COMMAND ${MAKE_EXE}
    COMMAND ${MAKE_EXE} install
    COMMENT "Building libunwind..."
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external/libunwind)

add_custom_target(
    clean-timemory-libunwind
    COMMAND ${CMAKE_COMMAND} -E rm -f ${TIMEMORY_LIBUNWIND_BUILD_BYPRODUCTS}
    COMMENT "Cleaning libunwind..."
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external/libunwind)

add_dependencies(build-timemory-libunwind clean-timemory-libunwind)

if(TIMEMORY_INSTALL_HEADERS)
    file(GLOB libunwind_headers
         "${PROJECT_BINARY_DIR}/external/libunwind/install/include/*.h")

    foreach(_HEADER ${libunwind_headers})
        install(
            FILES ${_HEADER}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory/libunwind
            OPTIONAL)
    endforeach()
endif()

file(GLOB libunwind_libs
     "${PROJECT_BINARY_DIR}/external/libunwind/install/${CMAKE_INSTALL_LIBDIR}/*")

foreach(_LIB ${libunwind_libs})
    if(IS_DIRECTORY ${_LIB})
        continue()
    endif()

    if("${_LIB}" MATCHES "\\.so($|\\.)")
        execute_process(COMMAND ${CMAKE_STRIP} ${_LIB})
        find_program(CHRPATH_EXECUTABLE chrpath)
        find_program(PATCHELF_EXECUTABLE patchelf)

        if(CHRPATH_EXECUTABLE)
            execute_process(COMMAND ${CHRPATH_EXECUTABLE} -r "$ORIGIN" ${_LIB})
        elseif(PATCHELF_EXECUTABLE)
            execute_process(COMMAND ${PATCHELF_EXECUTABLE} --set-rpath "$ORIGIN" ${_LIB})
        else()
            message(
                AUTHOR_WARNING
                    "[timemory] Neither chrpath nor patchelf found. Skipping rpath modification for libunwind."
                )
        endif()
    endif()

    install(
        FILES ${_LIB}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/libunwind
        OPTIONAL)
endforeach()

install(
    DIRECTORY ${PROJECT_BINARY_DIR}/external/libunwind/install/lib/pkgconfig
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/libunwind/pkgconfig
    OPTIONAL)

# Add include directories with BEFORE to ensure they come first in include search path
# This ensures GNU libunwind headers are found before LLVM libunwind headers
target_include_directories(
    timemory-libunwind BEFORE
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/libunwind/install/include>
              $<INSTALL_INTERFACE:include/timemory/libunwind>)
target_link_directories(
    timemory-libunwind
    INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/libunwind/install/${CMAKE_INSTALL_LIBDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}/timemory/libunwind>)
target_link_libraries(
    timemory-libunwind
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/libunwind/install/${CMAKE_INSTALL_LIBDIR}/libunwind${CMAKE_SHARED_LIBRARY_SUFFIX}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/timemory/libunwind/libunwind${CMAKE_SHARED_LIBRARY_SUFFIX}>
    )
timemory_target_compile_definitions(timemory-libunwind INTERFACE TIMEMORY_USE_LIBUNWIND
                                    UNW_LOCAL_ONLY)
