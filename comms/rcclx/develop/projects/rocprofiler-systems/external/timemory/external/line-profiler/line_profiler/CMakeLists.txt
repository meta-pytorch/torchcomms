set(cython_source
  ${CMAKE_CURRENT_SOURCE_DIR}/_line_profiler.pyx
  ${CMAKE_CURRENT_SOURCE_DIR}/python25.pxd
)
set(module_name "_line_profiler")
set(_install_dest ${CMAKE_INSTALL_PYTHONDIR}/${CMAKE_PROJECT_NAME}/line_profiler)

# Translate Cython into C/C++
add_cython_target(${module_name} "${cython_source}" C OUTPUT_VAR sources)

# Add any other non-cython dependencies to the sources
list(APPEND sources
  ${CMAKE_CURRENT_SOURCE_DIR}/unset_trace.c)
# message(STATUS "[OURS] sources = ${sources}")

# Create C++ library. Specify include dirs and link libs as normal
add_library(${module_name} MODULE ${TIMEMORY_EXCLUDE_FROM_ALL} ${sources})
target_include_directories(${module_name} PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}  # for the pure c files defined here
)
set_target_properties(${module_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}/line_profiler
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}/line_profiler
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}/line_profiler
)

# compile options
include(CheckCCompilerFlag)
check_c_compiler_flag("-Wno-deprecated-declarations" line_profiler_no_deprecated_declarations)
if(line_profiler_no_deprecated_declarations)
    target_compile_options(${module_name} PRIVATE "-Wno-deprecated-declarations")
endif()

file(GLOB pysources ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
foreach(f ${pysources})
    get_filename_component(fname "${f}" NAME)
    set(fout ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}/line_profiler/${fname})
    configure_file(${f} ${fout} COPYONLY)
    install(FILES ${fout} DESTINATION ${_install_dest} OPTIONAL)
endforeach()

# Transform the C++ library into an importable python module
python_extension_module(${module_name})

# Install the C++ module to the correct relative location
# (this will be an inplace build if you use `pip install -e`)
#file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
#set(_install_dest ".")

#message(STATUS "_install_dest = ${_install_dest}")
#install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
#file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
#message(STATUS "[OURS] _install_dest = ${_install_dest}")
install(TARGETS ${module_name} LIBRARY DESTINATION ${_install_dest} OPTIONAL)
