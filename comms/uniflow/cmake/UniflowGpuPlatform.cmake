# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# GPU platform detection and the AMD source translation step.
#
# uniflow's GPU layer is written once against the CUDA API and translated to HIP
# for AMD. The platform is resolved once into UNIFLOW_GPU_PLATFORM (CUDA or HIP)
# and everything branches on that, so the seam sits in one place.
#
# This translates with hipify-perl. It is not the only way the sources can reach
# a HIP compiler, and a build that translates them differently will not produce
# an identical result, so do not assume coverage carries across from another
# build of this tree.
#
# Only sources that name cuda*/cu* types directly need translating. Headers that
# stay in neutral types are shared verbatim, so their includers need no
# translation of their own.

set(UNIFLOW_GPU_PLATFORM "" CACHE STRING "GPU platform: CUDA, HIP, or empty to auto-detect")
set_property(CACHE UNIFLOW_GPU_PLATFORM PROPERTY STRINGS "" CUDA HIP)

if(NOT UNIFLOW_GPU_PLATFORM)
  find_package(CUDAToolkit ${UNIFLOW_MINIMUM_CUDA_VERSION} QUIET)
  find_package(hip QUIET)
  if(CUDAToolkit_FOUND AND hip_FOUND)
    # Guessing here is worse than stopping: the wrong guess produces a library
    # that cannot run on the machine that built it, and says so only in a
    # status line.
    message(FATAL_ERROR
        "Both a CUDA toolkit and ROCm were found; set -DUNIFLOW_GPU_PLATFORM=CUDA or HIP.")
  elseif(CUDAToolkit_FOUND)
    set(UNIFLOW_GPU_PLATFORM CUDA)
  elseif(hip_FOUND)
    set(UNIFLOW_GPU_PLATFORM HIP)
  endif()
endif()

# Write the resolved value back so it is visible and stays put across reconfigures.
set(UNIFLOW_GPU_PLATFORM "${UNIFLOW_GPU_PLATFORM}" CACHE STRING "GPU platform" FORCE)

if(NOT UNIFLOW_GPU_PLATFORM)
  message(FATAL_ERROR
      "No GPU platform found. Install the CUDA toolkit or ROCm, or set "
      "-DUNIFLOW_GPU_PLATFORM=CUDA|HIP with the matching toolkit on "
      "CMAKE_PREFIX_PATH.")
endif()

message(STATUS "UNIFLOW_GPU_PLATFORM = ${UNIFLOW_GPU_PLATFORM}")

if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  find_package(CUDAToolkit ${UNIFLOW_MINIMUM_CUDA_VERSION} REQUIRED)
  set(UNIFLOW_GPU_LIBRARIES CUDA::cudart CUDA::cuda_driver)
else()
  find_package(hip REQUIRED)
  set(UNIFLOW_GPU_LIBRARIES hip::host)

  find_program(UNIFLOW_HIPIFY_PERL
      NAMES hipify-perl
      HINTS ENV ROCM_PATH /opt/rocm
      PATH_SUFFIXES bin
  )
  if(NOT UNIFLOW_HIPIFY_PERL)
    message(FATAL_ERROR
        "hipify-perl not found; it ships with ROCm. Set ROCM_PATH or put it on PATH.")
  endif()
  message(STATUS "hipify-perl = ${UNIFLOW_HIPIFY_PERL}")
endif()

# uniflow_hipify(<out_var> <source>...)
#
# Translates each source with hipify-perl and returns the generated paths in
# <out_var>. On CUDA the inputs are returned unchanged, so callers list their
# sources once and stay platform-neutral.
#
# Outputs keep their path relative to the UniFlow source root under
# ${UNIFLOW_HIPIFY_DIR}, which precedes the build include tree on the include
# path. A translated header is therefore reachable through the same
# "comms/uniflow/..." include path as its original.
function(uniflow_hipify out_var)
  if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
    set(${out_var} ${ARGN} PARENT_SCOPE)
    return()
  endif()

  set(generated "")
  foreach(source IN LISTS ARGN)
    get_filename_component(absolute "${source}" ABSOLUTE)
    file(RELATIVE_PATH relative "${UNIFLOW_SOURCE_DIR}" "${absolute}")
    if(relative MATCHES "^\\.\\.")
      message(FATAL_ERROR
          "uniflow_hipify: ${source} is outside UniFlow (${UNIFLOW_SOURCE_DIR})")
    endif()
    set(output "${UNIFLOW_HIPIFY_DIR}/comms/uniflow/${relative}")
    get_filename_component(output_dir "${output}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_dir}")
    add_custom_command(
        OUTPUT "${output}"
        # -o rather than a redirect: a redirect needs a shell, which is not
        # guaranteed, and it truncates the output before hipify runs, so a
        # failed translation would leave an empty source that still looks
        # up to date.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
        COMMAND "${UNIFLOW_HIPIFY_PERL}" "-o=${output}" "${absolute}"
        DEPENDS "${absolute}" "${UNIFLOW_HIPIFY_PERL}"
        COMMENT "hipify ${relative}"
        VERBATIM
    )
    list(APPEND generated "${output}")
  endforeach()

  # A custom command's OUTPUT rule is only visible to targets in the directory
  # that declared it, so wrap the outputs in a target here. Targets elsewhere
  # reach them through add_dependencies, which does cross directories.
  get_property(index GLOBAL PROPERTY UNIFLOW_HIPIFY_INDEX)
  if(NOT index)
    set(index 0)
  endif()
  math(EXPR index "${index} + 1")
  set_property(GLOBAL PROPERTY UNIFLOW_HIPIFY_INDEX ${index})
  add_custom_target(uniflow_hipify_${index} DEPENDS ${generated})
  set_property(GLOBAL APPEND PROPERTY UNIFLOW_HIPIFY_PRODUCERS uniflow_hipify_${index})

  set(${out_var} ${generated} PARENT_SCOPE)
endfunction()

# The translated headers shadow the originals for everything built here, and the
# platform macro goes with them. Applied once at directory scope rather than per
# target: a target that missed it would compile against the CUDA declarations
# while its neighbours used the HIP ones, and the mangled names would match, so
# the link would succeed and the mismatch would surface only at runtime.
if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  include_directories(BEFORE "${UNIFLOW_HIPIFY_DIR}")
  add_compile_definitions(__HIP_PLATFORM_AMD__=1)
endif()

# uniflow_target_hipified(<target>)
#
# Links the HIP runtime into a target whose sources were translated. Kept
# PRIVATE so consumers of this project do not inherit a HIP dependency.
function(uniflow_target_hipified target)
  if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
    return()
  endif()
  target_link_libraries(${target} PRIVATE hip::host)
  set_property(GLOBAL APPEND PROPERTY UNIFLOW_HIPIFY_CONSUMERS ${target})
endfunction()

# uniflow_finalize_hipify()
#
# Call once after every subdirectory has been added.
#
# Translated headers are consumed across directory boundaries through #include,
# which CMake does not see: a rule in one directory produces the header, and a
# target in another compiles against it. Without an explicit edge a parallel
# build can compile before the translation runs, or while it is still writing,
# and the untranslated original is on the include path as a fallback, so the
# failure is a confusing compile error or a silently wrong object rather than a
# missing file.
function(uniflow_finalize_hipify)
  if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
    return()
  endif()
  get_property(producers GLOBAL PROPERTY UNIFLOW_HIPIFY_PRODUCERS)
  get_property(consumers GLOBAL PROPERTY UNIFLOW_HIPIFY_CONSUMERS)
  if(NOT producers OR NOT consumers)
    return()
  endif()
  foreach(target IN LISTS consumers)
    add_dependencies(${target} ${producers})
  endforeach()
endfunction()
