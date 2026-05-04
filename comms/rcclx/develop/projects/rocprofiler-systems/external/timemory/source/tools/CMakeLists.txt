# ########################################################################################
#
# timemory tools (stand-alone executables and libraries)
#
# ########################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tools)
set(CMAKE_UNITY_BUILD OFF)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-jump library
#
timemory_message(STATUS "Adding source/tools/timemory-jump...")
add_subdirectory(timemory-jump)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-stubs library
#
timemory_message(STATUS "Adding source/tools/timemory-stubs...")
add_subdirectory(timemory-stubs)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-compiler-instrument library
#
timemory_message(STATUS "Adding source/tools/timemory-compiler-instrument...")
add_subdirectory(timemory-compiler-instrument)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-avail tool
#
timemory_message(STATUS "Adding source/tools/timemory-avail...")
add_subdirectory(timemory-avail)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-pid tool
#
timemory_message(STATUS "Adding source/tools/timemory-pid...")
add_subdirectory(timemory-pid)

# ----------------------------------------------------------------------------------------#
# Build and install timem tool
#
timemory_message(STATUS "Adding source/tools/timemory-timem...")
add_subdirectory(timemory-timem)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-mpip library
#
timemory_message(STATUS "Adding source/tools/timemory-mpip...")
add_subdirectory(timemory-mpip)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-ompt library
#
timemory_message(STATUS "Adding source/tools/timemory-ompt...")
add_subdirectory(timemory-ompt)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-ncclp library
#
timemory_message(STATUS "Adding source/tools/timemory-ncclp...")
add_subdirectory(timemory-ncclp)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-mallocp library
#
timemory_message(STATUS "Adding source/tools/timemory-mallocp...")
add_subdirectory(timemory-mallocp)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
timemory_message(STATUS "Adding source/tools/timemory-run...")
add_subdirectory(timemory-run)

# ----------------------------------------------------------------------------------------#
# Build and install NVML daemon
#
timemory_message(STATUS "Adding source/tools/timemory-nvml...")
add_subdirectory(timemory-nvml)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-ert
#
timemory_message(STATUS "Adding source/tools/timemory-ert...")
add_subdirectory(timemory-ert)

# ----------------------------------------------------------------------------------------#
# Build and install timemory-connector libraries for kokkos
#
timemory_message(STATUS "Adding source/tools/kokkos-connector...")
add_subdirectory(kokkos-connector)
