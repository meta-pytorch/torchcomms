cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-Array-of-Bundles-Example LANGUAGES CXX)

# set the name of the target you want timemory to add include-directories, libraries, etc.
# to. Just link this library to your target and it will include all the necessary build
# data
set(timemory_FIND_COMPONENTS_INTERFACE timemory-array-of-bundles-example)

# find the timemory package and specify you want the C++ library (cxx)
find_package(
    timemory REQUIRED
    COMPONENTS cxx
    OPTIONAL_COMPONENTS compile-options)

# create your executable
add_executable(ex_array_of_bundles ex_array_of_bundles.cpp)

# link your executable to the interface target created by timemory
target_link_libraries(ex_array_of_bundles PRIVATE timemory-array-of-bundles-example)

# optionally install this executable
install(
    TARGETS ex_array_of_bundles
    DESTINATION bin
    OPTIONAL)
