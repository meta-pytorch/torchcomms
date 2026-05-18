# Contributing to timemory

This document will primarily cover introducing new components to timemory, however other pull requests for bug-fixes and such are encouraged.
The design of timemory enables there to be two categories for introducing new components to timemory:

1. Components whose code will reside in the timemory source code
2. Components whose definitions will reside in an external library

Either #1 or #2 is acceptable. In general, if the component(s) provide an interface to another tool, it is preferred that you just make the
necessary additions to the build system to find the package, creates a `types.hpp` header which declares the component types and provides
some metadata for the `timemory-avail` tool, and a `extern.hpp` + `extern.cpp` files for instantiating some of the templates when
your tool is enabled. Beyond this, every thing else can be freely modified in your source code. See the
[Version and Interface Compatibility](#version-and-interface-compatibility) section for long-term compatibility concerns.

## Basic Layout

- `source/timemory/components/hello_world/`
- `source/timemory/components/hello_world/CMakeLists.txt`
- `source/timemory/components/hello_world/types.hpp`
- `source/timemory/components/hello_world/backends.hpp` (optional)
- `source/timemory/components/hello_world/extern.hpp` (optional, recommended)
- `source/timemory/components/hello_world/extern.cpp` (optional, recommended)
- `source/timemory/components/hello_world/components.hpp` (optional, recommended)

See also: [source/timemory/components/README.md](source/timemory/components/README.md)

### Step 1. Create `source/timemory/components/hello_world/` folder

Folder containing all the necessary files for the component

### Step 2. `source/timemory/components/hello_world/CMakeLists.txt`

- Add an option: `timemory_add_option(TIMEMORY_USE_HELLO_WORLD "Enable HELLO_WORLD support" OFF)`
- Add an interface library: `timemory_add_interface_library(timemory-hello-world "Enables HELLO_WORLD support")`
  - Use this library to set all the build-flags, include paths, etc.
- Link the interface library to the `timemory-extensions` target
- Use the `timemory_build_intermediate_library(...)` to build the extern templates
- Include this directory in [source/timemory/components/CMakeLists.txt](source/timemory/components/CMakeLists.txt)

```cmake
# use ON if you want the package to be picked up in auto-detect mode
define_default_option(_HELLO_WORLD ON)
timemory_add_option(TIMEMORY_USE_HELLO_WORLD "Enable HELLO_WORLD support" ${_HELLO_WORLD})
timemory_add_interface_library(timemory-hello-world "Enables HELLOW_WORLD support")

if(TIMEMORY_USE_HELLO_WORLD)
    # if TIMEMORY_REQUIRE_PACKAGES=ON and user requested HELLO_WORLD, this will be fatal
    find_package(HelloWorld ${TIMEMORY_FIND_REQUIREMENT})
endif()

# this will only be entered if auto-detect mode is enabled, use FORCE so
# that this is correctly displayed in final report
if(NOT HelloWorld_FOUND)
    set(TIMEMORY_USE_HELLO_WORLD OFF CACHE BOOL "Hello World" FORCE)
    timemory_inform_empty_interface(timemory-hello-world "HelloWorld")
    return()
endif()

if(TIMEMORY_USE_HELLO_WORLD)
    # this ensure libtimemory has library in RPATH
    timemory_add_rpath(${HelloWorld_LIBRARIES})
    target_link_libraries(timemory-hello-world INTERFACE ${HelloWorld_LIBRARIES})
    target_link_directories(timemory-hello-world INTERFACE ${HelloWorld_LIBRARY_DIRS})
    target_include_directories(timemory-hello-world SYSTEM INTERFACE ${HelloWorld_INCLUDE_DIRS})
    target_compile_definitions(timemory-hello-world INTERFACE TIMEMORY_USE_HELLO_WORLD)
    add_target_flag_if_avail(timemory-hello-world "-any-flags-library-might-need")

    set(NAME hello-world)
    set(DEPS timemory-hello-world)

    file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

    timemory_build_intermediate_library(
        USE_INTERFACE
        NAME                ${NAME}
        TARGET              ${NAME}-component
        CATEGORY            COMPONENT
        FOLDER              components
        HEADERS             ${header_files}
        SOURCES             ${source_files}
        DEPENDS             ${DEPS}
        PROPERTY_DEPENDS    GLOBAL)
endif()
```

### Step 3. `source/timemory/components/hello_world/types.hpp`

- Declaration of the component(s)
- Set any neccesary type-traits
- Provide a `TIMEMORY_PROPERTY_SPECIALIZATION` for metadata
  - Will require creating an enumeration ID in [enum.h](source/timemory/enum.h)
    - Place your enumerations in the correct alphabetical place
    - Increment `TIMEMORY_NATIVE_COMPONENT_ENUM_SIZE` by the number of new enum fields
- Include this file in [source/timemory/components/types.hpp](source/timemory/components/types.hpp)

```cpp
#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

// provide an API tag if there are multiple components
// adds empty struct in tim::api namespace
TIMEMORY_DEFINE_API(hello_world)

// declares that there is a component named hello_world_printer in the tim::component namespace
TIMEMORY_DECLARE_COMPONENT(hello_world_printer)

// marks the API and components as unavailable so that a definition is not needed
#if !defined(TIMEMORY_USE_HELLO_WORLD)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, api::hello_world, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::hello_world_printer, false_type)
#endif

// provides mapping from enumeration to C++ type
// provides mapping from strings to C++ type
// information for timemory-avail
TIMEMORY_PROPERTY_SPECIALIZATION(hello_world_printer, HELLO_WORLD_PRINTER, "hello_world_printer", "")
```

### Step 4. `source/timemory/components/hello_world/backends.hpp`

Create any entries here that help with portability, e.g. timemory defines the PAPI presets when PAPI is not enabled
so that components like `papi_tuple<PAPI_L1_TCM>` do not cause compile errors when PAPI is not used.
Include this file in [source/timemory/components/backends.hpp](source/timemory/components/backends.hpp)

### Step 5. `source/timemory/components/hello_world/extern.hpp`

- Provide extern template instantiations, the `TIMEMORY_EXTERN_COMPONENT` macro is useful here
  - `TIMEMORY_EXTERN_COMPONENT(<C++ component type>, <bool for uses storage>, <data type used for storage>)`
    - E.g. the wall-clock timer stores 64-bit integers: `TIMEMORY_EXTERN_COMPONENT(wall_clock, true, int64_t)`
    - E.g. the caliper components handles it's own data: `TIMEMORY_EXTERN_COMPONENT(caliper_marker, false, void)`
- Include this file in [source/timemory/components/extern.hpp](source/timemory/components/extern.hpp)

```cpp
#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/hello_world/components.hpp"
#include "timemory/components/hello_world/types.hpp"

// provides minimum declarations or minimum definitions
#if defined(TIMEMORY_USE_EXTERN) && defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    include "timemory/components/extern/declaration.hpp"
#else
#    include "timemory/components/extern/definition.hpp"
#endif

TIMEMORY_EXTERN_COMPONENT(hello_world_printer, false, void)
```

### Step 6. `source/timemory/components/hello_world/extern.cpp`

Includes the `extern.hpp` header. `TIMEMORY_EXTERN_COMPONENT` will generate the instantiation.

```cpp
#include "timemory/components/hello_world/extern.hpp"
```

### Step 7. `source/timemory/components/hello_world/components.hpp`

The contents of this file depends on whether the component will be defined internally or externally.


- If the component is defined externally, include the timemory header your package provides

```cpp
#pragma once

#if defined(TIMEMORY_USE_HELLO_WORLD)
#    include "hello_world/timemory.hpp"
#endif
```

- If the component is defined internally, provide either the declaration or the full definition
  - If your component is not templated and requires linking to an external library, place large member functions in a `.cpp` file
  - If your component is templated or does not rely on on an external library, the full definition can be placed in the header file

```cpp
#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/hello_world/backends.hpp"
#include "timemory/components/hello_world/types.hpp"

#include <cstdio>

namespace tim
{
namespace component
{
struct hello_world_printer : public base<hello_world_printer, void>
{
    using value_type = void;
    using this_type  = hello_world_printer;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "hello_world_printer"; }
    static std::string description() { return "Prints hello world"; }

    void set_prefix(const char* _cstr) { m_prefix = _cstr; }

    void start() { printf("Hello World! %s has started", m_prefix); }
    void stop()  { printf("Hello World! %s has stopped", m_prefix); }

private:
    const char* m_prefix = nullptr;
};
}
}
```

## Version and Interface Compatibility

In general, version and interface compatibility issues _should be non-existent_. Most version and compatibility issues arise
from API changes and the changes causing these breaks are almost exclusively due to (1) function signature changes as
capabilities evolve and (2) the internal behavior changing in such a way that the extension does not function properly.

### Function Signature Changes

Depending on the package, this comes in three general forms: (1) the original function add/removes parameters or changes the return type,
e.g. `void foo(int)` being changed to `void foo(int, int)`, (2) the original function `void foo(int) is deprecated and it is suggested
that everyone start using `void foo(int, int), or (3) the function signature is opaque (i.e. `void*`) and there is a
new enumeration value for a switch statement which must be handled,
e.g. [CUpti_ActivityKernel5](https://docs.nvidia.com/cupti/Cupti/annotated.html#structCUpti__ActivityKernel5), etc. In general,
timemory is agnostic to return types and parameters being passed to the member functions of a component, it is left entirely
up to the component to decide which arguments a member function can handle and which it cannot -- if a component wants to disable
support for `mark_begin(cudaStream_t)`, then it can just remove the function. If the component gains new capabilities and now requires
`double get() const` to be become `std::vector<double> get() const`, the component can just make the change and the statistical
accumulation will adapt accordingly during compilation.

### Behavior changes

This issue tends to arise when internal changes lead to the extension no longer being invoked in the same way as previous versions.
There are multiple ways to handle these scenarios and ensure that
components retain the desired behavior in the event something changes internally and/or something that was previously available
is no longer available. In general, all that is required is a template specialization which provides the desired behavior and
doing so will fix that behavior for all previous and future versions of timemory.

For example, suppose if was found that the static polymorphic base was defined in such a way that the data
layout could be slightly tweaked to remove one data member for a 50% reduction in overhead and this data member wasn't used by
99% of the components but your component was part of the 1% using it and your component _depended_ on it. The solution is simple:
timemory would make this change and your component would integrate its requirements into the component, possibly even removing the
inheritance entirely since components are not required to inherit from a particular base class -- the base class only exists to
simplify the introduction of new components.

In the scenario where the behavior involves _how_ the component member functions are invoked, all member function invocations
are routed through the constructor of a templated operation class or the function operator of that type and these operation
classes can be specialized to perform the desired behavior.  This feature can be clearly demonstrated by the (fairly unlikely)
scenario provided: `Foo` should call `stop()` when every other component is calling `start()` and vice-versa:

```cpp
namespace tim {
namespace operation {

template <>
struct start<Foo>
{
    start(Foo& obj)
    {
        // call stop when every other tool calls start
        obj.stop();
    }
};

template <>
struct stop<Foo>
{
    start(Foo& obj)
    {
        // call start when every other tool calls stop
        obj.start();
    }
};
}
}
```

Thus, even in the event something changes in timemory and an undesirable behavior is introduced, a component can provide a
specialization to eliminate the issue without affecting any other tools and without requiring a change to the component
itself.

Higher-level specialization on categories are also possible. All specific operations, such as `operation::start<T>`,
are routed through the constructor of `operation::generic_operator<OpT, T, ApiT>`. This type is
responsible for ensuring that the specific operation (`OpT`) can be constructed with a reference to `T`.
The primary purpose of this class is to handle when the instance of `T` is runtime optional, and
therefore is a pointer and requires a nullptr check (might not be enabled). But the secondary purpose is to provide an API
specialization for higher-level modifications to the behavior of multiple components. Thus, you can disable certain
components from being utilized except for when the pre-processor definition `TIMEMORY_API` is `tim::api::SpecialFoo` or
when the components are part of `component_bundle<tim::api::SpecialFoo, ...>`:

```cpp
TIMEMORY_DEFINE_API(SpecialFoo)

namespace tim {
namespace operation {

// define default behavior of Foo to generally do nothing
template <typename Op, typename Tag>
struct generic_operator<Foo, Op, Tag>
{
    template <typename... Args>
    generic_operator(Args&&... args)
    {}
};

// When the API is SpecialFoo, you do something special
template <typename Tp, typename Op>
struct generic_operator<Tp, Op, api::SpecialFoo>
{
    // implementation of custom handling...
};
```

## Conclusion

Thanks for reading! Please file an issue if there are any remaining question or clarification is required.
