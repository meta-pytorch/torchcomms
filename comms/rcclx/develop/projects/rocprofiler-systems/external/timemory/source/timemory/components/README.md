# Timemory Components

This README is provided for those who wish to contribute new components to timemory. It describes
the basic folder structure of the components in timemory.

## Overview

The components provided by timemory use a standard folder layout for various reasons:

- Minimize unnecessary header file inclusion
- Avoid inadvertant template instantiations by not providing definition until necessary
- Easy look-up and reference for component names and type-traits

Until the actual component definition is needed (e.g. when instantiating an `tim::operation` class template),
only the declarations of the component types and their type-traits are required.

## Types headers

`timemory/components/types.hpp` is used as a bulk include of all the individual component `types.hpp` headers, e.g.:

```cpp
#include "timemory/components/timing/types.hpp"
#include "timemory/components/trip_count/types.hpp"
#include "timemory/components/user_bundle/types.hpp"
```

The individual `types.hpp` header for components is used to declare component types without providing the definition,
set any type-traits, and specialize the properties. One of these header would provide some or all of the following:

- Declare the components
- Declare any desired aliases
- Define the data type for collecting statistics for any of the components which store data
  - In general, this should match the return type of the `get()` member function
- Define any type-traits
- If the component relies on an external package, it should have a preprocessor definition
  - If this preprocess definition is not defined the `is_available` type-trait should be set to `false_type`
- Define the properties specialization
  - These entries are fixed values for the component because they are used to generate the bindings to C and Python and create lookup tables to identify the components via strings at runtime
  - `TIMEMORY_PROPERTY_SPECIALIZATION([C++ type], [Enumeration ID], [String ID], [Additional string IDs])`
    - This is a variadic macro -- any number of additional string IDs can be specified
- Define the information specialization
  - These entries can be overridden by member functions, if necessary
  - `TIMEMORY_METADATA_SPECIALIZATION([C++ type], [label], [short description], [additional information])`
- Define the default units specialization
  - These entries can be overridden by member functions, if necessary
  - `TIMEMORY_UNITDATA_SPECIALIZATION([C++ type], [numerical units], [string units])`

## Sample Types Header for CUDA components

```cpp
TIMEMORY_DECLARE_COMPONENT(cuda_event)
TIMEMORY_DECLARE_COMPONENT(cuda_profiler)
TIMEMORY_DECLARE_COMPONENT(nvtx_marker)
TIMEMORY_COMPONENT_ALIAS(cuda_nvtx, nvtx_marker)

TIMEMORY_STATISTICS_TYPE(component::cuda_event, float)

TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::cuda_event, true_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::cuda_event, true_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, component::cuda_event,
                               priority_constant<128>)

TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, component::cuda_event,
                               priority_constant<-128>)

TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::nvtx_marker, true_type)

#if !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_event, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cuda_profiler, false_type)
#endif

#if !defined(TIMEMORY_USE_NVTX) || !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::nvtx_marker, false_type)
#endif

// cuda_event can now be identified by CUDA_EVENT enumeration ID
// and a string "cuda_event"
TIMEMORY_PROPERTY_SPECIALIZATION(cuda_event,
    CUDA_EVENT,
    "cuda_event",
    "")

// nvtx_marker can now be identified by NVTX_MARKER enumeration ID
// and the strings "nvtx_marker" and "nvtx"
TIMEMORY_PROPERTY_SPECIALIZATION(nvtx_marker,
    NVTX_MARKER,
    "nvtx_marker",
    "nvtx")

TIMEMORY_PROPERTY_SPECIALIZATION(cuda_profiler,
    CUDA_PROFILER,
    "cuda_profiler",
    "")

// cuda event will generate output files ending in 'cuda_event' + extension
// and this label will be used in logs. A short description of the
// components functionality is provided along with some additional
// information about the accuracy
TIMEMORY_METADATA_SPECIALIZATION(cuda_event,
    "cuda_event",
    "Records the time interval between two points in a CUDA stream.",
    "Less accurate than 'cupti_activity' for kernel timing")

TIMEMORY_METADATA_SPECIALIZATION(nvtx_marker,
    "nvtx_marker",
    "Generates high-level region markers for CUDA profilers",
    "")

TIMEMORY_METADATA_SPECIALIZATION(cuda_profiler,
    "cuda_profiler",
    "Control switch for a CUDA profiler running on the application",
    "")

// the default units for cuda_event are seconds. The 'trait::uses_timing_units'
// will enable unit conversions for this type and 'trait::is_timing_category'
// will enable timing-specific output formatting to be applied
TIMEMORY_UNITDATA_SPECIALIZATION(cuda_event,
    units::sec,
    "sec")
```

## Backend Headers and Sources

These files, generally named `backend.hpp` and `backend.cpp` exist to provide the declarations and/or definitions
for external library calls. These are not strictly necessary but are generally useful for hiding numerous preprocessor
`#ifdef` blocks so that the definitions of components are easy to read.

### Backend Header Example

```cpp
#if defined(TIMEMORY_USE_CUDA)
#    include <cuda.h>
#    include <cuda_fp16.h>
#    include <cuda_fp16.hpp>
#    include <cuda_profiler_api.h>
#    include <cuda_runtime_api.h>
#endif

#if defined(TIMEMORY_USE_CUDA) && (defined(__NVCC__) || defined(__CUDACC__)) &&          \
    (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
#    if !defined(TIMEMORY_CUDA_FP16) && !defined(TIMEMORY_DISABLE_CUDA_HALF)
#        define TIMEMORY_CUDA_FP16
#    endif
#endif

namespace tim
{
namespace cuda
{
inline error_t
peek_at_last_error()
{
    /// gets last error but doesn't reset last error to cudaSuccess
#if defined(TIMEMORY_USE_CUDA)
    return cudaPeekAtLastError();
#else
    return success_v;
#endif
}

inline error_t
get_last_error()
{
    /// gets last error and resets to cudaSuccess
#if defined(TIMEMORY_USE_CUDA)
    return cudaGetLastError();
#else
    return success_v;
#endif
}
}  // namespace cuda
}  // namespace tim

```

`timemory/components/extern.hpp` is used as a bulk include of all the individual component `extern.hpp` headers
which already have template instantiations. The individual component `extern.hpp` header file either declares the
component(s) extern templates, instantiates the extern templates, or (in header-only mode) do nothing.
When the `extern.cpp` file for a component is compiled, the build system defines `TIMEMORY_SOURCE` and `TIMEMORY_COMPONENT_SOURCE`,
this results in the `TIMEMORY_EXTERN_COMPONENT(...)` macro performing an instantiation of the templates instead
of a declaration of an extern template. Thus, `extern.cpp` files tend to only have one line of code: including
the corresponding header file.

### Extern Header Example

```cpp
#include "timemory/components/base.hpp"
#include "timemory/components/macros.hpp"
//
#include "timemory/components/timing/components.hpp"
#include "timemory/components/timing/types.hpp"
//
#if defined(TIMEMORY_COMPONENT_SOURCE) ||                                                \
    (!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_COMPONENT_EXTERN))
// source/header-only requirements
#    include "timemory/environment/declaration.hpp"
#    include "timemory/operations/definition.hpp"
#    include "timemory/plotting/definition.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/storage/definition.hpp"
#else
// extern requirements
#    include "timemory/environment/declaration.hpp"
#    include "timemory/operations/definition.hpp"
#    include "timemory/plotting/declaration.hpp"
#    include "timemory/settings/declaration.hpp"
#    include "timemory/storage/declaration.hpp"
#endif

TIMEMORY_EXTERN_COMPONENT(wall_clock, true, int64_t)
TIMEMORY_EXTERN_COMPONENT(cpu_util, true, std::pair<int64_t, int64_t>>)
```

## Component Header and Sources

The `timemory/components/definition.hpp` header is used as a bulk include of all the individual component `components.hpp` or `definition.hpp` headers.
The individual components `components.hpp`, `definition.hpp`, or `components.cpp` should combine to provide a full definition of the component.

## Component Folder Layout Reivew

The declaration of the components and the type-traits are fully specified via
`#include "timemory/components/<CATEGORY>/types.hpp`.

- `mycomponent/`
  - `types.hpp` : declaration of all component(s), specializations of their type-traits, and the properties specialization (for `timemory-avail`)
  - `components.hpp` : definition of component(s)
  - `backends.hpp` : routines for handling functions used by component(s)
  - `extern.hpp` : header which either declares the components extern templates, instantiates the extern templates, or (in header-only mode) does nothing
  - `extern.cpp` : generally "empty" file with a single `#include` of the corresponding `extern.hpp`
    - When this file is compiled, the build system will define `TIMEMORY_SOURCE` and `TIMEMORY_COMPONENT_SOURCE`, which results in the `TIMEMORY_EXTERN_COMPONENT(...)` macro performing an instantiation of the templates
