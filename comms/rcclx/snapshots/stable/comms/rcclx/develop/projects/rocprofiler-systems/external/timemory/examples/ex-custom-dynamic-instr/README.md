# ex-custom-dynamic-instr

This example demonstrates the implementation of a custom instrumentation library for `timemory-run` dynamic instrumentation. The custom instrumentation library may also implement and use one or more custom instrumentation components.

## About custom components

A custom component can utilize the static polymorphic base class `tim::component::base` to inherit many features but ultiy, the goal is to not require a specific base class. The `tim::component::base class` provides the integration into the API and requires two template parameters:
1. component type (i.e. itself)
2. the data type being stored

See more details at [supplemental libraries](../../source/tools/timemory-run/README.md###Supplemental_Libraries)

## timemory-run

See [timemory-run](../../source/tools/timemory-run/README.md).

## Build

See [examples](../README.md##Build).

## Usage

```console
$ timemory-run --load libex_custom_dynamic_instr.so  [OPTIONS] -o [INSTRUMENTED_BINARY] -- [BINARY] [ARGS]
```

See [supplemental libraries](../../source/tools/timemory-run/README.md###Supplemental_Libraries)