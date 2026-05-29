# C++ Utilities

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

## Scope

A scope configuration handles how the component instances within a bundle
get inserted into the call-graph. The default behavior is `scope::tree`.
This can be combined with `scope::timeline` to form a hierarchical call-graph
where each entry is unique (lots of data): `scope::config cfg = scope::tree{} + scope::timeline{}`.
When the `scope::flat` is used, all component instances become a child
of the root node (i.e. the depth in call-stack is always zero). Similar
to `scope::tree`, `scope::flat` can be combined with `scope::timeline`.


```cpp
using bundle_t  = tim::component_tuple<wall_clock>;
namespace scope = tim::scope;

void foo()
{
    // always tree-scoped
    auto a = bundle_t("foo", scope::tree{});
    // always flat-scoped
    auto b = bundle_t("foo", scope::flat{});
    // always timeline-scoped
    auto c = bundle_t("foo", scope::timeline{});
    // subject to global settings for flat and timeline
    auto c = bundle_t("foo", scope::config{});
}
```

```eval_rst
.. doxygenstruct:: tim::scope::config
.. doxygenstruct:: tim::scope::tree
.. doxygenstruct:: tim::scope::flat
.. doxygenstruct:: tim::scope::timeline
```

## Quirks

Quirks are used to slightly tweak the default behavior of component bundlers.
Quirks can be included as template parameters and the property of the quirk will
be applied to all instances of that component bundle,
e.g. `component_tuple<wall_clock, quirk::flat_scope>` will cause all instances
of that bundler to propagate the flat-storage specification to the wall-clock
component regardless of the global setting. Quirks can also be applied
per-component bundle instance within a `tim::quirk::config<...>` specification, e.g.
an `auto_list` traditionally invokes `start()` on a `component_list` in the constructor
and thus, any attempts to activate the components in the list after construction are
ignored. Thus, if a special initialization case is desired for a particular instance
of `auto_list`, the `quirk::explicit_start` can be added to suppress this behavior so
that initialization can be performed before manually invoking `start()`.

```cpp
namespace quirk = tim::quirk;

using bundle_t = tim::auto_list<wall_clock, cpu_clock>;
using quirk_t  = quirk::config<quirk::explicit_start>;

void foo(bool condition)
{
    auto f = bundle_t("foo", quirk_t{});
    if(condition)
    {
        f.initialize<wall_clock>();
        f.disable<cpu_clock>();
        f.start();
    }
    // ...
}

void bar()
{
    auto f = bundle_t("bar");
    // ...
}
```

```eval_rst
.. doxygenstruct:: tim::quirk::config
.. doxygenstruct:: tim::quirk::auto_start
.. doxygenstruct:: tim::quirk::auto_stop
.. doxygenstruct:: tim::quirk::explicit_start
.. doxygenstruct:: tim::quirk::explicit_stop
.. doxygenstruct:: tim::quirk::explicit_push
.. doxygenstruct:: tim::quirk::explicit_pop
.. doxygenstruct:: tim::quirk::exit_report
.. doxygenstruct:: tim::quirk::no_init
.. doxygenstruct:: tim::quirk::no_store
.. doxygenstruct:: tim::quirk::tree_scope
.. doxygenstruct:: tim::quirk::flat_scope
.. doxygenstruct:: tim::quirk::timeline_scope
```

## Sampling

```eval_rst
.. doxygenfile:: timemory/sampling/sampler.hpp
```

## Conditional

```eval_rst
.. doxygenfile:: timemory/utility/conditional.hpp
```
