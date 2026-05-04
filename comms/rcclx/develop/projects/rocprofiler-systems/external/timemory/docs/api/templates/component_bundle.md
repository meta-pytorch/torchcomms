# component_bundle

The `component_bundle` variadic bundler is the recommended variadic bundler for
custom interface implementations with timemory when storage is desired. Custom interfaces should create
a unique API struct for their implementation, e.g. via the convenience macro, `TIMEMORY_DECLARE_API(myproject)`
and `TIMEMORY_DEFINE_API(myproject)` will declare/define `tim::api::myproject` and will ensure
`tim::concepts::is_api` is satisfied for your API. Once this is done, bundles of tools can be aliased as such:

```cpp
template <typename... Types>
using timemory_bundle_t = tim::component_bundle<tim::api::myproject, Types...>;

template <typename... Types>
using timemory_auto_bundle_t = tim::auto_bundle<tim::api::myproject, Types...>;
```

Using this scheme, you can disable all timemory instrumentation at compile-time or
run-time via:

```cpp
// disable at compile-time
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, api::myproject, false_type)

// disable at run-time
void disable_timemory_in_myproject()
{
    tim::trait::runtime_enabled<tim::api::myproject>::set(false);
}
```

```eval_rst
.. doxygenclass:: tim::component_bundle
.. doxygenclass:: tim::component_bundle< Tag, Types... >
   :members:
   :undoc-members:
```
