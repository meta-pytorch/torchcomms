# C++ Concepts

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

Concepts are used to provide the template meta-programming information
about the intented use of a type. For example, `concepts::is_component<T>::value` is used
to test whether the type `T` is a component and thus, should be continue to
try to provide some functionality for that component. In general, each `concept::is_<CONCEPT>`
can be activated by inheriting from `concepts::<CONCEPT>`. Every concept can be
activated or deactivated by specializing
`concept::is_<CONCEPT>` to inherit from `std::true_type` or `std::false_type`, respectively.
When available, timemory uses the inheritance method so that down-stream users can
perform a template specialization (which overrides the inheritance) is the need arises.

## Inheritable Concepts

```eval_rst
.. doxygenstruct:: tim::concepts::is_empty
   :members:
.. doxygenstruct:: tim::concepts::is_null_type
   :members:
.. doxygenstruct:: tim::concepts::is_placeholder
   :members:
.. doxygenstruct:: tim::concepts::is_component
   :members:
.. doxygenstruct:: tim::concepts::is_quirk_type
   :members:
.. doxygenstruct:: tim::concepts::is_api
   :members:
.. doxygenstruct:: tim::concepts::is_variadic
   :members:
.. doxygenstruct:: tim::concepts::is_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_stack_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_heap_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_mixed_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_tagged
   :members:
.. doxygenstruct:: tim::concepts::is_comp_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_auto_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_runtime_configurable
   :members:
.. doxygenstruct:: tim::concepts::is_external_function_wrapper
   :members:
.. doxygenstruct:: tim::concepts::is_phase_id
   :members:
```

## Non-Inheritable Concepts

```eval_rst
.. doxygenstruct:: tim::concepts::is_acceptable_conversion
   :members:
.. doxygenstruct:: tim::concepts::tuple_type
   :members:
.. doxygenstruct:: tim::concepts::auto_type
   :members:
.. doxygenstruct:: tim::concepts::component_type
   :members:
```
